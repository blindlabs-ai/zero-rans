#![forbid(unsafe_code)]

use zero_rans::{
    compress_lz_split, compress_lz_split_with_config, decode_4way, decode_8way,
    decode_residuals_4x4, decode_residuals_4x4_with_limit, decode_residuals_8x8,
    decode_residuals_8x8_with_limit, decode_with_limit_into, decompress_lz_split,
    decompress_lz_split_with_limit, encode_4way, encode_8way, encode_8way_with_scale,
    encode_order1_8way, encode_residuals_4x4, encode_residuals_8x8, forward_dct_4x4,
    forward_dct_8x8, quantize_block, ContextMap, Error, LzConfig, ScaleBits,
};

/// Deterministic SplitMix64 PRNG (zero external dependencies).
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

/// Precomputed exact Zipfian (`s = 1.2`) inverse-CDF table over 256 byte symbols.
struct Zipf12 {
    cdf: [u64; 256],
    total: u64,
}

impl Zipf12 {
    fn new() -> Self {
        let mut cdf = [0u64; 256];
        let mut acc = 0u64;
        for (rank, slot) in cdf.iter_mut().enumerate() {
            let k = (rank + 1) as f64;
            let w = (1_000_000_000.0 / k.powf(1.2)).round() as u64;
            acc += w.max(1);
            *slot = acc;
        }
        Self { cdf, total: acc }
    }

    fn sample(&self, rng: &mut Rng) -> u8 {
        let target = rng.next_u64() % self.total;
        match self.cdf.binary_search(&target) {
            Ok(i) => (i + 1).min(255) as u8,
            Err(i) => i.min(255) as u8,
        }
    }

    fn fill(&self, rng: &mut Rng, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.sample(rng));
        }
        out
    }
}

/// Generate realistic quantized 4x4 video transform residuals from deterministic synthetic motion deltas.
fn synth_video_residuals_4x4(num_blocks: usize, seed: u64, qstep: i32) -> Vec<[i16; 16]> {
    let mut rng = Rng::new(seed);
    let mut blocks = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        // 55% temporal skip/near-zero blocks, 45% active motion residual blocks.
        if (rng.next_u64() % 100) < 55 {
            blocks.push([0i16; 16]);
            continue;
        }
        let base_dc = ((rng.next_u64() % 41) as i16) - 20;
        let gx = ((rng.next_u64() % 17) as i16) - 8;
        let gy = ((rng.next_u64() % 17) as i16) - 8;
        let mut spatial = [0i16; 16];
        for r in 0..4usize {
            for c in 0..4usize {
                let noise = ((rng.next_u64() % 7) as i16) - 3;
                let edge = if (b & 7) == 0 && c >= 2 { 18 } else { 0 };
                spatial[r * 4 + c] = base_dc + gx * (c as i16) + gy * (r as i16) + noise + edge;
            }
        }
        let tr = forward_dct_4x4(&spatial);
        blocks.push(quantize_block(&tr, qstep));
    }
    blocks
}

/// Generate realistic quantized 8x8 video transform residuals.
fn synth_video_residuals_8x8(num_blocks: usize, seed: u64, qstep: i32) -> Vec<[i16; 64]> {
    let mut rng = Rng::new(seed);
    let mut blocks = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        if (rng.next_u64() % 100) < 50 {
            blocks.push([0i16; 64]);
            continue;
        }
        let base_dc = ((rng.next_u64() % 33) as i16) - 16;
        let gx = ((rng.next_u64() % 9) as i16) - 4;
        let gy = ((rng.next_u64() % 9) as i16) - 4;
        let mut spatial = [0i16; 64];
        for r in 0..8usize {
            for c in 0..8usize {
                let noise = ((rng.next_u64() % 5) as i16) - 2;
                spatial[r * 8 + c] = base_dc + gx * (c as i16) + gy * (r as i16) + noise;
            }
        }
        let tr = forward_dct_8x8(&spatial);
        blocks.push(quantize_block(&tr, qstep));
    }
    blocks
}

#[test]
fn zipfian_10000_deterministic_roundtrips() {
    let zipf = Zipf12::new();
    let mut rng = Rng::new(0xB11D_2A45_CAFE_0001);
    let mut total_cases = 0usize;

    // 10,000+ deterministic Zipfian s=1.2 roundtrips across 8-way (12b/14b), 4-way, and order-1.
    for case in 0..10_000usize {
        let len = match case % 20 {
            0 => 0,
            1 => 1,
            2 => 7,
            3 => 8,
            4 => 9,
            5 => 63,
            6 => 64,
            7 => 512,
            8 => 513,
            _ => 16 + rng.next_usize(384),
        };
        let data = zipf.fill(&mut rng, len);

        match case % 5 {
            0 | 1 => {
                let c = encode_8way(&data).expect("encode_8way");
                let d = decode_8way(&c).expect("decode_8way");
                assert_eq!(d, data, "8way 12b mismatch on case {case} len={len}");
            }
            2 => {
                let c = encode_8way_with_scale(&data, ScaleBits::Bits14).expect("encode_8way 14b");
                let d = decode_8way(&c).expect("decode_8way 14b");
                assert_eq!(d, data, "8way 14b mismatch on case {case} len={len}");
            }
            3 => {
                let c = encode_4way(&data).expect("encode_4way");
                let d = decode_4way(&c).expect("decode_4way");
                assert_eq!(d, data, "4way mismatch on case {case} len={len}");
            }
            _ => {
                let map = if (case & 1) == 0 {
                    ContextMap::HighNibble
                } else {
                    ContextMap::Low15
                };
                let c = encode_order1_8way(&data, map).expect("encode_order1_8way");
                let d = decode_8way(&c).expect("decode_order1_8way");
                assert_eq!(d, data, "order1 mismatch on case {case} len={len}");
            }
        }
        total_cases += 1;
    }

    assert_eq!(total_cases, 10_000);
}

#[test]
fn zipfian_and_lz_and_residual_bitflip_and_truncation_fail_closed() {
    let zipf = Zipf12::new();
    let mut rng = Rng::new(0xDEAD_BEEF_1234_5678);

    // 1. Exhaustive truncation and byte-level bit-flip tests on rANS8 stream.
    let sample = zipf.fill(&mut rng, 1024);
    let rans_wire = encode_8way(&sample).expect("encode rans8");
    assert!(rans_wire.len() < sample.len());

    for cut in 0..rans_wire.len() {
        assert!(
            decode_8way(&rans_wire[..cut]).is_err(),
            "truncated rans8 at {cut} must fail closed"
        );
    }
    let mut trailing = rans_wire.clone();
    trailing.push(0x00);
    assert_eq!(decode_8way(&trailing), Err(Error::TrailingBytes));

    for byte_idx in 0..rans_wire.len() {
        let mut flipped = rans_wire.clone();
        flipped[byte_idx] ^= 1u8 << (byte_idx & 7);
        assert!(
            decode_8way(&flipped).is_err(),
            "bit-flipped rans8 at byte {byte_idx} must fail closed"
        );
    }

    // 2. Exhaustive truncation + bit-flip fail-closed on 4x4 & 8x8 video residual streams.
    let blocks4 = synth_video_residuals_4x4(64, 0x4444, 16);
    let res4_wire = encode_residuals_4x4(&blocks4).expect("encode 4x4");
    for cut in 0..res4_wire.len() {
        assert!(decode_residuals_4x4(&res4_wire[..cut]).is_err());
    }
    for byte_idx in 0..res4_wire.len() {
        let mut flipped = res4_wire.clone();
        flipped[byte_idx] ^= 1u8 << ((byte_idx * 3) & 7);
        assert!(
            decode_residuals_4x4(&flipped).is_err(),
            "bit-flipped 4x4 residual at byte {byte_idx} must fail closed"
        );
    }

    let blocks8 = synth_video_residuals_8x8(32, 0x8888, 24);
    let res8_wire = encode_residuals_8x8(&blocks8).expect("encode 8x8");
    for byte_idx in 0..res8_wire.len() {
        let mut flipped = res8_wire.clone();
        flipped[byte_idx] ^= 1u8 << ((byte_idx * 5) & 7);
        assert!(
            decode_residuals_8x8(&flipped).is_err(),
            "bit-flipped 8x8 residual at byte {byte_idx} must fail closed"
        );
    }

    // 3. Exhaustive truncation + bit-flip fail-closed on split-stream LZ + rANS.
    let mut lz_input = Vec::new();
    for i in 0..64usize {
        lz_input.extend_from_slice(b"tile_residual_packet_");
        lz_input.extend_from_slice(&sample[..32]);
        lz_input.push(i as u8);
    }
    let lz_wire = compress_lz_split(&lz_input).expect("compress lz");
    for cut in 0..lz_wire.len() {
        assert!(decompress_lz_split(&lz_wire[..cut]).is_err());
    }
    for byte_idx in 0..lz_wire.len() {
        let mut flipped = lz_wire.clone();
        flipped[byte_idx] ^= 1u8 << ((byte_idx * 7) & 7);
        assert!(
            decompress_lz_split(&flipped).is_err(),
            "bit-flipped lz_split at byte {byte_idx} must fail closed"
        );
    }

    // 4. Random garbage fuzzing (2,000 malformed frames) must never panic.
    let mut out = Vec::new();
    for _ in 0..2_000usize {
        let len = rng.next_usize(128);
        let mut junk = vec![0u8; len];
        for b in &mut junk {
            *b = rng.next_u64() as u8;
        }
        let _ = decode_with_limit_into(&junk, 4096, &mut out);
        let _ = decode_residuals_4x4_with_limit(&junk, 64);
        let _ = decode_residuals_8x8_with_limit(&junk, 16);
        let _ = decompress_lz_split_with_limit(&junk, 4096);
    }

    // 5. Allocation-bomb header must fail closed immediately with OutputTooLarge.
    let mut bomb = rans_wire.clone();
    bomb[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        decode_with_limit_into(&bomb, 65_536, &mut out),
        Err(Error::OutputTooLarge)
    );
}

#[test]
fn lz_split_offset_boundary_256_and_65535() {
    // Exercise exact lo=255 -> offset=256 carry and max offset=65535 boundaries.
    let mut data = vec![0u8; 70_000];
    let mut rng = Rng::new(0x0FF5_E725_6000_FFFF);
    for b in &mut data {
        *b = (rng.next_u64() & 0xFF) as u8;
    }
    // Plant exact matches at offsets 255, 256, 257, and 65535.
    let pat_a = *b"OFFSET_256_EXACT_CARRY_CHECK_012";
    data[1000..1032].copy_from_slice(&pat_a);
    data[1255..1287].copy_from_slice(&pat_a); // offset 255
    data[1511..1543].copy_from_slice(&pat_a); // offset 256
    data[1768..1800].copy_from_slice(&pat_a); // offset 257

    let pat_b = *b"MAX_WINDOW_OFFSET_65535_BOUNDARY";
    data[2000..2032].copy_from_slice(&pat_b);
    data[2000 + 65_535..2032 + 65_535].copy_from_slice(&pat_b);

    let cfg = LzConfig {
        allow_stored_fallback: false,
        ..LzConfig::default()
    };
    let comp = compress_lz_split_with_config(&data, cfg).expect("compress offset boundaries");
    let dec = decompress_lz_split(&comp).expect("decompress offset boundaries");
    assert_eq!(dec, data);
}

#[test]
fn counterfactual_knockouts_compression_and_integrity() {
    // Knockout 1: Quantized video residuals MUST compress well below raw byte length,
    // whereas incompressible uniform random blocks do NOT shrink (assert_ne!).
    let video_blocks = synth_video_residuals_4x4(2_048, 0xC0DE_4040, 16);
    let raw_video_bytes = video_blocks.len() * 16 * 2;
    let wire_video = encode_residuals_4x4(&video_blocks).expect("encode video 4x4");
    assert!(
        wire_video.len() * 4 < raw_video_bytes,
        "expected >4x compression on quantized 4x4 video residuals, got {} vs {}",
        wire_video.len(),
        raw_video_bytes
    );
    assert_eq!(
        decode_residuals_4x4(&wire_video).expect("decode video 4x4"),
        video_blocks
    );

    let mut rng = Rng::new(0x5555_AAAA_1111_2222);
    let mut uniform_blocks = vec![[0i16; 16]; 2_048];
    for blk in &mut uniform_blocks {
        for slot in blk.iter_mut() {
            *slot = (rng.next_u64() & 0xFFFF) as i16;
        }
    }
    let wire_uniform = encode_residuals_4x4(&uniform_blocks).expect("encode uniform 4x4");
    assert_ne!(wire_video.len(), wire_uniform.len());
    assert!(wire_video.len() * 4 < wire_uniform.len());

    // Knockout 2: Single-bit flip in compressed video residual bitstream MUST fail closed.
    let mut flipped_video = wire_video.clone();
    let mid = flipped_video.len() / 2;
    flipped_video[mid] ^= 0x01;
    let valid_ok = decode_residuals_4x4(&wire_video).is_ok();
    let flipped_ok = decode_residuals_4x4(&flipped_video).is_ok();
    assert_ne!(valid_ok, flipped_ok);
    assert!(valid_ok && !flipped_ok);

    // Knockout 3: Order-1 context modeling strictly beats Order-0 on Markov skip-run data.
    let mut markov = Vec::with_capacity(16_384);
    let mut state = 0u8;
    for _ in 0..16_384 {
        let r = rng.next_u64() % 100;
        state = if state == 0 {
            if r < 92 { 0 } else { 1 + (r as u8 % 6) }
        } else if r < 85 {
            state
        } else {
            0
        };
        markov.push(state);
    }
    let ord0 = encode_8way(&markov).expect("ord0");
    let ord1 = encode_order1_8way(&markov, ContextMap::Low15).expect("ord1");
    assert_ne!(ord0.len(), ord1.len());
    assert!(
        ord1.len() < ord0.len(),
        "order-1 ({}) must beat order-0 ({}) on correlated Markov stream",
        ord1.len(),
        ord0.len()
    );
    assert_eq!(decode_8way(&ord1).expect("decode ord1"), markov);
}
