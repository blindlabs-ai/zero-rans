#![forbid(unsafe_code)]

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use zero_rans::nc_nibble::{
    compress_parallel_blocks, decode_adaptive_nc_nibble, decompress_parallel_blocks,
    encode_adaptive_nc_nibble, MAGIC_PARALLEL_BLOCKS,
};
use zero_rans::{Error, DEFAULT_MAX_DECODE_BYTES};

struct SplitMix64(u64);

impl SplitMix64 {
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

fn zstd19_len(data: &[u8]) -> Option<usize> {
    let mut child = Command::new("zstd")
        .args(["-19", "-c", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(data).ok()?;
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout.len())
}

fn synth_nc_trigram_corpus(len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(0xA1B2_C3D4_0001);
    let mut out = Vec::with_capacity(len);
    let alphabet: [u8; 8] = *b"ABCDEFGH";
    while out.len() + 3 <= len {
        let a_idx = rng.next_usize(8);
        let mut b_idx = rng.next_usize(7);
        if b_idx >= a_idx {
            b_idx += 1;
        }
        let a = alphabet[a_idx];
        let b = alphabet[b_idx];
        let target = if a_idx < b_idx {
            b'a' + (((a_idx * 3 + b_idx * 5) & 15) as u8)
        } else {
            b'q' + (((a_idx * 7 + b_idx * 2) & 7) as u8)
        };
        out.push(a);
        out.push(b);
        out.push(target);
    }
    while out.len() < len {
        out.push(b'A');
    }
    out
}

fn synth_nc_dihedral_d8_walk(len: usize) -> Vec<u8> {
    let mul_d8 = |lhs: u8, rhs: u8| -> u8 {
        let (r1, f1) = (lhs & 3, (lhs >> 2) & 1);
        let (r2, f2) = (rhs & 3, (rhs >> 2) & 1);
        let r = if f1 == 0 {
            (r1 + r2) & 3
        } else {
            (r1 + 4 - r2) & 3
        };
        let f = f1 ^ f2;
        (f << 2) | r
    };
    let gens: [u8; 4] = [1, 3, 4, 5];
    let mut rng = SplitMix64::new(0xD800_CA1E_0002);
    let mut state = 0u8;
    let mut out = Vec::with_capacity(len);
    while out.len() + 2 <= len {
        let g = gens[rng.next_usize(gens.len())];
        state = mul_d8(state, g);
        out.push(b'g' + g);
        out.push(b'0' + state);
    }
    while out.len() < len {
        out.push(b'0');
    }
    out
}

fn synth_nc_dyck_ast(len: usize) -> Vec<u8> {
    let opens: [u8; 3] = [b'{', b'(', b'['];
    let closes: [u8; 3] = [b'}', b')', b']'];
    let payload_alpha: &[u8] = b"abcdefghijklmnopqrstuvwxyz012345";
    let mut rng = SplitMix64::new(0xD1C4_A570_0003);
    let mut stack: Vec<usize> = Vec::with_capacity(16);
    let mut out = Vec::with_capacity(len);

    while out.len() + 16 < len {
        let do_push = stack.is_empty() || (stack.len() < 10 && rng.next_usize(10) < 5);
        if do_push {
            let kind = rng.next_usize(3);
            out.push(opens[kind]);
            stack.push(kind);
            let inner = 2 + rng.next_usize(4);
            for _ in 0..inner {
                out.push(payload_alpha[rng.next_usize(payload_alpha.len())]);
            }
        } else if let Some(kind) = stack.pop() {
            out.push(closes[kind]);
        }
    }
    while let Some(kind) = stack.pop() {
        if out.len() < len {
            out.push(closes[kind]);
        }
    }
    while out.len() < len {
        out.push(b' ');
    }
    out
}

fn synth_nc_stack_bytecode(len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(0xB17E_C0DE_0004);
    let mut out = Vec::with_capacity(len);
    while out.len() + 4 <= len {
        let ra = rng.next_usize(8) as u8;
        let rb = rng.next_usize(8) as u8;
        let op = if ra > rb {
            0x40u8
        } else if ra < rb {
            0x41u8
        } else {
            0x42u8
        };
        let dst = ((ra.wrapping_mul(5).wrapping_add(rb.wrapping_mul(3))) & 7) | 0x80;
        out.push(0x10 | ra);
        out.push(0x20 | rb);
        out.push(op);
        out.push(dst);
    }
    while out.len() < len {
        out.push(0x00);
    }
    out
}

#[test]
fn nc_nibble_nc1_nc4_ratio_speed_and_cross_profile() {
    let corpora: [(&str, Vec<u8>); 4] = [
        ("NC-1 Trigram (AB->X / BA->Y)", synth_nc_trigram_corpus(65_536)),
        ("NC-2 Non-Abelian D_8 Walk", synth_nc_dihedral_d8_walk(65_536)),
        ("NC-3 Nested Dyck/AST Stack", synth_nc_dyck_ast(65_536)),
        ("NC-4 Stack Bytecode (SUB/DIV)", synth_nc_stack_bytecode(65_536)),
    ];

    let mut rng = SplitMix64::new(0x5A0F_F1E0_7777);
    for (name, data) in &corpora {
        // Zen 1-3 L2 profile (`o2_log2=9`, `o3_log2=8`).
        let enc_z2 = encode_adaptive_nc_nibble(data, 9, 8);
        assert_eq!(enc_z2[0], 9);
        assert_eq!(enc_z2[1], 8);
        let dec_z2 = decode_adaptive_nc_nibble(&enc_z2, data.len()).expect("decode z2");
        assert_eq!(&dec_z2, data, "{name}: Zen 1-3 roundtrip mismatch");

        // Zen 4-5 L2 profile (`o2_log2=10`, `o3_log2=9`) decodable bit-exactly on Zen 2.
        let enc_z4 = encode_adaptive_nc_nibble(data, 10, 9);
        assert_eq!(enc_z4[0], 10);
        assert_eq!(enc_z4[1], 9);
        let dec_z4 = decode_adaptive_nc_nibble(&enc_z4, data.len()).expect("decode z4");
        assert_eq!(&dec_z4, data, "{name}: Zen 4-5 cross-profile roundtrip mismatch");

        // Also test boundary table sizes (`o2=6, o3=6` and `o2=12, o3=11`).
        let enc_min = encode_adaptive_nc_nibble(&data[..4096], 6, 6);
        assert_eq!(
            decode_adaptive_nc_nibble(&enc_min, 4096).expect("decode min"),
            &data[..4096]
        );
        let enc_max = encode_adaptive_nc_nibble(&data[..4096], 12, 11);
        assert_eq!(
            decode_adaptive_nc_nibble(&enc_max, 4096).expect("decode max"),
            &data[..4096]
        );

        // Counterfactual knockout: Fisher-Yates shuffle must strictly degrade size.
        let mut shuffled = data.clone();
        for i in (1..shuffled.len()).rev() {
            let j = rng.next_usize(i + 1);
            shuffled.swap(i, j);
        }
        let enc_shuf = encode_adaptive_nc_nibble(&shuffled, 9, 8);
        assert_ne!(enc_z2.len(), enc_shuf.len());
        assert!(
            enc_z2.len() < enc_shuf.len(),
            "{name}: ordered ({} B) must beat shuffled ({} B)",
            enc_z2.len(),
            enc_shuf.len()
        );

        // Timed encode & decode throughput (best of 5 runs after 1 warmup).
        let _ = encode_adaptive_nc_nibble(data, 9, 8);
        let _ = decode_adaptive_nc_nibble(&enc_z2, data.len()).expect("warmup dec");
        let iters = 5usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            let c = encode_adaptive_nc_nibble(data, 9, 8);
            assert_eq!(c.len(), enc_z2.len());
        }
        let enc_sec = t0.elapsed().as_secs_f64();
        let enc_mbps = (data.len() * iters) as f64 / (1_048_576.0 * enc_sec);

        let t1 = Instant::now();
        for _ in 0..iters {
            let d = decode_adaptive_nc_nibble(&enc_z2, data.len()).expect("timed dec");
            assert_eq!(d.len(), data.len());
        }
        let dec_sec = t1.elapsed().as_secs_f64();
        let dec_mbps = (data.len() * iters) as f64 / (1_048_576.0 * dec_sec);

        let z19 = zstd19_len(data).unwrap_or(0);
        let pct = if z19 > 0 {
            100.0 * (enc_z2.len() as f64 - z19 as f64) / (z19 as f64)
        } else {
            0.0
        };
        eprintln!(
            "{name:32} | nib_z2={} B | nib_z4={} B | zstd19={} B ({:+.1}%) | enc={:.1} MB/s | dec={:.1} MB/s",
            enc_z2.len(),
            enc_z4.len(),
            z19,
            pct,
            enc_mbps,
            dec_mbps
        );
    }

    // Verify zstd -19 dominance across all 4 NC corpora.
    for (name, data) in &corpora {
        let enc_z2 = encode_adaptive_nc_nibble(data, 9, 8);
        if let Some(z19) = zstd19_len(data) {
            assert!(
                enc_z2.len() < z19,
                "{name}: nc_nibble ({} B) must beat zstd -19 ({} B)",
                enc_z2.len(),
                z19
            );
        }
    }

    // Fail-closed header / state / trailing-byte checks on nc_nibble.
    let empty_enc = encode_adaptive_nc_nibble(&[], 9, 8);
    assert_eq!(decode_adaptive_nc_nibble(&empty_enc, 0), Ok(Vec::new()));
    let mut bad_hdr = empty_enc.clone();
    bad_hdr[0] = 13;
    assert_eq!(
        decode_adaptive_nc_nibble(&bad_hdr, 0),
        Err(Error::UnsupportedFormat)
    );
    let mut trailing = empty_enc.clone();
    trailing.push(0xAA);
    assert_eq!(
        decode_adaptive_nc_nibble(&trailing, 0),
        Err(Error::TrailingBytes)
    );
}

#[test]
fn brp2_parallel_blocks_determinism_scaling_and_security_laws() {
    // 1. Edge cases: empty, 1 byte, exact multiple, non-multiple.
    for len in [0usize, 1, 7, 1024, 4096, 4097] {
        let data = synth_nc_trigram_corpus(len);
        let f1 = compress_parallel_blocks(&data, 1024, 1, |b| encode_adaptive_nc_nibble(b, 9, 8));
        let f4 = compress_parallel_blocks(&data, 1024, 4, |b| encode_adaptive_nc_nibble(b, 9, 8));
        let f8 = compress_parallel_blocks(&data, 1024, 8, |b| encode_adaptive_nc_nibble(b, 9, 8));
        assert_eq!(&f1[0..4], &MAGIC_PARALLEL_BLOCKS);
        assert_eq!(f1, f4, "len={len}: t=1 must equal t=4");
        assert_eq!(f1, f8, "len={len}: t=1 must equal t=8");

        for dec_threads in [1usize, 4, 8] {
            let dec = decompress_parallel_blocks(&f1, DEFAULT_MAX_DECODE_BYTES, dec_threads, |c, l| {
                decode_adaptive_nc_nibble(c, l)
            })
            .expect("decode brp2");
            assert_eq!(dec, data, "len={len}, dec_threads={dec_threads}");
        }
    }

    // 2. Determinism & multi-core scaling on 512 KiB (128 KiB blocks) and 1 MiB (128 KiB & 256 KiB blocks).
    for &(total_bytes, block_bytes, o2_l, o3_l) in &[
        (512 * 1024usize, 128 * 1024usize, 9u8, 8u8),
        (1024 * 1024usize, 128 * 1024usize, 9u8, 8u8),
        (1024 * 1024usize, 256 * 1024usize, 10u8, 9u8),
    ] {
        let data = synth_nc_stack_bytecode(total_bytes);
        let t_e1 = Instant::now();
        let w1 = compress_parallel_blocks(&data, block_bytes, 1, |b| {
            encode_adaptive_nc_nibble(b, o2_l, o3_l)
        });
        let e1_sec = t_e1.elapsed().as_secs_f64();

        let t_e8 = Instant::now();
        let w8 = compress_parallel_blocks(&data, block_bytes, 8, |b| {
            encode_adaptive_nc_nibble(b, o2_l, o3_l)
        });
        let e8_sec = t_e8.elapsed().as_secs_f64();

        let w12 = compress_parallel_blocks(&data, block_bytes, 12, |b| {
            encode_adaptive_nc_nibble(b, o2_l, o3_l)
        });
        assert_eq!(w1, w8, "BRP2 t=1 vs t=8 determinism violated");
        assert_eq!(w1, w12, "BRP2 t=1 vs t=12 determinism violated");

        let t_d1 = Instant::now();
        let d1 = decompress_parallel_blocks(&w1, total_bytes, 1, |c, l| {
            decode_adaptive_nc_nibble(c, l)
        })
        .expect("dec t=1");
        let d1_sec = t_d1.elapsed().as_secs_f64();

        let t_d8 = Instant::now();
        let d8 = decompress_parallel_blocks(&w8, total_bytes, 8, |c, l| {
            decode_adaptive_nc_nibble(c, l)
        })
        .expect("dec t=8");
        let d8_sec = t_d8.elapsed().as_secs_f64();

        assert_eq!(d1, data);
        assert_eq!(d8, data);

        let mb = total_bytes as f64 / 1_048_576.0;
        eprintln!(
            "BRP2 {} KiB (blk={} KiB, o2={}, o3={}) | comp={} B | enc 1T={:.1} MB/s 8T={:.1} MB/s ({:.2}x) | dec 1T={:.1} MB/s 8T={:.1} MB/s ({:.2}x)",
            total_bytes / 1024,
            block_bytes / 1024,
            o2_l,
            o3_l,
            w1.len(),
            mb / e1_sec,
            mb / e8_sec,
            e1_sec / e8_sec,
            mb / d1_sec,
            mb / d8_sec,
            d1_sec / d8_sec
        );
    }

    // 3. Fail-closed adversarial security checks on BRP2 framing.
    let sample = synth_nc_dihedral_d8_walk(8192);
    let valid = compress_parallel_blocks(&sample, 2048, 4, |b| encode_adaptive_nc_nibble(b, 9, 8));

    // Counterfactual knockout: different block size produces distinct wire bytes.
    let valid_alt =
        compress_parallel_blocks(&sample, 4096, 4, |b| encode_adaptive_nc_nibble(b, 9, 8));
    assert_ne!(valid, valid_alt);

    // Output limit rejection.
    assert_eq!(
        decompress_parallel_blocks(&valid, sample.len() - 1, 4, |c, l| {
            decode_adaptive_nc_nibble(c, l)
        }),
        Err(Error::OutputTooLarge)
    );

    // Bad magic.
    let mut bad_magic = valid.clone();
    bad_magic[0] ^= 0xFF;
    assert_eq!(
        decompress_parallel_blocks(&bad_magic, sample.len(), 2, |c, l| {
            decode_adaptive_nc_nibble(c, l)
        }),
        Err(Error::InvalidMagic)
    );

    // Unsupported flags.
    let mut bad_flags = valid.clone();
    bad_flags[4] = 1;
    assert_eq!(
        decompress_parallel_blocks(&bad_flags, sample.len(), 2, |c, l| {
            decode_adaptive_nc_nibble(c, l)
        }),
        Err(Error::UnsupportedFormat)
    );

    // Header CRC tamper.
    let mut bad_hcrc = valid.clone();
    bad_hcrc[5] ^= 0x01;
    assert_eq!(
        decompress_parallel_blocks(&bad_hcrc, sample.len(), 2, |c, l| {
            decode_adaptive_nc_nibble(c, l)
        }),
        Err(Error::ChecksumMismatch)
    );

    // Trailing garbage rejection.
    let mut trailing = valid.clone();
    trailing.push(0x00);
    assert_eq!(
        decompress_parallel_blocks(&trailing, sample.len(), 2, |c, l| {
            decode_adaptive_nc_nibble(c, l)
        }),
        Err(Error::TrailingBytes)
    );

    // Truncated stream rejection.
    let truncated = &valid[..valid.len() - 1];
    assert!(decompress_parallel_blocks(truncated, sample.len(), 2, |c, l| {
        decode_adaptive_nc_nibble(c, l)
    })
    .is_err());

    // Payload single-bit tamper caught by decoder or per-block CRC.
    let mut bad_payload = valid.clone();
    let mid = bad_payload.len() - 32;
    bad_payload[mid] ^= 0x01;
    assert!(decompress_parallel_blocks(&bad_payload, sample.len(), 4, |c, l| {
        decode_adaptive_nc_nibble(c, l)
    })
    .is_err());

    // Block CRC tamper with identity codec (proves per-block CRC verification directly).
    let raw_id = compress_parallel_blocks(&sample, 2048, 4, |b| b.to_vec());
    let mut raw_corrupt = raw_id.clone();
    let last_idx = raw_corrupt.len() - 1;
    raw_corrupt[last_idx] ^= 0x80;
    assert_eq!(
        decompress_parallel_blocks(&raw_corrupt, sample.len(), 4, |c, _| Ok(c.to_vec())),
        Err(Error::ChecksumMismatch)
    );
}
