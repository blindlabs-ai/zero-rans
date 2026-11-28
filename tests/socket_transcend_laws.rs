#![forbid(unsafe_code)]

use zero_rans::{
    add_mod256_8, compress, compress_lz2d, compress_lz_split, compress_socket_lz2d,
    compress_socket_lz2d_with_effort, compress_with_time_budget, decode_hybrid_uint_socket,
    decode_socket_8way, decode_socket_8way_with_limit, decode_sparse_runs_8way,
    decode_substream_socket_auto, decompress, decompress_lz2d, decompress_socket_lz2d_with_limit,
    encode_8way_with_scale, encode_hybrid_uint_socket, encode_order1_8way, encode_socket_8way,
    encode_sparse_runs_8way, encode_substream_socket_auto, grad_socket8, grad_socket_u8,
    pack_radix_span, paeth_pred8, paeth_pred_u8, sub_mod256_8, unpack_radix_span, zigzag_decode_u32,
    zigzag_encode_i32, zigzag_u8_8, ClusteredSocketRans, ContextMap, Error, ScaleBits,
    SocketLz2dConfig,
};

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

/// Honest Oodle-class 1D static-menu baseline (`BRL1` 1D split-stream LZ77, `O0` 14-bit rANS,
/// `O1-HighNibble`, `O1-Low15`, and 1D `LitSubRep0` byte-delta over `BRL1`/`BRN1` without `RadixSpan`).
fn compress_static_menu_1d_baseline(data: &[u8]) -> Vec<u8> {
    let mut best = compress_lz_split(data).expect("brl1");
    let r0 = encode_8way_with_scale(data, ScaleBits::Bits14).expect("o0");
    if r0.len() < best.len() {
        best = r0;
    }
    let o1h = encode_order1_8way(data, ContextMap::HighNibble).expect("o1h");
    if o1h.len() < best.len() {
        best = o1h;
    }
    let o1l = encode_order1_8way(data, ContextMap::Low15).expect("o1l");
    if o1l.len() < best.len() {
        best = o1l;
    }
    if !data.is_empty() {
        let mut delta = Vec::with_capacity(data.len());
        let mut prev = 0u8;
        for &b in data {
            delta.push(b.wrapping_sub(prev));
            prev = b;
        }
        let d_lz = compress_lz_split(&delta).expect("delta brl1");
        if 1 + d_lz.len() < best.len() {
            let mut tagged = Vec::with_capacity(1 + d_lz.len());
            tagged.push(b'D');
            tagged.extend_from_slice(&d_lz);
            best = tagged;
        }
        let d_o0 = encode_8way_with_scale(&delta, ScaleBits::Bits14).expect("delta o0");
        if 1 + d_o0.len() < best.len() {
            let mut tagged = Vec::with_capacity(1 + d_o0.len());
            tagged.push(b'D');
            tagged.extend_from_slice(&d_o0);
            best = tagged;
        }
    }
    best
}

#[test]
fn socket_kernels_bit_exact_parity_and_bounds() {
    let mut rng = Rng::new(0x50CE_7101);
    let mut seen_buckets = [false; 64];

    for _ in 0..512 {
        let mut w = [0i32; 8];
        let mut n = [0i32; 8];
        let mut nw = [0i32; 8];
        let mut cur = [0i32; 8];
        let mut err = [0i32; 8];
        for k in 0..8usize {
            w[k] = (rng.next_u64() & 255) as i32;
            n[k] = (rng.next_u64() & 255) as i32;
            nw[k] = (rng.next_u64() & 255) as i32;
            cur[k] = (rng.next_u64() & 255) as i32;
            err[k] = ((rng.next_u64() & 255) as i32) - 128;
        }

        let mut pred = [0i32; 8];
        let mut sock = [0i32; 8];
        let mut res = [0i32; 8];
        let mut rec = [0i32; 8];
        let mut zz = [0i32; 8];

        paeth_pred8(&w, &n, &nw, &mut pred);
        grad_socket8(&w, &n, &nw, &mut sock);
        sub_mod256_8(&cur, &pred, &mut res);
        add_mod256_8(&res, &pred, &mut rec);
        zigzag_u8_8(&err, &mut zz);

        for k in 0..8usize {
            assert_eq!(pred[k], i32::from(paeth_pred_u8(w[k] as u8, n[k] as u8, nw[k] as u8)));
            assert_eq!(sock[k], i32::from(grad_socket_u8(w[k] as u8, n[k] as u8, nw[k] as u8)));
            assert!((0..=63).contains(&sock[k]));
            seen_buckets[sock[k] as usize] = true;
            assert_eq!(rec[k], cur[k]);
            let expected_zz = if err[k] >= 0 {
                err[k] * 2
            } else {
                (-err[k]) * 2 - 1
            };
            assert_eq!(zz[k], expected_zz);
        }
    }

    // Verify flat (0, 0) gradient hits exact flat bucket (3 << 3) | 3 == 27 and all 64 buckets exist.
    let flat = [100i32; 8];
    let mut flat_sock = [0i32; 8];
    grad_socket8(&flat, &flat, &flat, &mut flat_sock);
    assert_eq!(flat_sock, [27i32; 8]);

    let diffs = [-40i32, -15, -4, 0, 1, 5, 15, 40];
    for &dh in &diffs {
        for &dv in &diffs {
            let nw = [128i32; 8];
            let w = [128 + dh; 8];
            let n = [128 + dv; 8];
            let mut s = [0i32; 8];
            grad_socket8(&w, &n, &nw, &mut s);
            seen_buckets[s[0] as usize] = true;
        }
    }
    assert!(seen_buckets.iter().all(|&b| b));

    // Exhaustive -128..=127 bijection check on zigzag_u8_8.
    let mut seen_zz = [false; 256];
    for base in (0..256).step_by(8) {
        let mut e = [0i32; 8];
        for (k, slot) in e.iter_mut().enumerate() {
            *slot = (base + k) as i32 - 128;
        }
        let mut z = [0i32; 8];
        zigzag_u8_8(&e, &mut z);
        for k in 0..8 {
            assert!((0..=255).contains(&z[k]));
            seen_zz[z[k] as usize] = true;
        }
    }
    assert!(seen_zz.iter().all(|&b| b));
}

#[test]
fn radix_span_roundtrip_fail_closed_and_beats_rans() {
    let mut rng = Rng::new(0xAA11_5599);

    for &len in &[0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 513, 4099] {
        for &(min_v, span) in &[(0u8, 1usize), (42, 2), (10, 5), (120, 11), (5, 64), (128, 128)] {
            let data: Vec<u8> = (0..len)
                .map(|_| min_v + (rng.next_usize(span) as u8))
                .collect();
            let packed = pack_radix_span(&data).expect("pack_radix_span");
            let unpacked = unpack_radix_span(&packed, len.max(16)).expect("unpack_radix_span");
            assert_eq!(unpacked, data);
        }
    }

    // Span > 128 must return None.
    assert!(pack_radix_span(&[0u8, 128u8]).is_none());

    // Fail-closed checks.
    let sample: Vec<u8> = (0..64).map(|i| 10 + (i % 11) as u8).collect();
    let valid = pack_radix_span(&sample).expect("pack");
    assert_eq!(unpack_radix_span(&valid, 32), Err(Error::OutputTooLarge));
    assert!(unpack_radix_span(&valid[..valid.len() - 1], 128).is_err());
    let mut with_trailing = valid.clone();
    with_trailing.push(0x99);
    assert!(unpack_radix_span(&with_trailing, 128).is_err());
    let mut bad_span = valid.clone();
    bad_span[1] = 128;
    assert!(unpack_radix_span(&bad_span, 128).is_err());
    let mut overflow_range = valid.clone();
    overflow_range[0] = 250;
    overflow_range[1] = 10;
    assert!(unpack_radix_span(&overflow_range, 128).is_err());
    let mut leading_zero = valid.clone();
    leading_zero[3] = 0;
    assert!(unpack_radix_span(&leading_zero, 128).is_err());

    // Counterfactual Knockout #3: RadixSpan strictly beats rANS on bounded-span IID uniform noise.
    for &(span, seed) in &[(5usize, 0x5555_1111u64), (11usize, 0x1111_BBBBu64)] {
        let mut r = Rng::new(seed);
        let noise: Vec<u8> = (0..32_768)
            .map(|_| 100u8 + (r.next_usize(span) as u8))
            .collect();
        let rx = pack_radix_span(&noise).expect("radix pack");
        let r12 = encode_8way_with_scale(&noise, ScaleBits::Bits12).expect("rans12");
        let r14 = encode_8way_with_scale(&noise, ScaleBits::Bits14).expect("rans14");
        assert_eq!(unpack_radix_span(&rx, noise.len()).expect("rx dec"), noise);
        assert!(
            rx.len() < r12.len() && rx.len() < r14.len(),
            "RadixSpan ({}) must strictly beat rANS12 ({}) and rANS14 ({}) on span={span}",
            rx.len(),
            r12.len(),
            r14.len()
        );
        assert_ne!(rx.len(), r12.len());
    }
}

#[test]
fn socket_rans_counterfactual_knockout_and_mdl_cap() {
    let mut rng = Rng::new(0xC0DE_6464);
    let n = 32_768usize;
    let n_sockets = 64usize;

    let mut syms = Vec::with_capacity(n);
    let mut sockets = Vec::with_capacity(n);
    for i in 0..n {
        let s = ((i * 7 + (i >> 5)) % n_sockets) as u8;
        sockets.push(s);
        // Each socket has a distinct sharp centered residual regime.
        let center = s.wrapping_mul(4);
        let noise = ((rng.next_usize(5) as i32) - 2) as i8;
        syms.push(center.wrapping_add(noise as u8));
    }

    let wire_true =
        encode_socket_8way(&syms, &sockets, n_sockets, ScaleBits::Bits14).expect("true sockets");
    let dec_true = decode_socket_8way(&wire_true, &sockets).expect("decode true");
    assert_eq!(dec_true, syms);

    // Counterfactual knockout 1a: zeroed sockets destroy context separation.
    let zero_socks = vec![0u8; n];
    let wire_zero =
        encode_socket_8way(&syms, &zero_socks, n_sockets, ScaleBits::Bits14).expect("zero socks");
    assert_eq!(decode_socket_8way(&wire_zero, &zero_socks).expect("dec zero"), syms);

    // Counterfactual knockout 1b: shuffled sockets destroy context alignment.
    let mut shuf_socks = sockets.clone();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        shuf_socks.swap(i, j);
    }
    let wire_shuf =
        encode_socket_8way(&syms, &shuf_socks, n_sockets, ScaleBits::Bits14).expect("shuf socks");
    assert_eq!(decode_socket_8way(&wire_shuf, &shuf_socks).expect("dec shuf"), syms);

    assert!(
        wire_true.len() * 2 < wire_zero.len(),
        "informative sockets ({}) must crush zeroed sockets ({})",
        wire_true.len(),
        wire_zero.len()
    );
    assert!(
        wire_true.len() * 2 < wire_shuf.len(),
        "informative sockets ({}) must crush shuffled sockets ({})",
        wire_true.len(),
        wire_shuf.len()
    );
    assert_ne!(wire_true.len(), wire_zero.len());
    assert_ne!(wire_true.len(), wire_shuf.len());

    // Hard cluster cap M* <= 32 even with 256 active distinct sockets.
    let socks_256: Vec<u8> = (0..16_384).map(|i| (i & 255) as u8).collect();
    let syms_256: Vec<u8> = socks_256
        .iter()
        .map(|&s| s.wrapping_add((rng.next_usize(3)) as u8))
        .collect();
    let fit = ClusteredSocketRans::fit(&syms_256, &socks_256, 256, ScaleBits::Bits12)
        .expect("fit 256");
    assert!(fit.n_clusters >= 1 && fit.n_clusters <= 32);
    let w256 =
        encode_socket_8way(&syms_256, &socks_256, 256, ScaleBits::Bits12).expect("enc 256");
    assert_eq!(decode_socket_8way(&w256, &socks_256).expect("dec 256"), syms_256);

    // Auto mode strictly bounded on incompressible IID uniform 0..=255 noise.
    let raw_noise: Vec<u8> = (0..4096).map(|_| (rng.next_u64() & 255) as u8).collect();
    let auto_noise =
        encode_substream_socket_auto(&raw_noise, None, 1, ScaleBits::Bits14).expect("auto noise");
    assert!(auto_noise.len() <= raw_noise.len() + 14);
    assert_eq!(
        decode_substream_socket_auto(&auto_noise, None, raw_noise.len()).expect("dec noise"),
        raw_noise
    );

    // Fail-closed corruption checks on BRS1.
    assert_eq!(
        decode_socket_8way_with_limit(&wire_true, &sockets, n - 1),
        Err(Error::OutputTooLarge)
    );
    let mut bad_crc = wire_true.clone();
    let last = bad_crc.len() - 1;
    bad_crc[last] ^= 0xFF;
    assert!(decode_socket_8way(&bad_crc, &sockets).is_err());
    assert!(decode_socket_8way(&wire_true, &sockets[..n - 1]).is_err());
    let mut bad_magic = wire_true.clone();
    bad_magic[0] = b'X';
    assert_eq!(decode_socket_8way(&bad_magic, &sockets), Err(Error::InvalidMagic));
}

#[test]
fn hybrid_uint_and_sparse_runs_roundtrip_and_fail_closed() {
    for &v in &[0i32, -1, 1, -2, 2, 127, -128, 32_767, -32_768, i32::MAX, i32::MIN] {
        assert_eq!(zigzag_decode_u32(zigzag_encode_i32(v)), v);
    }

    let mut rng = Rng::new(0x7788_99AA);
    let mut vals = vec![
        0u32, 1, 2, 15, 16, 17, 31, 32, 63, 64, 255, 256, 65_535, 65_536, 1 << 20, u32::MAX,
    ];
    while vals.len() < 4096 {
        let s = vals.len() & 15;
        let v = if s < 8 {
            (rng.next_usize(12)) as u32
        } else {
            (rng.next_u64() & ((1u64 << (s + 4)) - 1)) as u32
        };
        vals.push(v);
    }
    let sockets: Vec<u8> = (0..vals.len()).map(|i| (i & 15) as u8).collect();
    let h_wire = encode_hybrid_uint_socket(&vals, &sockets, 16).expect("enc hybrid");
    let h_dec = decode_hybrid_uint_socket(&h_wire, &sockets, vals.len()).expect("dec hybrid");
    assert_eq!(h_dec, vals);
    assert_eq!(
        decode_hybrid_uint_socket(&h_wire, &sockets, vals.len() - 1),
        Err(Error::OutputTooLarge)
    );
    let mut h_bad = h_wire.clone();
    let idx = h_bad.len() - 1;
    h_bad[idx] ^= 0x01;
    assert!(decode_hybrid_uint_socket(&h_bad, &sockets, vals.len()).is_err());

    // Sparse runs with escapes (> 254) and extreme i16 values (including 0, i16::MIN, i16::MAX).
    let mut block_nnz = Vec::with_capacity(512);
    let mut gaps = Vec::new();
    let mut levels = Vec::new();
    for b in 0..512usize {
        let nnz = if b == 7 {
            260u16
        } else if (b % 3) == 0 {
            0u16
        } else {
            ((b % 7) + 1) as u16
        };
        block_nnz.push(nnz);
        for j in 0..usize::from(nnz) {
            let g = if j == 0 && b == 7 { 300u16 } else { (j % 5) as u16 };
            let lv = match (b + j) % 9 {
                0 => 0i16,
                1 => 1,
                2 => -1,
                3 => 127,
                4 => -128,
                5 => 400,
                6 => -900,
                7 => i16::MAX,
                _ => i16::MIN,
            };
            gaps.push(g);
            levels.push(lv);
        }
    }
    let s_wire = encode_sparse_runs_8way(&block_nnz, &gaps, &levels).expect("enc sparse");
    let (d_nnz, d_gaps, d_lvls) =
        decode_sparse_runs_8way(&s_wire, block_nnz.len(), gaps.len()).expect("dec sparse");
    assert_eq!(d_nnz, block_nnz);
    assert_eq!(d_gaps, gaps);
    assert_eq!(d_lvls, levels);
    assert_eq!(
        decode_sparse_runs_8way(&s_wire, block_nnz.len() - 1, gaps.len()),
        Err(Error::OutputTooLarge)
    );
}

#[test]
fn brl2_roundtrip_fail_closed_and_beats_1d_and_static_menu() {
    // Edge lengths.
    for len in 0..=24usize {
        let data: Vec<u8> = (0..len).map(|i| ((i * 19 + 3) & 255) as u8).collect();
        let c = compress_lz2d(&data, 8).expect("short brl2");
        assert_eq!(decompress_lz2d(&c, 8).expect("dec short brl2"), data);
    }

    // 1. 2D raster/UI surface (256 x 256 = 65,536 B, stride = 256).
    let w = 256usize;
    let h = 256usize;
    let mut surface = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 2 + y * 3) >> 2) as u8;
            let val = if ((32..96).contains(&x) || (128..192).contains(&x))
                && (24..120).contains(&y)
            {
                let lx = x & 63;
                let ly = y.wrapping_sub(24);
                if lx == 0 || lx == 63 || ly == 0 || ly == 95 {
                    240u8
                } else {
                    (40 + ((lx + ly * 2) & 63)) as u8
                }
            } else if (150..230).contains(&y) {
                // Smooth diagonal gradient with vertical correlation.
                ((x.wrapping_mul(3) + y.wrapping_mul(5)) >> 3) as u8
            } else {
                base
            };
            surface[y * w + x] = val;
        }
    }

    let brl2_2d = compress_lz2d(&surface, w).expect("brl2 2d");
    assert_eq!(decompress_lz2d(&brl2_2d, w).expect("dec brl2 2d"), surface);

    let brl2_1d = compress_lz2d(&surface, 0).expect("brl2 1d");
    let brl1 = compress_lz_split(&surface).expect("brl1");
    let static_1d = compress_static_menu_1d_baseline(&surface);

    assert!(
        brl2_2d.len() < brl2_1d.len(),
        "BRL2 2D ({}) must strictly beat BRL2 1D ({})",
        brl2_2d.len(),
        brl2_1d.len()
    );
    assert!(
        brl2_2d.len() < brl1.len(),
        "BRL2 2D ({}) must strictly beat BRL1 ({})",
        brl2_2d.len(),
        brl1.len()
    );
    assert!(
        brl2_2d.len() < static_1d.len(),
        "BRL2 2D ({}) must strictly beat static-menu 1D baseline ({})",
        brl2_2d.len(),
        static_1d.len()
    );
    assert_ne!(brl2_2d.len(), brl2_1d.len());

    // 2. Structured columnar binary table (1024 records x 32 B, stride = 32).
    let mut rng = Rng::new(0xB00C_3232);
    let mut table = Vec::with_capacity(1024 * 32);
    let mut ts = 1_700_000_000u64;
    let mut price = 50_000i32;
    let symbols: [&[u8; 8]; 4] = [b"BTCUSDT ", b"ETHUSDT ", b"SOLUSDT ", b"AVAXUSDT"];
    for i in 0..1024usize {
        ts += 10 + (rng.next_usize(7) as u64);
        price += (rng.next_usize(9) as i32) - 4;
        let vol = (rng.next_usize(120) as u32) * 10;
        let sym = symbols[i & 3];
        let flags = (i & 3) as u32;
        let seq = i as u32;
        table.extend_from_slice(&ts.to_le_bytes());
        table.extend_from_slice(&price.to_le_bytes());
        table.extend_from_slice(&vol.to_le_bytes());
        table.extend_from_slice(sym);
        table.extend_from_slice(&flags.to_le_bytes());
        table.extend_from_slice(&seq.to_le_bytes());
    }
    let brl2_col = compress_lz2d(&table, 32).expect("brl2 col");
    assert_eq!(decompress_lz2d(&brl2_col, 32).expect("dec col"), table);
    let static_col = compress_static_menu_1d_baseline(&table);
    assert!(
        brl2_col.len() < static_col.len(),
        "BRL2 columnar ({}) must strictly beat 1D static menu ({})",
        brl2_col.len(),
        static_col.len()
    );

    // Fail-closed checks on BRL2.
    assert_eq!(
        decompress_socket_lz2d_with_limit(&brl2_2d, None, w, surface.len() - 1),
        Err(Error::OutputTooLarge)
    );
    assert_eq!(
        decompress_socket_lz2d_with_limit(&brl2_2d, None, w + 1, surface.len()),
        Err(Error::InvalidParameter)
    );
    let mut bad = brl2_2d.clone();
    let idx = bad.len() - 1;
    bad[idx] ^= 0x80;
    assert!(decompress_lz2d(&bad, w).is_err());

    // 3. Workload 4 counterfactual: Bounded [-5, +5] IID uniform noise (span = 11).
    let mut iid_rng = Rng::new(0x11D5_5511);
    let iid: Vec<u8> = (0..65_536)
        .map(|_| 123u8 + (iid_rng.next_usize(11) as u8))
        .collect();
    let cfg_1d = SocketLz2dConfig {
        stride: 0,
        min_match: 3,
        arrival_passes: 2,
    };
    let brl2_iid = compress_socket_lz2d(&iid, None, &cfg_1d).expect("brl2 iid");
    assert_eq!(
        decompress_socket_lz2d_with_limit(&brl2_iid, None, 0, iid.len()).expect("dec brl2 iid"),
        iid
    );
    let brs1_iid =
        encode_substream_socket_auto(&iid, None, 1, ScaleBits::Bits14).expect("brs1 iid");
    assert_eq!(
        decode_substream_socket_auto(&brs1_iid, None, iid.len()).expect("dec brs1 iid"),
        iid
    );
    let oodle_iid = compress_static_menu_1d_baseline(&iid);
    assert!(
        brs1_iid.len() < oodle_iid.len(),
        "BRS1 ({}) must strictly beat static-menu 1D ({}) on bounded [-5,+5] IID noise",
        brs1_iid.len(),
        oodle_iid.len()
    );
    assert!(
        brl2_iid.len() < oodle_iid.len(),
        "BRL2 ({}) must strictly beat static-menu 1D ({}) on bounded [-5,+5] IID noise",
        brl2_iid.len(),
        oodle_iid.len()
    );
    assert_ne!(brs1_iid.len(), oodle_iid.len());
    assert_ne!(brl2_iid.len(), oodle_iid.len());
}

#[test]
fn unified_1d_adaptive_o1_and_time_budget_knockout() {
    let sample = include_bytes!("../src/rans8.rs");
    let wire = compress(sample).expect("compress 1d");
    assert_eq!(decompress(&wire).expect("decompress 1d"), sample);

    // Fail-closed bitflip inside adaptive order-1 / compact sequence substreams.
    let mut bad = wire.clone();
    let last = bad.len() - 5;
    bad[last] ^= 0x80;
    assert!(decompress(&bad).is_err());

    // Counterfactual effort/budget knockout: deeper tier strictly shrinks compressed data.
    let t1 = compress_socket_lz2d_with_effort(
        sample,
        None,
        &SocketLz2dConfig {
            stride: 0,
            min_match: 3,
            arrival_passes: 1,
        },
        4,
    )
    .expect("t1");
    let t4 = compress_socket_lz2d_with_effort(
        sample,
        None,
        &SocketLz2dConfig {
            stride: 0,
            min_match: 3,
            arrival_passes: 3,
        },
        256,
    )
    .expect("t4");
    assert_ne!(t1.len(), t4.len());
    assert!(
        t4.len() < t1.len(),
        "deeper effort tier must strictly shrink compressed data: t4={} vs t1={}",
        t4.len(),
        t1.len()
    );

    let (budget_wire, tier) = compress_with_time_budget(sample, 0, 500_000).expect("budget");
    assert!(tier >= 2);
    assert_eq!(decompress(&budget_wire).expect("dec budget"), sample);
}
