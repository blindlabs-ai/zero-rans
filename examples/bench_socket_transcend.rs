#![forbid(unsafe_code)]

use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use zero_rans::{
    compress_lz2d, compress_lz_split, compress_socket_lz2d, decode_8way, decode_socket_8way,
    decode_substream_socket_auto, decompress_lz2d, decompress_lz_split,
    decompress_socket_lz2d_with_limit, encode_8way_with_scale, encode_order1_8way,
    encode_substream_socket_auto, ContextMap, ScaleBits, SocketLz2dConfig,
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

fn mb_per_sec(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / 1_000_000.0) / secs.max(1e-9)
}

fn time_fn<T, F: FnMut() -> T>(iters: usize, mut f: F) -> (T, f64) {
    let warm = f();
    black_box(&warm);
    let t0 = Instant::now();
    let mut last = warm;
    for _ in 0..iters {
        last = black_box(f());
    }
    let elapsed = t0.elapsed().as_secs_f64() / (iters as f64);
    (last, elapsed)
}

/// Honest Oodle-class 1D static-menu baseline: competes `BRL1` 1D split-stream LZ77,
/// order-0 14-bit rANS, order-1 `HighNibble`/`Low15` rANS, and 1D `LitSubRep0` (`rep0 = 1`
/// byte-delta) over `BRL1`/`BRN1` with zero 2D stride, zero caller sockets, and zero `RadixSpan`.
fn encode_oodle_static_1d(data: &[u8]) -> Vec<u8> {
    let mut best = compress_lz_split(data).expect("brl1");
    let o0 = encode_8way_with_scale(data, ScaleBits::Bits14).expect("o0");
    if o0.len() < best.len() {
        best = o0;
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

fn decode_oodle_static_1d(wire: &[u8]) -> Vec<u8> {
    if let Some(rest) = wire.strip_prefix(b"D") {
        let mut out = if rest.starts_with(b"BRL1") {
            decompress_lz_split(rest).expect("dec delta brl1")
        } else {
            decode_8way(rest).expect("dec delta brn1")
        };
        let mut acc = 0u8;
        for b in &mut out {
            acc = acc.wrapping_add(*b);
            *b = acc;
        }
        out
    } else if wire.starts_with(b"BRL1") {
        decompress_lz_split(wire).expect("dec 1d brl1")
    } else {
        decode_8way(wire).expect("dec brn1")
    }
}

fn bench_zlib9(data: &[u8], iters: usize) -> Option<(usize, f64, f64)> {
    let script = format!(
        "import sys, time, zlib\n\
         raw = sys.stdin.buffer.read()\n\
         n = {iters}\n\
         c = zlib.compress(raw, 9)\n\
         t0 = time.perf_counter()\n\
         for _ in range(n): c = zlib.compress(raw, 9)\n\
         te = (time.perf_counter() - t0) / n\n\
         d = zlib.decompress(c)\n\
         assert d == raw\n\
         t1 = time.perf_counter()\n\
         for _ in range(n): d = zlib.decompress(c)\n\
         td = (time.perf_counter() - t1) / n\n\
         print(f'{{len(c)}} {{te:.9f}} {{td:.9f}}')\n"
    );
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
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
    let text = String::from_utf8(out.stdout).ok()?;
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() != 3 {
        return None;
    }
    let len: usize = parts[0].parse().ok()?;
    let te: f64 = parts[1].parse().ok()?;
    let td: f64 = parts[2].parse().ok()?;
    Some((len, mb_per_sec(data.len(), te), mb_per_sec(data.len(), td)))
}

fn bench_zstd19(data: &[u8]) -> Option<usize> {
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
    let enc_out = child.wait_with_output().ok()?;
    if !enc_out.status.success() {
        return None;
    }
    let comp = enc_out.stdout;
    let mut dec_child = Command::new("zstd")
        .args(["-d", "-c", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = dec_child.stdin.take() {
        stdin.write_all(&comp).ok()?;
    }
    let dec_out = dec_child.wait_with_output().ok()?;
    if !dec_out.status.success() || dec_out.stdout != data {
        return None;
    }
    Some(comp.len())
}

/// Workload 1: 2D Raster / UI surface (`256 x 256 = 65,536 B`, `stride = 256`).
fn synth_2d_ui_surface() -> (Vec<u8>, usize) {
    let w = 256usize;
    let h = 256usize;
    let mut buf = vec![0u8; w * h];
    let mut rng = Rng::new(0x2D01_FACE);
    for y in 0..h {
        for x in 0..w {
            let bg = (((x * 3 + y * 5) >> 3) + 20) as u8;
            let v = if y < 24 {
                // Header toolbar with subtle horizontal gradient + button glyphs.
                if (x & 31) < 4 {
                    220u8
                } else {
                    (32 + (x >> 2)) as u8
                }
            } else if ((16..112).contains(&x) || (136..232).contains(&x))
                && (36..140).contains(&y)
            {
                // Two identical UI cards separated horizontally with internal 2D smooth shading.
                let lx = if x >= 136 { x - 136 } else { x - 16 };
                let ly = y - 36;
                if lx == 0 || lx == 95 || ly == 0 || ly == 103 {
                    245u8
                } else if (ly % 16) == 8 && (8..88).contains(&lx) {
                    180u8
                } else {
                    (60 + ((lx + ly * 2) >> 1)) as u8
                }
            } else if (156..244).contains(&y) {
                // Smooth shaded 2D ramp with light sensor texture (+-1).
                let noise = (rng.next_usize(3) as i32) - 1;
                let base = ((x * 5 + y * 7) >> 3) as i32;
                let val = base + noise;
                u8::try_from(val).unwrap_or(if val < 0 { 0 } else { 255 })
            } else {
                bg
            };
            buf[y * w + x] = v;
        }
    }
    (buf, w)
}

/// Workload 2: Structured columnar / record binary (`2,048` records x `32 B = 65,536 B`, `stride = 32`).
fn synth_columnar_records() -> (Vec<u8>, usize) {
    let stride = 32usize;
    let n_rec = 2048usize;
    let mut out = Vec::with_capacity(n_rec * stride);
    let mut rng = Rng::new(0xC010_3232);
    let mut ts = 1_732_000_000_000u64;
    let mut px = 91_250_000i32;
    let tags: [&[u8; 8]; 8] = [
        b"BTC-PERP",
        b"ETH-PERP",
        b"SOL-PERP",
        b"AVAXPERP",
        b"LINKPERP",
        b"ARB-PERP",
        b"SUI-PERP",
        b"DOGEPERP",
    ];
    for i in 0..n_rec {
        ts += 25 + (rng.next_usize(15) as u64);
        px += (rng.next_usize(17) as i32) - 8;
        let qty = ((rng.next_usize(64) + 1) as u32) * 5;
        let tag = tags[(i / 4) & 7];
        let flags = ((i & 7) as u32) | (((i % 19 == 0) as u32) << 8);
        let seq = i as u32;
        out.extend_from_slice(&ts.to_le_bytes());
        out.extend_from_slice(&px.to_le_bytes());
        out.extend_from_slice(&qty.to_le_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&seq.to_le_bytes());
    }
    (out, stride)
}

/// Workload 3: 64-socket mixed multi-regime residual stream (`65,536 B`, `64` causal sockets).
fn synth_64_socket_residuals() -> (Vec<u8>, Vec<u8>) {
    let n = 65_536usize;
    let mut rng = Rng::new(0x6464_BEEF);
    let mut syms = Vec::with_capacity(n);
    let mut sockets = Vec::with_capacity(n);
    for i in 0..n {
        let s = ((i.wrapping_mul(13) ^ (i >> 4)) & 63) as u8;
        sockets.push(s);
        let regime = s & 15;
        let center = regime.wrapping_mul(16);
        let spread = match s >> 4 {
            0 => 3usize,
            1 => 5,
            2 => 9,
            _ => 15,
        };
        let delta = (rng.next_usize(spread) as i32) - ((spread / 2) as i32);
        syms.push(center.wrapping_add(delta as u8));
    }
    (syms, sockets)
}

/// Workload 4: Bounded-span `[-5, +5]` IID uniform noise (`65,536 B`, `span = 11`).
fn synth_bounded_iid_noise() -> Vec<u8> {
    let mut rng = Rng::new(0x11D5_5511);
    (0..65_536)
        .map(|_| 123u8 + (rng.next_usize(11) as u8))
        .collect()
}

struct BenchRow<'a> {
    workload: &'a str,
    raw: usize,
    brl2: usize,
    brs1: usize,
    oodle1d: usize,
    brl1: usize,
    zlib9: Option<usize>,
    zstd19: Option<usize>,
    brl2_enc_mbs: f64,
    brl2_dec_mbs: f64,
}

fn print_row(r: &BenchRow<'_>) {
    let z9_s = r.zlib9.map_or_else(|| "n/a".to_string(), |v| format!("{v}"));
    let zs_s = r.zstd19.map_or_else(|| "n/a".to_string(), |v| format!("{v}"));
    let ratio = (r.raw as f64) / (r.brl2.max(1) as f64);
    let vs_oodle = 100.0 * ((r.brl2 as f64) - (r.oodle1d as f64)) / (r.oodle1d as f64);
    let vs_zstd = r.zstd19.map_or_else(
        || "   n/a".to_string(),
        |z| format!("{:>+6.1}%", 100.0 * ((r.brl2 as f64) - (z as f64)) / (z as f64)),
    );
    println!(
        "| {:<28} | {:>6} | {:>6} ({ratio:>5.2}x) | {vs_oodle:>+6.1}% | {vs_zstd} | {:>6} | {:>6} | {:>6} | {z9_s:>6} | {zs_s:>6} | {:>6.1} | {:>6.1} |",
        r.workload, r.raw, r.brl2, r.brs1, r.oodle1d, r.brl1, r.brl2_enc_mbs, r.brl2_dec_mbs
    );
}

fn bench_zero_cc_kernels() {
    let zcc_bin = std::env::var_os("ZERO_CC_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            [
                "../../../zero-cc/target/release/zero-cc",
                "../zero-cc/target/release/zero-cc",
            ]
            .into_iter()
            .map(PathBuf::from)
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("zero-cc"))
        });
    println!();
    println!("=== zero-cc Pure-Integer 8-Lane Socket Kernel Compilation (`src/socket_kernels.rs`) ===");
    if !zcc_bin.exists() {
        println!("(zero-cc binary not found at {})", zcc_bin.display());
        return;
    }
    println!(
        "| {:<18} | {:<10} | {:<9} | {:>12} | {:>9} | {:>10} | {:<11} | {:>11} |",
        "Kernel", "Ops", "Divs", "Direct Insns", "ELF rel B", "ELF exec B", "Zero Spills", "Compile ms"
    );
    println!("| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: |");

    let kernels = [
        "paeth_pred8",
        "grad_socket8",
        "sub_mod256_8",
        "add_mod256_8",
        "zigzag_u8_8",
    ];
    for k in kernels {
        let out = Command::new(&zcc_bin)
            .args(["src/socket_kernels.rs", "--kernel", k])
            .output()
            .expect("invoke zero-cc");
        assert!(
            out.status.success(),
            "zero-cc failed on {k}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let line = String::from_utf8_lossy(&out.stdout);
        let mut ops = "-";
        let mut divs = "-";
        let mut insns = "-";
        let mut rel_b = "-";
        let mut exec_b = "-";
        let mut spills = "-";
        let mut comp_us = 0.0f64;
        for tok in line.split_whitespace() {
            if let Some(v) = tok.strip_prefix("ops=") {
                ops = v;
            } else if let Some(v) = tok.strip_prefix("divs=") {
                divs = v;
            } else if let Some(v) = tok.strip_prefix("direct_insns=") {
                insns = v;
            } else if let Some(v) = tok.strip_prefix("rel_bytes=") {
                rel_b = v;
            } else if let Some(v) = tok.strip_prefix("exec_bytes=") {
                exec_b = v;
            } else if let Some(v) = tok.strip_prefix("zero_spills=") {
                spills = v;
            } else if let Some(v) = tok.strip_prefix("compile_us=") {
                comp_us = v.parse().unwrap_or(0.0);
            }
        }
        assert_eq!(divs, "0->0", "expected divs=0->0 on {k}");
        assert_eq!(spills, "true", "expected zero_spills=true on {k}");
        println!(
            "| {:<18} | {:<10} | {:<9} | {:>12} | {:>9} | {:>10} | {:<11} | {:>11.3} |",
            k,
            ops,
            divs,
            insns,
            rel_b,
            exec_b,
            spills,
            comp_us / 1000.0
        );
    }
}

fn main() {
    println!("=== zero-rans Socket-LZ2D (`BRL2`) & Clustered Socket rANS (`BRS1`) Benchmark ===");
    println!(
        "| {:<28} | {:>6} | {:>15} | {:>7} | {:>7} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} |",
        "Workload",
        "Raw B",
        "BRL2 (Ratio)",
        "vsOod1D",
        "vsZstd",
        "BRS1 B",
        "Ood1D",
        "BRL1 B",
        "zlib-9",
        "zstd19",
        "EncMBs",
        "DecMBs"
    );
    println!("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");

    // 1. 2D Raster / UI Surface.
    {
        let (data, stride) = synth_2d_ui_surface();
        let (brl2, te) = time_fn(5, || compress_lz2d(&data, stride).expect("brl2 ui"));
        let (dec, td) = time_fn(15, || decompress_lz2d(&brl2, stride).expect("dec brl2 ui"));
        assert_eq!(dec, data);
        let brs1 = encode_substream_socket_auto(&data, None, 1, ScaleBits::Bits14).expect("brs1");
        assert_eq!(decode_substream_socket_auto(&brs1, None, data.len()).expect("d"), data);
        let oodle = encode_oodle_static_1d(&data);
        assert_eq!(decode_oodle_static_1d(&oodle), data);
        let brl1 = compress_lz_split(&data).expect("brl1");
        assert_eq!(decompress_lz_split(&brl1).expect("dec brl1"), data);
        let z9 = bench_zlib9(&data, 3).map(|v| v.0);
        let zs = bench_zstd19(&data);
        print_row(&BenchRow {
            workload: "1. 2D UI Surface (256x256)",
            raw: data.len(),
            brl2: brl2.len(),
            brs1: brs1.len(),
            oodle1d: oodle.len(),
            brl1: brl1.len(),
            zlib9: z9,
            zstd19: zs,
            brl2_enc_mbs: mb_per_sec(data.len(), te),
            brl2_dec_mbs: mb_per_sec(data.len(), td),
        });
    }

    // 2. Structured Columnar / Record Binary.
    {
        let (data, stride) = synth_columnar_records();
        let (brl2, te) = time_fn(5, || compress_lz2d(&data, stride).expect("brl2 col"));
        let (dec, td) = time_fn(15, || decompress_lz2d(&brl2, stride).expect("dec brl2 col"));
        assert_eq!(dec, data);
        let phase_socks: Vec<u8> = (0..data.len()).map(|p| (p % stride) as u8).collect();
        let brs1 =
            encode_substream_socket_auto(&data, Some(&phase_socks), stride, ScaleBits::Bits14)
                .expect("brs1 col");
        assert_eq!(decode_socket_8way(&brs1, &phase_socks).expect("d"), data);
        let oodle = encode_oodle_static_1d(&data);
        assert_eq!(decode_oodle_static_1d(&oodle), data);
        let brl1 = compress_lz_split(&data).expect("brl1");
        assert_eq!(decompress_lz_split(&brl1).expect("dec brl1"), data);
        let z9 = bench_zlib9(&data, 3).map(|v| v.0);
        let zs = bench_zstd19(&data);
        print_row(&BenchRow {
            workload: "2. Columnar Records (32B)",
            raw: data.len(),
            brl2: brl2.len(),
            brs1: brs1.len(),
            oodle1d: oodle.len(),
            brl1: brl1.len(),
            zlib9: z9,
            zstd19: zs,
            brl2_enc_mbs: mb_per_sec(data.len(), te),
            brl2_dec_mbs: mb_per_sec(data.len(), td),
        });
    }

    // 3. 64-Socket Mixed Multi-Regime Residual Stream.
    {
        let (data, sockets) = synth_64_socket_residuals();
        let cfg = SocketLz2dConfig {
            stride: 0,
            min_match: 3,
            arrival_passes: 2,
        };
        let (brl2, te) =
            time_fn(5, || compress_socket_lz2d(&data, Some(&sockets), &cfg).expect("brl2 sock"));
        let (dec, td) = time_fn(20, || {
            decompress_socket_lz2d_with_limit(&brl2, Some(&sockets), 0, data.len())
                .expect("dec brl2 sock")
        });
        assert_eq!(dec, data);
        let brs1 = encode_substream_socket_auto(&data, Some(&sockets), 64, ScaleBits::Bits14)
            .expect("brs1 sock");
        assert_eq!(decode_socket_8way(&brs1, &sockets).expect("d"), data);
        let oodle = encode_oodle_static_1d(&data);
        assert_eq!(decode_oodle_static_1d(&oodle), data);
        let brl1 = compress_lz_split(&data).expect("brl1");
        assert_eq!(decompress_lz_split(&brl1).expect("dec brl1"), data);
        let z9 = bench_zlib9(&data, 3).map(|v| v.0);
        let zs = bench_zstd19(&data);
        print_row(&BenchRow {
            workload: "3. 64-Socket Mixed Residuals",
            raw: data.len(),
            brl2: brl2.len(),
            brs1: brs1.len(),
            oodle1d: oodle.len(),
            brl1: brl1.len(),
            zlib9: z9,
            zstd19: zs,
            brl2_enc_mbs: mb_per_sec(data.len(), te),
            brl2_dec_mbs: mb_per_sec(data.len(), td),
        });
    }

    // 4. Bounded-Span [-5, +5] IID Uniform Noise.
    {
        let data = synth_bounded_iid_noise();
        let (brl2, te) = time_fn(5, || compress_lz2d(&data, 0).expect("brl2 iid"));
        let (dec, td) = time_fn(20, || decompress_lz2d(&brl2, 0).expect("dec brl2 iid"));
        assert_eq!(dec, data);
        let brs1 = encode_substream_socket_auto(&data, None, 1, ScaleBits::Bits14).expect("brs1");
        assert_eq!(decode_substream_socket_auto(&brs1, None, data.len()).expect("d"), data);
        let oodle = encode_oodle_static_1d(&data);
        assert_eq!(decode_oodle_static_1d(&oodle), data);
        let brl1 = compress_lz_split(&data).expect("brl1");
        assert_eq!(decompress_lz_split(&brl1).expect("dec brl1"), data);
        let z9 = bench_zlib9(&data, 3).map(|v| v.0);
        let zs = bench_zstd19(&data);
        print_row(&BenchRow {
            workload: "4. Bounded [-5,+5] IID Noise",
            raw: data.len(),
            brl2: brl2.len(),
            brs1: brs1.len(),
            oodle1d: oodle.len(),
            brl1: brl1.len(),
            zlib9: z9,
            zstd19: zs,
            brl2_enc_mbs: mb_per_sec(data.len(), te),
            brl2_dec_mbs: mb_per_sec(data.len(), td),
        });
    }

    bench_zero_cc_kernels();
}
