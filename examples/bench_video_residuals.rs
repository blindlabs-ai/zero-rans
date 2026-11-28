#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use zero_rans::{
    compress_lz_split, decode_4way, decode_8way, decode_residuals_4x4, decode_residuals_8x8,
    decode_video_frame, decompress_lz_split, encode_4way, encode_8way, encode_8way_with_scale,
    encode_order1_8way, encode_residuals_4x4, encode_residuals_8x8, encode_video_frame,
    forward_dct_4x4, forward_dct_8x8, quantize_block, AdaptiveStreamConfig, AdaptiveStreamDecoder,
    AdaptiveStreamEncoder, CodecProfile, ContextMap, FrameKind, ScaleBits, VideoConfig, Yuv420Frame,
};

struct BenchRow {
    name: String,
    raw_bytes: usize,
    comp_bytes: usize,
    enc_mbs: f64,
    dec_mbs: f64,
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

/// Optional real C `libz` (Deflate) baseline via system `python3` (zero Cargo dependencies).
fn bench_python_zlib(data: &[u8], level: u8, iters: usize) -> Option<(usize, f64, f64)> {
    let script = format!(
        "import sys, time, zlib\n\
         raw = sys.stdin.buffer.read()\n\
         lvl = {level}\n\
         n = {iters}\n\
         c = zlib.compress(raw, lvl)\n\
         t0 = time.perf_counter()\n\
         for _ in range(n): c = zlib.compress(raw, lvl)\n\
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
    let comp_len: usize = parts[0].parse().ok()?;
    let enc_s: f64 = parts[1].parse().ok()?;
    let dec_s: f64 = parts[2].parse().ok()?;
    Some((
        comp_len,
        mb_per_sec(data.len(), enc_s),
        mb_per_sec(data.len(), dec_s),
    ))
}

/// Deterministic Zipfian s=1.2 byte corpus (`4 MiB`).
fn generate_zipf_corpus(len: usize) -> Vec<u8> {
    let mut cdf = [0u64; 256];
    let mut acc = 0u64;
    for (rank, slot) in cdf.iter_mut().enumerate() {
        let k = (rank + 1) as f64;
        let w = (1_000_000_000.0 / k.powf(1.2)).round() as u64;
        acc += w.max(1);
        *slot = acc;
    }
    let mut state = 0x5A1F_1200_CAFE_BABEu64;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let target = (z ^ (z >> 31)) % acc;
        let sym = match cdf.binary_search(&target) {
            Ok(i) => (i + 1).min(255) as u8,
            Err(i) => i.min(255) as u8,
        };
        out.push(sym);
    }
    out
}

/// Synthesize realistic motion-compensated 1080p grayscale video frames (8 frames).
fn synthesize_video_frames(width: usize, height: usize, frames: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(frames);
    let mut base = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let grad = ((x * 180) / width + (y * 90) / height) as i32;
            let checker = if ((x >> 5) ^ (y >> 5)) & 1 == 0 { 14 } else { -14 };
            let wave = (((x as f64 * 0.035).sin() * (y as f64 * 0.025).cos()) * 28.0) as i32;
            let val = grad + checker + wave;
            base[y * width + x] = if val < 0 {
                0
            } else if val > 255 {
                255
            } else {
                val as u8
            };
        }
    }

    for t in 0..frames {
        let mut frame = vec![0u8; width * height];
        let pan_x = t * 2;
        let box_x = (width / 4) + t * 18;
        let box_y = (height / 3) + t * 6;

        for y in 0..height {
            for x in 0..width {
                let sx = (x + pan_x) % width;
                let mut v = i32::from(base[y * width + sx]);
                // Moving crisp foreground object with high-frequency edge detail.
                if x >= box_x && x < box_x + 240 && y >= box_y && y < box_y + 140 {
                    let lx = x - box_x;
                    let ly = y - box_y;
                    v = if !(8..232).contains(&lx) || !(8..132).contains(&ly) {
                        245
                    } else if ((lx / 16) + (ly / 16)) & 1 == 0 {
                        210
                    } else {
                        40
                    };
                }
                // Subtle camera sensor noise (+/- 1 luma level).
                let h = ((x as u64).wrapping_mul(0x9E37_79B9)
                    ^ (y as u64).wrapping_mul(0xBF58_476D)
                    ^ (t as u64).wrapping_mul(0x94D0_49BB))
                    >> 60;
                let noise = (h as i32 % 3) - 1;
                let p = v + noise;
                frame[y * width + x] = if p < 0 {
                    0
                } else if p > 255 {
                    255
                } else {
                    p as u8
                };
            }
        }
        out.push(frame);
    }
    out
}

/// Load Y-plane frames from raw YUV420p or packed Y8 file (`--yuv <path> <w> <h> <frames>`).
fn load_yuv420_luma_frames(path: &str, width: usize, height: usize, frames: usize) -> Vec<Vec<u8>> {
    let raw = fs::read(path).expect("failed to read YUV file");
    let y_bytes = width * height;
    let yuv420_frame_bytes = y_bytes + (y_bytes / 2);
    let stride = if raw.len() >= yuv420_frame_bytes * frames {
        yuv420_frame_bytes
    } else if raw.len() >= y_bytes * frames {
        y_bytes
    } else {
        panic!("YUV file too small for requested {width}x{height}x{frames}");
    };
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * stride;
        out.push(raw[start..start + y_bytes].to_vec());
    }
    out
}

/// Extract motion-compensated temporal deltas and quantized 4x4 + 8x8 DCT residual blocks.
fn build_video_residuals(
    frames: &[Vec<u8>],
    width: usize,
    height: usize,
    qstep4: i32,
    qstep8: i32,
) -> (Vec<u8>, Vec<[i16; 16]>, Vec<[i16; 64]>) {
    let mut raw_deltas = Vec::with_capacity(width * height * frames.len().saturating_sub(1));
    let mut blocks4 = Vec::new();
    let mut blocks8 = Vec::new();

    let bw4 = width / 4;
    let bh4 = height / 4;
    let bw8 = width / 8;
    let bh8 = height / 8;

    for pair in frames.windows(2) {
        let prev = &pair[0];
        let curr = &pair[1];
        let mut delta = vec![0i16; width * height];
        for i in 0..delta.len() {
            let d = i16::from(curr[i]) - i16::from(prev[i]);
            delta[i] = d;
            // Zigzag-mapped signed byte residual for general LZ/rANS comparison.
            let zz = ((d << 1) ^ (d >> 15)) as u16;
            raw_deltas.push(if zz > 255 { 255 } else { zz as u8 });
        }

        for by in 0..bh4 {
            for bx in 0..bw4 {
                let mut blk = [0i16; 16];
                for r in 0..4 {
                    let row_off = (by * 4 + r) * width + bx * 4;
                    for c in 0..4 {
                        blk[r * 4 + c] = delta[row_off + c];
                    }
                }
                let tr = forward_dct_4x4(&blk);
                blocks4.push(quantize_block(&tr, qstep4));
            }
        }

        for by in 0..bh8 {
            for bx in 0..bw8 {
                let mut blk = [0i16; 64];
                for r in 0..8 {
                    let row_off = (by * 8 + r) * width + bx * 8;
                    for c in 0..8 {
                        blk[r * 8 + c] = delta[row_off + c];
                    }
                }
                let tr = forward_dct_8x8(&blk);
                blocks8.push(quantize_block(&tr, qstep8));
            }
        }
    }

    (raw_deltas, blocks4, blocks8)
}

fn blocks_to_le_bytes<const N: usize>(blocks: &[[i16; N]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks.len() * N * 2);
    for blk in blocks {
        for &v in blk {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn print_table(title: &str, rows: &[BenchRow]) {
    println!("\n=== {title} ===");
    println!(
        "{:<36} | {:>10} | {:>10} | {:>7} | {:>10} | {:>10}",
        "Codec / Workload", "Raw (B)", "Comp (B)", "Ratio", "Enc MB/s", "Dec MB/s"
    );
    println!("{}", "-".repeat(98));
    for r in rows {
        let ratio = (r.raw_bytes as f64) / (r.comp_bytes.max(1) as f64);
        println!(
            "{:<36} | {:>10} | {:>10} | {:>6.2}x | {:>10.1} | {:>10.1}",
            r.name, r.raw_bytes, r.comp_bytes, ratio, r.enc_mbs, r.dec_mbs
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (label, width, height, frames) = if args.len() == 6 && args[1] == "--yuv" {
        let path = &args[2];
        let w: usize = args[3].parse().expect("width");
        let h: usize = args[4].parse().expect("height");
        let n: usize = args[5].parse().expect("frames");
        (
            format!("External YUV {w}x{h} ({n} frames)"),
            w,
            h,
            load_yuv420_luma_frames(path, w, h, n),
        )
    } else {
        let (w, h, n) = (1280usize, 720usize, 6usize);
        (
            format!("Synthetic 720p Motion Sequence {w}x{h} ({n} frames)"),
            w,
            h,
            synthesize_video_frames(w, h, n),
        )
    };

    // 1. Pure Entropy Coding Benchmark (4 MiB Zipfian s=1.2).
    let zipf = generate_zipf_corpus(4 * 1024 * 1024);
    let mut zipf_rows = Vec::new();

    let (c8_12, te) = time_fn(10, || encode_8way(&zipf).expect("enc8 12b"));
    let (d8_12, td) = time_fn(20, || decode_8way(&c8_12).expect("dec8 12b"));
    assert_eq!(d8_12, zipf);
    zipf_rows.push(BenchRow {
        name: "zero-rans rans8 (8-way, 12-bit)".to_string(),
        raw_bytes: zipf.len(),
        comp_bytes: c8_12.len(),
        enc_mbs: mb_per_sec(zipf.len(), te),
        dec_mbs: mb_per_sec(zipf.len(), td),
    });

    let (c8_14, te) = time_fn(10, || {
        encode_8way_with_scale(&zipf, ScaleBits::Bits14).expect("enc8 14b")
    });
    let (d8_14, td) = time_fn(20, || decode_8way(&c8_14).expect("dec8 14b"));
    assert_eq!(d8_14, zipf);
    zipf_rows.push(BenchRow {
        name: "zero-rans rans8 (8-way, 14-bit)".to_string(),
        raw_bytes: zipf.len(),
        comp_bytes: c8_14.len(),
        enc_mbs: mb_per_sec(zipf.len(), te),
        dec_mbs: mb_per_sec(zipf.len(), td),
    });

    let (c4_12, te) = time_fn(10, || encode_4way(&zipf).expect("enc4 12b"));
    let (d4_12, td) = time_fn(20, || decode_4way(&c4_12).expect("dec4 12b"));
    assert_eq!(d4_12, zipf);
    zipf_rows.push(BenchRow {
        name: "zero-rans rans4 (4-way, 12-bit)".to_string(),
        raw_bytes: zipf.len(),
        comp_bytes: c4_12.len(),
        enc_mbs: mb_per_sec(zipf.len(), te),
        dec_mbs: mb_per_sec(zipf.len(), td),
    });

    if let Some((clen, enc_mbs, dec_mbs)) = bench_python_zlib(&zipf, 1, 3) {
        zipf_rows.push(BenchRow {
            name: "zlib/Deflate level 1 (C libz)".to_string(),
            raw_bytes: zipf.len(),
            comp_bytes: clen,
            enc_mbs,
            dec_mbs,
        });
    }
    if let Some((clen, enc_mbs, dec_mbs)) = bench_python_zlib(&zipf, 6, 3) {
        zipf_rows.push(BenchRow {
            name: "zlib/Deflate level 6 (C libz)".to_string(),
            raw_bytes: zipf.len(),
            comp_bytes: clen,
            enc_mbs,
            dec_mbs,
        });
    }
    print_table("Zipfian s=1.2 Entropy Benchmark (4 MiB)", &zipf_rows);

    // 2. Real Quantized Video Transform Residuals (4x4, 8x8, and Frame Deltas).
    let (raw_deltas, blocks4, blocks8) = build_video_residuals(&frames, width, height, 24, 32);
    let raw4_bytes = blocks_to_le_bytes(&blocks4);
    let raw8_bytes = blocks_to_le_bytes(&blocks8);

    let mut vid_rows = Vec::new();

    let (r4_c, te) = time_fn(8, || encode_residuals_4x4(&blocks4).expect("enc res4"));
    let (r4_d, td) = time_fn(15, || decode_residuals_4x4(&r4_c).expect("dec res4"));
    assert_eq!(r4_d, blocks4);
    vid_rows.push(BenchRow {
        name: "zero-rans 4x4 residual (split+rans8)".to_string(),
        raw_bytes: raw4_bytes.len(),
        comp_bytes: r4_c.len(),
        enc_mbs: mb_per_sec(raw4_bytes.len(), te),
        dec_mbs: mb_per_sec(raw4_bytes.len(), td),
    });

    if let Some((clen, enc_mbs, dec_mbs)) = bench_python_zlib(&raw4_bytes, 6, 3) {
        vid_rows.push(BenchRow {
            name: "zlib level 6 on raw 4x4 i16 blocks".to_string(),
            raw_bytes: raw4_bytes.len(),
            comp_bytes: clen,
            enc_mbs,
            dec_mbs,
        });
    }

    let (r8_c, te) = time_fn(8, || encode_residuals_8x8(&blocks8).expect("enc res8"));
    let (r8_d, td) = time_fn(15, || decode_residuals_8x8(&r8_c).expect("dec res8"));
    assert_eq!(r8_d, blocks8);
    vid_rows.push(BenchRow {
        name: "zero-rans 8x8 residual (split+rans8)".to_string(),
        raw_bytes: raw8_bytes.len(),
        comp_bytes: r8_c.len(),
        enc_mbs: mb_per_sec(raw8_bytes.len(), te),
        dec_mbs: mb_per_sec(raw8_bytes.len(), td),
    });

    if let Some((clen, enc_mbs, dec_mbs)) = bench_python_zlib(&raw8_bytes, 6, 3) {
        vid_rows.push(BenchRow {
            name: "zlib level 6 on raw 8x8 i16 blocks".to_string(),
            raw_bytes: raw8_bytes.len(),
            comp_bytes: clen,
            enc_mbs,
            dec_mbs,
        });
    }

    let (lz_c, te) = time_fn(5, || compress_lz_split(&raw_deltas).expect("enc lz"));
    let (lz_d, td) = time_fn(15, || decompress_lz_split(&lz_c).expect("dec lz"));
    assert_eq!(lz_d, raw_deltas);
    vid_rows.push(BenchRow {
        name: "zero-rans lz_split (frame deltas)".to_string(),
        raw_bytes: raw_deltas.len(),
        comp_bytes: lz_c.len(),
        enc_mbs: mb_per_sec(raw_deltas.len(), te),
        dec_mbs: mb_per_sec(raw_deltas.len(), td),
    });

    let (o1_c, te) = time_fn(8, || {
        encode_order1_8way(&raw_deltas, ContextMap::Low15).expect("enc o1")
    });
    let (o1_d, td) = time_fn(15, || decode_8way(&o1_c).expect("dec o1"));
    assert_eq!(o1_d, raw_deltas);
    vid_rows.push(BenchRow {
        name: "zero-rans order-1 rans8 (deltas)".to_string(),
        raw_bytes: raw_deltas.len(),
        comp_bytes: o1_c.len(),
        enc_mbs: mb_per_sec(raw_deltas.len(), te),
        dec_mbs: mb_per_sec(raw_deltas.len(), td),
    });

    if let Some((clen, enc_mbs, dec_mbs)) = bench_python_zlib(&raw_deltas, 6, 3) {
        vid_rows.push(BenchRow {
            name: "zlib level 6 (frame deltas)".to_string(),
            raw_bytes: raw_deltas.len(),
            comp_bytes: clen,
            enc_mbs,
            dec_mbs,
        });
    }

    print_table(&format!("Video Residual Benchmark: {label}"), &vid_rows);

    // 3. Multi-Standard BVF1 Video Profiles & AdaptiveStream Engine Benchmark (320x240, 8 frames).
    let (vw, vh, vn) = (320usize, 240usize, 8usize);
    let nat_seq: Vec<Yuv420Frame> = (0..vn)
        .map(|t| Yuv420Frame::synth_natural_motion(vw, vh, t).expect("synth nat"))
        .collect();
    let ui_seq: Vec<Yuv420Frame> = (0..vn)
        .map(|t| Yuv420Frame::synth_screen_ui(vw, vh, t).expect("synth ui"))
        .collect();
    let raw_seq_bytes = vn * nat_seq[0].byte_len();

    println!("\n=== Multi-Codec BVF1 & AdaptiveStream Video Benchmark ({vw}x{vh}, {vn} frames, YUV420p) ===");
    println!(
        "{:<36} | {:>9} | {:>9} | {:>7} | {:>9} | {:>9} | {:>9}",
        "Profile / Mode", "Raw (B)", "Comp (B)", "Ratio", "PSNR dB", "Enc MB/s", "Dec MB/s"
    );
    println!("{}", "-".repeat(106));

    for (pname, prof, qstep) in [
        ("BVF1 AvcH264 (qstep=4)", CodecProfile::AvcH264, 4u8),
        ("BVF1 HevcH265 + SAO (qstep=4)", CodecProfile::HevcH265, 4u8),
        ("BVF1 Vp9 ADST/DCT (qstep=4)", CodecProfile::Vp9, 4u8),
        ("BVF1 Av1 CFL+CDEF (qstep=4)", CodecProfile::Av1, 4u8),
        ("BVF1 HevcH265 lossless-near (q=1)", CodecProfile::HevcH265, 1u8),
    ] {
        let cfg = VideoConfig {
            profile: prof,
            qstep,
            lambda_q8: u16::from(qstep) * 32,
            search_range: 12,
            loop_filter: true,
            intra_refresh_col: None,
        };
        let (pkts, te) = time_fn(2, || {
            let mut r_opt: Option<Yuv420Frame> = None;
            let mut out_pkts = Vec::with_capacity(vn);
            for (idx, f) in nat_seq.iter().enumerate() {
                let kind = if idx == 0 {
                    FrameKind::Key
                } else {
                    FrameKind::Inter
                };
                let (p, rec) = encode_video_frame(f, r_opt.as_ref(), kind, cfg).expect("enc");
                r_opt = Some(rec);
                out_pkts.push(p);
            }
            out_pkts
        });
        let (recs, td) = time_fn(4, || {
            let mut r_opt: Option<Yuv420Frame> = None;
            let mut out_recs = Vec::with_capacity(vn);
            for p in &pkts {
                let rec = decode_video_frame(p, r_opt.as_ref()).expect("dec");
                r_opt = Some(rec.clone());
                out_recs.push(rec);
            }
            out_recs
        });
        let comp_bytes: usize = pkts.iter().map(Vec::len).sum();
        let avg_psnr: f64 = nat_seq
            .iter()
            .zip(&recs)
            .map(|(a, b)| a.psnr_db(b))
            .sum::<f64>()
            / (vn as f64);
        println!(
            "{:<36} | {:>9} | {:>9} | {:>6.2}x | {:>8.2} | {:>9.1} | {:>9.1}",
            pname,
            raw_seq_bytes,
            comp_bytes,
            (raw_seq_bytes as f64) / (comp_bytes as f64),
            avg_psnr,
            mb_per_sec(raw_seq_bytes, te),
            mb_per_sec(raw_seq_bytes, td),
        );
    }

    for (aname, seq_ref, refresh) in [
        ("AdaptiveStream Auto (Natural, CDF)", &nat_seq, false),
        ("AdaptiveStream Auto (Screen UI, IBC)", &ui_seq, false),
    ] {
        let acfg = AdaptiveStreamConfig {
            profile: CodecProfile::Auto,
            initial_qstep: 4,
            target_bytes_per_frame: None,
            keyint: 30,
            rolling_intra_refresh: refresh,
            loop_filter: true,
            search_range: 12,
        };
        let ((pkts, enc_dig), te) = time_fn(2, || {
            let mut enc = AdaptiveStreamEncoder::new(acfg).expect("aenc");
            let mut out_pkts = Vec::with_capacity(vn);
            for f in seq_ref {
                let (p, _) = enc.encode_frame(f).expect("aenc frame");
                out_pkts.push(p);
            }
            (out_pkts, enc.state_digest())
        });
        let ((recs, dec_dig), td) = time_fn(4, || {
            let mut dec = AdaptiveStreamDecoder::new().expect("adec");
            let mut out_recs = Vec::with_capacity(vn);
            for p in &pkts {
                out_recs.push(dec.decode_frame(p).expect("adec frame"));
            }
            (out_recs, dec.state_digest())
        });
        assert_eq!(enc_dig, dec_dig);
        let comp_bytes: usize = pkts.iter().map(Vec::len).sum();
        let avg_psnr: f64 = seq_ref
            .iter()
            .zip(&recs)
            .map(|(a, b)| a.psnr_db(b))
            .sum::<f64>()
            / (vn as f64);
        println!(
            "{:<36} | {:>9} | {:>9} | {:>6.2}x | {:>8.2} | {:>9.1} | {:>9.1}",
            aname,
            raw_seq_bytes,
            comp_bytes,
            (raw_seq_bytes as f64) / (comp_bytes as f64),
            avg_psnr,
            mb_per_sec(raw_seq_bytes, te),
            mb_per_sec(raw_seq_bytes, td),
        );
    }
}
