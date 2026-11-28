//! `zero-rans`: zero-dependency, zero-spill pure-Rust interleaved rANS, split-stream
//! LZ77/BRL2, multi-standard video (`AVC`/`HEVC`/`VP9`/`AV1`), and adaptive nibble coding.
//!
//! Default builds enforce `#![forbid(unsafe_code)]` with `0` external crates; opt-in
//! `--features zero-cc-native` links static `zero-cc` AVX2 kernels on `x86_64` Unix.
//! Modules: [`rans8`], [`residual`], [`lz_split`], [`socket_kernels`], [`socket_rans`],
//! [`socket_lz2d`], [`nc_nibble`], [`video`], [`adaptive_stream`], [`cpu_spec`], [`compact_v2`].
//!
//! # Examples
//!
//! ```
//! use zero_rans::{compress, compress_lz2d, compress_with_time_budget, decompress, decompress_lz2d};
//!
//! let src = b"fn main() { let mut acc = 0u64; for i in 0..64 { acc += i; } }";
//! assert_eq!(decompress(&compress(src)?)?, src);
//! let (budgeted, tier) = compress_with_time_budget(src, 0, 5_000)?;
//! assert!(tier >= 1 && decompress(&budgeted)? == src);
//! let img: Vec<u8> = (0..512).map(|i| (((i % 32) * 3 + (i / 32) * 5) & 255) as u8).collect();
//! assert_eq!(decompress_lz2d(&compress_lz2d(&img, 32)?, 32)?, img);
//! # Ok::<(), zero_rans::Error>(())
//! ```
//!
//! ```
//! use zero_rans::adaptive_stream::{AdaptiveStreamConfig, AdaptiveStreamDecoder, AdaptiveStreamEncoder};
//! use zero_rans::video::Yuv420Frame;
//!
//! let mut enc = AdaptiveStreamEncoder::new(AdaptiveStreamConfig::default())?;
//! let mut dec = AdaptiveStreamDecoder::new()?;
//! let frame = Yuv420Frame::synth_natural_motion(64, 48, 0)?;
//! let (packet, report) = enc.encode_frame(&frame)?;
//! let decoded = dec.decode_frame(&packet)?;
//! assert_eq!(enc.state_digest(), dec.state_digest());
//! assert_eq!(report.state_digest, dec.state_digest());
//! assert!(frame.psnr_db(&decoded) >= 34.0);
//! # Ok::<(), zero_rans::Error>(())
//! ```
//!
//! ```
//! use zero_rans::lz_split::{compress_lz_split, decompress_lz_split};
//!
//! let payload = b"frame0_delta_00000000_frame0_delta_00000000_frame0_delta_00000000";
//! assert_eq!(decompress_lz_split(&compress_lz_split(payload)?)?, payload);
//! # Ok::<(), zero_rans::Error>(())
//! ```
//!
//! ```
//! use zero_rans::{
//!     compress_socket_lz2d, decode_socket_8way, decompress_socket_lz2d_with_limit,
//!     encode_socket_8way, grad_socket8, paeth_pred8, sub_mod256_8, ScaleBits, SocketLz2dConfig,
//! };
//!
//! let (w, h) = (32usize, 16usize);
//! let img: Vec<u8> = (0..w * h).map(|i| (((i % w) * 3 + (i / w) * 5) & 255) as u8).collect();
//! let (mut res, mut socks) = (vec![0u8; w * h], vec![0u8; w * h]);
//! for y in 0..h {
//!     for x0 in (0..w).step_by(8) {
//!         let (mut wl, mut nl, mut nwl, mut cur) = ([0i32; 8], [0i32; 8], [0i32; 8], [0i32; 8]);
//!         for k in 0..8 {
//!             let x = x0 + k;
//!             cur[k] = i32::from(img[y * w + x]);
//!             wl[k] = if x > 0 { i32::from(img[y * w + x - 1]) } else { 0 };
//!             nl[k] = if y > 0 { i32::from(img[(y - 1) * w + x]) } else { 0 };
//!             nwl[k] = if x > 0 && y > 0 { i32::from(img[(y - 1) * w + x - 1]) } else { 0 };
//!         }
//!         let (mut pred, mut sock, mut rv) = ([0i32; 8], [0i32; 8], [0i32; 8]);
//!         paeth_pred8(&wl, &nl, &nwl, &mut pred);
//!         grad_socket8(&wl, &nl, &nwl, &mut sock);
//!         sub_mod256_8(&cur, &pred, &mut rv);
//!         for k in 0..8 {
//!             res[y * w + x0 + k] = rv[k] as u8;
//!             socks[y * w + x0 + k] = sock[k] as u8;
//!         }
//!     }
//! }
//! assert_eq!(decode_socket_8way(&encode_socket_8way(&res, &socks, 64, ScaleBits::Bits12)?, &socks)?, res);
//! let cfg = SocketLz2dConfig { stride: w, min_match: 3, arrival_passes: 2 };
//! let brl2 = compress_socket_lz2d(&img, Some(&socks), &cfg)?;
//! assert_eq!(decompress_socket_lz2d_with_limit(&brl2, Some(&socks), w, img.len())?, img);
//! # Ok::<(), zero_rans::Error>(())
//! ```

#![no_std]
#![cfg_attr(not(feature = "zero-cc-native"), forbid(unsafe_code))]
#![warn(missing_docs)]

extern crate alloc;

pub mod adaptive_stream;
pub mod compact_v2;
pub mod cpu_spec;
pub mod lz_split;
pub mod nc_nibble;
pub mod rans8;
pub mod residual;
pub mod socket_kernels;
pub mod socket_lz2d;
pub mod socket_rans;
pub mod video;
mod wire;

pub use adaptive_stream::{
    AdaptivePrior256, AdaptiveStreamConfig, AdaptiveStreamDecoder, AdaptiveStreamEncoder,
    AdaptiveStreamReport, RateControlState,
};
pub use cpu_spec::{
    detect_host_profile, host_tuning_plan, parse_cpu_specs, profile_by_name, CpuProfile, SimdTier,
    ZenTuningPlan, EMBEDDED_AMD_ZEN_SPEC,
};
pub use lz_split::{
    compress_lz_split, compress_lz_split_with_config, decompress_lz_split,
    decompress_lz_split_with_limit, LzConfig, MAGIC_LZ_SPLIT,
};
pub use rans8::{
    decode_4way, decode_8way, decode_with_limit_into, decode_with_model_and_limit_into,
    encode_4way, encode_4way_with_scale, encode_8way, encode_8way_with_model,
    encode_8way_with_scale, encode_order1_8way, ContextMap, RansModel, ScaleBits, MAGIC_RANS,
    RANS_L,
};
pub use residual::{
    decode_residuals_4x4, decode_residuals_4x4_with_limit, decode_residuals_8x8,
    decode_residuals_8x8_with_limit, dequantize_block, encode_residuals_4x4,
    encode_residuals_4x4_with_config, encode_residuals_8x8, encode_residuals_8x8_with_config,
    forward_dct_4x4, forward_dct_8x8, forward_dst7_4x4, forward_idtx_4x4, inverse_dct_4x4,
    inverse_dct_8x8, inverse_dst7_4x4, inverse_idtx_4x4, quantize_block, zigzag_decode_u16,
    zigzag_encode_i16, ResidualConfig, MAGIC_RESIDUAL_4X4, MAGIC_RESIDUAL_8X8, ZIGZAG_4X4,
    ZIGZAG_8X8,
};
pub use socket_kernels::{
    add_mod256_8, grad_socket8, grad_socket_u8, has_socket_batch_accel, nc_hash4_8, nc_hash4_u8,
    nc_hash_o2_8, nc_hash_o2_u8, nc_mix3_prob12_8, paeth_pred8, paeth_pred_u8,
    register_socket_accel, register_socket_batch_accel, run_add_mod256_8, run_add_mod256_8_batch,
    run_grad_socket8, run_grad_socket8_batch, run_nc_hash4_8_batch, run_nc_hash_o2_8_batch,
    run_paeth_pred8, run_paeth_pred8_batch, run_sub_mod256_8, run_sub_mod256_8_batch,
    run_zigzag_u8_8, sub_mod256_8, zigzag_u8_8, SocketAccel, SocketBatch2Fn, SocketBatch3Fn,
    SocketBatch4Fn, SocketBatchAccel, SocketMod256Fn, SocketPred3Fn, SocketZigzagFn,
};
pub use socket_lz2d::{
    compress, compress_lz2d, compress_parallel, compress_socket_lz2d,
    compress_socket_lz2d_with_effort, compress_socket_lz2d_with_profile, compress_with_arena,
    compress_with_profile, compress_with_time_budget, decompress, decompress_lz2d,
    decompress_socket_lz2d_with_limit, NibbleArena, SocketLz2dConfig, MAGIC_PARALLEL_BLOCKS,
    MAGIC_SOCKET_LZ2D,
};
pub use socket_rans::{
    decode_hybrid_uint_socket, decode_socket_8way, decode_socket_8way_with_limit,
    decode_sparse_runs_8way, decode_substream_socket_auto, encode_hybrid_uint_socket,
    encode_socket_8way, encode_sparse_runs_8way, encode_substream_socket_auto, pack_radix_span,
    unpack_radix_span, zigzag_decode_u32, zigzag_encode_i32, ClusteredSocketRans,
    MAGIC_HYBRID_UINT, MAGIC_SOCKET_RANS, MAGIC_SPARSE_RUNS, MAX_SOCKET_CLUSTERS,
};
pub use video::{
    analyze_frame_structure, decode_video_frame, decode_video_frame_with_limit, encode_video_frame,
    select_auto_profile, CodecProfile, FrameKind, FrameStructuralStats, MotionVec, VideoConfig,
    Yuv420Frame, MAGIC_VIDEO_FRAME,
};
pub use wire::{crc32_ieee, Error, Result, DEFAULT_MAX_DECODE_BYTES};

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::rans8::{rans_enc_step_scalar, EncSym, MAX_STAGE_WORDS, RANS_UPPER};
    use crate::wire::{write_varint_u32, Reader};

    #[kani::proof]
    fn proof_reciprocal_table_exact_all_freq() {
        let scale_bits: u32 = if kani::any() { 14 } else { 12 };
        let total = 1u32 << scale_bits;
        let freq: u32 = kani::any();
        kani::assume(freq >= 1 && freq <= total);
        let enc = EncSym::from_freq_cum(freq, 0, scale_bits);
        assert_eq!(enc.x_max, ((RANS_L >> scale_bits) << 16) * freq);
        assert!(enc.x_max <= RANS_UPPER);
        let pow2 = 1u128 << enc.rcp_shift;
        let prod = u128::from(enc.rcp) * u128::from(freq);
        assert!(prod >= pow2 && prod - pow2 < u128::from(freq));
        assert!((prod - pow2) * u128::from(RANS_UPPER - 1) < pow2);
        assert!(u128::from(RANS_UPPER - 1) * u128::from(enc.rcp) <= u128::from(u64::MAX));
    }

    #[kani::proof]
    fn proof_rans_step_bijection_branchless() {
        let use_14: bool = kani::any();
        let scale_bits: u32 = if use_14 { 14 } else { 12 };
        let total = 1u32 << scale_bits;
        let idx: usize = kani::any();
        kani::assume(idx < 8);
        let freqs = if use_14 {
            [1, 2, 3, 17, 255, 4096, 16383, 16384]
        } else {
            [1, 2, 3, 17, 255, 2047, 4095, 4096]
        };
        let freq = freqs[idx];
        let cum: u32 = kani::any();
        let state: u32 = kani::any();
        kani::assume(cum <= total - freq && state >= RANS_L && state < RANS_UPPER);
        let enc = EncSym::from_freq_cum(freq, cum, scale_bits);
        let mut stage = [0u16; MAX_STAGE_WORDS];
        let mut lp = 0usize;
        let encoded = rans_enc_step_scalar(state, &enc, scale_bits, &mut stage, &mut lp);
        assert!(encoded >= RANS_L && encoded < RANS_UPPER && lp <= 1);
        if freq == total {
            assert_eq!(enc.x_max, RANS_UPPER);
            assert_eq!(lp, 0);
        }
        let slot = encoded & (total - 1);
        assert!(slot >= cum && slot < cum + freq);
        let next = freq * (encoded >> scale_bits) + (slot - cum);
        let win_word = if lp == 1 { stage[0] } else { kani::any::<u16>() };
        let need = usize::from(next < RANS_L);
        assert_eq!(need, lp);
        let refilled = (next << 16) | u32::from(win_word);
        let sel = 0u32.wrapping_sub(need as u32);
        assert_eq!(next ^ ((refilled ^ next) & sel), state);
    }

    #[kani::proof]
    fn proof_zigzag_i16_bijection() {
        let v: i16 = kani::any();
        assert_eq!(zigzag_decode_u16(zigzag_encode_i16(v)), v);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_varint_canonical_and_malformed_no_panic() {
        let value: u32 = kani::any();
        let mut buf = Vec::with_capacity(5);
        write_varint_u32(&mut buf, value);
        let mut r = Reader::new(&buf);
        assert_eq!(r.read_varint_u32(), Ok(value));
        assert_eq!(r.finish(), Ok(()));
        let len: usize = kani::any();
        kani::assume(len <= 8);
        let raw: [u8; 8] = kani::any();
        let _ = Reader::new(&raw[..len]).read_varint_u32();
        let _ = Reader::new(&raw[..len]).read_substream();
        let overlong = [0x80u8 | (kani::any::<u8>() & 0x7F), 0x00u8];
        assert_eq!(Reader::new(&overlong).read_varint_u32(), Err(Error::CorruptState));
    }

    #[kani::proof]
    #[kani::unwind(18)]
    fn proof_h264_4x4_dct_no_overflow_and_dc_exact() {
        let mut blk = [0i16; 16];
        for i in 0..16 {
            let v: i16 = kani::any();
            kani::assume((-255..=255).contains(&v));
            blk[i] = v;
        }
        let _ = inverse_dct_4x4(&forward_dct_4x4(&blk));
        let dc: i16 = kani::any();
        kani::assume((-255..=255).contains(&dc));
        let dc_tr = forward_dct_4x4(&[dc; 16]);
        assert_eq!(dc_tr[0], 16 * i32::from(dc));
        for k in 1..16 {
            assert_eq!(dc_tr[k], 0);
        }
        let mut scaled = [0i32; 16];
        scaled[0] = dc_tr[0] * 4;
        assert_eq!(inverse_dct_4x4(&scaled), [dc; 16]);
    }
}
