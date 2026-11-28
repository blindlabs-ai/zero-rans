//! Multi-standard video codec profiles (`H.264/AVC`, `H.265/HEVC`, `VP9`, `AV1`, `Auto`,
//! `Compact`) over 8-way interleaved `rANS` (`BRN1`) and split-stream `LZ77` (`BRL1`).

use alloc::vec;
use alloc::vec::Vec;

use crate::lz_split::{compress_lz_split, decompress_lz_split_with_limit};
use crate::rans8::{RansModel, ScaleBits};
use crate::residual::{
    decode_residual_blocks_with_priors, decode_substream_maybe_prior, dequantize_block,
    encode_residual_blocks_with_priors, encode_substream_best, forward_dct_4x4, forward_dct_8x8,
    forward_dst7_4x4, forward_idtx_4x4, inverse_dct_4x4, inverse_dct_8x8, inverse_dst7_4x4,
    inverse_idtx_4x4, quantize_block, ResidualConfig, ResidualPriors, ResidualStreamObs,
    MAGIC_RESIDUAL_4X4, MAGIC_RESIDUAL_8X8, ZIGZAG_4X4, ZIGZAG_8X8,
};
use crate::socket_kernels::paeth_pred_u8;
use crate::wire::{
    crc32_update, expect_magic, framed_header_crc, write_framed_header, write_substream, Error,
    Reader, Result, DEFAULT_MAX_DECODE_BYTES,
};

/// Framed multi-profile video stream magic (`BVF1`).
pub const MAGIC_VIDEO_FRAME: [u8; 4] = *b"BVF1";
const MAX_FRAME_DIM: usize = 16_384;
pub(crate) const VF_LOOP_FILTER: u8 = 0x01;
pub(crate) const VF_LZ_SIDE: u8 = 0x02;
pub(crate) const VF_INTRA_REFRESH: u8 = 0x04;
pub(crate) const VF_COMPACT_V2: u8 = 0x08;
pub(crate) const VF_KNOCKOUT_RAW_MV: u8 = 0x10;

/// Supported industry video codec profiles plus adaptive profile selection (`Auto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecProfile {
    /// ITU-T H.264 / AVC: DC/H/V/Plane intra, diamond ME, 4x4/8x8 DCT, 4-tap deblocking.
    AvcH264 = 0,
    /// ITU-T H.265 / HEVC: Quadtree 16->8->4 RDO split, Planar intra, 4x4 DST-VII, SAO.
    HevcH265 = 1,
    /// WebM VP9: True-Motion (`L + T - TL`), boundary-adaptive ADST/DCT, flatness LF.
    Vp9 = 2,
    /// AOMedia AV1: Paeth intra, CFL, IBC + IDTX screen mode, 8-dir CDEF.
    Av1 = 3,
    /// Adaptive online profile selection (resolved per frame into `0..=3` on wire).
    Auto = 4,
    /// Compact V2 ultra-low-bitrate profile (`<= 0.050 bpp`): `8x8` DCT, half-pel `PMV_SKIP`.
    Compact = 5,
}

impl CodecProfile {
    #[inline]
    pub(crate) const fn from_wire_id(id: u8) -> Result<Self> {
        match id {
            0 => Ok(Self::AvcH264), 1 => Ok(Self::HevcH265), 2 => Ok(Self::Vp9),
            3 => Ok(Self::Av1), 5 => Ok(Self::Compact), _ => Err(Error::UnsupportedFormat),
        }
    }
}
/// Frame prediction type (`Key` intra-only I-frame or `Inter` motion-compensated P-frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Self-contained intra keyframe (`I-frame`).
    Key = 0,
    /// Inter-predicted delta frame (`P-frame`) referencing the previous reconstructed frame.
    Inter = 1,
}
/// Integer-pel motion (or AV1 intra-block-copy) displacement vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVec {
    /// Horizontal luma pixel offset (`-64..=63`).
    pub dx: i8,
    /// Vertical luma pixel offset (`-64..=63`).
    pub dy: i8,
}
/// Encoder configuration for single-frame or sequence `BVF1` coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoConfig {
    /// Target codec architecture (`AvcH264`, `HevcH265`, `Vp9`, `Av1`, `Auto`, or `Compact`).
    pub profile: CodecProfile,
    /// Quantization step (`1..=64` standard, `1..=255` for `Compact`).
    pub qstep: u8,
    /// Lagrangian rate multiplier (`Q8` fixed-point) for mode/quadtree RDO.
    pub lambda_q8: u16,
    /// Integer diamond search radius in luma pixels (`1..=32`).
    pub search_range: u8,
    /// Enable codec-specific in-loop restoration filter (`Deblock`, `SAO`, `VP9-LF`, `CDEF`).
    pub loop_filter: bool,
    /// Optional forced rolling intra-refresh macroblock column (`None` disables forced refresh).
    pub intra_refresh_col: Option<u16>,
}
impl Default for VideoConfig {
    #[inline]
    fn default() -> Self {
        Self {
            profile: CodecProfile::HevcH265, qstep: 4, lambda_q8: 192,
            search_range: 12, loop_filter: true, intra_refresh_col: None,
        }
    }
}
/// Structural telemetry measured on an input frame (used by `Auto` and `AdaptiveStreamEncoder`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStructuralStats {
    /// Mean absolute temporal luma difference against reference (`0..=255`, `255` if none).
    pub temporal_sad_avg: u16,
    /// Percentage (`0..=100`) of pixel pairs with exact zero gradient.
    pub zero_grad_pct: u8,
    /// Percentage (`0..=100`) of pixel pairs with sharp step edges (`|dx| + |dy| >= 32`).
    pub sharp_step_pct: u8,
    /// Mean `8x8` block spatial variance proxy (`0..=65535`).
    pub spatial_var_avg: u16,
}
/// Planar 8-bit 4:2:0 video frame (`Y` full resolution, `U`/`V` half width and half height).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420Frame {
    /// Luma width in pixels (must be positive and even).
    pub width: usize,
    /// Luma height in pixels (must be positive and even).
    pub height: usize,
    /// Planar luma samples (`width * height` bytes).
    pub y: Vec<u8>,
    /// Planar Cb chroma samples (`(width / 2) * (height / 2)` bytes).
    pub u: Vec<u8>,
    /// Planar Cr chroma samples (`(width / 2) * (height / 2)` bytes).
    pub v: Vec<u8>,
}

impl Yuv420Frame {
    /// Allocate a mid-gray (`Y=128, U=128, V=128`) validated YUV420p frame.
    pub fn new_blank(width: usize, height: usize) -> Result<Self> {
        let (y_len, uv_len) = validate_yuv420_dims(width, height, DEFAULT_MAX_DECODE_BYTES)?;
        Ok(Self { width, height, y: vec![128; y_len], u: vec![128; uv_len], v: vec![128; uv_len] })
    }
    /// Validate planar buffer lengths and even dimensions against `max_bytes`.
    pub fn validate_with_limit(&self, max_bytes: usize) -> Result<()> {
        let (y_len, uv_len) = validate_yuv420_dims(self.width, self.height, max_bytes)?;
        if (self.y.len(), self.u.len(), self.v.len()) != (y_len, uv_len, uv_len) {
            return Err(Error::InvalidDimensions);
        }
        Ok(())
    }
    /// Total planar byte count (`width * height * 3 / 2`).
    #[inline] #[must_use]
    pub fn byte_len(&self) -> usize { self.y.len() + self.u.len() + self.v.len() }
    /// Compute IEEE 802.3 CRC-32 across `width || height || Y || U || V` for reference lockstep.
    #[must_use]
    pub fn digest_crc32(&self) -> u32 {
        let mut crc = crc32_update(0xFFFF_FFFFu32, &(self.width as u32).to_le_bytes());
        crc = crc32_update(crc, &(self.height as u32).to_le_bytes());
        crc = crc32_update(crc, &self.y);
        crc = crc32_update(crc, &self.u);
        !crc32_update(crc, &self.v)
    }
    /// Convert packed `RGB24` (`width * height * 3` bytes) into BT.601 `YUV420p`.
    pub fn from_rgb24(width: usize, height: usize, rgb: &[u8]) -> Result<Self> {
        let (y_len, uv_len) = validate_yuv420_dims(width, height, DEFAULT_MAX_DECODE_BYTES)?;
        if rgb.len() != y_len.checked_mul(3).ok_or(Error::OutputTooLarge)? {
            return Err(Error::InvalidParameter);
        }
        let mut f = Self { width, height, y: vec![0; y_len], u: vec![128; uv_len], v: vec![128; uv_len] };
        for i in 0..y_len {
            let (r, g, b) = (i32::from(rgb[i * 3]), i32::from(rgb[i * 3 + 1]), i32::from(rgb[i * 3 + 2]));
            f.y[i] = clip_u8(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
        }
        let (cw, ch) = (width / 2, height / 2);
        for cy in 0..ch {
            for cx in 0..cw {
                let (mut rs, mut gs, mut bs) = (0i32, 0i32, 0i32);
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let i = ((cy * 2 + dy) * width + cx * 2 + dx) * 3;
                    (rs, gs, bs) = (
                        rs + i32::from(rgb[i]), gs + i32::from(rgb[i + 1]), bs + i32::from(rgb[i + 2]),
                    );
                }
                let (r, g, b) = ((rs + 2) >> 2, (gs + 2) >> 2, (bs + 2) >> 2);
                f.u[cy * cw + cx] = clip_u8(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
                f.v[cy * cw + cx] = clip_u8(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
            }
        }
        Ok(f)
    }
    /// Convert BT.601 `YUV420p` into packed `RGB24` (`width * height * 3` bytes).
    pub fn to_rgb24(&self) -> Result<Vec<u8>> {
        self.validate_with_limit(DEFAULT_MAX_DECODE_BYTES)?;
        let (cw, mut rgb) = (self.width / 2, vec![0u8; self.width * self.height * 3]);
        for py in 0..self.height {
            for px in 0..self.width {
                let ci = (py / 2) * cw + (px / 2);
                let (uv, vv) = (i32::from(self.u[ci]) - 128, i32::from(self.v[ci]) - 128);
                let idx = py * self.width + px;
                let (c, d) = ((i32::from(self.y[idx]) - 16) * 298, idx * 3);
                (rgb[d], rgb[d + 1], rgb[d + 2]) = (
                    clip_u8((c + 409 * vv + 128) >> 8),
                    clip_u8((c - 100 * uv - 208 * vv + 128) >> 8),
                    clip_u8((c + 516 * uv + 128) >> 8),
                );
            }
        }
        Ok(rgb)
    }
    /// Compute peak signal-to-noise ratio (PSNR in `dB`) across `Y + U + V` (`99.0` if identical).
    #[must_use]
    pub fn psnr_db(&self, other: &Self) -> f64 {
        if (self.width, self.height, self.y.len(), self.u.len(), self.v.len())
            != (other.width, other.height, other.y.len(), other.u.len(), other.v.len())
        { return 0.0; }
        let sse: u64 = self.y.iter().zip(&other.y)
            .chain(self.u.iter().zip(&other.u)).chain(self.v.iter().zip(&other.v))
            .map(|(&a, &b)| { let d = i64::from(a) - i64::from(b); (d * d) as u64 }).sum();
        if sse == 0 { return 99.0; }
        10.0 * crate::wire::log10_f64((255.0 * 255.0) / ((sse as f64) / (self.byte_len() as f64)))
    }
    fn fill_synth_chroma(&mut self, f_uv: impl Fn(usize, usize, i32) -> (i32, i32)) {
        let (cw, ch) = (self.width / 2, self.height / 2);
        for cy in 0..ch {
            for cx in 0..cw {
                let (u, v) = f_uv(cx, cy, i32::from(self.y[(cy * 2) * self.width + cx * 2]) - 128);
                (self.u[cy * cw + cx], self.v[cy * cw + cx]) = (clip_u8(u), clip_u8(v));
            }
        }
    }
    /// Synthesize a natural smooth camera-pan + moving object YUV420p frame at time index `t`.
    pub fn synth_natural_motion(width: usize, height: usize, t: usize) -> Result<Self> {
        let mut f = Self::new_blank(width, height)?;
        let (pan_x, pan_y) = (t * 4, t * 2);
        let (obj_x, obj_y) = ((width / 5 + t * 6) % width.max(1), (height / 4 + t * 4) % height.max(1));
        for py in 0..height {
            for px in 0..width {
                let (sx, sy) = (px + pan_x, py + pan_y);
                let grad = ((sx * 96) / width.max(1) + (sy * 64) / height.max(1)) & 127;
                let tex = ((sx.wrapping_mul(17) ^ sy.wrapping_mul(31))
                    .wrapping_add((sx >> 2).wrapping_mul(13))
                    .wrapping_add((sy >> 2).wrapping_mul(7))) & 31;
                let mut lum = 44 + grad as i32 + (tex as i32 - 15);
                let (dx, dy) = (px as isize - obj_x as isize, py as isize - obj_y as isize);
                if (0..28).contains(&dx) && (0..24).contains(&dy) {
                    lum = 192 - ((dx + dy) as i32) + (((dx as usize * 7) ^ (dy as usize * 11)) & 15) as i32;
                }
                f.y[py * width + px] = clip_u8(lum);
            }
        }
        f.fill_synth_chroma(|cx, cy, d| (
            128 + (d >> 2) + (((cx + pan_x / 2) as i32) & 7),
            128 - (d >> 2) + (((cy + pan_y / 2) as i32) & 7),
        ));
        Ok(f)
    }
    /// Synthesize a crisp desktop UI / terminal text / repeated glyph screen frame at time `t`.
    pub fn synth_screen_ui(width: usize, height: usize, t: usize) -> Result<Self> {
        let mut f = Self::new_blank(width, height)?;
        let scroll = (t * 4) % 16;
        for py in 0..height {
            for px in 0..width {
                let wy = py + scroll;
                let border = px < 4 || py < 12 || px + 4 >= width || py + 4 >= height;
                let (cx, cy) = (px & 15, wy & 15);
                let glyph = (2..=12).contains(&cx)
                    && (cy == 2 || cy == 8 || cy == 13 || (cx == 3 && cy < 13) || (cx == 11 && cy > 2));
                f.y[py * width + px] = if border { 42 } else if py < 24 { 220 }
                    else if glyph && ((px >> 4) + (wy >> 4)) % 3 != 0 { 238 } else { 18 };
            }
        }
        f.fill_synth_chroma(|_, _, d| (128 - (d >> 2), 128 + (d >> 3)));
        Ok(f)
    }
}

/// Saturate a signed 32-bit sample into the 8-bit pixel domain `[0, 255]`.
/// @domain v: -2147483648..=2147483647
#[inline]
pub const fn clip_u8(v: i32) -> u8 { if v < 0 { 0 } else if v > 255 { 255 } else { v as u8 } }
/// Saturate a signed 32-bit value to `[-bound, bound]` (`bound >= 0`).
/// @domain v: -1048576..=1048576, bound: 0..=32767
#[inline]
pub const fn clip_sym_i32(v: i32, bound: i32) -> i32 {
    if v < -bound { -bound } else if v > bound { bound } else { v }
}
/// AV1 Paeth intra predictor (`p = L + T - TL`, selecting neighbor closest to `p`).
/// @domain left: 0..=255, top: 0..=255, top_left: 0..=255
#[inline]
pub const fn paeth_predictor(left: u8, top: u8, top_left: u8) -> u8 { paeth_pred_u8(left, top, top_left) }
/// VP9 True-Motion (`TM`) intra predictor (`clip_u8(L + T - TL)`).
/// @domain left: 0..=255, top: 0..=255, top_left: 0..=255
#[inline]
pub const fn vp9_tm_predictor(left: u8, top: u8, top_left: u8) -> u8 {
    clip_u8((left as i32) + (top as i32) - (top_left as i32))
}
/// AV1 Chroma-from-Luma (`CFL`) sample reconstruction.
/// @domain dc_uv: 0..=255, alpha_q3: -16..=16, luma_ac: -255..=255
#[inline]
pub const fn av1_cfl_sample(dc_uv: u8, alpha_q3: i32, luma_ac: i32) -> u8 {
    clip_u8((dc_uv as i32) + ((alpha_q3 * luma_ac + 4) >> 3))
}
/// HEVC Sample Adaptive Offset (`SAO`) 3-sample edge class index (`0..=4`).
/// @domain center: 0..=255, n0: 0..=255, n1: 0..=255
#[inline]
pub const fn sao_edge_class(center: u8, n0: u8, n1: u8) -> usize {
    let (c, a, b) = (center as i32, n0 as i32, n1 as i32);
    let s = (if c > a { 1 } else if c < a { -1 } else { 0 })
        + (if c > b { 1 } else if c < b { -1 } else { 0 });
    match s { -2 => 1, -1 => 2, 1 => 3, 2 => 4, _ => 0 }
}
/// Apply a signed HEVC SAO sample offset (`-31..=31`) to `center`.
/// @domain center: 0..=255, offset: -31..=31
#[inline]
pub const fn sao_apply_offset(center: u8, offset: i32) -> u8 { clip_u8((center as i32) + offset) }
/// AV1 Constrained Directional Enhancement Filter (`CDEF`) nonlinear tap weight (`damping = 4`).
/// @domain diff: -255..=255, threshold: 0..=63
#[inline]
pub const fn cdef_constrain(diff: i32, threshold: i32) -> i32 {
    if threshold <= 0 { return 0; }
    let abs_d = if diff < 0 { -diff } else { diff };
    let rem = threshold - (abs_d >> 4);
    let mag = if rem <= 0 { 0 } else if abs_d < rem { abs_d } else { rem };
    if diff < 0 { -mag } else { mag }
}
/// Apply AV1 directional CDEF 3-tap step given constrained neighbor deltas `(d0, d1)` in `[-63, 63]`.
/// @domain center: 0..=255, d0: -63..=63, d1: -63..=63
#[inline]
pub const fn cdef_dir_step(center: u8, d0: i32, d1: i32) -> u8 {
    clip_u8((center as i32) + ((d0 + d1 + 2) >> 2))
}
/// @domain p1: 0..=255, p0: 0..=255, q0: 0..=255, q1: 0..=255
/// @domain mpq: -1..=128, mf: -1..=64, w: 3..=4, co: 16..=255, cl: 0..=32
#[inline]
const fn lf_delta(p1: u8, p0: u8, q0: u8, q1: u8, mpq: i32, mf: i32, w: i32, co: i32, cl: i32) -> i32 {
    let (p1i, p0i, q0i, q1i) = (p1 as i32, p0 as i32, q0 as i32, q1 as i32);
    let dpq = if q0i >= p0i { q0i - p0i } else { p0i - q0i };
    let dp = if p1i >= p0i { p1i - p0i } else { p0i - p1i };
    let dq = if q1i >= q0i { q1i - q0i } else { q0i - q1i };
    if dpq <= mpq && dp <= mf && dq <= mf {
        clip_sym_i32((w * (q0i - p0i) + clip_sym_i32(p1i - q1i, co) + 4) >> 3, cl)
    } else { 0 }
}
/// H.264 4-tap boundary-strength deblocking delta across `(p1, p0 | q0, q1)`.
/// @domain p1: 0..=255, p0: 0..=255, q0: 0..=255, q1: 0..=255, alpha: 0..=64, beta: 0..=32, tc: 0..=16
#[inline]
pub const fn avc_deblock_delta(p1: u8, p0: u8, q0: u8, q1: u8, alpha: i32, beta: i32, tc: i32) -> i32 {
    lf_delta(p1, p0, q0, q1, alpha - 1, beta - 1, 4, 255, tc)
}
/// VP9 flatness-adaptive 4-tap loop filter delta across `(p1, p0 | q0, q1)`.
/// @domain p1: 0..=255, p0: 0..=255, q0: 0..=255, q1: 0..=255, limit: 0..=64
#[inline]
pub const fn vp9_loop_delta(p1: u8, p0: u8, q0: u8, q1: u8, limit: i32) -> i32 {
    lf_delta(p1, p0, q0, q1, limit * 2, limit, 3, 16, limit >> 1)
}

#[inline]
pub(crate) fn validate_yuv420_dims(width: usize, height: usize, max_bytes: usize) -> Result<(usize, usize)> {
    let bad = width == 0 || height == 0 || width > MAX_FRAME_DIM || height > MAX_FRAME_DIM
        || (width & 1) != 0 || (height & 1) != 0;
    if bad { return Err(Error::InvalidDimensions); }
    let y_len = width.checked_mul(height).ok_or(Error::OutputTooLarge)?;
    let uv_len = (width / 2).checked_mul(height / 2).ok_or(Error::OutputTooLarge)?;
    let total = y_len.checked_add(uv_len.checked_mul(2).ok_or(Error::OutputTooLarge)?)
        .ok_or(Error::OutputTooLarge)?;
    if total > max_bytes { return Err(Error::OutputTooLarge); }
    Ok((y_len, uv_len))
}
pub(crate) fn validate_encode_inputs(
    frame: &Yuv420Frame, reference: Option<&Yuv420Frame>, kind: FrameKind,
) -> Result<u32> {
    frame.validate_with_limit(DEFAULT_MAX_DECODE_BYTES)?;
    if frame.width > usize::from(u16::MAX) || frame.height > usize::from(u16::MAX) {
        return Err(Error::InvalidDimensions);
    }
    if kind == FrameKind::Inter {
        let ref_f = reference.ok_or(Error::InvalidParameter)?;
        ref_f.validate_with_limit(DEFAULT_MAX_DECODE_BYTES)?;
        if (ref_f.width, ref_f.height) != (frame.width, frame.height) {
            return Err(Error::InvalidDimensions);
        }
        return Ok(ref_f.digest_crc32());
    }
    Ok(0)
}
pub(crate) fn resolve_decode_reference(
    kind: FrameKind, reference: Option<&Yuv420Frame>, width: usize, height: usize,
    expected_ref_crc: u32, max_bytes: usize, resilient_heal: bool,
) -> Result<(Option<PaddedCanvas>, bool)> {
    if kind != FrameKind::Inter {
        if expected_ref_crc != 0 { return Err(Error::CorruptVideoFrame); }
        return Ok((None, false));
    }
    if let Some(rf) = reference.filter(|r| (r.width, r.height) == (width, height)) {
        rf.validate_with_limit(max_bytes)?;
        let mismatched = rf.digest_crc32() != expected_ref_crc;
        if mismatched && !resilient_heal { return Err(Error::ReferenceMismatch); }
        return Ok((Some(PaddedCanvas::from_frame(rf)), mismatched));
    }
    if !resilient_heal { return Err(Error::ReferenceMismatch); }
    Ok((Some(PaddedCanvas::new_blank(width, height)), true))
}
pub(crate) fn encode_side_stream(
    side: &[u8], prior: Option<&RansModel>, allow_empty: bool,
) -> Result<(Vec<u8>, bool)> {
    if allow_empty && side.is_empty() { return Ok((Vec::new(), false)); }
    let side_rans = encode_with_optional_prior(side, prior)?;
    if side.len() >= 48 {
        let side_lz = compress_lz_split(side)?;
        if side_lz.len() + 2 < side_rans.len() { return Ok((side_lz, true)); }
    }
    Ok((side_rans, false))
}
pub(crate) fn finish_bvf1_frame(
    profile_kind: u8, flags: u8, qstep_u8: u8, refresh_col_u8: u8,
    ref_crc: u32, rec_frame: &Yuv420Frame, mut body: Vec<u8>,
) -> Result<Vec<u8>> {
    let mut meta = [profile_kind, flags, qstep_u8, refresh_col_u8, 0, 0, 0, 0, 0, 0, 0, 0];
    meta[4..6].copy_from_slice(&(rec_frame.width as u16).to_le_bytes());
    meta[6..8].copy_from_slice(&(rec_frame.height as u16).to_le_bytes());
    meta[8..12].copy_from_slice(&ref_crc.to_le_bytes());
    let orig_len = u32::try_from(rec_frame.byte_len()).map_err(|_| Error::OutputTooLarge)?;
    body.extend_from_slice(&rec_frame.digest_crc32().to_le_bytes());
    let mut wire = Vec::with_capacity(24 + body.len());
    write_framed_header(&mut wire, &MAGIC_VIDEO_FRAME, &meta, orig_len, &body);
    wire.extend_from_slice(&body);
    Ok(wire)
}

/// Measure spatial edge sharpness, zero-gradient UI flatness, variance, and temporal SAD.
#[must_use]
pub fn analyze_frame_structure(frame: &Yuv420Frame, reference: Option<&Yuv420Frame>) -> FrameStructuralStats {
    let (w, h) = (frame.width, frame.height);
    if w < 2 || h < 2 || frame.y.len() < w * h { return FrameStructuralStats::default(); }
    let (mut zero_grad, mut sharp_step, mut pairs) = (0u64, 0u64, 0u64);
    for py in (0..h - 1).step_by(2) {
        let (row, next_row) = (py * w, (py + 1) * w);
        for px in (0..w - 1).step_by(2) {
            let c = i32::from(frame.y[row + px]);
            let grad = (i32::from(frame.y[row + px + 1]) - c).abs()
                + (i32::from(frame.y[next_row + px]) - c).abs();
            zero_grad += u64::from(grad == 0);
            sharp_step += u64::from(grad >= 32);
            pairs += 1;
        }
    }
    let temporal_sad_avg = match reference {
        Some(rf) if rf.y.len() == frame.y.len() && !frame.y.is_empty() => {
            let (mut sad, mut cnt) = (0u64, 0u64);
            for i in (0..frame.y.len()).step_by(4) {
                sad += u64::from(frame.y[i].abs_diff(rf.y[i]));
                cnt += 1;
            }
            (sad / cnt.max(1)).min(255) as u16
        }
        _ => 255,
    };
    let (mut var_acc, mut blk_cnt) = (0u64, 0u64);
    for by in (0..h.saturating_sub(7)).step_by(16) {
        for bx in (0..w.saturating_sub(7)).step_by(16) {
            let (mut sum, mut sum_sq) = (0u32, 0u64);
            for r in 0..8usize {
                for &pv in &frame.y[(by + r) * w + bx..(by + r) * w + bx + 8] {
                    let v = u32::from(pv);
                    (sum, sum_sq) = (sum + v, sum_sq + u64::from(v * v));
                }
            }
            var_acc += sum_sq.saturating_sub((u64::from(sum) * u64::from(sum)) >> 6) >> 6;
            blk_cnt += 1;
        }
    }
    let denom = pairs.max(1);
    FrameStructuralStats {
        temporal_sad_avg,
        zero_grad_pct: ((zero_grad * 100) / denom).min(100) as u8,
        sharp_step_pct: ((sharp_step * 100) / denom).min(100) as u8,
        spatial_var_avg: (var_acc / blk_cnt.max(1)).min(65_535) as u16,
    }
}

/// Resolve `CodecProfile::Auto` into the optimal concrete wire profile (`AvcH264..=Av1`).
#[must_use]
pub fn select_auto_profile(stats: FrameStructuralStats) -> CodecProfile {
    if stats.zero_grad_pct >= 35 && stats.sharp_step_pct >= 8 { CodecProfile::Av1 }
    else if stats.spatial_var_avg >= 450 && stats.temporal_sad_avg >= 18 { CodecProfile::Vp9 }
    else if stats.spatial_var_avg <= 35 && stats.sharp_step_pct <= 2 { CodecProfile::AvcH264 }
    else { CodecProfile::HevcH265 }
}

pub(crate) struct PaddedCanvas {
    pub(crate) pw: usize, pub(crate) ph: usize, pub(crate) cw: usize, pub(crate) ch: usize,
    pub(crate) y: Vec<u8>, pub(crate) u: Vec<u8>, pub(crate) v: Vec<u8>,
}
impl PaddedCanvas {
    pub(crate) fn new_blank(width: usize, height: usize) -> Self {
        let (pw, ph) = ((width + 15) & !15, (height + 15) & !15);
        let (cw, ch) = (pw / 2, ph / 2);
        Self { pw, ph, cw, ch, y: vec![128; pw * ph], u: vec![128; cw * ch], v: vec![128; cw * ch] }
    }
    pub(crate) fn from_frame(frame: &Yuv420Frame) -> Self {
        let mut c = Self::new_blank(frame.width, frame.height);
        blit_plane_pad(&frame.y, frame.width, frame.height, &mut c.y, c.pw, c.ph);
        blit_plane_pad(&frame.u, frame.width / 2, frame.height / 2, &mut c.u, c.cw, c.ch);
        blit_plane_pad(&frame.v, frame.width / 2, frame.height / 2, &mut c.v, c.cw, c.ch);
        c
    }
    pub(crate) fn to_frame(&self, width: usize, height: usize) -> Yuv420Frame {
        let crop = |src: &[u8], stride: usize, w: usize, h: usize| -> Vec<u8> {
            let mut out = vec![0u8; w * h];
            for r in 0..h { out[r * w..(r + 1) * w].copy_from_slice(&src[r * stride..r * stride + w]); }
            out
        };
        Yuv420Frame {
            width, height,
            y: crop(&self.y, self.pw, width, height),
            u: crop(&self.u, self.cw, width / 2, height / 2),
            v: crop(&self.v, self.cw, width / 2, height / 2),
        }
    }
}
fn blit_plane_pad(src: &[u8], w: usize, h: usize, dst: &mut [u8], pw: usize, ph: usize) {
    for r in 0..h {
        let (sr, dr) = (&src[r * w..(r + 1) * w], &mut dst[r * pw..(r + 1) * pw]);
        dr[..w].copy_from_slice(sr);
        dr[w..pw].fill(sr[w - 1]);
    }
    let last_src = (h - 1) * pw;
    for r in h..ph { dst.copy_within(last_src..last_src + pw, r * pw); }
}
#[inline]
pub(crate) fn read_patch<const N: usize>(plane: &[u8], stride: usize, bx: usize, by: usize) -> [[u8; N]; N] {
    let mut out = [[0u8; N]; N];
    for r in 0..N { out[r].copy_from_slice(&plane[(by + r) * stride + bx..(by + r) * stride + bx + N]); }
    out
}
#[inline]
pub(crate) fn write_patch<const N: usize>(
    plane: &mut [u8], stride: usize, bx: usize, by: usize, patch: &[[u8; N]; N],
) {
    for r in 0..N { plane[(by + r) * stride + bx..(by + r) * stride + bx + N].copy_from_slice(&patch[r]); }
}
#[inline]
pub(crate) fn extract_quad8(patch16: &[[u8; 16]; 16], qx: usize, qy: usize) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for r in 0..8usize { out[r].copy_from_slice(&patch16[qy * 8 + r][qx * 8..qx * 8 + 8]); }
    out
}
#[inline]
pub(crate) fn blit_quad8(dst16: &mut [[u8; 16]; 16], qx: usize, qy: usize, blk: &[[u8; 8]; 8]) {
    for r in 0..8usize { dst16[qy * 8 + r][qx * 8..qx * 8 + 8].copy_from_slice(&blk[r]); }
}

/// Unified `N x N` block distortion (`SQUARED = true` -> SSE, `false` -> SAD).
#[inline]
pub(crate) fn block_dist<const N: usize, const SQUARED: bool>(
    src: &[u8], stride: usize, bx: usize, by: usize, cand: &[[u8; N]; N],
) -> u64 {
    let mut acc = 0u64;
    for (r, row_c) in cand.iter().enumerate() {
        let off = (by + r) * stride + bx;
        if off + N <= src.len() {
            for (c, &cv) in row_c.iter().enumerate() {
                let d = i32::from(src[off + c]) - i32::from(cv);
                acc += if SQUARED { (d * d) as u64 } else { u64::from(d.unsigned_abs()) };
            }
        }
    }
    acc
}

#[inline]
fn clip_coord(v: isize, max_v: usize) -> usize {
    if v <= 0 { 0 } else if (v as usize) >= max_v { max_v } else { v as usize }
}
fn fetch_motion_patch<const W: usize, const H: usize>(
    plane: &[u8], stride: usize, ph: usize, bx: usize, by: usize, dx: i8, dy: i8,
) -> [[u8; W]; H] {
    let mut out = [[128u8; W]; H];
    if stride == 0 || ph == 0 || stride > 65_536 || ph > 65_536 || plane.len() < stride * ph { return out; }
    let (mx, my) = (stride - 1, ph - 1);
    for (r, row_out) in out.iter_mut().enumerate() {
        let cy = clip_coord((by + r) as isize + (dy as isize), my) * stride;
        for (c, slot) in row_out.iter_mut().enumerate() {
            *slot = plane[cy + clip_coord((bx + c) as isize + (dx as isize), mx)];
        }
    }
    out
}
fn load_intra_neighbors<const N: usize>(
    plane: &[u8], stride: usize, bx: usize, by: usize,
) -> ([u8; N], [u8; N], u8) {
    let (has_top, has_left, mut top, mut left) = (by > 0, bx > 0, [128u8; N], [128u8; N]);
    if has_top { top.copy_from_slice(&plane[(by - 1) * stride + bx..(by - 1) * stride + bx + N]); }
    else if has_left { top = [plane[by * stride + bx - 1]; N]; }
    if has_left { for r in 0..N { left[r] = plane[(by + r) * stride + bx - 1]; } }
    else if has_top { left = [top[0]; N]; }
    let tl = if has_top && has_left { plane[(by - 1) * stride + bx - 1] }
        else if has_top { top[0] } else if has_left { left[0] } else { 128 };
    (top, left, tl)
}

pub(crate) fn predict_intra_block<const N: usize>(
    profile: CodecProfile, submode: u8, plane: &[u8], stride: usize, bx: usize, by: usize,
) -> [[u8; N]; N] {
    let (top, left, tl) = load_intra_neighbors::<N>(plane, stride, bx, by);
    let mut out = [[128u8; N]; N];
    match submode & 3 {
        0 => {
            let sum: u32 = top.iter().zip(&left).map(|(&t, &l)| u32::from(t) + u32::from(l)).sum();
            out = [[((sum + (N as u32)) / ((2 * N) as u32)) as u8; N]; N];
        }
        1 => { for (row_out, &lv) in out.iter_mut().zip(&left) { *row_out = [lv; N]; } }
        2 => out.fill(top),
        _ if N > 0 => {
            let (tr_e, bl_e, n_i32) = (i32::from(top[N - 1]), i32::from(left[N - 1]), N as i32);
            let (mid, gx, gy) = (n_i32 / 2, tr_e - i32::from(tl), bl_e - i32::from(tl));
            let a = 16 * (tr_e + bl_e);
            for (r, row_out) in out.iter_mut().enumerate() {
                let lv = left[r];
                for (c, slot) in row_out.iter_mut().enumerate() {
                    let tv = top[c];
                    *slot = match profile {
                        CodecProfile::AvcH264 => {
                            clip_u8((a + gx * ((c as i32 - mid) * 2) + gy * ((r as i32 - mid) * 2) + 16) >> 5)
                        }
                        CodecProfile::HevcH265 => {
                            let wr = (n_i32 - 1 - c as i32) * i32::from(lv) + (c as i32 + 1) * tr_e;
                            let wc = (n_i32 - 1 - r as i32) * i32::from(tv) + (r as i32 + 1) * bl_e;
                            clip_u8((wr + wc + n_i32) / (2 * n_i32))
                        }
                        CodecProfile::Vp9 => vp9_tm_predictor(lv, tv, tl),
                        _ => paeth_predictor(lv, tv, tl),
                    };
                }
            }
        }
        _ => {}
    }
    out
}
#[derive(Clone, Copy)]
struct Tx4Spec { profile: CodecProfile, is_intra: bool, is_screen_idtx: bool, submode: u8, qstep: i32 }
const H264_NORM_Q6: [i32; 16] = [
    256, 205, 256, 205, 205, 164, 205, 164, 256, 205, 256, 205, 205, 164, 205, 164,
];
#[inline]
fn tx4_uses_dst(spec: Tx4Spec, plane_rec: &[u8], stride: usize, bx: usize, by: usize) -> bool {
    match spec.profile {
        CodecProfile::HevcH265 => spec.is_intra,
        CodecProfile::Vp9 => {
            if !spec.is_intra || spec.submode == 0 { return false; }
            let (top, left, tl) = load_intra_neighbors::<4>(plane_rec, stride, bx, by);
            top.iter().zip(&left).map(|(&t, &l)| {
                (i32::from(t) - i32::from(tl)).abs() + (i32::from(l) - i32::from(tl)).abs()
            }).sum::<i32>() >= 12
        }
        CodecProfile::Av1 => spec.is_intra && spec.submode >= 1,
        _ => false,
    }
}
#[inline]
fn inv_tx4_core(is_idtx: bool, use_dst: bool, qstep: i32, q: &[i16; 16]) -> [i16; 16] {
    let dq = dequantize_block(q, qstep * 4);
    if is_idtx { inverse_idtx_4x4(&dq) }
    else if use_dst { inverse_dst7_4x4(&dq) }
    else { inverse_dct_4x4(&dq) }
}
fn transform_roundtrip_4x4(
    spec: Tx4Spec, plane_rec: &[u8], stride: usize, bx: usize, by: usize, res: &[i16; 16],
) -> ([i16; 16], [i16; 16]) {
    let qstep = spec.qstep.max(1);
    let use_dst = !spec.is_screen_idtx && tx4_uses_dst(spec, plane_rec, stride, bx, by);
    let q = if spec.is_screen_idtx {
        quantize_block(&forward_idtx_4x4(res), qstep * 4)
    } else if use_dst {
        quantize_block(&forward_dst7_4x4(res), qstep * 4)
    } else {
        let (tr, mut norm) = (forward_dct_4x4(res), [0i32; 16]);
        for (dst, (&t, &n)) in norm.iter_mut().zip(tr.iter().zip(&H264_NORM_Q6)) {
            *dst = (t * n + 128) >> 8;
        }
        quantize_block(&norm, qstep)
    };
    (q, inv_tx4_core(spec.is_screen_idtx, use_dst, qstep, &q))
}
fn inverse_luma_4x4(
    spec: Tx4Spec, plane_rec: &[u8], stride: usize, bx: usize, by: usize, q: &[i16; 16],
) -> [i16; 16] {
    let qstep = spec.qstep.max(1);
    let use_dst = !spec.is_screen_idtx && tx4_uses_dst(spec, plane_rec, stride, bx, by);
    inv_tx4_core(spec.is_screen_idtx, use_dst, qstep, q)
}
fn transform_roundtrip_8x8(res: &[i16; 64], qstep: i32) -> ([i16; 64], [i16; 64]) {
    let step = qstep.max(1);
    let q = quantize_block(&forward_dct_8x8(res), step);
    (q, inverse_dct_8x8(&dequantize_block(&q, step)))
}
fn diamond_search_16x16(
    src_y: &[u8], ref_y: &[u8], stride: usize, ph: usize, bx: usize, by: usize, range: i8,
) -> (MotionVec, u32) {
    let sad_at = |dx: i8, dy: i8| -> u32 {
        let patch = fetch_motion_patch::<16, 16>(ref_y, stride, ph, bx, by, dx, dy);
        block_dist::<16, false>(src_y, stride, bx, by, &patch) as u32
            + u32::from(dx.unsigned_abs() + dy.unsigned_abs()) * 2
    };
    let (mut best_mv, mut best_sad) = (MotionVec::default(), sad_at(0, 0));
    if best_sad <= 64 { return (best_mv, best_sad); }
    let r_lim = i32::from(range).max(1).min(48);
    for step in [4i32, 2, 1] {
        for _ in 0..8usize {
            let mut improved = false;
            let (cx, cy) = (i32::from(best_mv.dx), i32::from(best_mv.dy));
            for (ox, oy) in [(-step, 0), (step, 0), (0, -step), (0, step)] {
                let (nx, ny) = (cx + ox, cy + oy);
                if nx < -r_lim || nx > r_lim || ny < -r_lim || ny > r_lim { continue; }
                let cand = sad_at(nx as i8, ny as i8);
                if cand < best_sad {
                    (best_sad, best_mv, improved) = (cand, MotionVec { dx: nx as i8, dy: ny as i8 }, true);
                }
            }
            if !improved { break; }
        }
    }
    (best_mv, best_sad)
}
fn search_av1_ibc_16x16(
    src_y: &[u8], rec_y: &[u8], stride: usize, bx: usize, by: usize,
) -> Option<(MotionVec, u32)> {
    if (bx < 16 && by < 16) || stride == 0 || bx + 16 > stride { return None; }
    let min_len = (by + 16).checked_mul(stride)?;
    if src_y.len() < min_len || rec_y.len() < min_len { return None; }
    let src_blk = read_patch::<16>(src_y, stride, bx, by);
    let (mut best, mut best_sad) = (None, u32::MAX);
    let mut eval = |dx: i32, dy: i32, sx: usize, sy: usize| -> bool {
        let (mut sad, mut r_off) = (0u32, sy * stride + sx);
        for s_row in &src_blk {
            let r_row = &rec_y[r_off..r_off + 16];
            sad += (0..16usize).map(|c| u32::from(s_row[c].abs_diff(r_row[c]))).sum::<u32>();
            if sad >= best_sad { return false; }
            r_off += stride;
        }
        (best_sad, best) = (sad, Some((MotionVec { dx: dx as i8, dy: dy as i8 }, sad)));
        sad == 0
    };
    if (bx >= 16 && eval(-16, 0, bx - 16, by)) || (by >= 16 && eval(0, -16, bx, by - 16)) { return best; }
    for dy in (-48i32..=0).step_by(4) {
        let sy = by as isize + dy as isize;
        if sy < 0 { continue; }
        let max_dx = if dy <= -16 { 48 } else { -16 };
        for dx in (-64i32..=max_dx).step_by(4) {
            if !((dx == -16 && dy == 0) || (dx == 0 && dy == -16)) {
                let sx = bx as isize + dx as isize;
                if sx >= 0 && (sx as usize) + 16 <= stride && eval(dx, dy, sx as usize, sy as usize) {
                    return best;
                }
            }
        }
    }
    best
}
#[inline]
fn validate_causal_ibc(stride: usize, bx: usize, by: usize, dx: i8, dy: i8) -> Result<(usize, usize)> {
    if stride > 65_536 || bx > stride || by > 65_536 { return Err(Error::CorruptVideoFrame); }
    let (sx, sy) = ((bx as i64) + (dx as i64), (by as i64) + (dy as i64));
    if sx < 0 || sy < 0 { return Err(Error::CorruptVideoFrame); }
    let (ux, uy) = (sx as usize, sy as usize);
    if ux + 16 > stride || (uy + 16 > by && !(uy <= by && ux + 16 <= bx)) {
        return Err(Error::CorruptVideoFrame);
    }
    Ok((ux, uy))
}
pub(crate) fn apply_8x8_edge_filter<F: Fn(u8, u8, u8, u8) -> i32>(
    plane: &mut [u8], stride: usize, width: usize, height: usize, delta_fn: F,
) {
    let step = |p: &mut [u8], i: usize, s: usize| {
        if let Some(s2) = s.checked_mul(2) {
            if i >= s2 && i + s < p.len() {
                let d = delta_fn(p[i - s2], p[i - s], p[i], p[i + s]);
                (p[i - s], p[i]) = (clip_u8(i32::from(p[i - s]) + d), clip_u8(i32::from(p[i]) - d));
            }
        }
    };
    for x in (8..width.saturating_sub(1)).step_by(8) {
        for y in 0..height { step(plane, y * stride + x, 1); }
    }
    for y in (8..height.saturating_sub(1)).step_by(8) {
        for x in 0..width { step(plane, y * stride + x, stride); }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HevcSaoParams { mode: u8, band_start: u8, offsets: [i8; 4] }
impl HevcSaoParams {
    fn to_bytes(self) -> [u8; 6] {
        [self.mode, self.band_start, self.offsets[0] as u8, self.offsets[1] as u8,
         self.offsets[2] as u8, self.offsets[3] as u8]
    }
    fn from_bytes(raw: &[u8]) -> Result<Self> {
        if raw.len() != 6 || raw[0] > 5 || raw[1] > 28 { return Err(Error::CorruptVideoFrame); }
        let mut offsets = [0i8; 4];
        for (slot, &b) in offsets.iter_mut().zip(&raw[2..6]) {
            let v = b as i8;
            if !(-31..=31).contains(&v) { return Err(Error::CorruptVideoFrame); }
            *slot = v;
        }
        Ok(Self { mode: raw[0], band_start: raw[1], offsets })
    }
}
fn plane_sse_roi(orig: &[u8], o_stride: usize, rec: &[u8], r_stride: usize, w: usize, h: usize) -> u64 {
    let mut sse = 0u64;
    for y in 0..h {
        let (or, rr) = (y * o_stride, y * r_stride);
        for x in 0..w {
            if or + x < orig.len() && rr + x < rec.len() {
                let d = i64::from(orig[or + x]) - i64::from(rec[rr + x]);
                sse += (d * d) as u64;
            }
        }
    }
    sse
}
fn for_sao_eo(
    src: &[u8], stride: usize, w: usize, h: usize, dx: isize, dy: isize,
    mut f: impl FnMut(usize, usize, u8, usize),
) {
    if w < 3 || h < 3 { return; }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (ic, i0, i1) = (
                y * stride + x,
                (y as isize - dy).max(0) as usize * stride + (x as isize - dx).max(0) as usize,
                (y as isize + dy).max(0) as usize * stride + (x as isize + dx).max(0) as usize,
            );
            if ic < src.len() && i0 < src.len() && i1 < src.len() {
                let cls = sao_edge_class(src[ic], src[i0], src[i1]);
                if (1..=4).contains(&cls) { f(x, y, src[ic], cls - 1); }
            }
        }
    }
}
fn derive_hevc_sao(orig: &[u8], rec: &[u8], stride: usize, width: usize, height: usize) -> HevcSaoParams {
    let mut best = HevcSaoParams::default();
    let mut best_sse = plane_sse_roi(orig, width, rec, stride, width, height);
    let mut try_cand = |cand: HevcSaoParams| {
        let mut trial = rec.to_vec();
        apply_hevc_sao(&mut trial, stride, width, height, cand);
        let sse = plane_sse_roi(orig, width, &trial, stride, width, height);
        if sse < best_sse { (best_sse, best) = (sse, cand); }
    };
    let (mut bdiff, mut bcnt) = ([0i64; 32], [0u32; 32]);
    for y in 0..height {
        for x in 0..width {
            let (r, o) = (rec[y * stride + x], orig[y * width + x]);
            let b = usize::from(r >> 3);
            (bdiff[b], bcnt[b]) = (bdiff[b] + i64::from(o) - i64::from(r), bcnt[b] + 1);
        }
    }
    let (mut best_s, mut best_e) = (0usize, 0i64);
    for s in 0..=28usize {
        let e: i64 = (0..4).map(|k| {
            bdiff[s + k] * i64::from(clip_sym_i32((bdiff[s + k] / i64::from(bcnt[s + k].max(1))) as i32, 15))
        }).sum();
        if e > best_e { (best_e, best_s) = (e, s); }
    }
    if best_e > 0 {
        let mut offsets = [0i8; 4];
        for (k, slot) in offsets.iter_mut().enumerate() {
            let cnt = i64::from(bcnt[best_s + k].max(1));
            *slot = clip_sym_i32((bdiff[best_s + k] / cnt) as i32, 15) as i8;
        }
        try_cand(HevcSaoParams { mode: 1, band_start: best_s as u8, offsets });
    }
    if width >= 3 && height >= 3 {
        for (mode, dx, dy) in [(2u8, 1isize, 0isize), (3u8, 0isize, 1isize)] {
            let (mut cdiff, mut ccnt) = ([0i64; 4], [0u32; 4]);
            for_sao_eo(rec, stride, width, height, dx, dy, |x, y, c, ci| {
                if ci < 4 {
                    cdiff[ci] += i64::from(orig[y * width + x]) - i64::from(c);
                    ccnt[ci] += 1;
                }
            });
            let mut offsets = [0i8; 4];
            for (k, slot) in offsets.iter_mut().enumerate() {
                *slot = clip_sym_i32((cdiff[k] / i64::from(ccnt[k].max(1))) as i32, 15) as i8;
            }
            try_cand(HevcSaoParams { mode, band_start: 0, offsets });
        }
    }
    best
}
fn apply_hevc_sao(plane: &mut [u8], stride: usize, width: usize, height: usize, params: HevcSaoParams) {
    match params.mode {
        1 => {
            let start = usize::from(params.band_start);
            for y in 0..height {
                for x in 0..width {
                    let (p, b) = (plane[y * stride + x], usize::from(plane[y * stride + x] >> 3));
                    if b >= start && b < start + 4 {
                        plane[y * stride + x] = sao_apply_offset(p, i32::from(params.offsets[b - start]));
                    }
                }
            }
        }
        2..=5 if width >= 3 && height >= 3 => {
            let (dx, dy) = [(1isize, 0isize), (0, 1), (1, 1), (1, -1)][usize::from(params.mode - 2)];
            let src = plane.to_vec();
            for_sao_eo(&src, stride, width, height, dx, dy, |x, y, c, ci| {
                if ci < 4 { plane[y * stride + x] = sao_apply_offset(c, i32::from(params.offsets[ci])); }
            });
        }
        _ => {}
    }
}

const CDEF_DIRS: [(isize, isize); 8] = [
    (1, -1), (2, -1), (1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2),
];

fn estimate_cdef_dir_8x8(plane: &[u8], stride: usize, bx: usize, by: usize) -> usize {
    let (mut best_dir, mut best_score) = (2usize, i64::MAX);
    for (dir_idx, &(dx, dy)) in CDEF_DIRS.iter().enumerate() {
        let mut diff_sum = 0i64;
        for r in 1..7usize {
            for c in 1..7usize {
                let (x, y) = ((bx + c) as isize, (by + r) as isize);
                let ci = (y as usize) * stride + x as usize;
                let ni = (y + dy).max(0) as usize * stride + (x + dx).max(0) as usize;
                if ci < plane.len() && ni < plane.len() {
                    let d = i64::from(plane[ci]) - i64::from(plane[ni]);
                    diff_sum += d * d;
                }
            }
        }
        if diff_sum < best_score { (best_score, best_dir) = (diff_sum, dir_idx); }
    }
    best_dir
}
fn apply_av1_cdef(plane: &mut [u8], stride: usize, width: usize, height: usize, strength_byte: u8) {
    let threshold = i32::from(strength_byte & 0x3F);
    if threshold == 0 { return; }
    let (src, wi, hi) = (plane.to_vec(), width as isize, height as isize);
    for by in (0..height).step_by(8) {
        for bx in (0..width).step_by(8) {
            let dir = if bx + 8 <= width && by + 8 <= height {
                estimate_cdef_dir_8x8(&src, stride, bx, by)
            } else { 2 };
            let (dx, dy) = CDEF_DIRS[dir];
            for y in by..(by + 8).min(height) {
                for x in bx..(bx + 8).min(width) {
                    let (xi, yi) = (x as isize, y as isize);
                    let (x0, x1, y0, y1) = (xi - dx, xi + dx, yi - dy, yi + dy);
                    if x0 < 0 || x0 >= wi || x1 < 0 || x1 >= wi || y0 < 0 || y0 >= hi || y1 < 0 || y1 >= hi {
                        continue;
                    }
                    let (ic, i0) = (y * stride + x, (y0 as usize) * stride + (x0 as usize));
                    let i1 = (y1 as usize) * stride + (x1 as usize);
                    if ic < src.len() && i0 < src.len() && i1 < src.len() && ic < plane.len() {
                        let c = src[ic];
                        let d0 = cdef_constrain(i32::from(src[i0]) - i32::from(c), threshold);
                        let d1 = cdef_constrain(i32::from(src[i1]) - i32::from(c), threshold);
                        plane[ic] = cdef_dir_step(c, d0, d1);
                    }
                }
            }
        }
    }
}
fn derive_av1_cdef_byte(
    orig: &[u8], rec: &[u8], stride: usize, width: usize, height: usize, qstep: u8,
) -> u8 {
    let (mut best_byte, mut best_sse) = (0u8, plane_sse_roi(orig, width, rec, stride, width, height));
    let t_base = ((qstep >> 1) + 2).min(24);
    for cand in [t_base | (1u8 << 6), (t_base >> 1).max(1) | (2u8 << 6)] {
        let mut trial = rec.to_vec();
        apply_av1_cdef(&mut trial, stride, width, height, cand);
        let sse = plane_sse_roi(orig, width, &trial, stride, width, height);
        if sse < best_sse { (best_sse, best_byte) = (sse, cand); }
    }
    best_byte
}
fn apply_loop_filter(
    profile: CodecProfile, plane: &mut [u8], stride: usize, width: usize, height: usize,
    qstep_u8: u8, sao: Option<HevcSaoParams>, cdef_byte: u8,
) {
    let q = i32::from(qstep_u8);
    match profile {
        CodecProfile::AvcH264 => {
            let (a, b, tc) = ((q * 3 + 6).min(64), (q + 4).min(32), ((q >> 1) + 2).min(16));
            apply_8x8_edge_filter(plane, stride, width, height, |p1, p0, q0, q1| {
                avc_deblock_delta(p1, p0, q0, q1, a, b, tc)
            });
        }
        CodecProfile::HevcH265 => if let Some(p) = sao { apply_hevc_sao(plane, stride, width, height, p); },
        CodecProfile::Vp9 => {
            let lim = (q * 2 + 4).min(48);
            apply_8x8_edge_filter(plane, stride, width, height, |p1, p0, q0, q1| {
                vp9_loop_delta(p1, p0, q0, q1, lim)
            });
        }
        _ => apply_av1_cdef(plane, stride, width, height, cdef_byte),
    }
}

const MB_INTRA_8X8_BASE: u8 = 0;
const MB_INTRA_4X4_BASE: u8 = 4;
const MB_INTER_SKIP: u8 = 8;
const MB_INTER_8X8: u8 = 9;
const MB_INTER_4X4: u8 = 10;
const MB_QUADTREE_MIX: u8 = 11;
const MB_AV1_IBC_IDTX: u8 = 12;
const MB_AV1_INTRA_IDTX_BASE: u8 = 13;

#[derive(Clone, Copy)]
pub(crate) struct FramePriors<'a> {
    pub(crate) mode: &'a RansModel, pub(crate) side: &'a RansModel, pub(crate) res: ResidualPriors<'a>,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameSymbolObs {
    pub(crate) modes: Vec<u8>, pub(crate) side: Vec<u8>,
    pub(crate) res4: ResidualStreamObs, pub(crate) res8: ResidualStreamObs,
}
fn fit_av1_cfl_alpha(orig_uv: &[[u8; 8]; 8], luma_ac: &[[i16; 8]; 8], dc_uv: u8) -> i8 {
    let (mut num, mut den) = (0i64, 0i64);
    for r in 0..8usize {
        for c in 0..8usize {
            let ac = i64::from(luma_ac[r][c]);
            (num, den) = (num + (i64::from(orig_uv[r][c]) - i64::from(dc_uv)) * ac, den + ac * ac);
        }
    }
    if den < 64 { return 0; }
    let a = ((num * 8) + (den >> 1)) / den;
    if a < -16 { -16 } else if a > 15 { 15 } else { a as i8 }
}
fn compute_subsampled_luma_ac(rec_y: &[u8], stride: usize, bx: usize, by: usize) -> [[i16; 8]; 8] {
    let (mut sub, mut sum) = ([[0i16; 8]; 8], 0i32);
    for (r, sub_row) in sub.iter_mut().enumerate() {
        for (c, slot) in sub_row.iter_mut().enumerate() {
            let (y0, y1) = ((by + r * 2) * stride + bx + c * 2, (by + r * 2 + 1) * stride + bx + c * 2);
            if y1 + 1 < rec_y.len() {
                let s = i32::from(rec_y[y0]) + i32::from(rec_y[y0 + 1])
                    + i32::from(rec_y[y1]) + i32::from(rec_y[y1 + 1]);
                let avg = (s + 2) >> 2;
                (*slot, sum) = (avg as i16, sum + avg);
            }
        }
    }
    let mean = ((sum + 32) >> 6) as i16;
    for row in &mut sub { for slot in row { *slot -= mean; } }
    sub
}
#[inline]
const fn pack_cfl_pair(alpha_u: i8, alpha_v: i8) -> u8 {
    (((alpha_u as i32 + 16) & 0x1F) as u8) | ((((alpha_v as i32 >> 2) + 4) & 0x07) as u8) << 5
}
#[inline]
const fn unpack_cfl_pair(byte: u8) -> (i32, i32) {
    ((byte & 0x1F) as i32 - 16, ((byte >> 5) as i32 - 4) * 4)
}
#[inline]
fn count_nonzero_cost<const N: usize>(q: &[i16; N]) -> u64 {
    let mut bits = 4u64;
    for &v in q { if v != 0 { bits += 10 + (u64::from(v.unsigned_abs().min(64)) >> 1); } }
    bits
}

/// Encode a single [`Yuv420Frame`] into a `BVF1` packet (standalone API without external priors).
pub fn encode_video_frame(
    frame: &Yuv420Frame, reference: Option<&Yuv420Frame>, kind: FrameKind, cfg: VideoConfig,
) -> Result<(Vec<u8>, Yuv420Frame)> {
    let (wire, rec, _) = encode_video_frame_internal(frame, reference, kind, cfg, None)?;
    Ok((wire, rec))
}
/// Decode a single `BVF1` packet into a reconstructed [`Yuv420Frame`].
pub fn decode_video_frame(packet: &[u8], reference: Option<&Yuv420Frame>) -> Result<Yuv420Frame> {
    decode_video_frame_with_limit(packet, reference, DEFAULT_MAX_DECODE_BYTES)
}
/// Decode a `BVF1` packet enforcing a caller-specified maximum output byte ceiling.
pub fn decode_video_frame_with_limit(
    packet: &[u8], reference: Option<&Yuv420Frame>, max_bytes: usize,
) -> Result<Yuv420Frame> {
    Ok(decode_video_frame_internal(packet, reference, max_bytes, false, None)?.0)
}

pub(crate) fn encode_video_frame_internal(
    frame: &Yuv420Frame, reference: Option<&Yuv420Frame>, kind: FrameKind,
    cfg: VideoConfig, priors: Option<FramePriors<'_>>,
) -> Result<(Vec<u8>, Yuv420Frame, FrameSymbolObs)> {
    if cfg.profile == CodecProfile::Compact {
        return crate::compact_v2::encode_compact_frame_internal(
            frame, reference, kind, cfg, priors, false,
        );
    }
    let ref_crc = validate_encode_inputs(frame, reference, kind)?;
    let profile = if cfg.profile == CodecProfile::Auto {
        select_auto_profile(analyze_frame_structure(frame, reference))
    } else { cfg.profile };
    let (qstep_u8, lambda_q8) = (cfg.qstep.max(1).min(64), u64::from(cfg.lambda_q8.max(16)));
    let src = PaddedCanvas::from_frame(frame);
    let ref_canvas = (kind == FrameKind::Inter).then(|| reference.map(PaddedCanvas::from_frame)).flatten();
    let mut rec = PaddedCanvas::new_blank(frame.width, frame.height);
    let (mb_cols, mb_rows) = (src.pw / 16, src.ph / 16);
    let (mut modes, mut side, mut blocks4, mut blocks8) = (
        Vec::with_capacity(mb_rows * mb_cols), Vec::with_capacity(mb_rows * mb_cols * 2),
        Vec::with_capacity(mb_rows * mb_cols * 8), Vec::with_capacity(mb_rows * mb_cols * 4),
    );
    for mby in 0..mb_rows {
        for mbx in 0..mb_cols {
            let mb_ctx = MbEncodeCtx {
                profile, src: &src, ref_canvas: ref_canvas.as_ref(),
                bx: mbx * 16, by: mby * 16, qstep: i32::from(qstep_u8), lambda_q8,
                search_range: cfg.search_range,
                force_intra: kind == FrameKind::Key
                    || cfg.intra_refresh_col.is_some_and(|col| usize::from(col) == mbx),
            };
            let chosen = encode_luma_macroblock(&mb_ctx, &mut rec);
            modes.push(chosen.mode);
            side.extend_from_slice(&chosen.side_bytes);
            blocks4.extend_from_slice(&chosen.q4);
            blocks8.extend_from_slice(&chosen.q8);
            encode_chroma_macroblock(&mb_ctx, chosen.mode, chosen.mv, &mut rec, &mut side, &mut blocks4);
        }
    }

    let mut filter_hdr = Vec::new();
    if cfg.loop_filter {
        let sao = (profile == CodecProfile::HevcH265)
            .then(|| derive_hevc_sao(&frame.y, &rec.y, rec.pw, frame.width, frame.height));
        let is_av1 = matches!(profile, CodecProfile::Av1 | CodecProfile::Auto | CodecProfile::Compact);
        let cdef = if is_av1 {
            derive_av1_cdef_byte(&frame.y, &rec.y, rec.pw, frame.width, frame.height, qstep_u8)
        } else { 0 };
        apply_loop_filter(profile, &mut rec.y, rec.pw, frame.width, frame.height, qstep_u8, sao, cdef);
        if let Some(p) = sao { filter_hdr.extend_from_slice(&p.to_bytes()); }
        else if is_av1 { filter_hdr.push(cdef); }
    }
    let rec_frame = rec.to_frame(frame.width, frame.height);
    let mode_enc = encode_with_optional_prior(&modes, priors.map(|p| p.mode))?;
    let (side_enc, use_lz_side) = encode_side_stream(&side, priors.map(|p| p.side), false)?;
    let res_cfg = ResidualConfig { scale: ScaleBits::Bits12, order1_cbp: true };
    let (res4_enc, res4_obs) = encode_residual_blocks_with_priors(
        &blocks4, &ZIGZAG_4X4, &MAGIC_RESIDUAL_4X4, res_cfg, priors.map(|p| p.res),
    )?;
    let (res8_enc, res8_obs) = encode_residual_blocks_with_priors(
        &blocks8, &ZIGZAG_8X8, &MAGIC_RESIDUAL_8X8, res_cfg, priors.map(|p| p.res),
    )?;
    let cap = filter_hdr.len() + mode_enc.len() + side_enc.len() + res4_enc.len() + res8_enc.len() + 32;
    let mut body = Vec::with_capacity(cap);
    body.extend_from_slice(&filter_hdr);
    for sub in [&mode_enc, &side_enc, &res4_enc, &res8_enc] { write_substream(&mut body, sub)?; }
    let profile_kind = ((profile as u8) & 0x07) | ((kind as u8) << 4);
    let mut flags = (u8::from(cfg.loop_filter) * VF_LOOP_FILTER) | (u8::from(use_lz_side) * VF_LZ_SIDE);
    let refresh_col_u8 = match cfg.intra_refresh_col {
        Some(col) if col <= 255 => { flags |= VF_INTRA_REFRESH; col as u8 }
        _ => 0,
    };
    let wire = finish_bvf1_frame(profile_kind, flags, qstep_u8, refresh_col_u8, ref_crc, &rec_frame, body)?;
    Ok((wire, rec_frame, FrameSymbolObs { modes, side, res4: res4_obs, res8: res8_obs }))
}
struct MbCandidate {
    mode: u8, mv: MotionVec, side_bytes: Vec<u8>,
    q4: Vec<[i16; 16]>, q8: Vec<[i16; 64]>, rec_patch: [[u8; 16]; 16], cost: u64,
}
impl MbCandidate {
    fn new(
        mode: u8, mv: MotionVec, side_bytes: Vec<u8>,
        q4: Vec<[i16; 16]>, q8: Vec<[i16; 64]>, rec_patch: [[u8; 16]; 16], cost: u64,
    ) -> Self { Self { mode, mv, side_bytes, q4, q8, rec_patch, cost } }
}
struct MbEncodeCtx<'a> {
    profile: CodecProfile, src: &'a PaddedCanvas, ref_canvas: Option<&'a PaddedCanvas>,
    bx: usize, by: usize, qstep: i32, lambda_q8: u64, search_range: u8, force_intra: bool,
}
#[inline]
fn consider_mb(best: &mut Option<MbCandidate>, cand: MbCandidate) {
    if best.as_ref().map_or(true, |b| cand.cost < b.cost) { *best = Some(cand); }
}
#[inline]
fn encode_mv_pair(mv: MotionVec) -> Vec<u8> {
    vec![(mv.dx as i16 + 128) as u8, (mv.dy as i16 + 128) as u8]
}
fn encode_tiled<const P: usize, const B: usize, const N: usize, const G: usize, F>(
    src: &[u8], stride: usize, bx: usize, by: usize, pred: &[[u8; P]; P], mut tx_fn: F,
) -> ([[i16; N]; G], [[u8; P]; P], u64)
where F: FnMut(usize, usize, &[i16; N]) -> ([i16; N], [i16; N]) {
    let grid = P / B;
    let (mut qs, mut rec, mut bits) = ([[0i16; N]; G], [[0u8; P]; P], 0u64);
    for ty in 0..grid {
        for tx in 0..grid {
            let mut res = [0i16; N];
            for r in 0..B {
                for c in 0..B {
                    let (py, px) = (ty * B + r, tx * B + c);
                    res[r * B + c] = i16::from(src[(by + py) * stride + bx + px]) - i16::from(pred[py][px]);
                }
            }
            let (q_blk, d_blk) = tx_fn(bx + tx * B, by + ty * B, &res);
            bits += count_nonzero_cost(&q_blk);
            qs[ty * grid + tx] = q_blk;
            for r in 0..B {
                for c in 0..B {
                    let (py, px) = (ty * B + r, tx * B + c);
                    rec[py][px] = clip_u8(i32::from(pred[py][px]) + i32::from(d_blk[r * B + c]));
                }
            }
        }
    }
    (qs, rec, bits)
}
fn decode_tiled<const P: usize, const B: usize, const N: usize, F>(
    dst: &mut [u8], stride: usize, bx: usize, by: usize, pred: &[[u8; P]; P], mut inv_fn: F,
) -> Result<()>
where F: FnMut(&[u8], usize, usize) -> Result<[i16; N]> {
    let grid = P / B;
    for ty in 0..grid {
        for tx in 0..grid {
            let d_blk = inv_fn(dst, bx + tx * B, by + ty * B)?;
            for r in 0..B {
                for c in 0..B {
                    let (py, px) = (ty * B + r, tx * B + c);
                    dst[(by + py) * stride + bx + px] =
                        clip_u8(i32::from(pred[py][px]) + i32::from(d_blk[r * B + c]));
                }
            }
        }
    }
    Ok(())
}
fn encode_luma_macroblock(ctx: &MbEncodeCtx<'_>, rec: &mut PaddedCanvas) -> MbCandidate {
    let (profile, src, bx, by, qstep, lambda_q8) =
        (ctx.profile, ctx.src, ctx.bx, ctx.by, ctx.qstep, ctx.lambda_q8);
    let mut best: Option<MbCandidate> = None;
    let rd = |sse: u64, bits: u64| -> u64 { sse + ((lambda_q8 * bits) >> 8) };
    let push = |best: &mut Option<MbCandidate>, m: u8, mv: MotionVec, side: Vec<u8>, q4: Vec<[i16; 16]>,
                q8: Vec<[i16; 64]>, rp: [[u8; 16]; 16], bits: u64, extra_sse: u64| {
        let sse = block_dist::<16, true>(&src.y, src.pw, bx, by, &rp) + extra_sse;
        consider_mb(best, MbCandidate::new(m, mv, side, q4, q8, rp, rd(sse, bits)));
    };
    if !ctx.force_intra {
        if let Some(ref_c) = ctx.ref_canvas {
            let (mv, _) =
                diamond_search_16x16(&src.y, &ref_c.y, src.pw, src.ph, bx, by, ctx.search_range as i8);
            let pred16 = fetch_motion_patch::<16, 16>(&ref_c.y, src.pw, src.ph, bx, by, mv.dx, mv.dy);
            if mv.dx == 0 && mv.dy == 0 {
                let (cbx, cby) = (bx >> 1, by >> 1);
                let pu = read_patch::<8>(&ref_c.u, src.cw, cbx, cby);
                let pv = read_patch::<8>(&ref_c.v, src.cw, cbx, cby);
                let csse = block_dist::<8, true>(&src.u, src.cw, cbx, cby, &pu)
                    + block_dist::<8, true>(&src.v, src.cw, cbx, cby, &pv);
                push(&mut best, MB_INTER_SKIP, mv, Vec::new(), Vec::new(), Vec::new(), pred16, 2, csse);
            }
            let (q8, rec8, bits8) = encode_tiled::<16, 8, 64, 4, _>(&src.y, src.pw, bx, by, &pred16, |_, _, r| {
                transform_roundtrip_8x8(r, qstep)
            });
            let mv_bits = 16 + u64::from((mv.dx.unsigned_abs() + mv.dy.unsigned_abs()) * 2);
            push(
                &mut best, MB_INTER_8X8, mv, encode_mv_pair(mv), Vec::new(), q8.to_vec(), rec8,
                mv_bits + bits8, 0,
            );
            if matches!(profile, CodecProfile::HevcH265 | CodecProfile::Vp9) {
                consider_mb(&mut best, eval_quadtree_inter_mix(ctx, &pred16, mv));
            } else {
                let spec4 = Tx4Spec { profile, is_intra: false, is_screen_idtx: false, submode: 0, qstep };
                let (q4, rec4, bits4) =
                    encode_tiled::<16, 4, 16, 16, _>(&src.y, src.pw, bx, by, &pred16, |sx, sy, r| {
                        transform_roundtrip_4x4(spec4, &rec.y, src.pw, sx, sy, r)
                    });
                let mv_p = encode_mv_pair(mv);
                push(
                    &mut best, MB_INTER_4X4, mv, mv_p, q4.to_vec(), Vec::new(), rec4,
                    mv_bits + bits4 + 6, 0,
                );
            }
        }
    }
    let (mut best_sub, mut best_sad) = (0u8, u32::MAX);
    for sub in 0..4u8 {
        let pred = predict_intra_block::<16>(profile, sub, &rec.y, rec.pw, bx, by);
        let sad = block_dist::<16, false>(&src.y, src.pw, bx, by, &pred) as u32;
        if sad < best_sad { (best_sad, best_sub) = (sad, sub); }
    }
    let pred_i16 = predict_intra_block::<16>(profile, best_sub, &rec.y, rec.pw, bx, by);
    let (iq8, irec8, ibits8) = encode_tiled::<16, 8, 64, 4, _>(&src.y, src.pw, bx, by, &pred_i16, |_, _, r| {
        transform_roundtrip_8x8(r, qstep)
    });
    let z = MotionVec::default();
    push(
        &mut best, MB_INTRA_8X8_BASE + best_sub, z, Vec::new(), Vec::new(), iq8.to_vec(), irec8,
        8 + ibits8, 0,
    );
    for (is_idtx, base_mode) in [(false, MB_INTRA_4X4_BASE), (true, MB_AV1_INTRA_IDTX_BASE)] {
        if is_idtx && profile != CodecProfile::Av1 { continue; }
        let spec = Tx4Spec { profile, is_intra: true, is_screen_idtx: is_idtx, submode: best_sub, qstep };
        let (q4, rec4, bits4) = eval_causal_intra_4x4(spec, &src.y, &mut rec.y, src.pw, bx, by);
        push(&mut best, base_mode + best_sub, z, Vec::new(), q4.to_vec(), Vec::new(), rec4, 10 + bits4, 0);
    }
    if profile == CodecProfile::Av1 {
        if let Some((ibc_mv, _)) = search_av1_ibc_16x16(&src.y, &rec.y, src.pw, bx, by) {
            let ibc_pred = fetch_motion_patch::<16, 16>(&rec.y, src.pw, src.ph, bx, by, ibc_mv.dx, ibc_mv.dy);
            let spec = Tx4Spec { profile, is_intra: false, is_screen_idtx: true, submode: 0, qstep };
            let (q4, rec4, bits4) =
                encode_tiled::<16, 4, 16, 16, _>(&src.y, src.pw, bx, by, &ibc_pred, |sx, sy, r| {
                    transform_roundtrip_4x4(spec, &rec.y, src.pw, sx, sy, r)
                });
            let mv_p = encode_mv_pair(ibc_mv);
            push(&mut best, MB_AV1_IBC_IDTX, ibc_mv, mv_p, q4.to_vec(), Vec::new(), rec4, 18 + bits4, 0);
        }
    }
    let winner = best.unwrap_or_else(|| MbCandidate::new(
        MB_INTRA_8X8_BASE, MotionVec::default(), Vec::new(), Vec::new(),
        vec![[0; 64]; 4], [[128; 16]; 16], 0,
    ));
    write_patch::<16>(&mut rec.y, rec.pw, bx, by, &winner.rec_patch);
    winner
}
fn eval_causal_intra_4x4(
    spec: Tx4Spec, src_y: &[u8], rec_y: &mut [u8], stride: usize, bx: usize, by: usize,
) -> ([[i16; 16]; 16], [[u8; 16]; 16], u64) {
    let saved = read_patch::<16>(rec_y, stride, bx, by);
    let (mut q4, mut out_patch, mut bits) = ([[0i16; 16]; 16], [[0u8; 16]; 16], 0u64);
    for ty in 0..4usize {
        for tx in 0..4usize {
            let (sbx, sby) = (bx + tx * 4, by + ty * 4);
            let pred = predict_intra_block::<4>(spec.profile, spec.submode, rec_y, stride, sbx, sby);
            let mut res = [0i16; 16];
            for r in 0..4usize {
                for c in 0..4usize {
                    res[r * 4 + c] = i16::from(src_y[(sby + r) * stride + sbx + c]) - i16::from(pred[r][c]);
                }
            }
            let (q_blk, d_blk) = transform_roundtrip_4x4(spec, rec_y, stride, sbx, sby, &res);
            bits += count_nonzero_cost(&q_blk);
            q4[ty * 4 + tx] = q_blk;
            for r in 0..4usize {
                for c in 0..4usize {
                    let v = clip_u8(i32::from(pred[r][c]) + i32::from(d_blk[r * 4 + c]));
                    (rec_y[(sby + r) * stride + sbx + c], out_patch[ty * 4 + r][tx * 4 + c]) = (v, v);
                }
            }
        }
    }
    write_patch::<16>(rec_y, stride, bx, by, &saved);
    (q4, out_patch, bits)
}
fn eval_quadtree_inter_mix(ctx: &MbEncodeCtx<'_>, pred: &[[u8; 16]; 16], mv: MotionVec) -> MbCandidate {
    let (sy, pw, lq) = (&ctx.src.y, ctx.src.pw, ctx.lambda_q8);
    let (mut split_mask, mut q4_all, mut q8_all) = (0u8, Vec::new(), Vec::new());
    let mut rec_patch = [[0u8; 16]; 16];
    let mv_bits = 24 + u64::from((mv.dx.unsigned_abs() + mv.dy.unsigned_abs()) * 2);
    let mut total_cost = (lq * mv_bits) >> 8;
    let spec4 = Tx4Spec {
        profile: ctx.profile, is_intra: false, is_screen_idtx: false, submode: 0, qstep: ctx.qstep,
    };
    for q_idx in 0..4usize {
        let (qx_i, qy_i) = (q_idx & 1, q_idx >> 1);
        let (ox, oy, qpred) = (ctx.bx + qx_i * 8, ctx.by + qy_i * 8, extract_quad8(pred, qx_i, qy_i));
        let (q8_v, rec8, b8) = encode_tiled::<8, 8, 64, 1, _>(sy, pw, ox, oy, &qpred, |_, _, r| {
            transform_roundtrip_8x8(r, ctx.qstep)
        });
        let cost8 = block_dist::<8, true>(sy, pw, ox, oy, &rec8) + ((lq * b8) >> 8);
        let (q4_v, rec4, b4) = encode_tiled::<8, 4, 16, 4, _>(sy, pw, ox, oy, &qpred, |sx, s_y, r| {
            transform_roundtrip_4x4(spec4, sy, pw, sx, s_y, r)
        });
        let cost4 = block_dist::<8, true>(sy, pw, ox, oy, &rec4) + ((lq * (4 + b4)) >> 8);
        if cost4 < cost8 {
            split_mask |= 1u8 << q_idx;
            q4_all.extend_from_slice(&q4_v);
            total_cost += cost4;
            blit_quad8(&mut rec_patch, qx_i, qy_i, &rec4);
        } else {
            q8_all.push(q8_v[0]);
            total_cost += cost8;
            blit_quad8(&mut rec_patch, qx_i, qy_i, &rec8);
        }
    }
    let mut side_bytes = encode_mv_pair(mv);
    side_bytes.push(split_mask);
    MbCandidate::new(MB_QUADTREE_MIX, mv, side_bytes, q4_all, q8_all, rec_patch, total_cost)
}
fn predict_chroma_pair(
    profile: CodecProfile, mode: u8, mv: MotionVec, rec: &PaddedCanvas,
    ref_canvas: Option<&PaddedCanvas>, bx: usize, by: usize,
    cfl_byte_fn: impl FnOnce(u8, u8, &[[i16; 8]; 8]) -> Result<u8>,
) -> Result<([[u8; 8]; 8], [[u8; 8]; 8])> {
    let (cbx, cby) = (bx >> 1, by >> 1);
    let is_inter = matches!(mode, MB_INTER_8X8 | MB_INTER_4X4 | MB_QUADTREE_MIX);
    if is_inter || mode == MB_AV1_IBC_IDTX {
        let src_c = if is_inter { ref_canvas.ok_or(Error::ReferenceMismatch)? } else { rec };
        let (cdx, cdy) = (mv.dx / 2, mv.dy / 2);
        return Ok((
            fetch_motion_patch::<8, 8>(&src_c.u, rec.cw, rec.ch, cbx, cby, cdx, cdy),
            fetch_motion_patch::<8, 8>(&src_c.v, rec.cw, rec.ch, cbx, cby, cdx, cdy),
        ));
    }
    if profile == CodecProfile::Av1 {
        let dc_u = predict_intra_block::<8>(CodecProfile::Av1, 0, &rec.u, rec.cw, cbx, cby)[0][0];
        let dc_v = predict_intra_block::<8>(CodecProfile::Av1, 0, &rec.v, rec.cw, cbx, cby)[0][0];
        let luma_ac = compute_subsampled_luma_ac(&rec.y, rec.pw, bx, by);
        let (au, av) = unpack_cfl_pair(cfl_byte_fn(dc_u, dc_v, &luma_ac)?);
        let (mut pu, mut pv) = ([[0u8; 8]; 8], [[0u8; 8]; 8]);
        for r in 0..8usize {
            for c in 0..8usize {
                pu[r][c] = av1_cfl_sample(dc_u, au, i32::from(luma_ac[r][c]));
                pv[r][c] = av1_cfl_sample(dc_v, av, i32::from(luma_ac[r][c]));
            }
        }
        return Ok((pu, pv));
    }
    let sub = mode & 3;
    Ok((
        predict_intra_block::<8>(profile, sub, &rec.u, rec.cw, cbx, cby),
        predict_intra_block::<8>(profile, sub, &rec.v, rec.cw, cbx, cby),
    ))
}
#[inline]
fn chroma_tx_spec(mode: u8, qstep: i32) -> Tx4Spec {
    Tx4Spec {
        profile: CodecProfile::AvcH264, is_intra: false,
        is_screen_idtx: mode == MB_AV1_IBC_IDTX || mode >= MB_AV1_INTRA_IDTX_BASE, submode: 0, qstep,
    }
}
fn encode_chroma_macroblock(
    ctx: &MbEncodeCtx<'_>, mode: u8, mv: MotionVec,
    rec: &mut PaddedCanvas, side: &mut Vec<u8>, blocks4: &mut Vec<[i16; 16]>,
) {
    let (cbx, cby) = (ctx.bx >> 1, ctx.by >> 1);
    if mode == MB_INTER_SKIP {
        if let Some(ref_c) = ctx.ref_canvas {
            write_patch::<8>(&mut rec.u, rec.cw, cbx, cby, &read_patch::<8>(&ref_c.u, rec.cw, cbx, cby));
            write_patch::<8>(&mut rec.v, rec.cw, cbx, cby, &read_patch::<8>(&ref_c.v, rec.cw, cbx, cby));
        }
        return;
    }
    let Ok((pred_u, pred_v)) = predict_chroma_pair(
        ctx.profile, mode, mv, rec, ctx.ref_canvas, ctx.bx, ctx.by,
        |dc_u, dc_v, luma_ac| {
            let ou = read_patch::<8>(&ctx.src.u, ctx.src.cw, cbx, cby);
            let ov = read_patch::<8>(&ctx.src.v, ctx.src.cw, cbx, cby);
            let packed = pack_cfl_pair(
                fit_av1_cfl_alpha(&ou, luma_ac, dc_u), fit_av1_cfl_alpha(&ov, luma_ac, dc_v),
            );
            side.push(packed);
            Ok(packed)
        },
    ) else { return; };
    let (spec, stride) = (chroma_tx_spec(mode, ctx.qstep), rec.cw);
    for (src_uv, rec_uv, pred) in [(&ctx.src.u, &mut rec.u, &pred_u), (&ctx.src.v, &mut rec.v, &pred_v)] {
        let (qs, rec_p, _) = encode_tiled::<8, 4, 16, 4, _>(src_uv, stride, cbx, cby, pred, |sx, sy, r| {
            transform_roundtrip_4x4(spec, rec_uv, stride, sx, sy, r)
        });
        blocks4.extend_from_slice(&qs);
        write_patch::<8>(rec_uv, stride, cbx, cby, &rec_p);
    }
}
pub(crate) fn encode_with_optional_prior(data: &[u8], prior: Option<&RansModel>) -> Result<Vec<u8>> {
    encode_substream_best(data, ScaleBits::Bits12, prior)
}
pub(crate) fn decode_with_optional_prior(
    slice: &[u8], max_len: usize, prior: Option<&RansModel>, out: &mut Vec<u8>,
) -> Result<()> { decode_substream_maybe_prior(slice, max_len, prior, out) }

pub(crate) fn decode_video_frame_internal(
    packet: &[u8], reference: Option<&Yuv420Frame>, max_bytes: usize,
    resilient_heal: bool, priors: Option<FramePriors<'_>>,
) -> Result<(Yuv420Frame, FrameSymbolObs)> {
    let mut reader = Reader::new(packet);
    expect_magic(&mut reader, &MAGIC_VIDEO_FRAME)?;
    let mut meta = [0u8; 12];
    meta.copy_from_slice(reader.read_bytes(12)?);
    let (profile_kind, flags, qstep_u8, refresh_col_u8) = (meta[0], meta[1], meta[2], meta[3]);
    let width = usize::from(u16::from_le_bytes([meta[4], meta[5]]));
    let height = usize::from(u16::from_le_bytes([meta[6], meta[7]]));
    let expected_ref_crc = u32::from_le_bytes([meta[8], meta[9], meta[10], meta[11]]);
    let (orig_len_u32, expected_hdr_crc) = (reader.read_u32_le()?, reader.read_u32_le()?);
    let body = reader.read_bytes(packet.len().saturating_sub(24))?;
    reader.finish()?;

    if (flags & VF_COMPACT_V2) != 0 {
        let allowed = VF_LOOP_FILTER | VF_LZ_SIDE | VF_INTRA_REFRESH | VF_COMPACT_V2 | VF_KNOCKOUT_RAW_MV;
        if (flags & !allowed) != 0 || (profile_kind & 0x0F) != (CodecProfile::Compact as u8) {
            return Err(Error::UnsupportedFormat);
        }
        return crate::compact_v2::decode_compact_frame_internal(
            meta, orig_len_u32, expected_hdr_crc, body, reference, max_bytes, resilient_heal, priors,
        );
    }
    if (flags & !(VF_LOOP_FILTER | VF_LZ_SIDE | VF_INTRA_REFRESH)) != 0 || (profile_kind & 0x08) != 0 {
        return Err(Error::UnsupportedFormat);
    }
    if qstep_u8 == 0 || qstep_u8 > 64 { return Err(Error::CorruptVideoFrame); }
    let profile = CodecProfile::from_wire_id(profile_kind & 0x07)?;
    if profile == CodecProfile::Compact { return Err(Error::UnsupportedFormat); }
    let kind = match profile_kind >> 4 {
        0 => FrameKind::Key, 1 => FrameKind::Inter, _ => return Err(Error::UnsupportedFormat),
    };
    let (y_len, uv_len) = validate_yuv420_dims(width, height, max_bytes)?;
    if (orig_len_u32 as usize) != y_len + uv_len * 2 { return Err(Error::CorruptVideoFrame); }
    if framed_header_crc(&MAGIC_VIDEO_FRAME, &meta, orig_len_u32, body) != expected_hdr_crc {
        return Err(Error::ChecksumMismatch);
    }
    let (ref_canvas, reference_mismatched) = resolve_decode_reference(
        kind, reference, width, height, expected_ref_crc, max_bytes, resilient_heal,
    )?;
    let mut br = Reader::new(body);
    let (loop_filter, use_lz_side) = ((flags & VF_LOOP_FILTER) != 0, (flags & VF_LZ_SIDE) != 0);
    let sao_params = if loop_filter && profile == CodecProfile::HevcH265 {
        Some(HevcSaoParams::from_bytes(br.read_bytes(6)?)?)
    } else { None };
    let cdef_byte = if loop_filter && profile == CodecProfile::Av1 { br.read_u8()? } else { 0 };
    let (mode_s, side_s, res4_s, res8_s) =
        (br.read_substream()?, br.read_substream()?, br.read_substream()?, br.read_substream()?);
    let expected_rec_crc = br.read_u32_le()?;
    br.finish()?;
    let mut rec = PaddedCanvas::new_blank(width, height);
    let (mb_cols, mb_rows) = (rec.pw / 16, rec.ph / 16);
    let num_mbs = mb_rows.checked_mul(mb_cols).ok_or(Error::OutputTooLarge)?;
    let (mut modes, mut side) = (Vec::new(), Vec::new());
    decode_with_optional_prior(mode_s, num_mbs, priors.map(|p| p.mode), &mut modes)?;
    if modes.len() != num_mbs { return Err(Error::CorruptVideoFrame); }
    let max_side = num_mbs.checked_mul(8).ok_or(Error::OutputTooLarge)?;
    if use_lz_side { side = decompress_lz_split_with_limit(side_s, max_side)?; }
    else { decode_with_optional_prior(side_s, max_side, priors.map(|p| p.side), &mut side)?; }
    let max_b4 = num_mbs.checked_mul(24).ok_or(Error::OutputTooLarge)?;
    let max_b8 = num_mbs.checked_mul(4).ok_or(Error::OutputTooLarge)?;
    let (blocks4, res4_obs) = decode_residual_blocks_with_priors(
        res4_s, &ZIGZAG_4X4, &MAGIC_RESIDUAL_4X4, max_b4, priors.map(|p| p.res),
    )?;
    let (blocks8, res8_obs) = decode_residual_blocks_with_priors(
        res8_s, &ZIGZAG_8X8, &MAGIC_RESIDUAL_8X8, max_b8, priors.map(|p| p.res),
    )?;
    let mut cursor = MbDecodeCursor {
        profile, kind, qstep: i32::from(qstep_u8), ref_canvas: ref_canvas.as_ref(),
        side: &side, side_pos: 0, blocks4: &blocks4, idx4: 0, blocks8: &blocks8, idx8: 0,
    };
    for mby in 0..mb_rows {
        for mbx in 0..mb_cols {
            let mode = modes[mby * mb_cols + mbx];
            if (flags & VF_INTRA_REFRESH) != 0 && kind == FrameKind::Inter
                && mbx == usize::from(refresh_col_u8)
                && matches!(mode, MB_INTER_SKIP | MB_INTER_8X8 | MB_INTER_4X4 | MB_QUADTREE_MIX)
            { return Err(Error::CorruptVideoFrame); }
            let (bx, by) = (mbx * 16, mby * 16);
            let mv = decode_luma_macroblock(&mut cursor, &mut rec, mode, bx, by)?;
            decode_chroma_macroblock(&mut cursor, &mut rec, mode, mv, bx, by)?;
        }
    }
    if cursor.side_pos != side.len() || cursor.idx4 != blocks4.len() || cursor.idx8 != blocks8.len() {
        return Err(Error::CorruptVideoFrame);
    }
    if loop_filter {
        apply_loop_filter(profile, &mut rec.y, rec.pw, width, height, qstep_u8, sao_params, cdef_byte);
    }
    let out_frame = rec.to_frame(width, height);
    if !reference_mismatched && out_frame.digest_crc32() != expected_rec_crc {
        return Err(Error::ChecksumMismatch);
    }
    Ok((out_frame, FrameSymbolObs { modes, side, res4: res4_obs, res8: res8_obs }))
}
struct MbDecodeCursor<'a> {
    profile: CodecProfile, kind: FrameKind, qstep: i32, ref_canvas: Option<&'a PaddedCanvas>,
    side: &'a [u8], side_pos: usize, blocks4: &'a [[i16; 16]], idx4: usize,
    blocks8: &'a [[i16; 64]], idx8: usize,
}
impl<'a> MbDecodeCursor<'a> {
    fn next_side(&mut self) -> Result<u8> {
        let b = *self.side.get(self.side_pos).ok_or(Error::CorruptVideoFrame)?;
        self.side_pos += 1; Ok(b)
    }
    fn next_mv(&mut self) -> Result<MotionVec> {
        let (bx, by) = (i16::from(self.next_side()?) - 128, i16::from(self.next_side()?) - 128);
        if !(-64..=63).contains(&bx) || !(-64..=63).contains(&by) { return Err(Error::CorruptVideoFrame); }
        Ok(MotionVec { dx: bx as i8, dy: by as i8 })
    }
    fn next_q4(&mut self) -> Result<&'a [i16; 16]> {
        let b = self.blocks4.get(self.idx4).ok_or(Error::CorruptVideoFrame)?;
        self.idx4 += 1; Ok(b)
    }
    fn next_q8(&mut self) -> Result<&'a [i16; 64]> {
        let b = self.blocks8.get(self.idx8).ok_or(Error::CorruptVideoFrame)?;
        self.idx8 += 1; Ok(b)
    }
    fn require_inter_ref(&self) -> Result<&'a PaddedCanvas> {
        if self.kind != FrameKind::Inter { return Err(Error::CorruptVideoFrame); }
        self.ref_canvas.ok_or(Error::ReferenceMismatch)
    }
}
fn decode_luma_macroblock(
    cur: &mut MbDecodeCursor<'_>, rec: &mut PaddedCanvas, mode: u8, bx: usize, by: usize,
) -> Result<MotionVec> {
    let stride = rec.pw;
    let qstep = cur.qstep.max(1);
    match mode {
        0..=3 | MB_INTER_8X8 => {
            let (pred, mv) = if mode == MB_INTER_8X8 {
                let (ref_c, mv) = (cur.require_inter_ref()?, cur.next_mv()?);
                (fetch_motion_patch::<16, 16>(&ref_c.y, stride, rec.ph, bx, by, mv.dx, mv.dy), mv)
            } else {
                (predict_intra_block::<16>(cur.profile, mode - MB_INTRA_8X8_BASE, &rec.y, stride, bx, by),
                 MotionVec::default())
            };
            decode_tiled::<16, 8, 64, _>(&mut rec.y, stride, bx, by, &pred, |_, _, _| {
                Ok(inverse_dct_8x8(&dequantize_block(cur.next_q8()?, qstep)))
            })?;
            Ok(mv)
        }
        4..=7 | 13..=16 => {
            let is_idtx = mode >= MB_AV1_INTRA_IDTX_BASE;
            if is_idtx && cur.profile != CodecProfile::Av1 { return Err(Error::CorruptVideoFrame); }
            let sub = if is_idtx { mode - MB_AV1_INTRA_IDTX_BASE } else { mode - MB_INTRA_4X4_BASE };
            let spec = Tx4Spec {
                profile: cur.profile, is_intra: true, is_screen_idtx: is_idtx, submode: sub, qstep: cur.qstep,
            };
            for ty in 0..4usize {
                for tx in 0..4usize {
                    let (sbx, sby) = (bx + tx * 4, by + ty * 4);
                    let pred = predict_intra_block::<4>(cur.profile, sub, &rec.y, stride, sbx, sby);
                    let d_blk = inverse_luma_4x4(spec, &rec.y, stride, sbx, sby, cur.next_q4()?);
                    for r in 0..4usize {
                        for c in 0..4usize {
                            rec.y[(sby + r) * stride + sbx + c] =
                                clip_u8(i32::from(pred[r][c]) + i32::from(d_blk[r * 4 + c]));
                        }
                    }
                }
            }
            Ok(MotionVec::default())
        }
        MB_INTER_SKIP => {
            let ref_y = &cur.require_inter_ref()?.y;
            let pred = fetch_motion_patch::<16, 16>(ref_y, stride, rec.ph, bx, by, 0, 0);
            write_patch::<16>(&mut rec.y, stride, bx, by, &pred);
            Ok(MotionVec::default())
        }
        MB_INTER_4X4 | MB_AV1_IBC_IDTX => {
            let is_ibc = mode == MB_AV1_IBC_IDTX;
            let (pred, mv) = if is_ibc {
                if cur.profile != CodecProfile::Av1 { return Err(Error::CorruptVideoFrame); }
                let mv = cur.next_mv()?;
                let _ = validate_causal_ibc(stride, bx, by, mv.dx, mv.dy)?;
                (fetch_motion_patch::<16, 16>(&rec.y, stride, rec.ph, bx, by, mv.dx, mv.dy), mv)
            } else {
                let (ref_c, mv) = (cur.require_inter_ref()?, cur.next_mv()?);
                (fetch_motion_patch::<16, 16>(&ref_c.y, stride, rec.ph, bx, by, mv.dx, mv.dy), mv)
            };
            let spec = Tx4Spec {
                profile: if is_ibc { CodecProfile::Av1 } else { cur.profile },
                is_intra: false, is_screen_idtx: is_ibc, submode: 0, qstep: cur.qstep,
            };
            decode_tiled::<16, 4, 16, _>(&mut rec.y, stride, bx, by, &pred, |p, sx, sy| {
                Ok(inverse_luma_4x4(spec, p, stride, sx, sy, cur.next_q4()?))
            })?;
            Ok(mv)
        }
        MB_QUADTREE_MIX => {
            let (ref_c, mv, split_mask) = (cur.require_inter_ref()?, cur.next_mv()?, cur.next_side()?);
            if (split_mask & 0xF0) != 0 { return Err(Error::CorruptVideoFrame); }
            let pred = fetch_motion_patch::<16, 16>(&ref_c.y, stride, rec.ph, bx, by, mv.dx, mv.dy);
            let spec4 = Tx4Spec {
                profile: cur.profile, is_intra: false, is_screen_idtx: false, submode: 0, qstep: cur.qstep,
            };
            for q_idx in 0..4usize {
                let (qx_i, qy_i) = (q_idx & 1, q_idx >> 1);
                let (ox, oy, qpred) = (bx + qx_i * 8, by + qy_i * 8, extract_quad8(&pred, qx_i, qy_i));
                if (split_mask & (1u8 << q_idx)) != 0 {
                    decode_tiled::<8, 4, 16, _>(&mut rec.y, stride, ox, oy, &qpred, |p, sx, sy| {
                        Ok(inverse_luma_4x4(spec4, p, stride, sx, sy, cur.next_q4()?))
                    })?;
                } else {
                    decode_tiled::<8, 8, 64, _>(&mut rec.y, stride, ox, oy, &qpred, |_, _, _| {
                        Ok(inverse_dct_8x8(&dequantize_block(cur.next_q8()?, qstep)))
                    })?;
                }
            }
            Ok(mv)
        }
        _ => Err(Error::CorruptVideoFrame),
    }
}
fn decode_chroma_macroblock(
    cur: &mut MbDecodeCursor<'_>, rec: &mut PaddedCanvas, mode: u8, mv: MotionVec, bx: usize, by: usize,
) -> Result<()> {
    let (cbx, cby) = (bx >> 1, by >> 1);
    if mode == MB_INTER_SKIP {
        let ref_c = cur.ref_canvas.ok_or(Error::ReferenceMismatch)?;
        write_patch::<8>(&mut rec.u, rec.cw, cbx, cby, &read_patch::<8>(&ref_c.u, rec.cw, cbx, cby));
        write_patch::<8>(&mut rec.v, rec.cw, cbx, cby, &read_patch::<8>(&ref_c.v, rec.cw, cbx, cby));
        return Ok(());
    }
    let (pred_u, pred_v) = predict_chroma_pair(
        cur.profile, mode, mv, rec, cur.ref_canvas, bx, by, |_, _, _| cur.next_side(),
    )?;
    let (spec, stride) = (chroma_tx_spec(mode, cur.qstep), rec.cw);
    for (plane, pred) in [(&mut rec.u, &pred_u), (&mut rec.v, &pred_v)] {
        decode_tiled::<8, 4, 16, _>(plane, stride, cbx, cby, pred, |p, sx, sy| {
            Ok(inverse_luma_4x4(spec, p, stride, sx, sy, cur.next_q4()?))
        })?;
    }
    Ok(())
}
#[cfg(test)]
mod ibc_causal_tests {
    use super::validate_causal_ibc;
    use crate::wire::Error;
    #[test]
    fn validate_causal_ibc_accepts_upper_left_overlap_and_rejects_non_causal() {
        let (stride, bx, by) = (1280usize, 64usize, 64usize);
        for dy in [-12i8, -8, -4, 0] {
            for dx in [-64i8, -32, -16] {
                let exp = Ok(((bx as isize + dx as isize) as usize, (by as isize + dy as isize) as usize));
                assert_eq!(validate_causal_ibc(stride, bx, by, dx, dy), exp);
            }
            for bad_dx in [-12i8, 0, 16] {
                assert_eq!(validate_causal_ibc(stride, bx, by, bad_dx, dy), Err(Error::CorruptVideoFrame));
            }
        }
        assert_eq!(validate_causal_ibc(stride, bx, by, 32, -16), Ok((96, 48)));
        assert_eq!(validate_causal_ibc(stride, bx, by, -32, 4), Err(Error::CorruptVideoFrame));
    }
}
