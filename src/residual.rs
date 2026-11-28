//! H.264/HEVC/VP9/AV1 4x4 and 8x8 video transform residual entropy coder (`BRV4` / `BRV8`).

use alloc::vec;
use alloc::vec::Vec;

use crate::rans8::{
    decode_with_limit_into, decode_with_model_and_limit_into, encode_8way_with_model,
    encode_8way_with_scale, encode_order1_8way, ContextMap, RansModel, ScaleBits,
};
use crate::wire::{expect_magic, write_substream, Error, Reader, Result, DEFAULT_MAX_DECODE_BYTES};

/// Framed 4x4 residual stream magic (`BRV4`).
pub const MAGIC_RESIDUAL_4X4: [u8; 4] = *b"BRV4";
/// Framed 8x8 residual stream magic (`BRV8`).
pub const MAGIC_RESIDUAL_8X8: [u8; 4] = *b"BRV8";
/// Standard ITU-T H.264 4x4 zig-zag scan table (Table 8-13).
pub const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// Standard ITU-T H.264 8x8 zig-zag scan table (Table 8-14).
pub const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const LEVEL_ESCAPE: u8 = 255;
const ORDER1_CBP_MIN_BLOCKS: usize = 512;
const DCT8_ODD: [[i64; 4]; 4] = [
    [12, 10, 6, 3], [10, -3, -12, -6], [6, -12, 3, 10], [3, -6, 10, -12],
];
const DST7_M: [[i64; 4]; 4] = [
    [29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29],
];

/// Configuration options for transform residual entropy coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualConfig {
    /// rANS probability precision (`Bits12` default for L1D-hot LUT decode).
    pub scale: ScaleBits,
    /// Enable order-1 context modeling on `cbp_stream` to capture spatial skip-block runs.
    pub order1_cbp: bool,
}

impl Default for ResidualConfig {
    #[inline]
    fn default() -> Self { Self { scale: ScaleBits::Bits12, order1_cbp: true } }
}

/// Bijective signed-to-unsigned zigzag mapping (`0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, ...`).
#[inline] #[must_use]
pub const fn zigzag_encode_i16(v: i16) -> u16 { ((v << 1) ^ (v >> 15)) as u16 }
/// Inverse of [`zigzag_encode_i16`], exact across all 65,536 `u16` values.
#[inline] #[must_use]
pub const fn zigzag_decode_u16(z: u16) -> i16 { ((z >> 1) as i16) ^ -((z & 1) as i16) }

/// Saturate a 32-bit signed integer to `[-32768, 32767]`.
/// @domain v: -2147483648..=2147483647
#[inline]
pub(crate) const fn saturate_i16(v: i32) -> i16 {
    if v > 32_767 { 32_767 } else if v < -32_768 { -32_768 } else { v as i16 }
}
#[inline]
const fn sat_i32(v: i64) -> i32 {
    if v > 2_147_483_647 { 2_147_483_647 } else if v < -2_147_483_648 { -2_147_483_648 } else { v as i32 }
}
#[inline]
const fn sat_i16(v: i64) -> i16 {
    if v > 32_767 { 32_767 } else if v < -32_768 { -32_768 } else { v as i16 }
}
#[inline]
pub(crate) fn map_arr<T: Copy, O: Copy + Default, const N: usize>(
    inp: &[T; N], f: impl Fn(T) -> O,
) -> [O; N] {
    let mut out = [O::default(); N];
    for (dst, &v) in out.iter_mut().zip(inp) { *dst = f(v); }
    out
}

#[inline]
fn dct4_1d(a: [i32; 4], is_inv: bool) -> [i32; 4] {
    if is_inv {
        let (e0, e1, o0, o1) = (a[0] + a[2], a[0] - a[2], (a[1] >> 1) - a[3], a[1] + (a[3] >> 1));
        [e0 + o1, e1 + o0, e1 - o0, e0 - o1]
    } else {
        let (s0, s1, d0, d1) = (a[0] + a[3], a[1] + a[2], a[0] - a[3], a[1] - a[2]);
        [s0 + s1, (d0 << 1) + d1, s0 - s1, d0 - (d1 << 1)]
    }
}
fn dct4_2d(inp: [i32; 16], is_inv: bool) -> [i32; 16] {
    let (mut tmp, mut out) = ([0i32; 16], [0i32; 16]);
    for r in 0..4usize {
        let b = r * 4;
        tmp[b..b + 4].copy_from_slice(&dct4_1d([inp[b], inp[b + 1], inp[b + 2], inp[b + 3]], is_inv));
    }
    for c in 0..4usize {
        let col = dct4_1d([tmp[c], tmp[4 + c], tmp[8 + c], tmp[12 + c]], is_inv);
        for r in 0..4usize { out[r * 4 + c] = col[r]; }
    }
    out
}
/// ITU-T H.264 4x4 forward integer transform (`C_f * X * C_f^T`).
#[must_use]
pub fn forward_dct_4x4(input: &[i16; 16]) -> [i32; 16] { dct4_2d(map_arr(input, i32::from), false) }
/// ITU-T H.264 4x4 inverse integer transform (with `1/64` normalization shift).
#[must_use]
pub fn inverse_dct_4x4(coeffs: &[i32; 16]) -> [i16; 16] {
    map_arr(&dct4_2d(*coeffs, true), |v| saturate_i16((v + 32) >> 6))
}
#[inline]
fn mul_dct8_odd(v: &[i64; 4]) -> [i64; 4] {
    let mut out = [0i64; 4];
    for (slot, m) in out.iter_mut().zip(&DCT8_ODD) {
        *slot = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3];
    }
    out
}

#[inline]
fn dct8_1d(x: &[i32; 8]) -> [i32; 8] {
    let (mut a, mut b) = ([0i64; 4], [0i64; 4]);
    for k in 0..4usize { (a[k], b[k]) = (x[k] as i64 + x[7 - k] as i64, x[k] as i64 - x[7 - k] as i64); }
    let (e0, e1, e2, e3) = (a[0] + a[3], a[1] + a[2], a[0] - a[3], a[1] - a[2]);
    let odd = mul_dct8_odd(&b);
    [
        sat_i32((e0 + e1) * 8), sat_i32(odd[0]), sat_i32(10 * e2 + 4 * e3), sat_i32(odd[1]),
        sat_i32((e0 - e1) * 8), sat_i32(odd[2]), sat_i32(4 * e2 - 10 * e3), sat_i32(odd[3]),
    ]
}
#[inline]
fn idct8_1d_halves(raw: &[i32; 8]) -> ([i64; 4], [i64; 4]) {
    let (y0, y4) = (raw[0] as i64, raw[4] as i64);
    let (y2, y6) = (((raw[2] as i64) * 32 + 14) / 29, ((raw[6] as i64) * 32 + 14) / 29);
    let (e0, e2, e1, e3) = (8 * (y0 + y4), 10 * y2 + 4 * y6, 8 * (y0 - y4), 4 * y2 - 10 * y6);
    let mut y_odd = [0i64; 4];
    for k in 0..4usize { y_odd[k] = ((raw[2 * k + 1] as i64) * 256 + 144) / 289; }
    ([e0 + e2, e1 + e3, e1 - e3, e0 - e2], mul_dct8_odd(&y_odd))
}
#[inline]
fn idct8_1d_combine(even: &[i64; 4], odd: &[i64; 4]) -> [i32; 8] {
    let mut out = [0i32; 8];
    for k in 0..4usize {
        let (a, b) = (even[k], odd[k]);
        if a >= -70_000_000_000 && a <= 70_000_000_000 && b >= -70_000_000_000 && b <= 70_000_000_000 {
            (out[k], out[7 - k]) = (sat_i32(a + b), sat_i32(a - b));
        }
    }
    out
}
/// Batched 8-row 1D forward or inverse integer DCT-8 accelerator callback.
pub type Dct8BatchFn = fn(&[[i32; 8]; 8], &mut [[i32; 8]; 8]);
/// Register optional 8-row batched 1D DCT-8 and IDCT-8 accelerator callbacks (`no_std` stub).
pub fn register_dct8_accel(fwd: Dct8BatchFn, inv: Dct8BatchFn) -> bool {
    let _ = (fwd, inv);
    false
}
fn dct8_pass(in_rows: &[[i32; 8]; 8], is_inv: bool) -> [[i32; 8]; 8] {
    let mut out = [[0i32; 8]; 8];
    for r in 0..8usize {
        out[r] = if is_inv {
            let (e, o) = idct8_1d_halves(&in_rows[r]);
            idct8_1d_combine(&e, &o)
        } else { dct8_1d(&in_rows[r]) };
    }
    out
}
fn dct8_apply<T: Copy, O: Copy + Default>(
    inp: &[T; 64], is_inv: bool, to_i32: fn(T) -> i32, fin: fn(i32) -> O,
) -> [O; 64] {
    let (mut rows, mut cols, mut out) = ([[0i32; 8]; 8], [[0i32; 8]; 8], [O::default(); 64]);
    for r in 0..8usize { for c in 0..8usize { rows[r][c] = to_i32(inp[r * 8 + c]); } }
    let tr_rows = dct8_pass(&rows, is_inv);
    for c in 0..8usize {
        for r in 0..8usize {
            let v = tr_rows[r][c];
            cols[c][r] = if is_inv { sat_i32(((v as i64) + 4) >> 3) } else { v };
        }
    }
    let tr_cols = dct8_pass(&cols, is_inv);
    for r in 0..8usize { for c in 0..8usize { out[r * 8 + c] = fin(tr_cols[c][r]); } }
    out
}

/// Separable 8x8 orthogonal integer forward DCT (`D_8 * X * D_8^T`, `1/64` -> `4x` gain).
#[must_use]
pub fn forward_dct_8x8(input: &[i16; 64]) -> [i32; 64] {
    dct8_apply(input, false, i32::from, |v| sat_i32(((v as i64) + 32) >> 6))
}
/// Separable 8x8 orthogonal integer inverse DCT (`D_8^T * Y * D_8`) paired with [`forward_dct_8x8`].
#[must_use]
pub fn inverse_dct_8x8(coeffs: &[i32; 64]) -> [i16; 64] {
    dct8_apply(coeffs, true, |v| v, |v| sat_i16(((v as i64) + 256) >> 9))
}
/// HEVC/VP9/AV1 1D 4-point DST-VII / ADST mode-0 projection (`29*x0 + 55*x1 + 74*x2 + 84*x3`).
/// @domain x0: -32768..=32767, x1: -32768..=32767, x2: -32768..=32767, x3: -32768..=32767
#[inline]
pub const fn dst7_4_step(x0: i32, x1: i32, x2: i32, x3: i32) -> i32 {
    sat_i32(29 * (x0 as i64) + 55 * (x1 as i64) + 74 * (x2 as i64) + 84 * (x3 as i64))
}
fn dst7_apply<T: Copy, O: Copy + Default>(
    inp: &[T; 16], to_i64: fn(T) -> i64, tr: bool, sh: (u32, u32), sat_mid: bool, fin: fn(i64) -> O,
) -> [O; 16] {
    let dot = |k: usize, v: [i64; 4]| -> i64 {
        let m = |j: usize| if tr { DST7_M[j][k] } else { DST7_M[k][j] };
        m(0) * v[0] + m(1) * v[1] + m(2) * v[2] + m(3) * v[3]
    };
    let (rnd1, rnd2, mut tmp, mut out) =
        (1i64 << (sh.0 - 1), 1i64 << (sh.1 - 1), [0i64; 16], [O::default(); 16]);
    for r in 0..4usize {
        let b = r * 4;
        let row = [to_i64(inp[b]), to_i64(inp[b + 1]), to_i64(inp[b + 2]), to_i64(inp[b + 3])];
        for k in 0..4usize {
            let v = (dot(k, row) + rnd1) >> sh.0;
            tmp[b + k] = if sat_mid { i64::from(sat_i16(v)) } else { v };
        }
    }
    for c in 0..4usize {
        let col = [tmp[c], tmp[4 + c], tmp[8 + c], tmp[12 + c]];
        for k in 0..4usize { out[k * 4 + c] = fin((dot(k, col) + rnd2) >> sh.1); }
    }
    out
}

/// Separable 4x4 integer DST-VII / ADST forward transform (`S_4 * X * S_4^T`, `4x` gain).
#[must_use]
pub fn forward_dst7_4x4(input: &[i16; 16]) -> [i32; 16] {
    dst7_apply(input, i64::from, false, (6, 6), false, sat_i32)
}
/// Separable 4x4 integer DST-VII / ADST inverse transform (`S_4^T * Y * S_4`).
#[must_use]
pub fn inverse_dst7_4x4(coeffs: &[i32; 16]) -> [i16; 16] {
    dst7_apply(coeffs, i64::from, true, (7, 9), true, sat_i16)
}
/// AV1 4x4 forward Identity Transform (`IDTX`, `4x` orthonormal energy scale).
#[must_use]
pub fn forward_idtx_4x4(input: &[i16; 16]) -> [i32; 16] { map_arr(input, |v| i32::from(v) << 2) }
/// AV1 4x4 inverse Identity Transform (`IDTX`) paired with [`forward_idtx_4x4`].
#[must_use]
pub fn inverse_idtx_4x4(coeffs: &[i32; 16]) -> [i16; 16] {
    map_arr(coeffs, |v| sat_i16(((v as i64) + 2) >> 2))
}
/// Dead-zone uniform quantizer for transform blocks (`qstep >= 1`).
#[must_use]
pub fn quantize_block<const N: usize>(coeffs: &[i32; N], qstep: i32) -> [i16; N] {
    let (step, dz) = (qstep.max(1), i64::from(qstep.max(1) / 3));
    map_arr(coeffs, |v| {
        let q = (v.unsigned_abs() as i64 + dz) / i64::from(step);
        saturate_i16((if v < 0 { -q } else { q }) as i32)
    })
}
/// Uniform inverse quantizer for transform blocks (`qstep >= 1`).
#[must_use]
pub fn dequantize_block<const N: usize>(qcoeffs: &[i16; N], qstep: i32) -> [i32; N] {
    let step = qstep.max(1);
    map_arr(qcoeffs, |v| i32::from(v) * step)
}

#[derive(Clone, Copy)]
pub(crate) struct ResidualPriors<'a> {
    pub(crate) cbp: &'a RansModel,
    pub(crate) run: &'a RansModel,
    pub(crate) level_lo: &'a RansModel,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct ResidualStreamObs {
    pub(crate) cbp: Vec<u8>,
    pub(crate) runs: Vec<u8>,
    pub(crate) levels_lo: Vec<u8>,
}

pub(crate) fn encode_substream_best(
    data: &[u8], scale: ScaleBits, prior: Option<&RansModel>,
) -> Result<Vec<u8>> {
    let inline = encode_8way_with_scale(data, scale)?;
    if let Some(model) = prior {
        if let Ok(ext) = encode_8way_with_model(data, model) {
            if ext.len() < inline.len() { return Ok(ext); }
        }
    }
    Ok(inline)
}
pub(crate) fn decode_substream_maybe_prior(
    slice: &[u8], max_len: usize, prior: Option<&RansModel>, out: &mut Vec<u8>,
) -> Result<()> {
    if let Some(m) = prior { decode_with_model_and_limit_into(slice, m, max_len, out) }
    else { decode_with_limit_into(slice, max_len, out) }
}
pub(crate) fn encode_residual_blocks_with_priors<const N: usize>(
    blocks: &[[i16; N]], zigzag: &[usize; N], magic: &[u8; 4],
    cfg: ResidualConfig, priors: Option<ResidualPriors<'_>>,
) -> Result<(Vec<u8>, ResidualStreamObs)> {
    let num_blocks = u32::try_from(blocks.len()).map_err(|_| Error::InvalidParameter)?;
    let (mut cbp, mut runs, mut levels_lo, mut levels_esc) = (
        Vec::with_capacity(blocks.len()), Vec::with_capacity(blocks.len() * 2),
        Vec::with_capacity(blocks.len() * 2), Vec::new(),
    );
    for block in blocks {
        let last_nz = (0..N).rev().find(|&p| block[zigzag[p]] != 0).map_or(0, |p| p + 1);
        cbp.push(last_nz as u8);
        let mut zero_run = 0u8;
        for &idx in &zigzag[..last_nz] {
            let coeff = block[idx];
            if coeff == 0 { zero_run += 1; continue; }
            runs.push(zero_run);
            zero_run = 0;
            let zz_m1 = zigzag_encode_i16(coeff) - 1;
            if zz_m1 < u16::from(LEVEL_ESCAPE) { levels_lo.push(zz_m1 as u8); }
            else { levels_lo.push(LEVEL_ESCAPE); levels_esc.extend_from_slice(&zz_m1.to_le_bytes()); }
        }
    }
    let mut cbp_enc = encode_substream_best(&cbp, cfg.scale, priors.map(|p| p.cbp))?;
    if cfg.order1_cbp && blocks.len() >= ORDER1_CBP_MIN_BLOCKS {
        let ord1 = encode_order1_8way(&cbp, ContextMap::Low15)?;
        if ord1.len() < cbp_enc.len() { cbp_enc = ord1; }
    }
    let runs_enc = encode_substream_best(&runs, cfg.scale, priors.map(|p| p.run))?;
    let lo_enc = encode_substream_best(&levels_lo, cfg.scale, priors.map(|p| p.level_lo))?;
    let esc_enc = encode_8way_with_scale(&levels_esc, cfg.scale)?;
    let mut out = Vec::with_capacity(16 + cbp_enc.len() + runs_enc.len() + lo_enc.len() + esc_enc.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(&num_blocks.to_le_bytes());
    for sub in [&cbp_enc, &runs_enc, &lo_enc, &esc_enc] { write_substream(&mut out, sub)?; }
    Ok((out, ResidualStreamObs { cbp, runs, levels_lo }))
}

pub(crate) fn decode_residual_blocks_with_priors<const N: usize>(
    input: &[u8], zigzag: &[usize; N], magic: &[u8; 4],
    max_blocks: usize, priors: Option<ResidualPriors<'_>>,
) -> Result<(Vec<[i16; N]>, ResidualStreamObs)> {
    let mut r = Reader::new(input);
    expect_magic(&mut r, magic)?;
    let num_blocks = r.read_u32_le()? as usize;
    if num_blocks > max_blocks { return Err(Error::OutputTooLarge); }
    let (cbp_s, runs_s, lo_s, esc_s) =
        (r.read_substream()?, r.read_substream()?, r.read_substream()?, r.read_substream()?);
    r.finish()?;
    let max_coeffs = num_blocks.checked_mul(N).ok_or(Error::OutputTooLarge)?;
    let max_esc = max_coeffs.checked_mul(2).ok_or(Error::OutputTooLarge)?;
    let (mut cbp, mut runs, mut levels_lo, mut levels_esc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    decode_substream_maybe_prior(cbp_s, num_blocks, priors.map(|p| p.cbp), &mut cbp)?;
    decode_substream_maybe_prior(runs_s, max_coeffs, priors.map(|p| p.run), &mut runs)?;
    decode_substream_maybe_prior(lo_s, max_coeffs, priors.map(|p| p.level_lo), &mut levels_lo)?;
    decode_with_limit_into(esc_s, max_esc, &mut levels_esc)?;
    if cbp.len() != num_blocks || runs.len() != levels_lo.len() || (levels_esc.len() & 1) != 0 {
        return Err(Error::CorruptResidualBlock);
    }

    let (mut blocks, mut nz_idx, mut esc_idx) = (vec![[0i16; N]; num_blocks], 0usize, 0usize);
    for (block, &last_u8) in blocks.iter_mut().zip(&cbp) {
        let last_plus1 = usize::from(last_u8);
        if last_plus1 == 0 { continue; }
        if last_plus1 > N { return Err(Error::CorruptResidualBlock); }
        let (target, mut pos) = (last_plus1 - 1, 0usize);
        loop {
            let run = usize::from(*runs.get(nz_idx).ok_or(Error::CorruptResidualBlock)?);
            let lo = *levels_lo.get(nz_idx).ok_or(Error::CorruptResidualBlock)?;
            nz_idx += 1;
            pos = pos.checked_add(run).ok_or(Error::CorruptResidualBlock)?;
            if pos > target { return Err(Error::CorruptResidualBlock); }
            let zz_m1 = if lo < LEVEL_ESCAPE { u16::from(lo) } else {
                let p = levels_esc.get(esc_idx..esc_idx + 2).ok_or(Error::CorruptResidualBlock)?;
                esc_idx += 2;
                let v = u16::from_le_bytes([p[0], p[1]]);
                if v < u16::from(LEVEL_ESCAPE) || v == u16::MAX { return Err(Error::CorruptResidualBlock); }
                v
            };
            block[zigzag[pos]] = zigzag_decode_u16(zz_m1 + 1);
            if pos == target { break; }
            pos += 1;
        }
    }
    if nz_idx != runs.len() || esc_idx != levels_esc.len() { return Err(Error::CorruptResidualBlock); }
    Ok((blocks, ResidualStreamObs { cbp, runs, levels_lo }))
}
macro_rules! residual_api {
    ($enc:ident, $enc_cfg:ident, $dec:ident, $dec_lim:ident, $n:expr, $zz:expr, $mag:expr, $d:literal) => {
        #[doc = concat!("Encode quantized ", $d, " video transform residual blocks.")]
        pub fn $enc(blocks: &[[i16; $n]]) -> Result<Vec<u8>> { $enc_cfg(blocks, ResidualConfig::default()) }
        #[doc = concat!("Encode quantized ", $d, " video transform residual blocks with config.")]
        pub fn $enc_cfg(blocks: &[[i16; $n]], cfg: ResidualConfig) -> Result<Vec<u8>> {
            Ok(encode_residual_blocks_with_priors(blocks, &$zz, &$mag, cfg, None)?.0)
        }
        #[doc = concat!("Decode framed ", $d, " quantized transform residual blocks.")]
        pub fn $dec(input: &[u8]) -> Result<Vec<[i16; $n]>> {
            $dec_lim(input, DEFAULT_MAX_DECODE_BYTES / ($n * 2))
        }
        #[doc = concat!("Decode framed ", $d, " residual blocks with explicit block ceiling.")]
        pub fn $dec_lim(input: &[u8], max_blocks: usize) -> Result<Vec<[i16; $n]>> {
            Ok(decode_residual_blocks_with_priors(input, &$zz, &$mag, max_blocks, None)?.0)
        }
    };
}
residual_api!(
    encode_residuals_4x4, encode_residuals_4x4_with_config,
    decode_residuals_4x4, decode_residuals_4x4_with_limit, 16, ZIGZAG_4X4, MAGIC_RESIDUAL_4X4, "4x4"
);
residual_api!(
    encode_residuals_8x8, encode_residuals_8x8_with_config,
    decode_residuals_8x8, decode_residuals_8x8_with_limit, 64, ZIGZAG_8X8, MAGIC_RESIDUAL_8X8, "8x8"
);

#[cfg(test)]
mod tests {
    use super::*;

    fn sat_i32_128(v: i128) -> i32 {
        if v > i128::from(i32::MAX) { i32::MAX }
        else if v < i128::from(i32::MIN) { i32::MIN }
        else { v as i32 }
    }
    #[test]
    fn zigzag_bijection_all_i16() {
        for raw in i16::MIN..=i16::MAX { assert_eq!(zigzag_decode_u16(zigzag_encode_i16(raw)), raw); }
    }
    #[test]
    fn residual_4x4_and_8x8_roundtrip_extremes() {
        let mut b4 = vec![[0i16; 16]; 8];
        (b4[1][0], b4[1][15], b4[7][5]) = (i16::MIN, i16::MAX, -32767);
        b4[3] = [1, -1, 2, -2, 127, -127, 128, -128, 0, 0, 500, -500, 0, 0, 0, 1];
        assert_eq!(decode_residuals_4x4(&encode_residuals_4x4(&b4).expect("enc4")).expect("dec4"), b4);
        let mut b8 = vec![[0i16; 64]; 6];
        (b8[0][0], b8[0][63], b8[4][17], b8[4][18]) = (i16::MAX, i16::MIN, 128, -128);
        assert_eq!(decode_residuals_8x8(&encode_residuals_8x8(&b8).expect("enc8")).expect("dec8"), b8);
        assert!(decode_residuals_4x4(&encode_residuals_4x4(&[]).expect("empty")).expect("dec").is_empty());
    }
    #[test]
    fn h264_4x4_dct_exact_inverse_norm() {
        const NORM: [i32; 16] = [
            256, 205, 256, 205, 205, 164, 205, 164, 256, 205, 256, 205, 205, 164, 205, 164,
        ];
        let blk: [i16; 16] = core::array::from_fn(|i| (i as i16) * 9 - 60);
        let tr = forward_dct_4x4(&blk);
        let rec = inverse_dct_4x4(&core::array::from_fn(|i| (tr[i] * NORM[i] + 32) >> 6));
        for i in 0..16 { assert!((blk[i] - rec[i]).abs() <= 1); }
    }
    #[test]
    fn dct8_dst7_idtx_near_lossless_accuracy() {
        let blk8: [i16; 64] = core::array::from_fn(|i| {
            ((i / 8) as i16) * 11 - ((i % 8) as i16) * 9 + ((i as i16 & 3) * 5) - 20
        });
        let rec8 = inverse_dct_8x8(&forward_dct_8x8(&blk8));
        for i in 0..64 { assert!((blk8[i] - rec8[i]).abs() <= 2); }
        let blk4: [i16; 16] = core::array::from_fn(|i| (i as i16) * 13 - 95);
        let tr_dst = forward_dst7_4x4(&blk4);
        let rec_dst = inverse_dst7_4x4(&tr_dst);
        for i in 0..16 { assert!((blk4[i] - rec_dst[i]).abs() <= 2); }
        let tr_id = forward_idtx_4x4(&blk4);
        assert_eq!(inverse_idtx_4x4(&tr_id), blk4);
        assert_ne!(tr_dst, tr_id);
    }
    #[test]
    fn dct8_idct8_and_dst7_extreme_i32_corners_exact_and_envelope_bound() {
        let (mut max_even, mut max_odd) = (0i64, 0i64);
        for mask in 0u16..256u16 {
            let v: [i32; 8] =
                core::array::from_fn(|k| if ((mask >> k) & 1) != 0 { i32::MAX } else { i32::MIN });
            assert_eq!(dct8_1d(&v)[0], sat_i32_128(v.iter().map(|&x| i128::from(x)).sum::<i128>() * 8));
            let (even, odd) = idct8_1d_halves(&v);
            let out = idct8_1d_combine(&even, &odd);
            for k in 0..4usize {
                (max_even, max_odd) = (max_even.max(even[k].abs()), max_odd.max(odd[k].abs()));
                assert!(even[k].abs() < 70_000_000_000 && odd[k].abs() < 70_000_000_000);
                assert_eq!(out[k], sat_i32_128(i128::from(even[k]) + i128::from(odd[k])));
                assert_eq!(out[7 - k], sat_i32_128(i128::from(even[k]) - i128::from(odd[k])));
            }
        }
        assert!(max_even > 60_000_000_000 && max_odd > 50_000_000_000);
        let ext = [
            i32::MAX, i32::MIN, i32::MAX, i32::MIN, i32::MIN, i32::MAX, i32::MIN, i32::MAX,
            i32::MAX, i32::MAX, i32::MIN, i32::MIN, 0, i32::MAX, i32::MIN, 1,
        ];
        assert_ne!(inverse_dst7_4x4(&ext), [0i16; 16]);
    }
}
