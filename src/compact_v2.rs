//! Compact ultra-low-bitrate (`<= 0.050 bpp`) video profile (`BVF1` + `VF_COMPACT_V2`).

use alloc::vec;
use alloc::vec::Vec;

use crate::lz_split::decompress_lz_split_with_limit;
use crate::rans8::{encode_order1_8way, ContextMap, RansModel};
use crate::residual::{
    forward_dct_8x8, inverse_dct_8x8, saturate_i16, zigzag_decode_u16, zigzag_encode_i16,
    ResidualStreamObs, ZIGZAG_8X8,
};
use crate::video::{
    apply_8x8_edge_filter, avc_deblock_delta, blit_quad8, block_dist, clip_u8,
    decode_with_optional_prior, encode_side_stream, encode_with_optional_prior, extract_quad8,
    finish_bvf1_frame, predict_intra_block, resolve_decode_reference, validate_encode_inputs,
    validate_yuv420_dims, write_patch, CodecProfile, FrameKind, FramePriors, FrameSymbolObs,
    PaddedCanvas, VideoConfig, Yuv420Frame, MAGIC_VIDEO_FRAME, VF_COMPACT_V2, VF_INTRA_REFRESH,
    VF_KNOCKOUT_RAW_MV, VF_LOOP_FILTER, VF_LZ_SIDE,
};
use crate::wire::{expect_magic, framed_header_crc, write_substream, Error, Reader, Result};

const COMPACT_INTRA_BASE: u8 = 0;
const COMPACT_PMV_SKIP: u8 = 4;
const COMPACT_MVD_SKIP: u8 = 5;
const COMPACT_PMV_RES: u8 = 6;
const COMPACT_MVD_RES: u8 = 7;

const TOK_LAST_BASE: u8 = 0xF0;
const TOK_ZRL15: u8 = 0xFE;
const TOK_EOB: u8 = 0xFF;
const LAMBDA_QUAD_COEFF: u64 = 36;

/// Compute perceptual `8x8` quantizer step table weighted by `(16 + 3 * (r + c)) / 16`.
/// @domain q_plane: 0..=4096
#[inline]
const fn build_quant_steps(q_plane: i32) -> [i32; 64] {
    let qp = if q_plane < 1 { 1 } else if q_plane > 4096 { 4096 } else { q_plane };
    let mut steps = [1i32; 64];
    let mut r = 0usize;
    while r < 8 {
        let mut c = 0usize;
        while c < 8 {
            let s = (qp * 8 * (16 + 3 * ((r + c) as i32)) + 8) >> 4;
            steps[r * 8 + c] = if s < 1 { 1 } else { s };
            c += 1;
        }
        r += 1;
    }
    steps
}

/// Quadratic Lagrangian rate multiplier `lambda_q8(q_px) = 36 * q_px^2`.
#[inline] #[must_use]
pub const fn quadratic_lambda_q8(q_px: u8) -> u64 {
    let q = if q_px == 0 { 1u64 } else { q_px as u64 };
    LAMBDA_QUAD_COEFF * q * q
}

#[inline]
fn quantize_8x8_perceptual(coeffs: &[i32; 64], steps: &[i32; 64], is_intra: bool) -> [i16; 64] {
    let mut out = [0i16; 64];
    for (slot, (&v, &step)) in out.iter_mut().zip(coeffs.iter().zip(steps)) {
        let dz = if is_intra { step / 3 } else { step / 6 };
        let q = (v.unsigned_abs() as i64 + i64::from(dz)) / i64::from(step);
        *slot = saturate_i16((if v < 0 { -q } else { q }) as i32);
    }
    out
}

#[inline]
fn dequant_idct_8x8(qcoeffs: &[i16; 64], steps: &[i32; 64]) -> [i16; 64] {
    let mut dq = [0i32; 64];
    for (slot, (&q, &s)) in dq.iter_mut().zip(qcoeffs.iter().zip(steps)) {
        *slot = i32::from(q) * s;
    }
    inverse_dct_8x8(&dq)
}

#[inline]
fn clip_idx(v: isize, hi: usize) -> usize {
    if hi == 0 || v <= 0 { 0 } else if (v as usize) >= hi { hi - 1 } else { v as usize }
}

fn fetch_halfpel_patch<const W: usize, const H: usize>(
    plane: &[u8], stride: usize, ph: usize, bx: usize, by: usize, mv_x_hp: i8, mv_y_hp: i8,
) -> [[u8; W]; H] {
    let (fx, fy) = ((mv_x_hp & 1) as usize, (mv_y_hp & 1) as usize);
    let (sx0, sy0) = ((bx as isize) + ((mv_x_hp >> 1) as isize), (by as isize) + ((mv_y_hp >> 1) as isize));
    let mut out = [[128u8; W]; H];
    if fx == 0 && fy == 0 && sx0 >= 0 && sy0 >= 0
        && (sx0 as usize) + W <= stride && (sy0 as usize) + H <= ph
    {
        let (ux, uy) = (sx0 as usize, sy0 as usize);
        for r in 0..H {
            let off = (uy + r) * stride + ux;
            out[r].copy_from_slice(&plane[off..off + W]);
        }
        return out;
    }
    let (mut ys0, mut ys1, mut xs0, mut xs1) = ([0usize; H], [0usize; H], [0usize; W], [0usize; W]);
    for r in 0..H {
        ys0[r] = clip_idx(sy0 + r as isize, ph) * stride;
        ys1[r] = clip_idx(sy0 + r as isize + fy as isize, ph) * stride;
    }
    for c in 0..W {
        xs0[c] = clip_idx(sx0 + c as isize, stride);
        xs1[c] = clip_idx(sx0 + c as isize + fx as isize, stride);
    }
    for (r, row_out) in out.iter_mut().enumerate() {
        let (r0, r1) = (ys0[r], ys1[r]);
        for (c, slot) in row_out.iter_mut().enumerate() {
            let (x0, x1) = (xs0[c], xs1[c]);
            let a = u16::from(plane[r0 + x0]);
            *slot = match (fx, fy) {
                (0, 0) => a as u8,
                (1, 0) => ((a + u16::from(plane[r0 + x1]) + 1) >> 1) as u8,
                (0, 1) => ((a + u16::from(plane[r1 + x0]) + 1) >> 1) as u8,
                _ => {
                    let b = u16::from(plane[r0 + x1]) + u16::from(plane[r1 + x0]);
                    ((a + b + u16::from(plane[r1 + x1]) + 2) >> 2) as u8
                }
            };
        }
    }
    out
}

fn fetch_mb_halfpel(
    ref_c: &PaddedCanvas, bx: usize, by: usize, mv: (i8, i8),
) -> ([[u8; 16]; 16], [[u8; 8]; 8], [[u8; 8]; 8]) {
    let (cbx, cby) = (bx >> 1, by >> 1);
    (
        fetch_halfpel_patch::<16, 16>(&ref_c.y, ref_c.pw, ref_c.ph, bx, by, mv.0, mv.1),
        fetch_halfpel_patch::<8, 8>(&ref_c.u, ref_c.cw, ref_c.ch, cbx, cby, mv.0 >> 1, mv.1 >> 1),
        fetch_halfpel_patch::<8, 8>(&ref_c.v, ref_c.cw, ref_c.ch, cbx, cby, mv.0 >> 1, mv.1 >> 1),
    )
}

#[inline]
fn sse16_hp(
    cur: &[u8], ref_p: &[u8], stride: usize, ph: usize, bx: usize, by: usize, mx: i8, my: i8,
) -> u64 {
    let patch = fetch_halfpel_patch::<16, 16>(ref_p, stride, ph, bx, by, mx, my);
    block_dist::<16, true>(cur, stride, bx, by, &patch)
}

#[inline]
const fn mvd_eg_bits(delta: i8) -> u64 {
    let a = delta.unsigned_abs();
    if a == 0 { 1 } else { 1 + 2 * ((8 - a.leading_zeros()) as u64) }
}

#[inline]
const fn mvd_pair_bits(mv: (i8, i8), pmv: (i8, i8)) -> u64 {
    if mv.0 == pmv.0 && mv.1 == pmv.1 { 0 }
    else { 2 + mvd_eg_bits(mv.0.wrapping_sub(pmv.0)) + mvd_eg_bits(mv.1.wrapping_sub(pmv.1)) }
}

#[inline]
const fn clip_i8(v: i32) -> i8 {
    if v < -128 { -128 } else if v > 127 { 127 } else { v as i8 }
}

#[inline]
const fn median3_i8(a: i8, b: i8, c: i8) -> i8 {
    if a <= b { if b <= c { b } else if a <= c { c } else { a } }
    else if a <= c { a } else if b <= c { c } else { b }
}

/// Compute causal spatial median motion predictor (`PMV`) in half-pel units.
#[inline] #[must_use]
pub fn spatial_median_pmv(mv_grid: &[(i8, i8)], mb_cols: usize, mb_x: usize, mb_y: usize) -> (i8, i8) {
    if mb_y == 0 { return if mb_x == 0 { (0, 0) } else { mv_grid[mb_x - 1] }; }
    let top = mv_grid[(mb_y - 1) * mb_cols + mb_x];
    if mb_x == 0 {
        return if mb_cols > 1 {
            let tr = mv_grid[(mb_y - 1) * mb_cols + 1];
            (median3_i8(0, top.0, tr.0), median3_i8(0, top.1, tr.1))
        } else { top };
    }
    let left = mv_grid[mb_y * mb_cols + mb_x - 1];
    let tr = if mb_x + 1 < mb_cols {
        mv_grid[(mb_y - 1) * mb_cols + mb_x + 1]
    } else {
        mv_grid[(mb_y - 1) * mb_cols + mb_x - 1]
    };
    (median3_i8(left.0, top.0, tr.0), median3_i8(left.1, top.1, tr.1))
}

fn diamond_mv_search<F: FnMut(i8, i8) -> u64>(
    start: (i8, i8), bounds: (i32, i32, i32, i32), max_iters: usize,
    mut eval: F, mut best: (i8, i8, u64),
) -> (i8, i8, u64) {
    let (mut cx, mut cy) = start;
    for step in [16i32, 8, 4, 2] {
        for _ in 0..max_iters {
            let mut improved = false;
            for (dx, dy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
                let (nx, ny) = (i32::from(cx) + dx, i32::from(cy) + dy);
                if nx < bounds.0 || nx > bounds.1 || ny < bounds.2 || ny > bounds.3 { continue; }
                let j = eval(nx as i8, ny as i8);
                if j < best.2 {
                    (best, cx, cy, improved) = ((nx as i8, ny as i8, j), nx as i8, ny as i8, true);
                }
            }
            if !improved { break; }
        }
    }
    best
}

/// Quadratic Lagrangian half-pel motion search (`[-128..=127]` half-pel) around `pmv_hp`.
#[must_use]
pub fn search_halfpel_mvd(
    cur: &[u8], ref_p: &[u8], stride: usize, mb_x: usize, mb_y: usize, pmv_hp: (i8, i8), lambda_q8: u64,
) -> (i8, i8, u64) {
    if stride == 0 { return (0, 0, u64::MAX); }
    let (ph, bx, by) = (ref_p.len() / stride, mb_x * 16, mb_y * 16);
    let eval = |mx: i8, my: i8| -> u64 {
        sse16_hp(cur, ref_p, stride, ph, bx, by, mx, my)
            + ((lambda_q8 * mvd_pair_bits((mx, my), pmv_hp)) >> 8)
    };
    let mut best = (pmv_hp.0, pmv_hp.1, eval(pmv_hp.0, pmv_hp.1));
    for cand in [(0, 0), (pmv_hp.0 & !1, pmv_hp.1 & !1)] {
        if cand != (best.0, best.1) {
            let j = eval(cand.0, cand.1);
            if j < best.2 { best = (cand.0, cand.1, j); }
        }
    }
    if best.2 <= 96 { return best; }
    let bounds = (
        i32::from(clip_i8(i32::from(pmv_hp.0) - 48)), i32::from(clip_i8(i32::from(pmv_hp.0) + 48)),
        i32::from(clip_i8(i32::from(pmv_hp.1) - 48)), i32::from(clip_i8(i32::from(pmv_hp.1) + 48)),
    );
    best = diamond_mv_search((best.0 & !1, best.1 & !1), bounds, 4, &eval, best);
    let (bx0, by0) = (i32::from(best.0), i32::from(best.1));
    for dy in -1..=1i32 {
        for dx in -1..=1i32 {
            let (nx, ny) = (bx0 + dx, by0 + dy);
            if (dx != 0 || dy != 0) && (-128..=127).contains(&nx) && (-128..=127).contains(&ny) {
                let j = eval(nx as i8, ny as i8);
                if j < best.2 { best = (nx as i8, ny as i8, j); }
            }
        }
    }
    best
}

fn search_intpel_knockout(
    cur: &[u8], ref_p: &[u8], stride: usize, mb_x: usize, mb_y: usize, lambda_q8: u64,
) -> (i8, i8, u64) {
    let (ph, bx, by) = (ref_p.len() / stride, mb_x * 16, mb_y * 16);
    let eval = |mx: i8, my: i8| -> u64 {
        let bits = 16 + mvd_eg_bits(mx) + mvd_eg_bits(my);
        sse16_hp(cur, ref_p, stride, ph, bx, by, mx, my) + ((lambda_q8 * bits) >> 8)
    };
    let start = (0i8, 0i8, eval(0, 0));
    diamond_mv_search((0, 0), (-48, 48, -48, 48), 1, eval, start)
}

#[inline]
fn estimate_block8_bits(q: &[i16; 64]) -> u64 {
    let Some(last) = (0..64usize).rev().find(|&p| q[ZIGZAG_8X8[p]] != 0) else { return 0; };
    let (mut bits, mut zrun) = (0u64, 0usize);
    for pos in 0..=last {
        let v = q[ZIGZAG_8X8[pos]];
        if v == 0 { zrun += 1; continue; }
        while zrun >= 15 { bits += 6; zrun -= 15; }
        let abs = v.unsigned_abs();
        if pos == last && zrun <= 6 && abs == 1 {
            bits += if zrun == 0 { 3 } else { 5 };
        } else {
            let mag = if abs == 1 { 3 }
                else if abs <= 3 { 5 }
                else { 6 + 2 * u64::from(16 - abs.leading_zeros()) };
            bits += mag + u64::from(zrun != 0) * 2 + u64::from(pos == last && last < 63) * 2;
        }
        zrun = 0;
    }
    bits
}

fn push_block8_tokens(q: &[i16; 64], tokens: &mut Vec<u8>, esc: &mut Vec<u8>) {
    let Some(last_pos) = (0..64usize).rev().find(|&p| q[ZIGZAG_8X8[p]] != 0) else {
        tokens.push(TOK_EOB);
        return;
    };
    let mut zrun = 0u8;
    for pos in 0..=last_pos {
        let coeff = q[ZIGZAG_8X8[pos]];
        if coeff == 0 { zrun += 1; continue; }
        while zrun >= 15 { tokens.push(TOK_ZRL15); zrun -= 15; }
        let zz_m1 = zigzag_encode_i16(coeff) - 1;
        if pos == last_pos && zrun <= 6 && zz_m1 <= 1 {
            tokens.push(TOK_LAST_BASE + (zrun << 1) + (zz_m1 as u8));
            return;
        }
        if zz_m1 < 15 { tokens.push((zrun << 4) | (zz_m1 as u8)); }
        else { tokens.push((zrun << 4) | 0x0F); esc.extend_from_slice(&zz_m1.to_le_bytes()); }
        if pos == last_pos && last_pos < 63 { tokens.push(TOK_EOB); }
        zrun = 0;
    }
}

fn decode_block8_tokens(
    tokens: &[u8], tok_pos: &mut usize, esc: &[u8], esc_pos: &mut usize,
) -> Result<[i16; 64]> {
    let (mut out, mut idx) = ([0i16; 64], 0usize);
    while idx < 64 {
        let tok = *tokens.get(*tok_pos).ok_or(Error::CorruptVideoFrame)?;
        *tok_pos += 1;
        if tok == TOK_EOB { break; }
        if tok == TOK_ZRL15 {
            idx += 15;
            if idx >= 64 { return Err(Error::CorruptVideoFrame); }
            continue;
        }
        let is_last = tok >= TOK_LAST_BASE;
        idx += if is_last { ((tok - TOK_LAST_BASE) >> 1) as usize } else { (tok >> 4) as usize };
        if idx >= 64 { return Err(Error::CorruptVideoFrame); }
        let zz_m1 = if is_last { u16::from((tok - TOK_LAST_BASE) & 1) }
            else if (tok & 0x0F) < 15 { u16::from(tok & 0x0F) }
            else {
                let p = esc.get(*esc_pos..*esc_pos + 2).ok_or(Error::CorruptVideoFrame)?;
                *esc_pos += 2;
                u16::from_le_bytes([p[0], p[1]])
            };
        out[ZIGZAG_8X8[idx]] = zigzag_decode_u16(zz_m1 + 1);
        if is_last { break; }
        idx += 1;
    }
    Ok(out)
}

#[repr(C, align(64))]
struct CodedBlock8 { coded: bool, q: [i16; 64], rec: [[u8; 8]; 8], sse: u64, bits: u64 }

fn eval_block8_rdo(
    src: &[u8], stride: usize, bx: usize, by: usize, pred: &[[u8; 8]; 8],
    steps: &[i32; 64], is_intra: bool, lambda_q8: u64,
) -> CodedBlock8 {
    let sse_zero = block_dist::<8, true>(src, stride, bx, by, pred);
    let mut res = [0i16; 64];
    for r in 0..8usize {
        let row = (by + r) * stride + bx;
        for c in 0..8usize { res[r * 8 + c] = i16::from(src[row + c]) - i16::from(pred[r][c]); }
    }
    let q = quantize_8x8_perceptual(&forward_dct_8x8(&res), steps, is_intra);
    if q.iter().all(|&v| v == 0) {
        return CodedBlock8 { coded: false, q, rec: *pred, sse: sse_zero, bits: 0 };
    }
    let inv = dequant_idct_8x8(&q, steps);
    let mut rec = [[0u8; 8]; 8];
    for r in 0..8usize {
        for c in 0..8usize { rec[r][c] = clip_u8(i32::from(pred[r][c]) + i32::from(inv[r * 8 + c])); }
    }
    let (sse_coded, bits) = (block_dist::<8, true>(src, stride, bx, by, &rec), estimate_block8_bits(&q));
    if sse_coded + ((lambda_q8 * bits) >> 8) < sse_zero {
        CodedBlock8 { coded: true, q, rec, sse: sse_coded, bits }
    } else {
        CodedBlock8 { coded: false, q: [0; 64], rec: *pred, sse: sse_zero, bits: 0 }
    }
}

#[repr(C, align(64))]
struct CompactMbEval {
    mode: u8,
    mv: (i8, i8),
    cbp6: u8,
    q_blocks: [[i16; 64]; 6],
    rec_y: [[u8; 16]; 16],
    rec_u: [[u8; 8]; 8],
    rec_v: [[u8; 8]; 8],
    cost: u64,
}

fn eval_mb_skip(
    src: &PaddedCanvas, bx: usize, by: usize, preds: ([[u8; 16]; 16], [[u8; 8]; 8], [[u8; 8]; 8]),
    mode: u8, mv: (i8, i8), bits: u64, lambda_q8: u64,
) -> (CompactMbEval, u64) {
    let (cbx, cby) = (bx >> 1, by >> 1);
    let sse = block_dist::<16, true>(&src.y, src.pw, bx, by, &preds.0)
        + block_dist::<8, true>(&src.u, src.cw, cbx, cby, &preds.1)
        + block_dist::<8, true>(&src.v, src.cw, cbx, cby, &preds.2);
    let ev = CompactMbEval {
        mode, mv, cbp6: 0, q_blocks: [[0; 64]; 6],
        rec_y: preds.0, rec_u: preds.1, rec_v: preds.2, cost: sse + ((lambda_q8 * bits) >> 8),
    };
    (ev, sse)
}

fn eval_mb_residual(
    src: &PaddedCanvas, bx: usize, by: usize,
    pred_y: &[[u8; 16]; 16], pred_u: &[[u8; 8]; 8], pred_v: &[[u8; 8]; 8],
    steps_y: &[i32; 64], steps_uv: &[i32; 64],
    is_intra: bool, lambda_q8: u64, mode: u8, mv: (i8, i8), overhead_bits: u64,
) -> CompactMbEval {
    let (mut cbp6, mut q_blocks, mut rec_y) = (0u8, [[0i16; 64]; 6], [[0u8; 16]; 16]);
    let (mut total_sse, mut total_bits) = (0u64, overhead_bits);
    let mut record = |bi: usize, cb: &CodedBlock8| {
        if cb.coded { cbp6 |= 1u8 << bi; q_blocks[bi] = cb.q; }
        total_sse += cb.sse;
        total_bits += cb.bits;
    };
    for qy in 0..2usize {
        for qx in 0..2usize {
            let pq = extract_quad8(pred_y, qx, qy);
            let (ox, oy) = (bx + qx * 8, by + qy * 8);
            let cb = eval_block8_rdo(&src.y, src.pw, ox, oy, &pq, steps_y, is_intra, lambda_q8);
            record(qy * 2 + qx, &cb);
            blit_quad8(&mut rec_y, qx, qy, &cb.rec);
        }
    }
    let (cbx, cby) = (bx >> 1, by >> 1);
    let cb_u = eval_block8_rdo(&src.u, src.cw, cbx, cby, pred_u, steps_uv, is_intra, lambda_q8);
    record(4, &cb_u);
    let cb_v = eval_block8_rdo(&src.v, src.cw, cbx, cby, pred_v, steps_uv, is_intra, lambda_q8);
    record(5, &cb_v);
    CompactMbEval {
        mode, mv, cbp6, q_blocks, rec_y, rec_u: cb_u.rec, rec_v: cb_v.rec,
        cost: total_sse + ((lambda_q8 * total_bits) >> 8),
    }
}

fn apply_compact_deblock(rec: &mut PaddedCanvas, width: usize, height: usize, q_px: u8) {
    let q = i32::from(q_px);
    let (a, b, tc) = ((((q * 3) >> 1) + 4).min(56), ((q >> 1) + 3).min(24), ((q >> 2) + 1).min(12));
    apply_8x8_edge_filter(&mut rec.y, rec.pw, width, height, |p1, p0, q0, q1| {
        avc_deblock_delta(p1, p0, q0, q1, a, b, tc)
    });
    let (ca, cb, ctc) = ((a * 3) >> 2, (b * 3) >> 2, ((tc * 3) >> 2).max(1));
    for plane in [&mut rec.u, &mut rec.v] {
        apply_8x8_edge_filter(plane, rec.cw, width / 2, height / 2, |p1, p0, q0, q1| {
            avc_deblock_delta(p1, p0, q0, q1, ca, cb, ctc)
        });
    }
}

fn encode_best_modes(modes: &[u8], prior: Option<&RansModel>) -> Result<Vec<u8>> {
    let mut best = encode_with_optional_prior(modes, prior)?;
    if modes.len() >= 128 {
        if let Ok(ord1) = encode_order1_8way(modes, ContextMap::Low15) {
            if ord1.len() < best.len() { best = ord1; }
        }
    }
    Ok(best)
}

#[inline]
fn consider_compact(best: &mut Option<CompactMbEval>, cand: CompactMbEval) {
    if best.as_ref().map_or(true, |b| cand.cost < b.cost) { *best = Some(cand); }
}

#[inline]
fn compact_steps(qstep_u8: u8) -> ([i32; 64], [i32; 64]) {
    let qy = i32::from(qstep_u8);
    (build_quant_steps(qy), build_quant_steps(((qy * 3) >> 1).max(1)))
}

pub(crate) fn encode_compact_frame_internal(
    frame: &Yuv420Frame, reference: Option<&Yuv420Frame>, kind: FrameKind,
    cfg: VideoConfig, priors: Option<FramePriors<'_>>, knockout_no_halfpel_pmv: bool,
) -> Result<(Vec<u8>, Yuv420Frame, FrameSymbolObs)> {
    let ref_crc = validate_encode_inputs(frame, reference, kind)?;
    let qstep_u8 = cfg.qstep.max(1);
    let ((steps_y, steps_uv), lambda_q8) = (compact_steps(qstep_u8), quadratic_lambda_q8(qstep_u8));
    let src = PaddedCanvas::from_frame(frame);
    let ref_canvas = (kind == FrameKind::Inter).then(|| reference.map(PaddedCanvas::from_frame)).flatten();
    let mut rec = PaddedCanvas::new_blank(frame.width, frame.height);
    let (mb_cols, mb_rows) = (src.pw / 16, src.ph / 16);
    let num_mbs = mb_rows * mb_cols;
    let (mut mv_grid, mut modes, mut side, mut tokens, mut esc) = (
        vec![(0i8, 0i8); num_mbs], Vec::with_capacity(num_mbs),
        Vec::with_capacity(num_mbs * 2), Vec::with_capacity(num_mbs * 4), Vec::new(),
    );

    for mby in 0..mb_rows {
        for mbx in 0..mb_cols {
            let (bx, by, cbx, cby) = (mbx * 16, mby * 16, mbx * 8, mby * 8);
            let force_intra = kind == FrameKind::Key
                || cfg.intra_refresh_col.is_some_and(|col| usize::from(col) == mbx);
            let pmv_hp = if !force_intra && !knockout_no_halfpel_pmv {
                spatial_median_pmv(&mv_grid, mb_cols, mbx, mby)
            } else { (0, 0) };
            let mut best: Option<CompactMbEval> = None;

            if !force_intra {
                if let Some(ref_c) = ref_canvas.as_ref() {
                    if !knockout_no_halfpel_pmv {
                        let preds_p = fetch_mb_halfpel(ref_c, bx, by, pmv_hp);
                        let (skip_ev, skip_sse) =
                            eval_mb_skip(&src, bx, by, preds_p, COMPACT_PMV_SKIP, pmv_hp, 1, lambda_q8);
                        best = Some(skip_ev);
                        if skip_sse > (lambda_q8 >> 8) * 3 {
                            let cand = eval_mb_residual(
                                &src, bx, by, &preds_p.0, &preds_p.1, &preds_p.2,
                                &steps_y, &steps_uv, false, lambda_q8, COMPACT_PMV_RES, pmv_hp, 8,
                            );
                            if cand.cbp6 != 0 { consider_compact(&mut best, cand); }
                            let (mx, my, _) =
                                search_halfpel_mvd(&src.y, &ref_c.y, src.pw, mbx, mby, pmv_hp, lambda_q8);
                            if (mx, my) != pmv_hp {
                                let preds_m = fetch_mb_halfpel(ref_c, bx, by, (mx, my));
                                let mv_bits = 4 + mvd_pair_bits((mx, my), pmv_hp);
                                let (mskip, _) = eval_mb_skip(
                                    &src, bx, by, preds_m, COMPACT_MVD_SKIP, (mx, my), mv_bits, lambda_q8,
                                );
                                consider_compact(&mut best, mskip);
                                let mres = eval_mb_residual(
                                    &src, bx, by, &preds_m.0, &preds_m.1, &preds_m.2, &steps_y,
                                    &steps_uv, false, lambda_q8, COMPACT_MVD_RES, (mx, my), mv_bits + 5,
                                );
                                if mres.cbp6 != 0 { consider_compact(&mut best, mres); }
                            }
                        }
                    } else {
                        let (mx, my, _) =
                            search_intpel_knockout(&src.y, &ref_c.y, src.pw, mbx, mby, lambda_q8);
                        let preds_m = fetch_mb_halfpel(ref_c, bx, by, (mx, my));
                        best = Some(eval_mb_residual(
                            &src, bx, by, &preds_m.0, &preds_m.1, &preds_m.2,
                            &steps_y, &steps_uv, false, lambda_q8, COMPACT_MVD_RES, (mx, my), 24,
                        ));
                    }
                }
            }

            if force_intra || best.as_ref().map_or(true, |b| b.cost > (lambda_q8 >> 8) * 14) {
                let (mut s0, mut d0, mut s1, mut d1) = (0u8, u32::MAX, 1u8, u32::MAX);
                for sub in 0..4u8 {
                    let pred = predict_intra_block::<16>(CodecProfile::Av1, sub, &rec.y, src.pw, bx, by);
                    let sad = block_dist::<16, false>(&src.y, src.pw, bx, by, &pred) as u32;
                    if sad < d0 { (s1, d1, s0, d0) = (s0, d0, sub, sad); }
                    else if sad < d1 { (s1, d1) = (sub, sad); }
                }
                let subs: &[u8] = if force_intra { &[s0, s1] } else { &[s0] };
                for &sub in subs {
                    let py = predict_intra_block::<16>(CodecProfile::Av1, sub, &rec.y, rec.pw, bx, by);
                    let pu = predict_intra_block::<8>(CodecProfile::Av1, sub, &rec.u, rec.cw, cbx, cby);
                    let pv = predict_intra_block::<8>(CodecProfile::Av1, sub, &rec.v, rec.cw, cbx, cby);
                    consider_compact(&mut best, eval_mb_residual(
                        &src, bx, by, &py, &pu, &pv, &steps_y, &steps_uv, true, lambda_q8,
                        COMPACT_INTRA_BASE + sub, (0, 0), if force_intra { 6 } else { 11 },
                    ));
                }
            }

            let w = best.unwrap_or(CompactMbEval {
                mode: COMPACT_INTRA_BASE, mv: (0, 0), cbp6: 0, q_blocks: [[0; 64]; 6],
                rec_y: [[128; 16]; 16], rec_u: [[128; 8]; 8], rec_v: [[128; 8]; 8], cost: 0,
            });
            write_patch::<16>(&mut rec.y, rec.pw, bx, by, &w.rec_y);
            write_patch::<8>(&mut rec.u, rec.cw, cbx, cby, &w.rec_u);
            write_patch::<8>(&mut rec.v, rec.cw, cbx, cby, &w.rec_v);
            mv_grid[mby * mb_cols + mbx] = if w.mode >= COMPACT_PMV_SKIP { w.mv } else { (0, 0) };
            modes.push(w.mode);
            if matches!(w.mode, COMPACT_MVD_SKIP | COMPACT_MVD_RES) {
                side.push((w.mv.0.wrapping_sub(pmv_hp.0) as u8).wrapping_add(128));
                side.push((w.mv.1.wrapping_sub(pmv_hp.1) as u8).wrapping_add(128));
            }
            if matches!(w.mode, 0..=3 | COMPACT_PMV_RES | COMPACT_MVD_RES) { side.push(w.cbp6); }
            for bi in 0..6usize {
                if (w.cbp6 & (1u8 << bi)) != 0 { push_block8_tokens(&w.q_blocks[bi], &mut tokens, &mut esc); }
            }
        }
    }

    if cfg.loop_filter { apply_compact_deblock(&mut rec, frame.width, frame.height, qstep_u8); }
    let rec_frame = rec.to_frame(frame.width, frame.height);
    let mode_enc = encode_best_modes(&modes, priors.map(|p| p.mode))?;
    let (side_enc, use_lz_side) = encode_side_stream(&side, priors.map(|p| p.side), true)?;
    let tok_enc = if tokens.is_empty() { Vec::new() }
        else { encode_with_optional_prior(&tokens, priors.map(|p| p.res.level_lo))? };
    let mut body = Vec::with_capacity(mode_enc.len() + side_enc.len() + tok_enc.len() + esc.len() + 24);
    for sub in [&mode_enc, &side_enc, &tok_enc, &esc] { write_substream(&mut body, sub)?; }

    let profile_kind = ((CodecProfile::Compact as u8) & 0x07) | ((kind as u8) << 4);
    let mut flags = VF_COMPACT_V2
        | (u8::from(knockout_no_halfpel_pmv) * VF_KNOCKOUT_RAW_MV)
        | (u8::from(cfg.loop_filter) * VF_LOOP_FILTER)
        | (u8::from(use_lz_side) * VF_LZ_SIDE);
    let refresh_col_u8 = match cfg.intra_refresh_col {
        Some(col) if col <= 255 => { flags |= VF_INTRA_REFRESH; col as u8 }
        _ => 0,
    };
    let wire = finish_bvf1_frame(profile_kind, flags, qstep_u8, refresh_col_u8, ref_crc, &rec_frame, body)?;
    let obs = FrameSymbolObs {
        modes, side, res4: ResidualStreamObs::default(),
        res8: ResidualStreamObs { cbp: Vec::new(), runs: Vec::new(), levels_lo: tokens },
    };
    Ok((wire, rec_frame, obs))
}

/// Public entry point for the counterfactual knockout benchmark (`disable_halfpel_and_pmv = true`).
pub fn encode_compact_frame_knockout(
    frame: &Yuv420Frame, reference: Option<&Yuv420Frame>, kind: FrameKind,
    cfg: VideoConfig, disable_halfpel_and_pmv: bool,
) -> Result<(Vec<u8>, Yuv420Frame)> {
    let (wire, rec, _) =
        encode_compact_frame_internal(frame, reference, kind, cfg, None, disable_halfpel_and_pmv)?;
    Ok((wire, rec))
}

fn recon_block8(
    dst: &mut [u8], stride: usize, ox: usize, oy: usize, pred: &[[u8; 8]; 8], coded: bool,
    steps: &[i32; 64], tokens: &[u8], tok_pos: &mut usize, esc: &[u8], esc_pos: &mut usize,
) -> Result<()> {
    if !coded { write_patch::<8>(dst, stride, ox, oy, pred); return Ok(()); }
    let q = decode_block8_tokens(tokens, tok_pos, esc, esc_pos)?;
    let inv = dequant_idct_8x8(&q, steps);
    for r in 0..8usize {
        for c in 0..8usize {
            dst[(oy + r) * stride + ox + c] = clip_u8(i32::from(pred[r][c]) + i32::from(inv[r * 8 + c]));
        }
    }
    Ok(())
}

fn next_cbp(side: &[u8], pos: &mut usize) -> Result<u8> {
    let c = *side.get(*pos).ok_or(Error::CorruptVideoFrame)?;
    *pos += 1;
    if c > 63 { return Err(Error::CorruptVideoFrame); }
    Ok(c)
}

pub(crate) fn decode_compact_frame_internal(
    meta: [u8; 12], orig_len_u32: u32, expected_hdr_crc: u32, body: &[u8],
    reference: Option<&Yuv420Frame>, max_bytes: usize, resilient_heal: bool, priors: Option<FramePriors<'_>>,
) -> Result<(Yuv420Frame, FrameSymbolObs)> {
    let (profile_kind, flags, qstep_u8, refresh_col_u8) = (meta[0], meta[1], meta[2], meta[3]);
    let width = usize::from(u16::from_le_bytes([meta[4], meta[5]]));
    let height = usize::from(u16::from_le_bytes([meta[6], meta[7]]));
    let expected_ref_crc = u32::from_le_bytes([meta[8], meta[9], meta[10], meta[11]]);
    if qstep_u8 == 0 { return Err(Error::CorruptVideoFrame); }
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
    let (mode_s, side_s, tok_s, esc_s) =
        (br.read_substream()?, br.read_substream()?, br.read_substream()?, br.read_substream()?);
    let expected_rec_crc = br.read_u32_le()?;
    br.finish()?;
    if (esc_s.len() & 1) != 0 { return Err(Error::CorruptVideoFrame); }

    let mut rec = PaddedCanvas::new_blank(width, height);
    let (mb_cols, mb_rows) = (rec.pw / 16, rec.ph / 16);
    let num_mbs = mb_rows.checked_mul(mb_cols).ok_or(Error::OutputTooLarge)?;
    let (mut modes, mut side, mut tokens) = (Vec::new(), Vec::new(), Vec::new());
    decode_with_optional_prior(mode_s, num_mbs, priors.map(|p| p.mode), &mut modes)?;
    if modes.len() != num_mbs { return Err(Error::CorruptVideoFrame); }
    let max_side = num_mbs.checked_mul(4).ok_or(Error::OutputTooLarge)?;
    if !side_s.is_empty() {
        if (flags & VF_LZ_SIDE) != 0 { side = decompress_lz_split_with_limit(side_s, max_side)?; }
        else { decode_with_optional_prior(side_s, max_side, priors.map(|p| p.side), &mut side)?; }
    }
    if !tok_s.is_empty() {
        let max_tok = num_mbs.checked_mul(6 * 65).ok_or(Error::OutputTooLarge)?;
        decode_with_optional_prior(tok_s, max_tok, priors.map(|p| p.res.level_lo), &mut tokens)?;
    }

    let (steps_y, steps_uv) = compact_steps(qstep_u8);
    let (mut mv_grid, mut side_pos, mut tok_pos, mut esc_pos) =
        (vec![(0i8, 0i8); num_mbs], 0usize, 0usize, 0usize);

    for mby in 0..mb_rows {
        for mbx in 0..mb_cols {
            let mode = modes[mby * mb_cols + mbx];
            if (flags & VF_INTRA_REFRESH) != 0 && kind == FrameKind::Inter
                && mbx == usize::from(refresh_col_u8) && mode >= COMPACT_PMV_SKIP
            {
                return Err(Error::CorruptVideoFrame);
            }
            let (bx, by, cbx, cby) = (mbx * 16, mby * 16, mbx * 8, mby * 8);
            let (pred_y, pred_u, pred_v, cbp6, mv_out) = match mode {
                0..=3 => {
                    let sub = mode - COMPACT_INTRA_BASE;
                    (
                        predict_intra_block::<16>(CodecProfile::Av1, sub, &rec.y, rec.pw, bx, by),
                        predict_intra_block::<8>(CodecProfile::Av1, sub, &rec.u, rec.cw, cbx, cby),
                        predict_intra_block::<8>(CodecProfile::Av1, sub, &rec.v, rec.cw, cbx, cby),
                        next_cbp(&side, &mut side_pos)?, (0i8, 0i8),
                    )
                }
                COMPACT_PMV_SKIP | COMPACT_MVD_SKIP | COMPACT_PMV_RES | COMPACT_MVD_RES => {
                    if kind != FrameKind::Inter { return Err(Error::CorruptVideoFrame); }
                    let ref_c = ref_canvas.as_ref().ok_or(Error::ReferenceMismatch)?;
                    let pmv = if (flags & VF_KNOCKOUT_RAW_MV) != 0 { (0, 0) }
                        else { spatial_median_pmv(&mv_grid, mb_cols, mbx, mby) };
                    let mv = if matches!(mode, COMPACT_MVD_SKIP | COMPACT_MVD_RES) {
                        let b0 = *side.get(side_pos).ok_or(Error::CorruptVideoFrame)?;
                        let b1 = *side.get(side_pos + 1).ok_or(Error::CorruptVideoFrame)?;
                        side_pos += 2;
                        (pmv.0.wrapping_add(b0.wrapping_sub(128) as i8),
                         pmv.1.wrapping_add(b1.wrapping_sub(128) as i8))
                    } else { pmv };
                    let cbp = if matches!(mode, COMPACT_PMV_RES | COMPACT_MVD_RES) {
                        next_cbp(&side, &mut side_pos)?
                    } else { 0 };
                    let (py, pu, pv) = fetch_mb_halfpel(ref_c, bx, by, mv);
                    (py, pu, pv, cbp, mv)
                }
                _ => return Err(Error::CorruptVideoFrame),
            };
            if let Some(slot) = mv_grid.get_mut(mby * mb_cols + mbx) { *slot = mv_out; }
            for qy in 0..2usize {
                for qx in 0..2usize {
                    let pq = extract_quad8(&pred_y, qx, qy);
                    recon_block8(
                        &mut rec.y, rec.pw, bx + qx * 8, by + qy * 8, &pq,
                        (cbp6 & (1u8 << (qy * 2 + qx))) != 0,
                        &steps_y, &tokens, &mut tok_pos, esc_s, &mut esc_pos,
                    )?;
                }
            }
            for (plane, pred, bit) in [(&mut rec.u, &pred_u, 16u8), (&mut rec.v, &pred_v, 32u8)] {
                recon_block8(
                    plane, rec.cw, cbx, cby, pred, (cbp6 & bit) != 0,
                    &steps_uv, &tokens, &mut tok_pos, esc_s, &mut esc_pos,
                )?;
            }
        }
    }

    if side_pos != side.len() || tok_pos != tokens.len() || esc_pos != esc_s.len() {
        return Err(Error::CorruptVideoFrame);
    }
    if (flags & VF_LOOP_FILTER) != 0 { apply_compact_deblock(&mut rec, width, height, qstep_u8); }
    let rec_frame = rec.to_frame(width, height);
    if !reference_mismatched && rec_frame.digest_crc32() != expected_rec_crc {
        return Err(Error::ChecksumMismatch);
    }
    let obs = FrameSymbolObs {
        modes, side, res4: ResidualStreamObs::default(),
        res8: ResidualStreamObs { cbp: Vec::new(), runs: Vec::new(), levels_lo: tokens },
    };
    Ok((rec_frame, obs))
}

/// Verify counterfactual knockout byte-reduction law (`p_opt_bytes * 3 < p_ko_bytes`).
#[inline] #[must_use]
pub const fn verify_pmv_knockout(p_opt_bytes: usize, p_ko_bytes: usize) -> bool {
    p_opt_bytes.saturating_mul(3) < p_ko_bytes
}

/// Decode only the macroblock mode histogram `[intra0..3, pmv_skip, mvd_skip, pmv_res, mvd_res]`.
pub fn compact_frame_mode_histogram(packet: &[u8], mode_prior: Option<&RansModel>) -> Result<[usize; 8]> {
    let mut r = Reader::new(packet);
    expect_magic(&mut r, &MAGIC_VIDEO_FRAME)?;
    let meta = r.read_bytes(12)?;
    let width = usize::from(u16::from_le_bytes([meta[4], meta[5]]));
    let height = usize::from(u16::from_le_bytes([meta[6], meta[7]]));
    let (orig_len, hdr_crc) = (r.read_u32_le()?, r.read_u32_le()?);
    if orig_len == 0 && hdr_crc == 0 && packet.len() < 24 { return Err(Error::TruncatedStream); }
    let body = r.read_bytes(packet.len().saturating_sub(24))?;
    r.finish()?;
    let mode_s = Reader::new(body).read_substream()?;
    let num_mbs = width.div_ceil(16) * height.div_ceil(16);
    let mut modes = Vec::with_capacity(num_mbs);
    decode_with_optional_prior(mode_s, num_mbs, mode_prior, &mut modes)?;
    let mut hist = [0usize; 8];
    for m in modes { if usize::from(m) < 8 { hist[usize::from(m)] += 1; } }
    Ok(hist)
}

/// Measure signed DC reconstruction residual (`recon_31 - v_dc`) after `1` I-frame + `31` P-frames.
#[must_use]
pub fn check_dc_drift_31p(v_dc: u8, q_inter: u8) -> i16 {
    let q_p = q_inter.max(1);
    let steps_i = build_quant_steps(i32::from((q_p / 2).max(1)));
    let steps_p = build_quant_steps(i32::from(q_p));
    let mut rec = [128u8; 64];
    for step_idx in 0..=31usize {
        let mut res = [0i16; 64];
        for (slot, &r) in res.iter_mut().zip(&rec) { *slot = i16::from(v_dc) - i16::from(r); }
        let (is_intra, steps) = (step_idx == 0, if step_idx == 0 { &steps_i } else { &steps_p });
        let q = quantize_8x8_perceptual(&forward_dct_8x8(&res), steps, is_intra);
        let inv = dequant_idct_8x8(&q, steps);
        for (slot, &d) in rec.iter_mut().zip(&inv) { *slot = clip_u8(i32::from(*slot) + i32::from(d)); }
    }
    i16::from(rec[0]) - i16::from(v_dc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::{decode_video_frame, encode_video_frame};

    #[test]
    fn compact_zero_dc_drift_31p_all_q_and_luma() {
        let (mut max_abs, mut sum_signed, mut count) = (0i16, 0i64, 0i64);
        for q_inter in [1u8, 8, 16, 24, 32, 48, 64, 96, 128, 200, 255] {
            let bound = i16::from(q_inter.div_ceil(2));
            for v_dc in (0u8..=255).step_by(5) {
                let err = check_dc_drift_31p(v_dc, q_inter);
                max_abs = max_abs.max(err.abs());
                sum_signed += i64::from(err);
                count += 1;
                assert!(err.abs() <= bound, "v_dc={v_dc}, q={q_inter}, err={err}, bound={bound}");
            }
        }
        assert!(count > 0 && max_abs <= 22 && sum_signed.unsigned_abs() < (count as u64));
    }

    #[test]
    fn compact_roundtrip_normal_and_knockout() {
        let f0 = Yuv420Frame::synth_natural_motion(128, 96, 0).expect("f0");
        let f1 = Yuv420Frame::synth_natural_motion(128, 96, 1).expect("f1");
        let cfg_k = VideoConfig {
            profile: CodecProfile::Compact, qstep: 24, lambda_q8: 1024,
            search_range: 16, loop_filter: true, intra_refresh_col: None,
        };
        let cfg_p = VideoConfig { qstep: 48, ..cfg_k };
        let (pkt0, rec0) = encode_video_frame(&f0, None, FrameKind::Key, cfg_k).expect("enc k");
        let dec0 = decode_video_frame(&pkt0, None).expect("dec k");
        assert_eq!(rec0, dec0);
        let (pkt1, rec1) =
            encode_compact_frame_knockout(&f1, Some(&rec0), FrameKind::Inter, cfg_p, false).expect("opt");
        assert_eq!(rec1, decode_video_frame(&pkt1, Some(&dec0)).expect("dec opt"));
        let (pkt_ko, rec_ko) =
            encode_compact_frame_knockout(&f1, Some(&rec0), FrameKind::Inter, cfg_p, true).expect("ko");
        assert_eq!(rec_ko, decode_video_frame(&pkt_ko, Some(&dec0)).expect("dec ko"));
        assert_ne!(pkt1.len(), pkt_ko.len());
    }
}
