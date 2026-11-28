//! Cost-guided 1D/2D arrival-parsed split-stream LZ (`BRL2`/`BRP2`) backed by MDL-clustered
//! socket-plugged 8-way interleaved rANS (`BRS1`), `1 MiB` dual-hash (`hash3`/`hash4`/`hash6`)
//! matchfinding, half-log2 offset coding, empirical Order-0/Order-1 high-nibble priors, and
//! adaptive 4-bit nibble coding.

use alloc::vec;
use alloc::vec::Vec;

use crate::cpu_spec::{detect_host_profile, host_tuning_plan, CpuProfile, ZenTuningPlan};
use crate::lz_split::{copy_lz_match, match_len_at};
use crate::nc_nibble::{
    compress_blocks_pooled, decode_adaptive_nc_nibble, decode_nibble_seq,
    decompress_blocks_pooled, encode_adaptive_nc_nibble, encode_nibble_o1_match,
    encode_nibble_seq, NibbleO1MatchDecoder,
};
use crate::rans8::{
    finish_shared_stream, parse_shared_stream, rans_enc_step_scalar, verify_terminal_shared,
    RansModel, ScaleBits, MAX_STAGE_WORDS, RANS_L,
};
use crate::socket_kernels::{
    grad_socket_u8, has_socket_batch_accel, nc_hash4_u8, paeth_pred_u8, run_add_mod256_8,
    run_grad_socket8, run_grad_socket8_batch, run_nc_hash4_8_batch, run_paeth_pred8,
    run_paeth_pred8_batch, run_sub_mod256_8_batch,
};
use crate::socket_rans::{
    decode_substream_socket_auto, encode_substream_socket_auto, read_compact_model,
    read_socket_map, write_compact_model, write_socket_map, ClusteredSocketRans,
};
use crate::wire::{
    expect_magic, framed_header_crc, write_framed_header, write_substream, write_varint_u32, Error,
    Reader, Result, DEFAULT_MAX_DECODE_BYTES,
};

/// Framed 1D/2D arrival-parsed socket LZ magic header (`BRL2`).
pub const MAGIC_SOCKET_LZ2D: [u8; 4] = *b"BRL2";
/// Framed deterministic multi-core L2-pinned parallel block magic header (`BRP2`).
pub const MAGIC_PARALLEL_BLOCKS: [u8; 4] = crate::nc_nibble::MAGIC_PARALLEL_BLOCKS;

const BRL2_MODE_STORED: u8 = 0;
const BRL2_MODE_LITERAL_ONLY: u8 = 1;
const BRL2_MODE_CAUSAL_2D_GRAD: u8 = 2;
const BRL2_MODE_LZ2D: u8 = 3;

const LIT_XFORM_RAW: u8 = 0;
const LIT_XFORM_SUB_REP0: u8 = 1;
const LIT_XFORM_SUB_STRIDE: u8 = 2;
const LIT_XFORM_SUB_PAETH2D: u8 = 3;

const SOCK_MODE_NONE: u8 = 0;
const SOCK_MODE_EXTERNAL: u8 = 1;
const SOCK_MODE_STRIDE_PHASE: u8 = 2;
const SOCK_MODE_PREV_BYTE: u8 = 3;
const SOCK_MODE_ADAPTIVE_O1: u8 = 4;
const SOCK_MODE_ADAPTIVE_NC: u8 = 5;

const WINDOW_RING: usize = 1 << 20;
const MAX_OFFSET: usize = WINDOW_RING - 1;
const HASH3_SIZE: usize = 1 << 12;
const HASH3_MASK: usize = HASH3_SIZE - 1;
const PREV3_RING: usize = 4096;
const PREV3_MASK: usize = PREV3_RING - 1;
const HASH4_BITS: u32 = 17;
const HASH4_MASK: usize = (1 << HASH4_BITS) - 1;
const HASH6_BITS: u32 = 17;
const NO_POS: u32 = u32::MAX;
const DEFAULT_FAST_CHAIN: usize = 24;
const DEFAULT_MAX_CHAIN: usize = 64;

const OFF_CODE_2D_BASE: u8 = 36;
const OFF_CODE_1D_BASE: u8 = 4;
const HALF_LOG2_SLOTS: usize = 40;
const NUM_OFF_CODES: usize = (OFF_CODE_2D_BASE as usize) + HALF_LOG2_SLOTS;
const LL_EXACT_MAX: usize = 16;
const NUM_LL_CODES: usize = 44;
const ML_EXACT_MAX: usize = 32;
const ML_CODE_NONE: u8 = 60;
const NUM_ML_CODES: usize = 61;

#[inline]
fn bound_f64(v: f64, lo: f64, hi: f64) -> f64 { if v < lo { lo } else if v > hi { hi } else { v } }

#[inline]
fn bound_usize(v: usize, lo: usize, hi: usize) -> usize {
    if v < lo { lo } else if v > hi { hi } else { v }
}

const SHELL_2D_DELTAS: [(i32, i32); 32] = [
    (0, 2), (0, 3), (0, 4), (0, 8), (0, 16), (1, -8), (1, -4), (1, -3),
    (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 8),
    (2, -4), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 4), (3, -1),
    (3, 0), (3, 1), (4, -4), (4, 0), (4, 4), (6, 0), (8, 0), (16, 0),
];

const SHELL_1D_SHORT: [usize; 32] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24,
    28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 384, 512,
];

/// Configuration for the `BRL2` 1D/2D arrival-parsed socket-plugged LZ compressor.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SocketLz2dConfig {
    /// Row or record stride in bytes (`0` disables 2D stride offsets and 2D predictors).
    pub stride: usize,
    /// Minimum match length for general 1D hash-chain matches (`2..=8`, default `3`).
    pub min_match: usize,
    /// Number of forward arrival DP passes (`1` = empirical prior, `2..=3` = multi-pass Q4 DP).
    pub arrival_passes: u8,
}

impl Default for SocketLz2dConfig {
    #[inline]
    fn default() -> Self { Self { stride: 0, min_match: 3, arrival_passes: 1 } }
}

/// Knuth multiplicative 3-byte hash mapped into `0..HASH3_SIZE`.
/// @domain v: 0..=16777215
#[inline(always)]
const fn hash3_u32(v: u32) -> usize { (((v.wrapping_mul(2_654_435_761)) >> 16) & 0xFFFF) as usize }

#[inline(always)]
const fn hash6_u64(v: u64) -> usize {
    (((v.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 47) & 0x1_FFFF) as usize
}

fn compute_hashes4_into(data: &[u8], out: &mut Vec<u32>) {
    let count = data.len().saturating_sub(3);
    out.resize(count, 0);
    if count == 0 { return; }
    let mut i_start = 0usize;
    if has_socket_batch_accel() && count >= 64 {
        const TILE: usize = 256;
        let (mut b0, mut b1, mut b2, mut b3, mut h) = (
            [[0i32; 8]; TILE], [[0i32; 8]; TILE], [[0i32; 8]; TILE],
            [[0i32; 8]; TILE], [[0i32; 8]; TILE],
        );
        let (n_blocks, mut blk) = (count / 8, 0usize);
        while blk < n_blocks {
            let chunk = (n_blocks - blk).min(TILE);
            for t in 0..chunk {
                let base = (blk + t) * 8;
                for k in 0..8usize {
                    let p = base + k;
                    if p + 3 < data.len() {
                        b0[t][k] = i32::from(data[p]); b1[t][k] = i32::from(data[p + 1]);
                        b2[t][k] = i32::from(data[p + 2]); b3[t][k] = i32::from(data[p + 3]);
                    }
                }
            }
            run_nc_hash4_8_batch(&b0[..chunk], &b1[..chunk], &b2[..chunk], &b3[..chunk], &mut h[..chunk]);
            for t in 0..chunk {
                let base = (blk + t) * 8;
                for k in 0..8usize {
                    let p = base + k;
                    if p < out.len() { out[p] = (h[t][k] as u32) & (HASH4_MASK as u32); }
                }
            }
            blk += chunk;
        }
        i_start = n_blocks * 8;
    }
    for i in i_start..count {
        if i + 3 < data.len() && i < out.len() {
            out[i] = nc_hash4_u8(data[i], data[i + 1], data[i + 2], data[i + 3]) as u32;
        }
    }
}

#[inline]
fn offset_to_half_log2(off: usize) -> (u8, u32, u8) {
    let d = (off.saturating_sub(1)) as u32;
    if d < 4 { return (d as u8, 0, 0); }
    let msb = 31 - d.leading_zeros();
    let slot = ((msb << 1) | ((d >> (msb - 1)) & 1)) as u8;
    (slot, d & ((1u32 << (msb - 1)) - 1), (msb - 1) as u8)
}

#[inline]
fn len_to_code(val: usize, exact_max: usize) -> (u8, u32, u8) {
    if val < exact_max { return (val as u8, 0, 0); }
    let r = (val - (exact_max - 1)) as u32;
    let msb = 31 - r.leading_zeros();
    let mask = if msb == 0 { 0 } else { (1u32 << msb) - 1 };
    ((exact_max as u32 + msb) as u8, r & mask, msb as u8)
}

struct RawBitWriter { buf: Vec<u8>, acc: u64, bits: u32 }

impl RawBitWriter {
    fn with_capacity(cap: usize) -> Self { Self { buf: Vec::with_capacity(cap), acc: 0, bits: 0 } }
    #[inline]
    fn write(&mut self, val: u32, nbits: u8) {
        if nbits == 0 { return; }
        self.acc |= u64::from(val) << self.bits;
        self.bits += u32::from(nbits);
        while self.bits >= 8 {
            self.buf.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 { self.buf.push((self.acc & 0xFF) as u8); }
        self.buf
    }
}

struct RawBitReader<'a> { data: &'a [u8], pos: usize, acc: u64, bits: u32 }

impl<'a> RawBitReader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0, acc: 0, bits: 0 } }
    #[inline]
    fn read(&mut self, nbits: u8) -> Result<u32> {
        if nbits == 0 { return Ok(0); }
        let need = u32::from(nbits);
        while self.bits < need {
            let b = *self.data.get(self.pos).ok_or(Error::CorruptLzStream)?;
            self.pos += 1;
            self.acc |= u64::from(b) << self.bits;
            self.bits += 8;
        }
        let v = (self.acc & ((1u64 << need) - 1)) as u32;
        self.acc >>= need;
        self.bits -= need;
        Ok(v)
    }
    fn finish(&self) -> Result<()> {
        if self.pos != self.data.len() || self.acc != 0 { return Err(Error::CorruptLzStream); }
        Ok(())
    }
}

#[inline]
fn initial_reps(stride: usize) -> [usize; 4] {
    let r1 = if stride >= 2 { stride.min(MAX_OFFSET) } else { 2 };
    let r2 = if stride >= 2 { (stride.saturating_mul(2)).min(MAX_OFFSET) } else { 3 };
    [1, r1, r2, 4]
}

#[inline]
fn build_shell_table(stride: usize) -> [usize; 32] {
    if stride < 4 { return SHELL_1D_SHORT; }
    let mut tbl = [1usize; 32];
    let s = stride as i64;
    for (idx, &(dy, dx)) in SHELL_2D_DELTAS.iter().enumerate() {
        let disp = i64::from(dy) * s + i64::from(dx);
        tbl[idx] = if disp >= 1 && disp <= MAX_OFFSET as i64 { disp as usize } else { idx + 1 };
    }
    tbl
}

#[inline]
fn mtf_update(reps: &mut [usize; 4], used_off: usize) {
    if reps[0] == used_off { return; }
    if reps[1] == used_off {
        reps.swap(0, 1);
    } else if reps[2] == used_off {
        reps[2] = reps[1]; reps[1] = reps[0]; reps[0] = used_off;
    } else {
        reps[3] = reps[2]; reps[2] = reps[1]; reps[1] = reps[0]; reps[0] = used_off;
    }
}

#[inline]
fn stride_phase_n_sockets(stride: usize) -> usize {
    if stride < 2 { 1 } else if stride <= 256 { stride } else { 64 }
}

#[inline]
fn stride_phase_socket(pos: usize, stride: usize) -> u8 {
    if stride < 2 { 0 } else if stride <= 256 { (pos % stride) as u8 } else { ((pos % stride) & 63) as u8 }
}

#[inline]
fn paeth_at(buf: &[u8], pos: usize, stride: usize) -> u8 {
    let w = if pos >= 1 && (pos % stride) != 0 {
        buf[pos - 1]
    } else if pos >= stride {
        buf[pos - stride]
    } else {
        0
    };
    let n = if pos >= stride { buf[pos - stride] } else { w };
    let nw = if pos > stride && (pos % stride) != 0 { buf[pos - stride - 1] } else { n };
    paeth_pred_u8(w, n, nw)
}

#[inline]
fn predict_literal(buf: &[u8], pos: usize, rep0: usize, stride: usize, xform: u8) -> u8 {
    match xform {
        LIT_XFORM_SUB_REP0 => if pos >= rep0 { buf[pos - rep0] } else { 0 },
        LIT_XFORM_SUB_STRIDE => if stride > 0 && pos >= stride { buf[pos - stride] } else { 0 },
        LIT_XFORM_SUB_PAETH2D => if stride >= 4 { paeth_at(buf, pos, stride) } else { 0 },
        _ => 0,
    }
}

fn apply_full_buffer_xform(src: &[u8], stride: usize, xform: u8, inverse: bool) -> Vec<u8> {
    if xform == LIT_XFORM_RAW { return src.to_vec(); }
    let mut out = Vec::with_capacity(src.len());
    for (pos, &b) in src.iter().enumerate() {
        let pred = if inverse {
            predict_literal(&out, pos, 1, stride, xform)
        } else {
            predict_literal(src, pos, 1, stride, xform)
        };
        out.push(if inverse { b.wrapping_add(pred) } else { b.wrapping_sub(pred) });
    }
    out
}

fn counts_to_q4_n<const N: usize>(counts: &[u64; N], total: u64) -> [u16; N] {
    let mut out = [96u16; N];
    let log_tot = crate::wire::log2_f64(total as f64);
    for i in 0..N {
        let bits = bound_f64(log_tot - crate::wire::log2_f64(counts[i].max(1) as f64), 0.25, 14.0);
        out[i] = crate::wire::round_f64(bits * 16.0) as u16;
    }
    out
}

fn slice_to_q4_small<const N: usize>(data: &[u8]) -> [u16; N] {
    let mut counts = [1u64; N];
    let mut total = N as u64;
    for &c in data {
        let idx = usize::from(c);
        if idx < N { counts[idx] += 1; total += 1; }
    }
    counts_to_q4_n(&counts, total)
}

/// Empirical Q4 bit-cost model (16 units = 1.0 bit) with L1D Order-0 + Order-1 high-nibble prior.
struct CostModelQ4 {
    lit_cost: [u16; 256],
    hi_delta: [[i16; 16]; 16],
    lit_o1_map: Option<[u8; 256]>,
    lit_o1_costs: Vec<[u16; 256]>,
    ll_code_cost: [u16; NUM_LL_CODES],
    ml_code_cost: [[u16; NUM_ML_CODES]; 2],
    off_code_cost: [[u16; NUM_OFF_CODES]; 4],
    off_base: u8,
}

impl CostModelQ4 {
    fn analytical(data: &[u8], stride: usize, lit_xform: u8) -> Self {
        let (mut byte_counts, mut hi_trans) = ([0u32; 256], [[0u32; 16]; 16]);
        let (mut hi_row, mut hi_col, mut prev_hi) = ([0u32; 16], [0u32; 16], 0usize);
        for (pos, &b) in data.iter().enumerate() {
            let sym = if lit_xform == LIT_XFORM_RAW { b }
                else { b.wrapping_sub(predict_literal(data, pos, 1, stride, lit_xform)) };
            let (ch, ph) = (usize::from(sym >> 4) & 15, prev_hi & 15);
            byte_counts[usize::from(sym) & 0xFF] += 1;
            hi_trans[ph][ch] += 1;
            hi_row[ph] += 1;
            hi_col[ch] += 1;
            prev_hi = usize::from(b >> 4) & 15;
        }
        let mut counts64 = [1u64; 256];
        for (dst, &c) in counts64.iter_mut().zip(byte_counts.iter()) { *dst = u64::from(c) + 1; }
        let total = (data.len() as u64) + 256;
        let lit_cost = counts_to_q4_n(&counts64, total);

        let mut hi_delta = [[0i16; 16]; 16];
        if data.len() >= 64 {
            let (log_tot, mut log_marg) = (crate::wire::log2_f64(total as f64), [0.0f64; 16]);
            for (dst, &hc) in log_marg.iter_mut().zip(hi_col.iter()) {
                *dst = crate::wire::log2_f64(f64::from(hc) + 16.0) - log_tot;
            }
            for (ph, row) in hi_delta.iter_mut().enumerate() {
                let log_r = crate::wire::log2_f64(f64::from(hi_row[ph]) + 16.0);
                for (ch, slot) in row.iter_mut().enumerate() {
                    let log_cond = crate::wire::log2_f64(f64::from(hi_trans[ph][ch]) + 1.0) - log_r;
                    *slot = crate::wire::round_f64(bound_f64((log_marg[ch] - log_cond) * 16.0, -96.0, 96.0)) as i16;
                }
            }
        }
        let off_base = if stride >= 4 { OFF_CODE_2D_BASE } else { OFF_CODE_1D_BASE };
        let mut oc_row = [88u16; NUM_OFF_CODES];
        oc_row[0..4].copy_from_slice(&[24, 42, 54, 64]);
        if stride >= 4 { oc_row[4..36].fill(72); }
        for s in 0..HALF_LOG2_SLOTS { oc_row[usize::from(off_base) + s] = 76 + ((s as u16) >> 2) * 2; }
        let mut ll_code_cost = [72u16; NUM_LL_CODES];
        ll_code_cost[0..5].copy_from_slice(&[18, 36, 48, 56, 64]);
        let mut ml_row = [92u16; NUM_ML_CODES];
        for i in 0..24usize { ml_row[i] = 44 + (i as u16) * 2; }
        ml_row[0] = 76;
        ml_row[1] = 52;
        ml_row[2] = 42;
        Self {
            lit_cost, hi_delta, lit_o1_map: None, lit_o1_costs: Vec::new(),
            ll_code_cost, ml_code_cost: [ml_row; 2], off_code_cost: [oc_row; 4], off_base,
        }
    }

    fn from_streams(
        streams: &Lz2dStreams, stride: usize, lit_bytes: &[u8], o1_cl: Option<&ClusteredSocketRans>,
    ) -> Self {
        let lit_cost = slice_to_q4_small::<256>(lit_bytes);
        let (lit_o1_map, lit_o1_costs) = if let Some(cl) = o1_cl {
            let mut map = [0u8; 256];
            let n = cl.socket_to_cluster.len().min(256);
            map[..n].copy_from_slice(&cl.socket_to_cluster[..n]);
            let mut tables = Vec::with_capacity(cl.models.len());
            for m in &cl.models {
                let (mut counts, mut total) = ([1u64; 256], 256u64);
                for (sym, &f) in m.freqs().iter().enumerate() {
                    let add = u64::from(f); counts[sym] += add; total += add;
                }
                tables.push(counts_to_q4_n(&counts, total));
            }
            (Some(map), tables)
        } else {
            (None, Vec::new())
        };
        let off_base = if stride >= 4 { OFF_CODE_2D_BASE } else { OFF_CODE_1D_BASE };
        let ll_code_cost = slice_to_q4_small::<NUM_LL_CODES>(&streams.ll_codes);
        let mut ml_b: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        for (&ll, &ml) in streams.ll_codes.iter().zip(streams.ml_codes.iter()) {
            ml_b[usize::from(ll == 0)].push(ml);
        }
        let ml_code_cost = [slice_to_q4_small(&ml_b[0]), slice_to_q4_small(&ml_b[1])];
        let mut oc_b: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut oi = 0usize;
        for &ml in &streams.ml_codes {
            if ml != ML_CODE_NONE && oi < streams.off_codes.len() {
                oc_b[usize::from(ml.min(3))].push(streams.off_codes[oi]); oi += 1;
            }
        }
        let mut off_code_cost = [slice_to_q4_small::<NUM_OFF_CODES>(&streams.off_codes); 4];
        for b in 0..4usize {
            if oc_b[b].len() >= 16 { off_code_cost[b] = slice_to_q4_small(&oc_b[b]); }
        }
        Self {
            lit_cost, hi_delta: [[0i16; 16]; 16], lit_o1_map, lit_o1_costs,
            ll_code_cost, ml_code_cost, off_code_cost, off_base,
        }
    }

    #[inline]
    fn literal_cost(&self, lit_sym: u8, prev_byte: u8) -> u32 {
        if let Some(ref map) = self.lit_o1_map {
            let cid = usize::from(map[usize::from(prev_byte)]);
            if let Some(tbl) = self.lit_o1_costs.get(cid) {
                return u32::from(tbl[usize::from(lit_sym)]);
            }
        }
        let base = i32::from(self.lit_cost[usize::from(lit_sym)]);
        let delta = i32::from(self.hi_delta[usize::from(prev_byte >> 4)][usize::from(lit_sym >> 4)]);
        (base + delta).max(4).min(224) as u32
    }

    #[inline]
    fn ll_and_off_base_cost(&self, off_code: u8, lit_run: u16) -> (u32, usize, usize) {
        let (ll_code, _, ll_ebits) = len_to_code(usize::from(lit_run), LL_EXACT_MAX);
        let ll_idx = usize::from(ll_code).min(NUM_LL_CODES - 1);
        let oc_idx = usize::from(off_code).min(NUM_OFF_CODES - 1);
        let mut base = u32::from(self.ll_code_cost[ll_idx]) + u32::from(ll_ebits) * 16;
        if off_code >= self.off_base {
            let slot = off_code - self.off_base;
            if slot >= 4 { base += u32::from((slot >> 1) - 1) * 16; }
        }
        (base, usize::from(lit_run == 0), oc_idx)
    }

    #[inline]
    fn match_len_cost(&self, ml_ctx: usize, oc_idx: usize, mlen: usize) -> u32 {
        let (ml_code, _, ml_ebits) = len_to_code(mlen.saturating_sub(2), ML_EXACT_MAX);
        let ml_idx = usize::from(ml_code).min(NUM_ML_CODES - 1);
        u32::from(self.ml_code_cost[ml_ctx][ml_idx])
            + u32::from(ml_ebits) * 16
            + u32::from(self.off_code_cost[usize::from(ml_code.min(3))][oc_idx])
    }
}

#[derive(Clone, Copy, Default)]
struct DpLink { prev: u32, offset: u32, lit_run: u16, off_code: u8, is_match: u8 }

struct Lz2dStreams {
    ll_codes: Vec<u8>,
    ml_codes: Vec<u8>,
    off_codes: Vec<u8>,
    raw_bits: Vec<u8>,
    lit_positions: Vec<usize>,
    lit_rep0: Vec<usize>,
}

impl Lz2dStreams {
    fn empty() -> Self {
        Self {
            ll_codes: Vec::new(), ml_codes: Vec::new(), off_codes: Vec::new(),
            raw_bits: Vec::new(), lit_positions: Vec::new(), lit_rep0: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.ll_codes.clear(); self.ml_codes.clear(); self.off_codes.clear();
        self.raw_bits.clear(); self.lit_positions.clear(); self.lit_rep0.clear();
    }

    fn push_packet(
        &mut self, bw: &mut RawBitWriter, off_base: u8, lit_start: usize, lit_len: usize,
        rep0_during_lits: usize, match_spec: Option<(usize, u8, usize)>,
    ) {
        let (ll_code, ll_ex, ll_eb) = len_to_code(lit_len, LL_EXACT_MAX);
        self.ll_codes.push(ll_code);
        bw.write(ll_ex, ll_eb);
        for i in 0..lit_len {
            self.lit_positions.push(lit_start + i);
            self.lit_rep0.push(rep0_during_lits);
        }
        if let Some((mlen, off_code, offset)) = match_spec {
            let (ml_code, ml_ex, ml_eb) = len_to_code(mlen.saturating_sub(2), ML_EXACT_MAX);
            self.ml_codes.push(ml_code);
            bw.write(ml_ex, ml_eb);
            self.off_codes.push(off_code);
            if off_code >= off_base {
                let (_, extra, ebits) = offset_to_half_log2(offset);
                bw.write(extra, ebits);
            }
        } else {
            self.ml_codes.push(ML_CODE_NONE);
        }
    }
}

#[inline]
fn active_rep_offsets(reps: &[usize; 4], zero_lits: bool) -> [usize; 4] {
    if zero_lits { [reps[1], reps[2], reps[3], reps[0].saturating_sub(1).max(1)] } else { *reps }
}

struct DpScratch {
    dp_cost: Vec<u32>,
    dp_link: Vec<DpLink>,
    dp_reps: Vec<[u32; 4]>,
    head3: [u32; HASH3_SIZE],
    prev3: [u32; PREV3_RING],
    head4: Vec<[u32; 2]>,
    prev4: Vec<u32>,
    head6: Vec<u32>,
    prev6: Vec<u32>,
    hashes4: Vec<u32>,
    streams: Lz2dStreams,
    path: Vec<(usize, usize, bool, u8, usize)>,
    ring_mask: usize,
    h4_mask: usize,
    h6_mask: usize,
}

impl DpScratch {
    fn empty() -> Self {
        Self {
            dp_cost: Vec::new(), dp_link: Vec::new(), dp_reps: Vec::new(),
            head3: [NO_POS; HASH3_SIZE], prev3: [NO_POS; PREV3_RING],
            head4: Vec::new(), prev4: Vec::new(), head6: Vec::new(), prev6: Vec::new(),
            hashes4: Vec::new(), streams: Lz2dStreams::empty(), path: Vec::new(),
            ring_mask: 0, h4_mask: 0, h6_mask: 0,
        }
    }

    fn prepare(&mut self, data: &[u8], plan: &ZenTuningPlan) {
        let n = data.len();
        let ring_len = n.next_power_of_two().min(WINDOW_RING).max(256);
        let h4_len = (n.saturating_mul(2)).next_power_of_two().max(4096)
            .min(1usize << plan.l2_hash4_bits.min(HASH4_BITS));
        let h6_len = (n.saturating_mul(2)).next_power_of_two().max(4096)
            .min(1usize << plan.l2_hash6_bits.min(HASH6_BITS));
        self.dp_cost.resize(n + 1, u32::MAX);
        self.dp_link.resize(n + 1, DpLink::default());
        self.dp_reps.resize(n + 1, [1, 2, 3, 4]);
        self.head4.resize(h4_len, [NO_POS, NO_POS]);
        self.prev4.resize(ring_len, NO_POS);
        self.head6.resize(h6_len, NO_POS);
        self.prev6.resize(ring_len, NO_POS);
        compute_hashes4_into(data, &mut self.hashes4);
        self.ring_mask = ring_len - 1; self.h4_mask = h4_len - 1; self.h6_mask = h6_len - 1;
    }

    fn reset(&mut self) {
        self.dp_cost.fill(u32::MAX); self.head3.fill(NO_POS);
        self.head4.fill([NO_POS, NO_POS]); self.head6.fill(NO_POS);
    }

    #[inline(always)]
    fn insert_hashes(&mut self, data: &[u8], pos: usize) -> (u32, u32, u32, u32, u32, u32) {
        let n = data.len();
        let (mut cand3, mut cur_v3) = (NO_POS, 0u32);
        if pos + 3 <= n {
            cur_v3 = u32::from(data[pos]) | (u32::from(data[pos + 1]) << 8)
                | (u32::from(data[pos + 2]) << 16);
            let h3 = (hash3_u32(cur_v3) >> 4) & HASH3_MASK;
            cand3 = self.head3[h3];
            self.head3[h3] = pos as u32;
            self.prev3[pos & PREV3_MASK] = cand3;
        }
        let (mut c4_0, mut c4_1, mut cur_v4) = (NO_POS, NO_POS, 0u32);
        if pos < self.hashes4.len() {
            cur_v4 = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            let h4 = (self.hashes4[pos] as usize) & self.h4_mask;
            let b = self.head4[h4];
            c4_0 = b[0]; c4_1 = b[1];
            self.prev4[pos & self.ring_mask] = c4_0;
            self.head4[h4] = [pos as u32, c4_0];
        }
        let mut cand6 = NO_POS;
        if pos + 6 <= n {
            let v6 = u64::from(cur_v4)
                | (u64::from(data[pos + 4]) << 32)
                | (u64::from(data[pos + 5]) << 40);
            let h6 = hash6_u64(v6) & self.h6_mask;
            cand6 = self.head6[h6];
            self.head6[h6] = pos as u32;
            self.prev6[pos & self.ring_mask] = cand6;
        }
        (cand3, cur_v3, c4_0, c4_1, cur_v4, cand6)
    }

    #[inline]
    fn relax_match(
        &mut self, pos: usize, base_cost: u32, lit_run: u16, reps: [usize; 4], code: u8,
        off: usize, mlen: usize, min_relax: usize, relax_limit: usize, costs: &CostModelQ4,
        fast_jump: &mut Option<usize>,
    ) {
        if mlen >= 256 && fast_jump.map_or(true, |b| mlen > b) { *fast_jump = Some(mlen); }
        let mut nr = reps;
        mtf_update(&mut nr, off);
        let next_reps = [nr[0] as u32, nr[1] as u32, nr[2] as u32, nr[3] as u32];
        let (prefix_cost, ml_ctx, oc_idx) = costs.ll_and_off_base_cost(code, lit_run);
        let fixed = base_cost.saturating_add(prefix_cost);
        let relax_cap = mlen.min(relax_limit);
        let link = DpLink {
            prev: pos as u32, offset: off as u32, lit_run: 0, off_code: code, is_match: 1,
        };
        let mut relax_one = |len: usize| {
            let dst = pos + len;
            let c = fixed.saturating_add(costs.match_len_cost(ml_ctx, oc_idx, len));
            if c < self.dp_cost[dst] {
                self.dp_cost[dst] = c; self.dp_link[dst] = link; self.dp_reps[dst] = next_reps;
            }
        };
        for len in min_relax..=relax_cap { relax_one(len); }
        if mlen > relax_cap { relax_one(mlen); }
    }
}

fn with_dp_scratch<R, F: FnOnce(&mut DpScratch) -> R>(f: F) -> R {
    let mut local = DpScratch::empty();
    f(&mut local)
}

fn parse_arrival_dp(
    data: &[u8], stride: usize, min_match_1d: usize, max_chain: usize,
    relax_limit: usize, lit_xform: u8, costs: &CostModelQ4, scratch: &mut DpScratch,
) {
    let n = data.len();
    scratch.streams.clear();
    if n == 0 { return; }

    let use_2d_shell = stride >= 4;
    let off_base = if use_2d_shell { OFF_CODE_2D_BASE } else { OFF_CODE_1D_BASE };
    let shell = build_shell_table(stride);
    let init_r = initial_reps(stride);
    scratch.reset();
    scratch.dp_cost[0] = 0;
    scratch.dp_link[0] = DpLink::default();
    scratch.dp_reps[0] = [init_r[0] as u32, init_r[1] as u32, init_r[2] as u32, init_r[3] as u32];

    let ring_mask = scratch.ring_mask;
    let chain3_cap = max_chain.min(16);
    let chain4_cap = bound_usize(max_chain, 4, 256);
    let chain6_cap = bound_usize(max_chain, 4, 128);
    let interior_skip_min: usize = if relax_limit <= 12 { 24 } else { 256 };
    let (mut skip_chain_start, mut skip_chain_until) = (0usize, 0usize);

    let mut pos = 0usize;
    while pos < n {
        let base_cost = scratch.dp_cost[pos];
        if base_cost == u32::MAX { pos += 1; continue; }
        let cur_lit_run = scratch.dp_link[pos].lit_run;
        let cur_reps_u32 = scratch.dp_reps[pos];
        let (cand3, cur_v3, cand4_s0, cand4_s1, cur_v4, cand6) = scratch.insert_hashes(data, pos);

        let max_len = n - pos;
        let reps_usize = [
            cur_reps_u32[0] as usize, cur_reps_u32[1] as usize,
            cur_reps_u32[2] as usize, cur_reps_u32[3] as usize,
        ];

        // 1. Literal step `pos -> pos + 1`.
        let pred = predict_literal(data, pos, reps_usize[0], stride, lit_xform);
        let lit_sym = data[pos.min(n - 1)].wrapping_sub(pred);
        let prev_b = if pos > 0 { data[pos - 1] } else { 0 };
        let next_lit_cost = base_cost.saturating_add(costs.literal_cost(lit_sym, prev_b));
        if next_lit_cost < scratch.dp_cost[pos + 1] {
            scratch.dp_cost[pos + 1] = next_lit_cost;
            scratch.dp_link[pos + 1] = DpLink {
                prev: pos as u32, offset: 0, lit_run: cur_lit_run.saturating_add(1),
                off_code: 0, is_match: 0,
            };
            scratch.dp_reps[pos + 1] = cur_reps_u32;
        }
        if max_len < 2 { pos += 1; continue; }

        let mut fast_jump: Option<usize> = None;
        let mut best_rep_len = 0usize;
        let rep_cands = active_rep_offsets(&reps_usize, cur_lit_run == 0);

        // 2. Check 4 MTF repeat offsets (`0..4`).
        for (r_idx, &off) in rep_cands.iter().enumerate() {
            if off == 0 || off > pos { continue; }
            let cpos = pos - off;
            if data[cpos] == data[pos] && data[cpos + 1] == data[pos + 1] {
                let mlen = match_len_at(data, cpos, pos, max_len);
                best_rep_len = best_rep_len.max(mlen);
                scratch.relax_match(
                    pos, base_cost, cur_lit_run, reps_usize, r_idx as u8, off, mlen, 2,
                    relax_limit, costs, &mut fast_jump,
                );
            }
        }

        // 3. Check 32 zero-extra-bit 2D stride-shell offsets (`4..36`) in 2D mode.
        let mut best_shell_len = 0usize;
        if use_2d_shell {
            let mut best_shell = 1usize;
            for (slot, &off) in shell.iter().enumerate() {
                if off == 0 || off > pos || rep_cands.contains(&off) { continue; }
                let cpos = pos - off;
                if data[cpos] == data[pos] && data[cpos + 1] == data[pos + 1] {
                    let mlen = match_len_at(data, cpos, pos, max_len);
                    if mlen > best_shell {
                        best_shell = mlen;
                        best_shell_len = mlen;
                        scratch.relax_match(
                            pos, base_cost, cur_lit_run, reps_usize, (4 + slot) as u8, off, mlen,
                            2, relax_limit, costs, &mut fast_jump,
                        );
                    }
                }
            }
        }

        let skip_chains = pos >= skip_chain_start && pos < skip_chain_until;

        // 4a. Near 3-byte L1D hash-chain matches (`off < 4096`).
        if !skip_chains && min_match_1d <= 3 && max_len >= 3 && best_rep_len < 16 && fast_jump.is_none() {
            let (mut best3, mut steps, mut cur_cand) = (min_match_1d - 1, 0usize, cand3);
            while cur_cand != NO_POS && steps < chain3_cap && best3 < max_len {
                let cpos = cur_cand as usize;
                if pos <= cpos || pos - cpos >= PREV3_RING { break; }
                let (off, next_cand) = (pos - cpos, scratch.prev3[cpos & PREV3_MASK]);
                let cv3 = u32::from(data[cpos])
                    | (u32::from(data[cpos + 1]) << 8)
                    | (u32::from(data[cpos + 2]) << 16);
                if data[cpos + best3] == data[pos + best3] && cv3 == cur_v3 {
                    let mlen = match_len_at(data, cpos, pos, max_len);
                    if mlen > best3 {
                        best3 = mlen;
                        let code = off_base + offset_to_half_log2(off).0;
                        scratch.relax_match(
                            pos, base_cost, cur_lit_run, reps_usize, code, off, mlen,
                            min_match_1d, relax_limit, costs, &mut fast_jump,
                        );
                        if best3 >= 8 { break; }
                    }
                }
                cur_cand = next_cand; steps += 1;
            }
        }

        let mut best4 = 3usize;
        if max_len >= 4 && best_rep_len < 32 && fast_jump.is_none() {
            let (mut steps, mut cur_cand) = (0usize, cand4_s0);
            while cur_cand != NO_POS && steps < chain4_cap && best4 < max_len {
                let cpos = cur_cand as usize;
                if pos <= cpos || pos - cpos > MAX_OFFSET { break; }
                let off = pos - cpos;
                let next_cand = if steps == 0 { cand4_s1 }
                    else if skip_chains { NO_POS }
                    else { scratch.prev4[cpos & ring_mask] };
                let cv4 = u32::from_le_bytes([data[cpos], data[cpos + 1], data[cpos + 2], data[cpos + 3]]);
                if data[cpos + best4] == data[pos + best4] && cv4 == cur_v4 {
                    let mlen = match_len_at(data, cpos, pos, max_len);
                    if mlen > best4 {
                        best4 = mlen;
                        let code = off_base + offset_to_half_log2(off).0;
                        scratch.relax_match(
                            pos, base_cost, cur_lit_run, reps_usize, code, off, mlen,
                            min_match_1d.max(4), relax_limit, costs, &mut fast_jump,
                        );
                        if best4 == max_len || best4 >= 128 { break; }
                    }
                }
                cur_cand = next_cand; steps += 1;
            }
        }

        let mut best6 = 5usize;
        if !skip_chains && max_len >= 6 && best4 < 24 && best_rep_len < 32 && fast_jump.is_none() {
            let (mut steps, mut cur_cand) = (0usize, cand6);
            while cur_cand != NO_POS && steps < chain6_cap && best6 < max_len {
                let cpos = cur_cand as usize;
                if pos <= cpos || pos - cpos > MAX_OFFSET { break; }
                let (off, next_cand) = (pos - cpos, scratch.prev6[cpos & ring_mask]);
                let cv4 = u32::from_le_bytes([data[cpos], data[cpos + 1], data[cpos + 2], data[cpos + 3]]);
                if data[cpos + best6] == data[pos + best6]
                    && cv4 == cur_v4
                    && data[cpos + 4] == data[pos + 4]
                    && data[cpos + 5] == data[pos + 5]
                {
                    let mlen = match_len_at(data, cpos, pos, max_len);
                    if mlen > best6 {
                        best6 = mlen;
                        let code = off_base + offset_to_half_log2(off).0;
                        scratch.relax_match(
                            pos, base_cost, cur_lit_run, reps_usize, code, off, mlen,
                            min_match_1d.max(6), relax_limit, costs, &mut fast_jump,
                        );
                        if best6 == max_len || best6 >= 128 { break; }
                    }
                }
                cur_cand = next_cand; steps += 1;
            }
        }

        let best_pos_len = best_rep_len
            .max(best_shell_len)
            .max(if best4 > 3 { best4 } else { 0 })
            .max(if best6 > 5 { best6 } else { 0 });
        if best_pos_len >= interior_skip_min {
            let skip_to = pos.saturating_add(best_pos_len.saturating_sub(2)).min(n);
            if skip_to > skip_chain_until {
                if pos >= skip_chain_until { skip_chain_start = pos.saturating_add(2); }
                skip_chain_until = skip_to;
            }
        }

        if let Some(jump_len) = fast_jump {
            let end_pos = pos + jump_len;
            let cap = (pos + 32).min(end_pos.min(n.saturating_sub(2)));
            for k in (pos + 1)..cap {
                let _ = scratch.insert_hashes(data, k);
            }
            pos = end_pos;
        } else {
            pos += 1;
        }
    }

    scratch.path.clear();
    let mut cur = n;
    while cur > 0 {
        let nd = scratch.dp_link[cur];
        let prev = nd.prev as usize;
        scratch.path.push((prev, cur - prev, nd.is_match != 0, nd.off_code, nd.offset as usize));
        cur = prev;
    }
    scratch.path.reverse();

    let mut bw = RawBitWriter::with_capacity(n / 16 + 16);
    let mut reps = init_r;
    let (mut lit_start, mut lit_len) = (0usize, 0usize);
    for &(start, len, is_match, off_code, off) in &scratch.path {
        if !is_match {
            if lit_len == 0 { lit_start = start; }
            lit_len += len;
        } else {
            scratch.streams.push_packet(&mut bw, off_base, lit_start, lit_len, reps[0], Some((len, off_code, off)));
            mtf_update(&mut reps, off);
            lit_len = 0;
        }
    }
    if lit_len > 0 { scratch.streams.push_packet(&mut bw, off_base, lit_start, lit_len, reps[0], None); }
    scratch.streams.raw_bits = bw.finish();
}

fn write_compact_o1_map(prev_bytes: &[u8], n_clusters: usize, map: &[u8], out: &mut Vec<u8>) {
    let mut seen = [0u8; 32];
    for &p in prev_bytes {
        let bi = usize::from(p) >> 3;
        if bi < seen.len() { seen[bi] |= 1u8 << (p & 7); }
    }
    out.push((n_clusters - 1) as u8);
    out.extend_from_slice(&seen);
    let (mut nib_buf, mut high) = (0u8, false);
    for s in 0..256usize {
        if (seen[s >> 3] & (1u8 << (s & 7))) == 0 { continue; }
        let cid = map[s];
        if n_clusters <= 16 {
            if !high { nib_buf = cid & 0x0F; high = true; }
            else { out.push(nib_buf | ((cid & 0x0F) << 4)); high = false; }
        } else {
            out.push(cid);
        }
    }
    if high { out.push(nib_buf); }
}

fn read_compact_o1_map(reader: &mut Reader<'_>) -> Result<(usize, [u8; 256])> {
    let n_clusters = usize::from(reader.read_u8()?) + 1;
    if n_clusters > 32 { return Err(Error::CorruptProbabilityTable); }
    let seen = reader.read_bytes(32)?;
    let mut map = [0u8; 256];
    let (mut cur_byte, mut high) = (0u8, false);
    for s in 0..256usize {
        let bi = s >> 3;
        if bi >= seen.len() || (seen[bi] & (1u8 << (s & 7))) == 0 { continue; }
        let cid = if n_clusters <= 16 {
            if !high { cur_byte = reader.read_u8()?; high = true; cur_byte & 0x0F }
            else { high = false; cur_byte >> 4 }
        } else {
            reader.read_u8()?
        };
        if usize::from(cid) >= n_clusters { return Err(Error::CorruptProbabilityTable); }
        map[s] = cid;
    }
    if high && (cur_byte >> 4) != 0 { return Err(Error::CorruptProbabilityTable); }
    Ok((n_clusters, map))
}

fn encode_1lane_rans_words(
    syms: &[u8], sockets: Option<(&[u8], &[u8])>, models: &[RansModel], scale_bits: u32,
) -> ([u32; 1], Vec<u16>) {
    let mut states = [RANS_L; 1];
    let mut words = Vec::with_capacity(syms.len() / 2 + 16);
    let mut stage = [0u16; MAX_STAGE_WORDS];
    let mut i_end = syms.len();
    while i_end > 0 {
        let i_start = i_end.saturating_sub(256);
        let mut lp = 0usize;
        if let Some((socks, map)) = sockets {
            for i in (i_start..i_end).rev() {
                let cid = usize::from(map[usize::from(socks[i])]);
                let enc = &models[cid].enc_table()[usize::from(syms[i])];
                states[0] = rans_enc_step_scalar(states[0], enc, scale_bits, &mut stage, &mut lp);
            }
        } else {
            let enc_tbl = models[0].enc_table();
            for i in (i_start..i_end).rev() {
                let enc = &enc_tbl[usize::from(syms[i])];
                states[0] = rans_enc_step_scalar(states[0], enc, scale_bits, &mut stage, &mut lp);
            }
        }
        words.extend_from_slice(&stage[..lp]);
        i_end = i_start;
    }
    (states, words)
}

#[inline]
fn keep_smaller(best: &mut Vec<u8>, cand: Vec<u8>) {
    if cand.len() < best.len() { *best = cand; }
}

fn encode_compact_seq_1lane(syms: &[u8], sockets: Option<&[u8]>, n_sockets: usize) -> Result<Vec<u8>> {
    if syms.is_empty() { return Ok(Vec::new()); }
    let encode_one = |ctx_opt: Option<&[u8]>, ns: usize| -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(syms.len() / 2 + 32);
        write_varint_u32(&mut out, syms.len() as u32);
        let sb = ScaleBits::Bits12.bits();
        if let Some(socks) = ctx_opt {
            let cl = ClusteredSocketRans::fit(syms, socks, ns, ScaleBits::Bits12)?;
            let map = &cl.socket_to_cluster;
            let (st, wd) = encode_1lane_rans_words(syms, Some((socks, map)), &cl.models, sb);
            out.push(1u8);
            write_socket_map(&mut out, cl.n_sockets, cl.n_clusters, map);
            for m in &cl.models { write_compact_model(m, &mut out); }
            finish_shared_stream(&mut out, &st, &wd)?;
        } else {
            let model = RansModel::from_bytes(syms, ScaleBits::Bits12)?;
            let (st, wd) = encode_1lane_rans_words(syms, None, core::slice::from_ref(&model), sb);
            out.push(0u8);
            write_compact_model(&model, &mut out);
            finish_shared_stream(&mut out, &st, &wd)?;
        }
        Ok(out)
    };
    let encode_nibble = |ctx_opt: Option<&[u8]>, ns: usize| -> Result<Vec<u8>> {
        let body = encode_nibble_seq(syms, ctx_opt, ns)?;
        let mut out = Vec::with_capacity(body.len() + 6);
        write_varint_u32(&mut out, syms.len() as u32);
        out.push(2u8);
        out.extend_from_slice(&body);
        Ok(out)
    };
    let mut best = encode_one(None, 1)?;
    if let Ok(c) = encode_nibble(None, 1) { keep_smaller(&mut best, c); }
    if let Some(socks) = sockets {
        if n_sockets > 1 && syms.len() >= 64 {
            if let Ok(c) = encode_one(Some(socks), n_sockets) { keep_smaller(&mut best, c); }
            if n_sockets <= 16 {
                if let Ok(c) = encode_nibble(Some(socks), n_sockets) { keep_smaller(&mut best, c); }
            }
        }
    }
    Ok(best)
}

fn decode_compact_seq_1lane(wire: &[u8], sockets: Option<&[u8]>, max_len: usize) -> Result<Vec<u8>> {
    if wire.is_empty() { return Ok(Vec::new()); }
    let mut reader = Reader::new(wire);
    let len = reader.read_varint_u32()? as usize;
    if len > max_len { return Err(Error::OutputTooLarge); }
    let mode = reader.read_u8()?;
    let mut out = Vec::with_capacity(len);
    match mode {
        0 => {
            let model = read_compact_model(&mut reader, ScaleBits::Bits12)?;
            let (mut states, stream) = parse_shared_stream::<1>(&mut reader)?;
            let mut pos = 0usize;
            for _ in 0..len { out.push(decode_one_model_step(&model, &mut states[0], stream, &mut pos)?); }
            verify_terminal_shared(&states, stream.len(), pos)?;
        }
        1 => {
            let socks = sockets.ok_or(Error::InvalidParameter)?;
            if socks.len() != len { return Err(Error::CorruptLzStream); }
            let (n_sockets, n_clusters, map) = read_socket_map(&mut reader)?;
            let mut models = Vec::with_capacity(n_clusters);
            for _ in 0..n_clusters { models.push(read_compact_model(&mut reader, ScaleBits::Bits12)?); }
            let (mut states, stream) = parse_shared_stream::<1>(&mut reader)?;
            let mut pos = 0usize;
            for &s in socks {
                if usize::from(s) >= n_sockets { return Err(Error::CorruptLzStream); }
                let cid = usize::from(map[usize::from(s)]);
                out.push(decode_one_model_step(&models[cid], &mut states[0], stream, &mut pos)?);
            }
            verify_terminal_shared(&states, stream.len(), pos)?;
        }
        2 => out = decode_nibble_seq(reader.read_rest(), len, sockets)?,
        _ => return Err(Error::UnsupportedFormat),
    }
    Ok(out)
}

fn encode_causal_1lane_socket_literals(
    literals: &[u8], prev_bytes: &[u8], scale: ScaleBits,
) -> Result<(Vec<u8>, ClusteredSocketRans)> {
    let cl = ClusteredSocketRans::fit(literals, prev_bytes, 256, scale)?;
    let (st, wd) = encode_1lane_rans_words(
        literals, Some((prev_bytes, &cl.socket_to_cluster)), &cl.models, scale.bits(),
    );
    let mut out = Vec::with_capacity(wd.len() * 2 + 128);
    out.push(scale as u8);
    write_compact_o1_map(prev_bytes, cl.n_clusters, &cl.socket_to_cluster, &mut out);
    for model in &cl.models { write_compact_model(model, &mut out); }
    finish_shared_stream(&mut out, &st, &wd)?;
    Ok((out, cl))
}

fn extract_literals(data: &[u8], streams: &Lz2dStreams, stride: usize, xform: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(streams.lit_positions.len());
    for (&pos, &r0) in streams.lit_positions.iter().zip(streams.lit_rep0.iter()) {
        if pos < data.len() {
            let pred = predict_literal(data, pos, r0, stride, xform);
            out.push(data[pos].wrapping_sub(pred));
        }
    }
    out
}

fn prev_bytes_for(data: &[u8], positions: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(positions.len());
    for &p in positions {
        out.push(if p > 0 && p <= data.len() { data[p - 1] } else { 0 });
    }
    out
}

#[inline]
fn row_byte_bounds(n_rows: usize, stride: usize) -> [(usize, usize); 8] {
    let (base, rem) = (n_rows / 8, n_rows % 8);
    let mut bounds = [(0usize, 0usize); 8];
    let mut cursor = 0usize;
    for (k, slot) in bounds.iter_mut().enumerate() {
        let rows = base + usize::from(k < rem);
        *slot = (cursor * stride, rows * stride);
        cursor += rows;
    }
    bounds
}

#[inline]
fn load_opt_u8(strip: &[u8], idx: isize) -> i32 {
    if idx >= 0 {
        let u = idx as usize;
        if u < strip.len() { return i32::from(strip[u]); }
    }
    0
}

#[inline]
fn strip_neighbor_indices(j: usize, stride: usize) -> (isize, isize, isize) {
    if stride == 0 { return (-1, -1, -1); }
    let (col, ji) = (j % stride, j as isize);
    let n_raw = if j >= stride { ji - (stride as isize) } else { -1 };
    let w = if col > 0 { ji - 1 } else { n_raw };
    let n = if n_raw >= 0 { n_raw } else { w };
    let nw = if n_raw > 0 && col > 0 { n_raw - 1 } else { n };
    (w, n, nw)
}

#[inline]
fn load_strip_neighbors_u8(strip: &[u8], j: usize, stride: usize) -> (u8, u8, u8) {
    let (wi, ni, nwi) = strip_neighbor_indices(j, stride);
    (load_opt_u8(strip, wi) as u8, load_opt_u8(strip, ni) as u8, load_opt_u8(strip, nwi) as u8)
}

fn encode_causal_2d_grad_frame(
    data: &[u8], stride: usize, scale: ScaleBits, plan: &ZenTuningPlan,
) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(data.len()).map_err(|_| Error::InvalidParameter)?;
    let n_rows = data.len() / stride;
    let main_bytes = n_rows * stride;
    let byte_bounds = row_byte_bounds(n_rows, stride);
    let (min_strip_len, rem_lanes) = ((n_rows / 8) * stride, n_rows % 8);
    let (mut residuals, mut sockets) = (vec![0u8; main_bytes], vec![0u8; main_bytes]);

    let tile_blocks = ((plan.l1_tile_bytes / (7 * 32)) & !7).max(8).min(256);
    let (mut w_t, mut n_t, mut nw_t) =
        (vec![[0i32; 8]; tile_blocks], vec![[0i32; 8]; tile_blocks], vec![[0i32; 8]; tile_blocks]);
    let (mut cur_t, mut pred_t, mut sock_t, mut res_t) = (
        vec![[0i32; 8]; tile_blocks], vec![[0i32; 8]; tile_blocks],
        vec![[0i32; 8]; tile_blocks], vec![[0i32; 8]; tile_blocks],
    );

    let mut j0 = 0usize;
    while j0 < min_strip_len {
        let chunk = (min_strip_len - j0).min(tile_blocks);
        for t in 0..chunk {
            let j = j0 + t;
            let (wi, ni, nwi) = strip_neighbor_indices(j, stride);
            for k in 0..8usize {
                let s0 = byte_bounds[k].0;
                let strip = &data[s0..];
                w_t[t][k] = load_opt_u8(strip, wi);
                n_t[t][k] = load_opt_u8(strip, ni);
                nw_t[t][k] = load_opt_u8(strip, nwi);
                cur_t[t][k] = i32::from(data[s0 + j]);
            }
        }
        run_paeth_pred8_batch(&w_t[..chunk], &n_t[..chunk], &nw_t[..chunk], &mut pred_t[..chunk]);
        run_grad_socket8_batch(&w_t[..chunk], &n_t[..chunk], &nw_t[..chunk], &mut sock_t[..chunk]);
        run_sub_mod256_8_batch(&cur_t[..chunk], &pred_t[..chunk], &mut res_t[..chunk]);
        for t in 0..chunk {
            let j = j0 + t;
            for k in 0..8usize {
                let idx = byte_bounds[k].0 + j;
                residuals[idx] = res_t[t][k] as u8;
                sockets[idx] = sock_t[t][k] as u8;
            }
        }
        j0 += chunk;
    }

    for &(s0, slen) in byte_bounds.iter().take(rem_lanes) {
        let strip = &data[s0..];
        for j in min_strip_len..slen {
            let (wk, nk, nwk) = load_strip_neighbors_u8(strip, j, stride);
            residuals[s0 + j] = data[s0 + j].wrapping_sub(paeth_pred_u8(wk, nk, nwk));
            sockets[s0 + j] = grad_socket_u8(wk, nk, nwk);
        }
    }

    let cl = ClusteredSocketRans::fit(&residuals, &sockets, 64, scale)?;
    let scale_bits = scale.bits();
    let (mut states, mut words, mut stage) =
        ([RANS_L; 8], Vec::with_capacity(main_bytes / 2 + 64), [0u16; MAX_STAGE_WORDS]);
    if rem_lanes > 0 {
        for j in (min_strip_len..min_strip_len + stride).rev() {
            let mut lp = 0usize;
            for k in (0..rem_lanes).rev() {
                let idx = byte_bounds[k].0 + j;
                let cid = usize::from(cl.socket_to_cluster[usize::from(sockets[idx])]);
                let enc = &cl.models[cid].enc_table()[usize::from(residuals[idx])];
                states[k] = rans_enc_step_scalar(states[k], enc, scale_bits, &mut stage, &mut lp);
            }
            words.extend_from_slice(&stage[..lp]);
        }
    }
    let mut j_end = min_strip_len;
    while j_end > 0 {
        let j_start = j_end.saturating_sub(64);
        let mut lp = 0usize;
        for j in (j_start..j_end).rev() {
            for k in (0..8usize).rev() {
                let idx = byte_bounds[k].0 + j;
                let cid = usize::from(cl.socket_to_cluster[usize::from(sockets[idx])]);
                let enc = &cl.models[cid].enc_table()[usize::from(residuals[idx])];
                states[k] = rans_enc_step_scalar(states[k], enc, scale_bits, &mut stage, &mut lp);
            }
        }
        words.extend_from_slice(&stage[..lp]);
        j_end = j_start;
    }

    let meta = [BRL2_MODE_CAUSAL_2D_GRAD, scale as u8];
    let mut out = Vec::with_capacity(words.len() * 2 + 256);
    write_framed_header(&mut out, &MAGIC_SOCKET_LZ2D, &meta, orig_len, data);
    write_varint_u32(&mut out, stride as u32);
    write_substream(&mut out, &data[main_bytes..])?;
    write_socket_map(&mut out, cl.n_sockets, cl.n_clusters, &cl.socket_to_cluster);
    for model in &cl.models { write_compact_model(model, &mut out); }
    finish_shared_stream(&mut out, &states, &words)?;
    Ok(out)
}

#[inline]
fn decode_one_model_step(
    model: &RansModel, state: &mut u32, stream: &[u8], pos: &mut usize,
) -> Result<u8> {
    let s = *state;
    let (sym, freq, bias, shift) = match model.scale() {
        ScaleBits::Bits12 => {
            let lut = model.lut_12().ok_or(Error::CorruptProbabilityTable)?;
            let e = lut[(s & 0x0FFF) as usize];
            ((e & 0xFF) as u8, (e >> 20) + 1, (e >> 8) & 0x0FFF, 12u32)
        }
        ScaleBits::Bits14 => {
            let lut = model.lut_14().ok_or(Error::CorruptProbabilityTable)?;
            let e = lut[(s & 0x3FFF) as usize];
            ((e & 0xFF) as u8, (((e >> 22) as u32) & 0x3FFF) + 1, ((e >> 8) & 0x3FFF) as u32, 14u32)
        }
    };
    let mut next = freq * (s >> shift) + bias;
    if next < RANS_L {
        let end = pos.checked_add(2).ok_or(Error::CorruptState)?;
        let pair = stream.get(*pos..end).ok_or(Error::CorruptState)?;
        next = (next << 16) | u32::from(u16::from_le_bytes([pair[0], pair[1]]));
        *pos = end;
    }
    *state = next;
    Ok(sym)
}

fn verify_brl2_crc(meta: &[u8; 2], orig_len_u32: u32, out: &[u8], expected_crc: u32) -> Result<()> {
    if framed_header_crc(&MAGIC_SOCKET_LZ2D, meta, orig_len_u32, out) != expected_crc {
        return Err(Error::ChecksumMismatch);
    }
    Ok(())
}

fn decode_causal_2d_grad_frame(
    reader: &mut Reader<'_>, meta: [u8; 2], orig_len_u32: u32,
    expected_crc: u32, expected_stride: usize,
) -> Result<Vec<u8>> {
    let scale = ScaleBits::from_u8(meta[1])?;
    let stride = reader.read_varint_u32()? as usize;
    if stride < 4 || (expected_stride != 0 && expected_stride != stride) {
        return Err(Error::InvalidParameter);
    }
    let orig_len = orig_len_u32 as usize;
    let n_rows = orig_len / stride;
    if n_rows < 8 { return Err(Error::CorruptLzStream); }
    let main_bytes = n_rows * stride;
    let tail_bytes = reader.read_substream()?;
    if tail_bytes.len() != orig_len - main_bytes { return Err(Error::CorruptLzStream); }
    let (n_sockets, n_clusters, map) = read_socket_map(reader)?;
    if n_sockets != 64 { return Err(Error::CorruptProbabilityTable); }
    let mut models = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters { models.push(read_compact_model(reader, scale)?); }
    let (mut states, stream) = parse_shared_stream::<8>(reader)?;

    let byte_bounds = row_byte_bounds(n_rows, stride);
    let (min_strip_len, rem_lanes) = ((n_rows / 8) * stride, n_rows % 8);
    let (mut out, mut pos) = (vec![0u8; orig_len], 0usize);

    for j in 0..min_strip_len {
        let (wi, ni, nwi) = strip_neighbor_indices(j, stride);
        let (mut w, mut n, mut nw) = ([0i32; 8], [0i32; 8], [0i32; 8]);
        for k in 0..8usize {
            let strip = &out[byte_bounds[k].0..];
            w[k] = load_opt_u8(strip, wi);
            n[k] = load_opt_u8(strip, ni);
            nw[k] = load_opt_u8(strip, nwi);
        }
        let (mut pred, mut sock, mut res, mut pix) = ([0i32; 8], [0i32; 8], [0i32; 8], [0i32; 8]);
        run_paeth_pred8(&w, &n, &nw, &mut pred);
        run_grad_socket8(&w, &n, &nw, &mut sock);
        for k in 0..8usize {
            let cid = usize::from(map[(sock[k] as usize) & 63]);
            res[k] = i32::from(decode_one_model_step(&models[cid], &mut states[k], stream, &mut pos)?);
        }
        run_add_mod256_8(&res, &pred, &mut pix);
        for k in 0..8usize { out[byte_bounds[k].0 + j] = pix[k] as u8; }
    }

    if rem_lanes > 0 {
        for j in min_strip_len..min_strip_len + stride {
            for k in 0..rem_lanes {
                let s0 = byte_bounds[k].0;
                let (wk, nk, nwk) = load_strip_neighbors_u8(&out[s0..], j, stride);
                let pred = paeth_pred_u8(wk, nk, nwk);
                let cid = usize::from(map[usize::from(grad_socket_u8(wk, nk, nwk)) & 63]);
                let sym = decode_one_model_step(&models[cid], &mut states[k], stream, &mut pos)?;
                out[s0 + j] = sym.wrapping_add(pred);
            }
        }
    }

    verify_terminal_shared(&states, stream.len(), pos)?;
    out[main_bytes..].copy_from_slice(tail_bytes);
    verify_brl2_crc(&meta, orig_len_u32, &out, expected_crc)?;
    Ok(out)
}

#[inline]
fn is_l1d_incompressible(data: &[u8]) -> bool {
    let n = data.len();
    if n < 256 { return false; }
    let step = n >> 6;
    let (mut seen, mut hi_hist, mut adj_eq) = ([0u64; 4], [0u8; 16], 0u32);
    for i in 0..64usize {
        let pos = i * step;
        let b = data[pos];
        seen[usize::from(b >> 6) & 3] |= 1u64 << (b & 63);
        hi_hist[usize::from(b >> 4) & 15] += 1;
        if pos + 1 < n && b == data[pos + 1] { adj_eq += 1; }
    }
    let uniq = seen[0].count_ones() + seen[1].count_ones() + seen[2].count_ones() + seen[3].count_ones();
    if uniq < 52 || adj_eq > 1 { return false; }
    let mut sum_sq = 0u32;
    for &c in &hi_hist { sum_sq += u32::from(c) * u32::from(c); }
    sum_sq <= 420
}

fn store_raw_brl2(data: &[u8]) -> Vec<u8> {
    let orig_len = data.len().min(u32::MAX as usize) as u32;
    let mut out = Vec::with_capacity(14 + data.len());
    write_framed_header(&mut out, &MAGIC_SOCKET_LZ2D, &[BRL2_MODE_STORED, 0], orig_len, data);
    out.extend_from_slice(data);
    out
}

fn frame_literal_only(data: &[u8], orig_len: u32, stride: usize, aux: u8, body: &[u8]) -> Result<Vec<u8>> {
    let mut cand = Vec::with_capacity(24 + body.len());
    write_framed_header(&mut cand, &MAGIC_SOCKET_LZ2D, &[BRL2_MODE_LITERAL_ONLY, aux], orig_len, data);
    write_varint_u32(&mut cand, stride as u32);
    write_substream(&mut cand, body)?;
    Ok(cand)
}

fn compress_single_block_with_plan(
    data: &[u8], sockets: Option<&[u8]>, cfg: &SocketLz2dConfig, max_chain: usize, plan: &ZenTuningPlan,
) -> Result<Vec<u8>> {
    with_dp_scratch(|dp_scratch| {
        compress_single_block_with_scratch(data, sockets, cfg, max_chain, plan, dp_scratch)
    })
}

fn compress_single_block_with_scratch(
    data: &[u8], sockets: Option<&[u8]>, cfg: &SocketLz2dConfig, max_chain: usize, plan: &ZenTuningPlan,
    dp_scratch: &mut DpScratch,
) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(data.len()).map_err(|_| Error::InvalidParameter)?;
    if cfg.min_match < 2 || cfg.min_match > 8 || sockets.map_or(false, |s| s.len() != data.len()) {
        return Err(Error::InvalidParameter);
    }
    if data.is_empty() { return Ok(store_raw_brl2(&[])); }

    let stride = cfg.stride;
    if stride == 0 && sockets.is_none() && data.len() >= 256 && is_l1d_incompressible(data) {
        return Ok(store_raw_brl2(data));
    }
    let ext_n_sockets = sockets.map_or(1usize, |s| s.iter().copied().max().map_or(1, |m| usize::from(m) + 1));
    let mut best = store_raw_brl2(data);

    let lit_only_xforms: &[u8] = if stride >= 4 && sockets.is_none() {
        &[LIT_XFORM_RAW, LIT_XFORM_SUB_STRIDE, LIT_XFORM_SUB_PAETH2D]
    } else if stride >= 2 {
        &[LIT_XFORM_RAW, LIT_XFORM_SUB_STRIDE]
    } else {
        &[LIT_XFORM_RAW]
    };
    let stride_socks: Option<(Vec<u8>, usize)> = if sockets.is_none() && stride >= 2 {
        let sv: Vec<u8> = (0..data.len()).map(|p| stride_phase_socket(p, stride)).collect();
        Some((sv, stride_phase_n_sockets(stride)))
    } else {
        None
    };

    let (mut best_prior_xform, mut best_prior_len) = (LIT_XFORM_RAW, usize::MAX);
    for &xf in lit_only_xforms {
        let transformed = apply_full_buffer_xform(data, stride, xf, false);
        let (sock_mode, sock_slice, n_sock) = if let Some(ext) = sockets {
            (SOCK_MODE_EXTERNAL, Some(ext), ext_n_sockets)
        } else if let Some((ref sv, ns)) = stride_socks {
            (SOCK_MODE_STRIDE_PHASE, Some(sv.as_slice()), ns)
        } else {
            (SOCK_MODE_NONE, None, 1)
        };
        let body = encode_substream_socket_auto(&transformed, sock_slice, n_sock, ScaleBits::Bits14)?;
        if body.len() < best_prior_len { best_prior_len = body.len(); best_prior_xform = xf; }
        let aux = (xf & 0x0F) | (sock_mode << 4);
        keep_smaller(&mut best, frame_literal_only(data, orig_len, stride, aux, &body)?);
    }

    if sockets.is_none() && stride >= 4 && data.len() >= stride.saturating_mul(8) {
        if let Ok(cand) = encode_causal_2d_grad_frame(data, stride, ScaleBits::Bits14, plan) {
            keep_smaller(&mut best, cand);
        }
    }

    let min_m = cfg.min_match.max(3);
    let use_o1_1d = sockets.is_none() && stride < 2;
    let passes = bound_usize(usize::from(cfg.arrival_passes), 1, 3);
    let mut costs = CostModelQ4::analytical(data, stride, best_prior_xform);
    let pass1_chain = if passes >= 2 { (max_chain / 2).max(16) } else { max_chain };
    let pass1_relax = if passes >= 2 { 12usize } else { 24usize };

    dp_scratch.prepare(data, plan);
    parse_arrival_dp(
        data, stride, min_m, pass1_chain, pass1_relax, best_prior_xform, &costs, dp_scratch,
    );
        let pass1_lits = dp_scratch.streams.lit_positions.len();
        let pass1_matches = dp_scratch.streams.off_codes.len();
        let dp_q4 = dp_scratch.dp_cost[data.len()];
        let est_bytes = if dp_q4 == u32::MAX { usize::MAX } else { (dp_q4 as usize) / 128 };
        let avg_m_x10 = if pass1_matches > 0 {
            data.len().saturating_sub(pass1_lits).saturating_mul(10) / pass1_matches
        } else {
            0
        };
        let lz_dominates = pass1_lits.saturating_mul(100) < data.len().saturating_mul(38)
            && avg_m_x10 >= 75
            && best_prior_len.saturating_mul(100) >= data.len().saturating_mul(55)
            && est_bytes.saturating_mul(100) < best_prior_len.saturating_mul(70);
        let mut cand1b_len: Option<usize> = None;
        if use_o1_1d && data.len() >= 64 && !lz_dominates {
            let body = encode_adaptive_nc_nibble(data, plan.nc_o2_log2, plan.nc_o3_log2);
            let aux = LIT_XFORM_RAW | (SOCK_MODE_ADAPTIVE_NC << 4);
            let cand = frame_literal_only(data, orig_len, stride, aux, &body)?;
            cand1b_len = Some(cand.len());
            keep_smaller(&mut best, cand);
        }
        let bail_pct = if cand1b_len.is_some() { 108usize } else { 118usize };
        if est_bytes > best.len().saturating_mul(bail_pct) / 100 {
            return Ok(best);
        }
        for _ in 1..passes {
            let lits_pass = extract_literals(data, &dp_scratch.streams, stride, best_prior_xform);
            let o1_fit = if use_o1_1d && lits_pass.len() >= 128 {
                let prevs = prev_bytes_for(data, &dp_scratch.streams.lit_positions);
                ClusteredSocketRans::fit(&lits_pass, &prevs, 256, ScaleBits::Bits12).ok()
            } else {
                None
            };
            costs = CostModelQ4::from_streams(&dp_scratch.streams, stride, &lits_pass, o1_fit.as_ref());
            parse_arrival_dp(
                data, stride, min_m, max_chain, 32, best_prior_xform, &costs, dp_scratch,
            );
        }

        let streams = &dp_scratch.streams;
        let ll_w = encode_compact_seq_1lane(&streams.ll_codes, None, 1)?;
        let ll_zero: Vec<u8> = streams.ll_codes.iter().map(|&ll| u8::from(ll == 0)).collect();
        let ml_w = encode_compact_seq_1lane(&streams.ml_codes, Some(&ll_zero), 2)?;
        let ml_socks: Vec<u8> =
            streams.ml_codes.iter().filter(|&&ml| ml != ML_CODE_NONE).map(|&ml| ml.min(3)).collect();
        let oc_w = encode_compact_seq_1lane(&streams.off_codes, Some(&ml_socks), 4)?;

        let (base_sock_mode, lit_socks, n_lit_socks) = if let Some(ext) = sockets {
            (SOCK_MODE_EXTERNAL, Some(streams.lit_positions.iter().map(|&p| ext[p]).collect()), ext_n_sockets)
        } else if stride >= 2 {
            let ls: Vec<u8> = streams.lit_positions.iter().map(|&p| stride_phase_socket(p, stride)).collect();
            (SOCK_MODE_STRIDE_PHASE, Some(ls), stride_phase_n_sockets(stride))
        } else {
            (SOCK_MODE_NONE, None, 1)
        };

        let lz_xforms: &[u8] = if stride >= 4 && sockets.is_none() {
            &[LIT_XFORM_RAW, LIT_XFORM_SUB_REP0, LIT_XFORM_SUB_STRIDE, LIT_XFORM_SUB_PAETH2D]
        } else if stride >= 2 {
            &[LIT_XFORM_RAW, LIT_XFORM_SUB_REP0, LIT_XFORM_SUB_STRIDE]
        } else {
            &[LIT_XFORM_RAW, LIT_XFORM_SUB_REP0]
        };

        let (mut best_xf, mut best_sm, mut best_lit) = (LIT_XFORM_RAW, base_sock_mode, None::<Vec<u8>>);
        let mut raw_o0_len = usize::MAX;
        for &xf in lz_xforms {
            let lits = extract_literals(data, streams, stride, xf);
            let w = encode_substream_socket_auto(&lits, lit_socks.as_deref(), n_lit_socks, ScaleBits::Bits14)?;
            if xf == LIT_XFORM_RAW { raw_o0_len = w.len(); }
            if best_lit.as_ref().map_or(true, |cur| w.len() < cur.len()) {
                best_xf = xf; best_sm = base_sock_mode; best_lit = Some(w);
            }
            if !use_o1_1d || lits.is_empty() { continue; }

            let prevs = prev_bytes_for(data, &streams.lit_positions);
            let mut fast_o1_won = false;
            if xf == LIT_XFORM_RAW {
                let n_l = lits.len();
                let (mut match_bytes, mut is_pm) = (Vec::with_capacity(n_l), Vec::with_capacity(n_l));
                let mut prev_pos = usize::MAX;
                for (&p, &r0) in streams.lit_positions.iter().zip(streams.lit_rep0.iter()) {
                    let pm = p > 0 && p != prev_pos.wrapping_add(1) && p >= r0 && r0 > 0;
                    match_bytes.push(if pm { data[p - r0] } else { 0 });
                    is_pm.push(pm);
                    prev_pos = p;
                }
                if let Ok(w_fast) = encode_nibble_o1_match(&lits, &prevs, &match_bytes, &is_pm) {
                    fast_o1_won = w_fast.len().saturating_mul(100) < raw_o0_len.saturating_mul(97);
                    if best_lit.as_ref().map_or(true, |cur| w_fast.len() < cur.len()) {
                        best_xf = xf; best_sm = SOCK_MODE_ADAPTIVE_O1; best_lit = Some(w_fast);
                    }
                }
                if passes == 1 && fast_o1_won { break; }
                if !streams.off_codes.is_empty() || cand1b_len.is_none() {
                    let w_nc = encode_adaptive_nc_nibble(&lits, plan.nc_o2_log2, plan.nc_o3_log2);
                    if best_lit.as_ref().map_or(true, |cur| w_nc.len() < cur.len()) {
                        best_xf = xf; best_sm = SOCK_MODE_ADAPTIVE_NC; best_lit = Some(w_nc);
                    }
                }
            }
            let run_o1 = (passes >= 2 || !fast_o1_won)
                && (xf == LIT_XFORM_RAW || best_lit.as_ref().map_or(true, |c| c.len() >= raw_o0_len));
            if run_o1 {
                let sc = if lits.len() >= 16_384 { ScaleBits::Bits14 } else { ScaleBits::Bits12 };
                if let Ok((w_o1, _)) = encode_causal_1lane_socket_literals(&lits, &prevs, sc) {
                    if best_lit.as_ref().map_or(true, |cur| w_o1.len() < cur.len()) {
                        best_xf = xf; best_sm = SOCK_MODE_PREV_BYTE; best_lit = Some(w_o1);
                    }
                }
            }
        }

        if let Some(lit_w) = best_lit {
            let aux = (best_xf & 0x0F) | (best_sm << 4);
            let cap = 40 + ll_w.len() + ml_w.len() + oc_w.len() + streams.raw_bits.len() + lit_w.len();
            let mut cand = Vec::with_capacity(cap);
            write_framed_header(&mut cand, &MAGIC_SOCKET_LZ2D, &[BRL2_MODE_LZ2D, aux], orig_len, data);
            write_varint_u32(&mut cand, stride as u32);
            write_substream(&mut cand, &ll_w)?;
            write_substream(&mut cand, &ml_w)?;
            write_substream(&mut cand, &oc_w)?;
            write_substream(&mut cand, &streams.raw_bits)?;
            write_substream(&mut cand, &lit_w)?;
            keep_smaller(&mut best, cand);
        }
        Ok(best)
}

#[inline]
fn default_chain_for(cfg: &SocketLz2dConfig) -> usize {
    if cfg.arrival_passes <= 1 { DEFAULT_FAST_CHAIN } else { DEFAULT_MAX_CHAIN }
}

/// Compress `data` using `BRL2` with explicit `max_chain` matchfinder depth (`4..=256`).
///
/// # Errors
/// Returns [`Error::InvalidParameter`] on invalid length, socket mismatch, or `min_match`.
pub fn compress_socket_lz2d_with_effort(
    data: &[u8], sockets: Option<&[u8]>, cfg: &SocketLz2dConfig, max_chain: usize,
) -> Result<Vec<u8>> {
    compress_single_block_with_plan(data, sockets, cfg, max_chain, &host_tuning_plan())
}

/// Compress `data` under an explicit [`CpuProfile`] and thread budget (`0` = profile default).
///
/// # Errors
/// Returns [`Error::InvalidParameter`] if `cfg.min_match` is outside `2..=8` or payload exceeds `u32::MAX`.
pub fn compress_socket_lz2d_with_profile(
    data: &[u8], cfg: &SocketLz2dConfig, profile: &CpuProfile, max_threads: usize,
) -> Result<Vec<u8>> {
    if cfg.min_match < 2 || cfg.min_match > 8 { return Err(Error::InvalidParameter); }
    let plan = profile.tuning_plan();
    let chain = default_chain_for(cfg);
    if data.len() >= plan.parallel_threshold_bytes {
        let mut block_bytes = plan.l2_block_bytes;
        if cfg.stride >= 2 && cfg.stride <= block_bytes {
            block_bytes = (block_bytes / cfg.stride).max(1) * cfg.stride;
        }
        let eff_threads = if max_threads == 0 { plan.max_threads } else { max_threads.min(64) };
        let cfg_copy = *cfg;
        Ok(compress_blocks_pooled(data, block_bytes, eff_threads, move |blk| {
            compress_single_block_with_plan(blk, None, &cfg_copy, chain, &plan)
                .unwrap_or_else(|_| store_raw_brl2(blk))
        }))
    } else {
        compress_single_block_with_plan(data, None, cfg, chain, &plan)
    }
}

/// Compress `data` using the `BRL2`/`BRP2` 1D/2D arrival-parsed socket-plugged LZ architecture.
///
/// # Errors
/// Returns [`Error::InvalidParameter`] on invalid length, socket mismatch, or `min_match`.
pub fn compress_socket_lz2d(
    data: &[u8], sockets: Option<&[u8]>, cfg: &SocketLz2dConfig,
) -> Result<Vec<u8>> {
    if sockets.is_some() {
        compress_single_block_with_plan(data, sockets, cfg, default_chain_for(cfg), &host_tuning_plan())
    } else {
        let profile = detect_host_profile();
        compress_socket_lz2d_with_profile(data, cfg, profile, profile.tuning_plan().max_threads)
    }
}

/// Decompress a `BRL2` or `BRP2` frame with optional caller `sockets`, row `stride`, and `max_out`.
///
/// # Errors
/// Returns an [`Error`] on invalid magic, stride/socket mismatch, corrupt streams, CRC, or `max_out`.
pub fn decompress_socket_lz2d_with_limit(
    wire: &[u8], sockets: Option<&[u8]>, stride: usize, max_out: usize,
) -> Result<Vec<u8>> {
    if wire.len() < 4 { return Err(Error::TruncatedStream); }
    if wire[0..4] == MAGIC_PARALLEL_BLOCKS {
        if sockets.is_some() { return Err(Error::InvalidParameter); }
        let threads = host_tuning_plan().max_threads;
        decompress_blocks_pooled(wire, max_out, threads, move |blk, lim| {
            decompress_socket_lz2d_single_with_limit(blk, None, stride, lim)
        })
    } else if wire[0..4] == MAGIC_SOCKET_LZ2D {
        decompress_socket_lz2d_single_with_limit(wire, sockets, stride, max_out)
    } else {
        Err(Error::InvalidMagic)
    }
}

#[inline]
fn push_lit_xform(out: &mut Vec<u8>, r: u8, r0: usize, enc_stride: usize, xform: u8) {
    if xform == LIT_XFORM_RAW {
        out.push(r);
    } else {
        let pred = predict_literal(out, out.len(), r0, enc_stride, xform);
        out.push(r.wrapping_add(pred));
    }
}

#[inline]
fn replay_match_step(
    out: &mut Vec<u8>, pkt_offsets: &[usize], off_idx: &mut usize,
    match_len: usize, replay_reps: &mut [usize; 4],
) -> Result<()> {
    if match_len > 0 {
        let offset = pkt_offsets[*off_idx];
        *off_idx += 1;
        copy_lz_match(out, offset, match_len)?;
        mtf_update(replay_reps, offset);
    }
    Ok(())
}

fn replay_bulk_literals(
    literals: &[u8], pkt_lit_lens: &[usize], pkt_match_lens: &[usize], pkt_offsets: &[usize],
    enc_stride: usize, xform: u8, out: &mut Vec<u8>, replay_reps: &mut [usize; 4], off_idx: &mut usize,
) -> Result<()> {
    let mut lit_pos = 0usize;
    for (&lit_len, &match_len) in pkt_lit_lens.iter().zip(pkt_match_lens.iter()) {
        let lit_end = lit_pos.checked_add(lit_len).ok_or(Error::CorruptLzStream)?;
        let lit_slice = literals.get(lit_pos..lit_end).ok_or(Error::CorruptLzStream)?;
        if xform == LIT_XFORM_RAW {
            out.extend_from_slice(lit_slice);
        } else {
            for &r in lit_slice { push_lit_xform(out, r, replay_reps[0], enc_stride, xform); }
        }
        lit_pos = lit_end;
        replay_match_step(out, pkt_offsets, off_idx, match_len, replay_reps)?;
    }
    if lit_pos != literals.len() { return Err(Error::CorruptLzStream); }
    Ok(())
}

fn sum_lit_lens(pkt_lit_lens: &[usize]) -> Result<usize> {
    let mut total = 0usize;
    for &ll in pkt_lit_lens { total = total.checked_add(ll).ok_or(Error::CorruptLzStream)?; }
    Ok(total)
}

fn decompress_socket_lz2d_single_with_limit(
    wire: &[u8], sockets: Option<&[u8]>, stride: usize, max_out: usize,
) -> Result<Vec<u8>> {
    let mut reader = Reader::new(wire);
    expect_magic(&mut reader, &MAGIC_SOCKET_LZ2D)?;
    let (mode, aux) = (reader.read_u8()?, reader.read_u8()?);
    let orig_len_u32 = reader.read_u32_le()?;
    let orig_len = orig_len_u32 as usize;
    let expected_crc = reader.read_u32_le()?;
    if orig_len > max_out { return Err(Error::OutputTooLarge); }

    let meta = [mode, aux];
    if orig_len == 0 {
        if mode != BRL2_MODE_STORED || aux != 0 { return Err(Error::UnsupportedFormat); }
        reader.finish()?;
        verify_brl2_crc(&meta, orig_len_u32, &[], expected_crc)?;
        return Ok(Vec::new());
    }

    match mode {
        BRL2_MODE_STORED => {
            if aux != 0 { return Err(Error::UnsupportedFormat); }
            let raw = reader.read_bytes(orig_len)?;
            reader.finish()?;
            verify_brl2_crc(&meta, orig_len_u32, raw, expected_crc)?;
            Ok(raw.to_vec())
        }
        BRL2_MODE_LITERAL_ONLY => {
            let (xform, sock_mode) = (aux & 0x0F, aux >> 4);
            if xform > LIT_XFORM_SUB_PAETH2D
                || (sock_mode > SOCK_MODE_STRIDE_PHASE && sock_mode != SOCK_MODE_ADAPTIVE_NC)
            {
                return Err(Error::UnsupportedFormat);
            }
            let enc_stride = reader.read_varint_u32()? as usize;
            if stride != 0 && enc_stride != stride { return Err(Error::InvalidParameter); }
            let body_s = reader.read_substream()?;
            reader.finish()?;
            let res = if sock_mode == SOCK_MODE_ADAPTIVE_NC {
                decode_adaptive_nc_nibble(body_s, orig_len)?
            } else {
                let sock_vec: Option<Vec<u8>> = match sock_mode {
                    SOCK_MODE_NONE => None,
                    SOCK_MODE_EXTERNAL => {
                        let ext = sockets.ok_or(Error::InvalidParameter)?;
                        if ext.len() != orig_len { return Err(Error::InvalidParameter); }
                        Some(ext.to_vec())
                    }
                    SOCK_MODE_STRIDE_PHASE => {
                        if enc_stride < 2 { return Err(Error::CorruptLzStream); }
                        Some((0..orig_len).map(|p| stride_phase_socket(p, enc_stride)).collect())
                    }
                    _ => return Err(Error::UnsupportedFormat),
                };
                decode_substream_socket_auto(body_s, sock_vec.as_deref(), orig_len)?
            };
            if res.len() != orig_len { return Err(Error::CorruptLzStream); }
            let out = apply_full_buffer_xform(&res, enc_stride, xform, true);
            verify_brl2_crc(&meta, orig_len_u32, &out, expected_crc)?;
            Ok(out)
        }
        BRL2_MODE_CAUSAL_2D_GRAD => {
            decode_causal_2d_grad_frame(&mut reader, meta, orig_len_u32, expected_crc, stride)
        }
        BRL2_MODE_LZ2D => {
            let (xform, sock_mode) = (aux & 0x0F, aux >> 4);
            if xform > LIT_XFORM_SUB_PAETH2D || sock_mode > SOCK_MODE_ADAPTIVE_NC {
                return Err(Error::UnsupportedFormat);
            }
            let enc_stride = reader.read_varint_u32()? as usize;
            if stride != 0 && enc_stride != stride { return Err(Error::InvalidParameter); }
            let (ll_s, ml_s, oc_s, raw_bits_s, lit_s) = (
                reader.read_substream()?, reader.read_substream()?, reader.read_substream()?,
                reader.read_substream()?, reader.read_substream()?,
            );
            reader.finish()?;

            let ll_codes = decode_compact_seq_1lane(ll_s, None, orig_len)?;
            let ll_zero: Vec<u8> = ll_codes.iter().map(|&ll| u8::from(ll == 0)).collect();
            let ml_codes = decode_compact_seq_1lane(ml_s, Some(&ll_zero), orig_len)?;
            if ll_codes.len() != ml_codes.len() { return Err(Error::CorruptLzStream); }
            let ml_socks: Vec<u8> =
                ml_codes.iter().filter(|&&ml| ml != ML_CODE_NONE).map(|&ml| ml.min(3)).collect();
            let off_codes = decode_compact_seq_1lane(oc_s, Some(&ml_socks), orig_len)?;

            let use_2d_shell = enc_stride >= 4;
            let off_base = if use_2d_shell { OFF_CODE_2D_BASE } else { OFF_CODE_1D_BASE };
            let shell = build_shell_table(enc_stride);
            let mut reps = initial_reps(enc_stride);
            let mut bit_reader = RawBitReader::new(raw_bits_s);
            let (mut pkt_ll, mut pkt_ml, mut pkt_off) = (
                Vec::with_capacity(ll_codes.len()),
                Vec::with_capacity(ll_codes.len()),
                Vec::with_capacity(off_codes.len()),
            );
            let (mut lit_positions, mut oc_pos, mut out_cursor) = (Vec::new(), 0usize, 0usize);

            for (&llc, &mlc) in ll_codes.iter().zip(ml_codes.iter()) {
                let lit_len = decode_len_code(llc, LL_EXACT_MAX, &mut bit_reader)?;
                if lit_len > orig_len.saturating_sub(out_cursor) { return Err(Error::OutputTooLarge); }
                if sock_mode == SOCK_MODE_EXTERNAL || sock_mode == SOCK_MODE_STRIDE_PHASE {
                    for i in 0..lit_len { lit_positions.push(out_cursor + i); }
                }
                out_cursor += lit_len;
                let match_len = if mlc == ML_CODE_NONE {
                    0usize
                } else {
                    let ml = decode_len_code(mlc, ML_EXACT_MAX, &mut bit_reader)?
                        .checked_add(2)
                        .ok_or(Error::OutputTooLarge)?;
                    if ml > orig_len.saturating_sub(out_cursor) { return Err(Error::OutputTooLarge); }
                    let code = *off_codes.get(oc_pos).ok_or(Error::CorruptLzStream)?;
                    oc_pos += 1;
                    let rep_cands = active_rep_offsets(&reps, lit_len == 0);
                    let offset = decode_offset_code(
                        code, use_2d_shell, off_base, &rep_cands, &shell, &mut bit_reader,
                    )?;
                    mtf_update(&mut reps, offset);
                    pkt_off.push(offset);
                    out_cursor += ml;
                    ml
                };
                pkt_ll.push(lit_len);
                pkt_ml.push(match_len);
            }
            bit_reader.finish()?;
            if out_cursor != orig_len || oc_pos != off_codes.len() { return Err(Error::CorruptLzStream); }

            let (mut out, mut replay_reps, mut off_idx) =
                (Vec::with_capacity(orig_len), initial_reps(enc_stride), 0usize);
            match sock_mode {
                SOCK_MODE_ADAPTIVE_O1 => {
                    let mut dec = NibbleO1MatchDecoder::new(lit_s, sum_lit_lens(&pkt_ll)?)?;
                    for (&lit_len, &match_len) in pkt_ll.iter().zip(pkt_ml.iter()) {
                        let r0 = replay_reps[0];
                        for li in 0..lit_len {
                            let cur_pos = out.len();
                            let prev = if cur_pos >= 1 { out[cur_pos - 1] } else { 0 };
                            let (mb, is_pm) = if li == 0 && cur_pos > 0 && cur_pos >= r0 && r0 > 0 {
                                (out[cur_pos - r0], true)
                            } else {
                                (0u8, false)
                            };
                            let r = dec.decode_byte(prev, mb, is_pm)?;
                            push_lit_xform(&mut out, r, r0, enc_stride, xform);
                        }
                        replay_match_step(&mut out, &pkt_off, &mut off_idx, match_len, &mut replay_reps)?;
                    }
                    dec.finish()?;
                }
                SOCK_MODE_ADAPTIVE_NC => {
                    let literals = decode_adaptive_nc_nibble(lit_s, sum_lit_lens(&pkt_ll)?)?;
                    replay_bulk_literals(
                        &literals, &pkt_ll, &pkt_ml, &pkt_off, enc_stride, xform,
                        &mut out, &mut replay_reps, &mut off_idx,
                    )?;
                }
                SOCK_MODE_PREV_BYTE => {
                    let mut lr = Reader::new(lit_s);
                    let scale = ScaleBits::from_u8(lr.read_u8()?)?;
                    let (n_clusters, map) = read_compact_o1_map(&mut lr)?;
                    let mut models = Vec::with_capacity(n_clusters);
                    for _ in 0..n_clusters { models.push(read_compact_model(&mut lr, scale)?); }
                    let (mut states, lit_stream) = parse_shared_stream::<1>(&mut lr)?;
                    let mut ls_pos = 0usize;
                    for (&lit_len, &match_len) in pkt_ll.iter().zip(pkt_ml.iter()) {
                        for _ in 0..lit_len {
                            let cid = usize::from(map[usize::from(out.last().copied().unwrap_or(0))]);
                            let m = &models[cid];
                            let r = decode_one_model_step(m, &mut states[0], lit_stream, &mut ls_pos)?;
                            push_lit_xform(&mut out, r, replay_reps[0], enc_stride, xform);
                        }
                        replay_match_step(&mut out, &pkt_off, &mut off_idx, match_len, &mut replay_reps)?;
                    }
                    verify_terminal_shared(&states, lit_stream.len(), ls_pos)?;
                }
                _ => {
                    let lit_socks: Option<Vec<u8>> = match sock_mode {
                        SOCK_MODE_NONE => None,
                        SOCK_MODE_EXTERNAL => {
                            let ext = sockets.ok_or(Error::InvalidParameter)?;
                            if ext.len() != orig_len { return Err(Error::InvalidParameter); }
                            Some(lit_positions.iter().map(|&p| ext[p]).collect())
                        }
                        SOCK_MODE_STRIDE_PHASE => {
                            if enc_stride < 2 { return Err(Error::CorruptLzStream); }
                            Some(lit_positions.iter().map(|&p| stride_phase_socket(p, enc_stride)).collect())
                        }
                        _ => return Err(Error::UnsupportedFormat),
                    };
                    let literals = decode_substream_socket_auto(lit_s, lit_socks.as_deref(), orig_len)?;
                    replay_bulk_literals(
                        &literals, &pkt_ll, &pkt_ml, &pkt_off, enc_stride, xform,
                        &mut out, &mut replay_reps, &mut off_idx,
                    )?;
                }
            }
            if out.len() != orig_len || off_idx != pkt_off.len() { return Err(Error::CorruptLzStream); }
            verify_brl2_crc(&meta, orig_len_u32, &out, expected_crc)?;
            Ok(out)
        }
        _ => Err(Error::UnsupportedFormat),
    }
}

#[inline]
fn decode_len_code(code: u8, exact_max: usize, bit_reader: &mut RawBitReader<'_>) -> Result<usize> {
    let c = usize::from(code);
    if c < exact_max { return Ok(c); }
    let msb = c - exact_max;
    if msb > 27 { return Err(Error::CorruptLzStream); }
    Ok((exact_max - 1) + (1usize << msb) + (bit_reader.read(msb as u8)? as usize))
}

#[inline]
fn decode_offset_code(
    code: u8, use_2d_shell: bool, off_base: u8, reps: &[usize; 4],
    shell: &[usize; 32], bit_reader: &mut RawBitReader<'_>,
) -> Result<usize> {
    if code < 4 {
        Ok(reps[usize::from(code)])
    } else if use_2d_shell && code < OFF_CODE_2D_BASE {
        Ok(shell[usize::from(code - 4)])
    } else if code >= off_base {
        let slot = code - off_base;
        if usize::from(slot) >= HALF_LOG2_SLOTS { return Err(Error::CorruptLzStream); }
        let d = if slot < 4 {
            u32::from(slot)
        } else {
            let (msb, half) = (u32::from(slot >> 1), u32::from(slot & 1));
            ((2 | half) << (msb - 1)) | bit_reader.read((msb - 1) as u8)?
        };
        let off = (d as usize) + 1;
        if off > MAX_OFFSET { return Err(Error::CorruptLzStream); }
        Ok(off)
    } else {
        Err(Error::CorruptLzStream)
    }
}

/// Convenience self-contained 1D/2D arrival-parsed LZ compressor (`BRL2`, empirical 1-pass DP).
///
/// # Errors
/// Returns [`Error::InvalidParameter`] if `data.len() > u32::MAX`.
pub fn compress_lz2d(data: &[u8], stride: usize) -> Result<Vec<u8>> {
    compress_socket_lz2d(data, None, &SocketLz2dConfig { stride, min_match: 3, arrival_passes: 1 })
}

/// Progressive anytime `BRL2` compressor mapping `budget_us` deterministically to search tiers,
/// returning `(wire, tier)` (`no_std`, zero OS clock dependencies).
///
/// # Errors
/// Returns [`Error::InvalidParameter`] if `data.len() > u32::MAX`.
pub fn compress_with_time_budget(data: &[u8], stride: usize, budget_us: u64) -> Result<(Vec<u8>, u8)> {
    let tiers: [(usize, u8); 4] = [(DEFAULT_FAST_CHAIN, 1), (48, 2), (96, 2), (256, 3)];
    let target_tier: usize = if budget_us < 10_000 { 1 } else if budget_us < 40_000 { 2 }
        else if budget_us < 120_000 { 3 } else { 4 };
    let mut best = compress_socket_lz2d_with_effort(
        data, None, &SocketLz2dConfig { stride, min_match: 3, arrival_passes: tiers[0].1 }, tiers[0].0,
    )?;
    let mut reached = 1u8;
    for (idx, &(chain, passes)) in tiers.iter().enumerate().skip(1).take(target_tier - 1) {
        let cfg = SocketLz2dConfig { stride, min_match: 3, arrival_passes: passes };
        keep_smaller(&mut best, compress_socket_lz2d_with_effort(data, None, &cfg, chain)?);
        reached = (idx + 1) as u8;
    }
    Ok((best, reached))
}

/// Caller-owned reusable scratch arena for zero-reallocation `no_std` compression across packets.
pub struct NibbleArena {
    scratch: alloc::boxed::Box<DpScratch>,
}

impl NibbleArena {
    /// Allocate a fresh L2-reusable scratch arena for [`compress_with_arena`].
    #[must_use]
    pub fn new() -> Self { Self { scratch: alloc::boxed::Box::new(DpScratch::empty()) } }
}

impl Default for NibbleArena {
    #[inline]
    fn default() -> Self { Self::new() }
}

/// Compress 1D `data` reusing a caller-owned [`NibbleArena`] with zero `DpScratch` reallocations.
///
/// # Errors
/// Returns [`Error::InvalidParameter`] if `data.len() > u32::MAX`.
pub fn compress_with_arena(data: &[u8], arena: &mut NibbleArena) -> Result<Vec<u8>> {
    let cfg = SocketLz2dConfig::default();
    let plan = host_tuning_plan();
    if data.len() >= plan.parallel_threshold_bytes {
        compress_socket_lz2d_with_profile(data, &cfg, detect_host_profile(), plan.max_threads)
    } else {
        compress_single_block_with_scratch(
            data, None, &cfg, default_chain_for(&cfg), &plan, &mut arena.scratch,
        )
    }
}

/// Unified flagship single-call `zero-rans` compressor (`BRL2`/`BRP2`, fast empirical 1-pass DP).
///
/// # Errors
/// Returns [`Error::InvalidParameter`] if a single-block payload exceeds `u32::MAX`.
pub fn compress(data: &[u8]) -> Result<Vec<u8>> { compress_lz2d(data, 0) }

/// Compress 1D `data` under an explicit [`CpuProfile`] and thread budget.
///
/// # Errors
/// Returns [`Error::InvalidParameter`] if a single-block payload exceeds `u32::MAX`.
pub fn compress_with_profile(data: &[u8], profile: &CpuProfile, max_threads: usize) -> Result<Vec<u8>> {
    compress_socket_lz2d_with_profile(data, &SocketLz2dConfig::default(), profile, max_threads)
}

/// Compress 1D `data` using the detected host [`CpuProfile`] and explicit `max_threads` worker cap.
///
/// # Errors
/// Returns [`Error::InvalidParameter`] if a single-block payload exceeds `u32::MAX`.
pub fn compress_parallel(data: &[u8], max_threads: usize) -> Result<Vec<u8>> {
    compress_with_profile(data, detect_host_profile(), max_threads)
}

/// Unified flagship single-call `zero-rans` decompressor (`BRL2`/`BRP2`, `256 MiB` default limit).
///
/// # Errors
/// Returns an [`Error`] on malformed frames, CRC-32 mismatch, or output exceeding `256 MiB`.
pub fn decompress(wire: &[u8]) -> Result<Vec<u8>> { decompress_lz2d(wire, 0) }

/// Convenience self-contained 2D arrival-parsed LZ decompressor (`BRL2`).
///
/// # Errors
/// Returns an [`Error`] on malformed headers, corrupt substreams, stride mismatch, or CRC failure.
pub fn decompress_lz2d(wire: &[u8], stride: usize) -> Result<Vec<u8>> {
    decompress_socket_lz2d_with_limit(wire, None, stride, DEFAULT_MAX_DECODE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix64_bytes(len: usize, mut z: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            let take = (len - out.len()).min(8);
            out.extend_from_slice(&x.to_le_bytes()[..take]);
        }
        out
    }

    #[test]
    fn l1d_incompressible_vpn_pregate_roundtrips_and_skips_compressible() {
        for &n in &[4096usize, 16_384, 65_536] {
            let ct = splitmix64_bytes(n, 0xCAFE_BABE_1234_5678 ^ (n as u64));
            assert!(is_l1d_incompressible(&ct), "ciphertext len={n} must hit pre-gate");
            let wire = compress(&ct).expect("compress ciphertext");
            assert_eq!(wire.len(), 14 + n);
            assert_eq!(decompress(&wire).expect("decompress ciphertext"), ct);
        }
        let text = b"fn compress(data: &[u8]) -> Result<Vec<u8>> { /* compressible */ }\n".repeat(64);
        assert!(!is_l1d_incompressible(&text));
        let wire = compress(&text).expect("compress text");
        assert!(wire.len() < text.len() / 2);
        assert_eq!(decompress(&wire).expect("decompress text"), text);
    }

    #[test]
    fn compress_with_arena_matches_compress_across_packets() {
        let mut arena = NibbleArena::new();
        let packets: [Vec<u8>; 4] = [
            Vec::new(),
            b"packet0_header_0000_packet0_header_0000_packet0_header_0000".to_vec(),
            splitmix64_bytes(4096, 0x1122_3344_5566_7788),
            b"fn step(x: u64) -> u64 { x.wrapping_mul(6364136223846793005) }\n".repeat(48),
        ];
        for pkt in &packets {
            let expected = compress(pkt).expect("compress");
            let actual = compress_with_arena(pkt, &mut arena).expect("compress_with_arena");
            assert_eq!(actual, expected);
            assert_eq!(decompress(&actual).expect("decompress"), *pkt);
        }
    }
}
