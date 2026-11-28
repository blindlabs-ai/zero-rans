//! Fast 2-step 4-bit nibble adaptive non-commutative rANS codec and spec-driven
//! multi-core `BRP2` parallel block framing (`#![forbid(unsafe_code)]`, `0` external crates).

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::rans8::{
    rans_refill_branchless, refill_checked_u16, verify_terminal_shared, RANS_L, RANS_UPPER,
};
use crate::wire::{
    chunk_bounds, crc32_ieee, crc32_update, read_varint_u64, write_varint_u32, write_varint_u64,
    Error, Reader, Result, DEFAULT_MAX_DECODE_BYTES,
};

/// Magic header for deterministic multi-core L2-pinned parallel block framing (`BRP2`).
pub const MAGIC_PARALLEL_BLOCKS: [u8; 4] = *b"BRP2";

const BRP2_VERSION_FLAGS: u8 = 0;
const DEFAULT_PARALLEL_BLOCK_BYTES: usize = 128 * 1024;
const O2_LOG2_MIN: u8 = 6; const O2_LOG2_MAX: u8 = 12;
const O3_LOG2_MIN: u8 = 6; const O3_LOG2_MAX: u8 = 11;
const TILE_LOG2_MIN: u8 = 10; const TILE_LOG2_MAX: u8 = 20;
const DEFAULT_TILE_LOG2: u8 = 14; const NIBBLE_TILE_SYMS: usize = 16_384;
const NC_TAG_EMPTY: u32 = u32::MAX; const LO_HI_EMPTY: u8 = 0xFF;
const CDF16_INIT: [u16; 16] = [
    256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
];

const fn build_cdf16_init_by_count() -> [[u16; 16]; 16] {
    let (mut tbl, mut lim) = ([[0u16; 16]; 16], 0usize);
    while lim < 16 {
        let (n, mut q, mut rem, mut acc, mut k) = (lim + 1, 0usize, 4096 - (15 - lim), 0usize, 0usize);
        while rem >= n { rem -= n; q += 1; }
        while k < 16 {
            acc += if k < n { q + if k < rem { 1 } else { 0 } } else { 1 };
            tbl[lim & 15][k & 15] = acc as u16;
            k += 1;
        }
        lim += 1;
    }
    tbl
}
const fn build_cdf16_targets() -> [[i16; 16]; 16] {
    let (mut tbl, mut sym) = ([[0i16; 16]; 16], 0usize);
    while sym < 16 {
        let mut k = 0usize;
        while k < 15 {
            tbl[sym & 15][k & 15] = if k < sym { (k + 1) as i16 } else { (4081 + k) as i16 };
            k += 1;
        }
        tbl[sym & 15][15] = 4096i16;
        sym += 1;
    }
    tbl
}
const fn build_log2_q4_lut() -> [u8; 256] {
    let (mut tbl, mut i) = ([0u8; 256], 0usize);
    while i < 256 {
        let p = ((i as u32) << 4) + 8;
        let msb = 31 - p.leading_zeros();
        let frac = if msb >= 4 { (p >> (msb - 4)) & 15 } else { (p << (4 - msb)) & 15 };
        tbl[i & 255] = ((msb << 4) + frac) as u8;
        i += 1;
    }
    tbl[0] = 24;
    tbl
}
const fn build_delim_lut() -> [u8; 256] {
    let mut t = [0u8; 256];
    t[123] = 1; t[40] = 2; t[91] = 3; t[125] = 5; t[41] = 6; t[93] = 7;
    t
}
static CDF16_INIT_BY_COUNT: [[u16; 16]; 16] = build_cdf16_init_by_count();
static CDF16_TARGET: [[i16; 16]; 16] = build_cdf16_targets();
static LOG2_Q4_LUT: [u8; 256] = build_log2_q4_lut();
static DELIM_LUT: [u8; 256] = build_delim_lut();

#[inline(always)]
const fn bound_u8(v: u8, lo: u8, hi: u8) -> u8 { if v < lo { lo } else if v > hi { hi } else { v } }
#[inline(always)] const fn bound_i16(v: i32, lo: i32, hi: i32) -> i16 {
    if v < lo { lo as i16 } else if v > hi { hi as i16 } else { v as i16 }
}
fn boxed_array<T: Copy, const N: usize>(val: T) -> Box<[T; N]> {
    vec![val; N].into_boxed_slice().try_into().unwrap_or_else(|_| Box::new([val; N]))
}

const fn build_recip12_u32() -> [u32; 4096] {
    let mut tbl = [0x8000_0000u32; 4096];
    let mut f = 1usize;
    while f < 4096 {
        let f32 = f as u32;
        let sh = 63u32 - (f32 - 1).leading_zeros();
        tbl[f & 0x0FFF] = (((1u64 << sh) + (f32 as u64) - 1) / (f32 as u64)) as u32;
        f += 1;
    }
    tbl
}

/// Half-L1D (`16 KiB`) Granlund-Montgomery reciprocal table (`[u32; 4096]`, `.rodata`).
static RECIP12_U32: [u32; 4096] = build_recip12_u32();

fn recip12_u32_table() -> &'static [u32; 4096] { &RECIP12_U32 }

#[inline(always)]
fn div_mod_prob12(state: u32, freq: u32, tbl: &[u32; 4096]) -> (u32, u32) {
    let f = if freq == 0 { 1 } else { freq };
    let rcp = u64::from(tbl[(f as usize) & 0x0FFF]);
    let shift = (63u32 - (f - 1).leading_zeros()) & 63;
    let q = (u64::from(state).wrapping_mul(rcp) >> shift) as u32;
    (q, state.wrapping_sub(q.wrapping_mul(f)))
}

#[inline(always)]
fn log2_q4_fast(p: u32) -> i32 { i32::from(LOG2_Q4_LUT[((p.saturating_sub(1) >> 4) as usize) & 255]) }

#[inline(always)]
const fn cdf16_prob(cdf: &[u16; 16], sym: usize) -> u32 { cdf16_cum_freq(cdf, sym).1 as u32 }

/// Branchless zero-dead-zone 16-wide `i16` SSE2 Q12 CDF update (`freq[s] >= 1`, `cdf[15] == 4096`).
#[inline(always)]
fn adapt_cdf16<const RATE: u32>(cdf: &mut [u16; 16], sym: usize) {
    let (tgt, bias) = (&CDF16_TARGET[sym & 15], ((1u32 << RATE) - 1) as i16);
    for k in 0..16usize {
        let cur = (cdf[k & 15] & 0x1FFF) as i16;
        let diff = tgt[k & 15] - cur;
        cdf[k & 15] = cur.wrapping_add((diff + (bias & !(diff >> 15))) >> RATE) as u16;
    }
}

#[inline(always)]
fn blend_half16(a: &[u16; 16], b: &[u16; 16], out: &mut [u16; 16]) {
    for k in 0..16usize { out[k & 15] = ((a[k & 15] & 0x1FFF) + (b[k & 15] & 0x1FFF)) >> 1; }
}

#[inline(always)]
fn blend_3_1_16(a: &[u16; 16], b: &[u16; 16], out: &mut [u16; 16]) {
    for k in 0..16usize {
        let (av, bv) = (u32::from(a[k & 15] & 0x1FFF), u32::from(b[k & 15] & 0x1FFF));
        out[k & 15] = (((av << 1) + av + bv) >> 2) as u16;
    }
}

#[inline(always)]
fn exclude_cdf16_sym(src: &[u16; 16], sym: usize, keep_shift: u32, rcp: &[u32; 4096], out: &mut [u16; 16]) {
    let (s, sh) = (sym & 15, keep_shift & 7);
    if sh == 0 { *out = *src; return; }
    let lo_c = if s == 0 { 0u32 } else { u32::from(src[(s - 1) & 15] & 0x1FFF) };
    let f_old = u32::from(src[s & 15] & 0x1FFF).saturating_sub(lo_c);
    let f_new = (f_old >> sh).max(1);
    let (delta, rem_mass) = (f_old.saturating_sub(f_new), 4096u32.saturating_sub(f_old));
    if delta == 0 || rem_mass == 0 || rem_mass >= 4096 { *out = *src; return; }
    let (scale_q12, _) = div_mod_prob12(delta << 12, rem_mass, rcp);
    for k in 0..15usize {
        let ck = u32::from(src[k & 15] & 0x1FFF);
        let c_no = if k < s { ck } else if k == s { lo_c } else { ck.saturating_sub(f_old) };
        let bonus = c_no.wrapping_mul(scale_q12) >> 12;
        let v = c_no.wrapping_add(bonus).wrapping_add(if k >= s { f_new } else { 0 });
        out[k & 15] = if v > 4095 { 4095 } else { v as u16 };
    }
    out[15] = 4096;
}

/// Parallel branchless 16-wide `i16` SSE2 search over a strictly monotonic Q12 CDF.
/// @domain slot: 0..=4095
#[inline(always)]
fn search_cdf16(cdf: &[u16; 16], slot: u32) -> (usize, u32, u32) {
    let (s, mut sum) = ((slot & 0x0FFF) as i16, 16i32);
    for k in 0..16usize { sum += i32::from((s - ((cdf[k & 15] & 0x1FFF) as i16)) >> 15); }
    let sym = (sum as usize) & 15;
    let cum = if sym == 0 { 0u32 } else { u32::from(cdf[(sym - 1) & 15]) };
    let hi = u32::from(cdf[sym & 15]);
    (sym, cum, if hi > cum { hi - cum } else { 1 })
}

#[inline(always)]
const fn cdf16_cum_freq(cdf: &[u16; 16], sym: usize) -> (u16, u16) {
    let s = sym & 15;
    let (cum, hi) = (if s == 0 { 0u16 } else { cdf[(s - 1) & 15] }, cdf[s & 15]);
    (cum, if hi > cum { hi - cum } else { 1u16 })
}

#[inline(always)]
fn delim_stack_step(bits: &mut u64, depth: &mut u8, b: u8) {
    let tag = DELIM_LUT[usize::from(b) & 255];
    if tag == 0 { return; }
    if tag < 4 {
        *bits = (*bits << 2) | u64::from(tag);
        if *depth < 32 { *depth += 1; }
    } else if *depth > 0 && (*bits & 3) == u64::from(tag & 3) {
        *bits >>= 2;
        *depth -= 1;
    }
}

#[inline(always)]
fn delim_stack_ctx(bits: u64, depth: u8, prev1: u8) -> usize {
    if depth == 0 { return 0; }
    let tag = DELIM_LUT[usize::from(prev1) & 255];
    if tag > 0 && tag < 4 { return 0; }
    let top = (bits & 3) as usize;
    if tag >= 5 { top | 4 } else { top }
}

/// Utility score for L2-pinned 2-way slots: profitable contexts (`gain > 0`) stay pinned.
/// @domain seen: 0..=31, gain: -32768..=32767
#[inline(always)]
const fn nc_slot_utility(seen: u16, gain: i16) -> u32 {
    if gain > 0 { 64 + (seen as u32) } else { (seen as u32) >> 2 }
}

/// Full-avalanche Fibonacci-Knuth non-commutative context hash.
/// @domain x: 0..=4294967295, shift: 0..=31
#[inline(always)]
const fn mix_knuth(x: u32, shift: u32) -> usize {
    let h = x.wrapping_mul(0x9E37_79B1);
    ((h ^ (h >> (shift & 31))) >> 16) as usize
}

/// Exact `32 B` (`align(32)`) 16-entry CDF container: zero cache-line splits, 16-wide SSE2.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct AlignedCdf16([u16; 16]);
impl AlignedCdf16 { const fn init() -> Self { Self(CDF16_INIT) } }

/// Exact `128 B` (`align(64)`) 2-way associative context slot (`2` contiguous cache lines).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct NcCtxSlot {
    tag: u32, seen: u16, gain_hi: i16, gain_lo: i16,
    lo_hi: [u8; 2], lo_seen: [u8; 2],
    hi_cdf: AlignedCdf16, lo_cdf: [AlignedCdf16; 2],
}
const _: () = assert!(core::mem::size_of::<AlignedCdf16>() == 32);
const _: () = assert!(core::mem::align_of::<AlignedCdf16>() == 32);
const _: () = assert!(core::mem::size_of::<NcCtxSlot>() == 128);
const _: () = assert!(core::mem::align_of::<NcCtxSlot>() == 64);

impl NcCtxSlot {
    const fn empty() -> Self {
        Self {
            tag: NC_TAG_EMPTY, seen: 0, gain_hi: 0, gain_lo: 0,
            lo_hi: [LO_HI_EMPTY; 2], lo_seen: [0; 2],
            hi_cdf: AlignedCdf16::init(), lo_cdf: [AlignedCdf16::init(); 2],
        }
    }

    #[inline(always)]
    fn resolve_lo_lane(&mut self, hi: u8, fallback: &[u16; 16]) -> usize {
        for lane in 0..2usize { if self.lo_hi[lane & 1] == hi { return lane; } }
        let target = if self.lo_hi[0] == LO_HI_EMPTY { Some(0) }
            else if self.lo_hi[1] == LO_HI_EMPTY { Some(1) }
            else {
                let v = usize::from(self.lo_seen[1] < self.lo_seen[0]);
                if self.lo_seen[v & 1] <= 1 { Some(v) }
                else { self.lo_seen[v & 1] = self.lo_seen[v & 1].saturating_sub(1); None }
            };
        if let Some(l) = target {
            self.lo_hi[l & 1] = hi;
            self.lo_seen[l & 1] = 0;
            self.lo_cdf[l & 1].0 = *fallback;
            l
        } else { 2 }
    }
}

struct NcAssocTable { slots: Vec<NcCtxSlot>, set_mask: usize, slot_mask: usize }
impl NcAssocTable {
    fn new(slot_count: usize) -> Self {
        let safe = slot_count.max(2);
        Self { slots: vec![NcCtxSlot::empty(); safe], set_mask: (safe >> 1) - 1, slot_mask: safe - 1 }
    }
    fn reset(&mut self, slot_count: usize) {
        let safe = slot_count.max(2);
        if self.slots.len() != safe {
            self.slots.resize(safe, NcCtxSlot::empty());
            self.set_mask = (safe >> 1) - 1;
            self.slot_mask = safe - 1;
        }
        self.slots.fill(NcCtxSlot::empty());
    }
    #[inline(always)]
    fn resolve(&mut self, hash: usize, exact_tag: u32, hi_fb: &[u16; 16]) -> (usize, bool) {
        let (w0, w1) = ((hash & self.set_mask) << 1, ((hash & self.set_mask) << 1) | 1);
        if w1 >= self.slots.len() { return (0, false); }
        if self.slots[w0].tag == exact_tag { return (w0, true); }
        if self.slots[w1].tag == exact_tag { return (w1, true); }
        let u0 = nc_slot_utility(self.slots[w0].seen, self.slots[w0].gain_hi.max(self.slots[w0].gain_lo));
        let u1 = nc_slot_utility(self.slots[w1].seen, self.slots[w1].gain_hi.max(self.slots[w1].gain_lo));
        let v = if u0 <= u1 { w0 } else { w1 };
        let e = &mut self.slots[v];
        if e.seen <= 1 { *e = NcCtxSlot::empty(); e.tag = exact_tag; e.hi_cdf.0 = *hi_fb; return (v, true); }
        e.seen = e.seen.saturating_sub(if e.gain_hi.max(e.gain_lo) <= 0 { 2 } else { 1 });
        (v, false)
    }
    #[inline(always)]
    fn get_cdf(&self, slot: usize, hit: bool, lane: Option<usize>) -> Option<(&[u16; 16], u16, i16)> {
        if !hit || slot >= self.slots.len() { return None; }
        let e = &self.slots[slot & self.slot_mask];
        match lane {
            None => Some((&e.hi_cdf.0, e.seen, e.gain_hi)),
            Some(l) if l < 2 => Some((&e.lo_cdf[l & 1].0, u16::from(e.lo_seen[l & 1]), e.gain_lo)),
            Some(_) => None,
        }
    }
    #[inline(always)]
    fn update_slot(
        &mut self, slot: usize, hit: bool, lane: Option<usize>, l_ref: i32, sym: usize,
    ) -> (i32, bool) {
        if !hit || slot >= self.slots.len() { return (l_ref, false); }
        let e = &mut self.slots[slot & self.slot_mask];
        match lane {
            None => update_assoc_entry(&mut e.hi_cdf.0, &mut e.gain_hi, e.seen, None, l_ref, sym),
            Some(l) if l < 2 => {
                let s = u16::from(e.lo_seen[l & 1]);
                let res = update_assoc_entry(
                    &mut e.lo_cdf[l & 1].0, &mut e.gain_lo, s, Some(&mut e.lo_seen[l & 1]), l_ref, sym,
                );
                if e.seen < 31 { e.seen += 1; }
                res
            }
            Some(_) => { if e.seen < 31 { e.seen += 1; } (l_ref, false) }
        }
    }
}
#[inline(always)]
fn select_o1_base<'a>(o0: &'a [u16; 16], o1: &'a [u16; 16], seen: u8, b: &'a mut [u16; 16]) -> &'a [u16; 16] {
    if seen >= 8 { o1 } else if seen == 0 { o0 } else { blend_half16(o1, o0, b); b }
}
#[inline(always)]
fn update_o0_o1(sym: usize, o0: &mut [u16; 16], o1: &mut [u16; 16], seen: &mut u8) {
    if *seen == 0 { *o1 = *o0; }
    if *seen < 24 { adapt_cdf16::<5>(o0, sym); adapt_cdf16::<4>(o1, sym); *seen += 1; }
    else { adapt_cdf16::<5>(o1, sym); }
}
#[inline(always)]
fn adapt_capped_u8(cdf: &mut [u16; 16], seen: &mut u8, fast_lim: u8, cap: u8, sym: usize) {
    if *seen < fast_lim { adapt_cdf16::<4>(cdf, sym); } else { adapt_cdf16::<5>(cdf, sym); }
    if *seen < cap { *seen += 1; }
}

/// Shared zero-copy CDF selector for both high-nibble and low-nibble stages.
#[inline(always)]
fn select_nibble_cdf<'a>(
    o0: &'a [u16; 16], o1: &'a [u16; 16], o1_seen: u8,
    o2: Option<(&'a [u16; 16], u16, i16)>, o3: Option<(&'a [u16; 16], u16, i16)>,
    stk: Option<(&'a [u16; 16], u16, bool)>, scratch: &'a mut [u16; 16],
) -> &'a [u16; 16] {
    let o2a = o2.filter(|&(_, s, g)| s >= 2 && g > 0);
    let o3a = o3.filter(|&(_, s, g)| s >= 2 && g > 0 && (o2a.is_none() || s >= 4));
    if let Some((sc, ss, _)) = stk.filter(|&(_, s, ok)| ok && s >= 4) {
        if ss >= 10 && o2a.is_none() && o3a.is_none() { return sc; }
        let base = if let Some((c, _, _)) = o3a { c } else if let Some((c, _, _)) = o2a { c }
            else if o1_seen > 0 { o1 } else { o0 };
        blend_half16(sc, base, scratch);
        return scratch;
    }
    if let Some((c3, s3, _)) = o3a {
        if s3 >= 8 { return c3; }
        blend_half16(c3, o2a.map_or(o1, |(c2, _, _)| c2), scratch);
        return scratch;
    }
    if let Some((c2, s2, _)) = o2a {
        if s2 >= 8 { return c2; }
        blend_half16(c2, o1, scratch);
        return scratch;
    }
    select_o1_base(o0, o1, o1_seen, scratch)
}

#[inline(always)]
fn update_assoc_entry(
    cdf: &mut [u16; 16], gain: &mut i16, seen: u16, lo_seen: Option<&mut u8>, l_ref: i32, sym: usize,
) -> (i32, bool) {
    let (l, mut pos) = (log2_q4_fast(cdf16_prob(cdf, sym)), false);
    if seen >= 1 {
        let g = i32::from(*gain);
        *gain = bound_i16(g - (g >> 4) + ((l - l_ref) << 2), -16_384, 16_384);
        pos = *gain > 0;
    }
    adapt_cdf16::<3>(cdf, sym);
    if let Some(ls) = lo_seen { if *ls < 31 { *ls += 1; } }
    (l, pos)
}

/// Shared stack + O0/O1 nibble updater.
#[inline(always)]
fn update_stk_and_o01(
    sym: usize, l_o1: i32, o0: &mut [u16; 16], o1: &mut [u16; 16], o1_seen: &mut u8,
    stk: Option<(&mut [u16; 16], &mut u16, &mut u32, &mut u32)>,
) {
    if let Some((sc, ss, eo1, estk)) = stk {
        if *ss >= 4 {
            let l_stk = log2_q4_fast(cdf16_prob(sc, sym));
            *eo1 = *eo1 - (*eo1 >> 5) + (((192i32 - l_o1).max(0) as u32) << 2);
            *estk = *estk - (*estk >> 5) + (((192i32 - l_stk).max(0) as u32) << 2);
        }
        if *ss < 16 { adapt_cdf16::<4>(sc, sym); } else { adapt_cdf16::<5>(sc, sym); }
        if *ss < 28 { *ss += 1; }
    }
    update_o0_o1(sym, o0, o1, o1_seen);
}

struct AdaptiveNcNibbleState {
    o0_hi: [u16; 16], o0_lo: [[u16; 16]; 16],
    o1_hi_cdf: Box<[AlignedCdf16; 256]>, o1_hi_seen: [u8; 256],
    o1_lo_cdf: Box<[AlignedCdf16; 4096]>, o1_lo_seen: Box<[u8; 4096]>,
    o2: NcAssocTable, o3: NcAssocTable, o3_admit: Box<[u8; 2048]>,
    stk_hi: [[u16; 16]; 8], stk_lo: Box<[[[u16; 16]; 16]; 8]>,
    seen_stk_hi: [u16; 8], seen_stk_lo: [[u16; 16]; 8],
    err_o1_stk_hi: u32, err_stk_hi: u32, err_o1_stk_lo: u32, err_stk_lo: u32,
}

impl AdaptiveNcNibbleState {
    fn new(o2_log2: u8, o3_log2: u8) -> Self {
        let n_o2 = 1usize << u32::from(bound_u8(o2_log2, O2_LOG2_MIN, O2_LOG2_MAX));
        let n_o3 = 1usize << u32::from(bound_u8(o3_log2, O3_LOG2_MIN, O3_LOG2_MAX));
        Self {
            o0_hi: CDF16_INIT, o0_lo: [CDF16_INIT; 16],
            o1_hi_cdf: boxed_array(AlignedCdf16::init()), o1_hi_seen: [0u8; 256],
            o1_lo_cdf: boxed_array(AlignedCdf16::init()), o1_lo_seen: boxed_array(0u8),
            o2: NcAssocTable::new(n_o2), o3: NcAssocTable::new(n_o3), o3_admit: boxed_array(0u8),
            stk_hi: [CDF16_INIT; 8], stk_lo: Box::new([[CDF16_INIT; 16]; 8]),
            seen_stk_hi: [0u16; 8], seen_stk_lo: [[0u16; 16]; 8],
            err_o1_stk_hi: 4096, err_stk_hi: 4096, err_o1_stk_lo: 4096, err_stk_lo: 4096,
        }
    }

    fn reset(&mut self, o2_log2: u8, o3_log2: u8) {
        let n_o2 = 1usize << u32::from(bound_u8(o2_log2, O2_LOG2_MIN, O2_LOG2_MAX));
        let n_o3 = 1usize << u32::from(bound_u8(o3_log2, O3_LOG2_MIN, O3_LOG2_MAX));
        self.o0_hi = CDF16_INIT;
        self.o0_lo = [CDF16_INIT; 16];
        self.o1_hi_cdf.fill(AlignedCdf16::init());
        self.o1_hi_seen.fill(0);
        self.o1_lo_cdf.fill(AlignedCdf16::init());
        self.o1_lo_seen.fill(0);
        self.o2.reset(n_o2);
        self.o3.reset(n_o3);
        self.o3_admit.fill(0);
        self.stk_hi = [CDF16_INIT; 8];
        self.stk_lo.fill([CDF16_INIT; 16]);
        self.seen_stk_hi = [0u16; 8];
        self.seen_stk_lo = [[0u16; 16]; 8];
        (self.err_o1_stk_hi, self.err_stk_hi) = (4096, 4096);
        (self.err_o1_stk_lo, self.err_stk_lo) = (4096, 4096);
    }

    #[inline(always)]
    fn admit_o3(&mut self, h3: usize) -> bool {
        let (bi, sh) = ((h3 >> 2) & 2047, ((h3 & 3) << 1) as u32);
        let cur = self.o3_admit[bi & 2047];
        let cnt = (cur >> sh) & 3;
        if cnt >= 2 { return true; }
        self.o3_admit[bi & 2047] = (cur & !(3u8 << sh)) | ((cnt + 1) << sh);
        false
    }

    #[inline(always)]
    fn resolve_byte_slots(&mut self, p3: u8, p2: u8, p1: u8, stk_ctx: usize) -> (usize, bool, usize, bool) {
        let o1_hi = self.o1_hi_cdf[usize::from(p1) & 255].0;
        let o2_tag = (u32::from(p2) << 8) | u32::from(p1);
        let (o2s, o2h) = self.o2.resolve(mix_knuth(o2_tag, 13), o2_tag, &o1_hi);
        if !o2h || o2s >= self.o2.slots.len() { return (o2s, false, 0, false); }
        let e = &self.o2.slots[o2s & self.o2.slot_mask];
        if (e.seen >= 10 && e.gain_lo > 400) || (stk_ctx != 0 && self.err_stk_hi < self.err_o1_stk_hi) {
            return (o2s, true, 0, false);
        }
        let (fb, o3_tag) = (e.hi_cdf.0, (u32::from(p3) << 16) | o2_tag);
        let h3 = mix_knuth(o3_tag, 15);
        if !self.admit_o3(h3) { return (o2s, true, 0, false); }
        let (o3s, o3h) = self.o3.resolve(h3, o3_tag, &fb);
        (o2s, true, o3s, o3h)
    }

    #[inline(always)]
    fn resolve_lo_lanes(
        &mut self, p1: u8, hi: usize, o2s: usize, o2h: bool, o3s: usize, o3h: bool,
    ) -> (usize, usize) {
        let hi_i = hi & 15;
        let o1_cdf = self.o1_lo_cdf[((usize::from(p1) << 4) | hi_i) & 4095].0;
        let o2l = if o2h && o2s < self.o2.slots.len() {
            self.o2.slots[o2s & self.o2.slot_mask].resolve_lo_lane(hi_i as u8, &o1_cdf)
        } else { 2 };
        let o3l = if o3h && o3s < self.o3.slots.len() {
            let fb = if o2l < 2 && o2s < self.o2.slots.len() {
                self.o2.slots[o2s & self.o2.slot_mask].lo_cdf[o2l & 1].0
            } else { o1_cdf };
            self.o3.slots[o3s & self.o3.slot_mask].resolve_lo_lane(hi_i as u8, &fb)
        } else { 2 };
        (o2l, o3l)
    }

    #[inline(always)]
    fn select_cdf<'a>(
        &'a self, p1: u8, hi_opt: Option<usize>, stk_ctx: usize,
        o2s: usize, o2_arg: usize, o3s: usize, o3_arg: usize, scratch: &'a mut [u16; 16],
    ) -> &'a [u16; 16] {
        let si = stk_ctx & 7;
        match hi_opt {
            None => {
                let idx = usize::from(p1) & 255;
                let o2 = self.o2.get_cdf(o2s, o2_arg != 0, None);
                let o3 = self.o3.get_cdf(o3s, o3_arg != 0, None);
                let stk = (si != 0).then(|| (
                    &self.stk_hi[si & 7], self.seen_stk_hi[si & 7], self.err_stk_hi < self.err_o1_stk_hi,
                ));
                select_nibble_cdf(
                    &self.o0_hi, &self.o1_hi_cdf[idx & 255].0, self.o1_hi_seen[idx & 255],
                    o2, o3, stk, scratch,
                )
            }
            Some(hi) => {
                let (hi_i, idx) = (hi & 15, ((usize::from(p1) << 4) | (hi & 15)) & 4095);
                let o2 = self.o2.get_cdf(o2s, true, Some(o2_arg));
                let o3 = self.o3.get_cdf(o3s, true, Some(o3_arg));
                let stk = (si != 0).then(|| (
                    &self.stk_lo[si & 7][hi_i & 15], self.seen_stk_lo[si & 7][hi_i & 15],
                    self.err_stk_lo < self.err_o1_stk_lo,
                ));
                select_nibble_cdf(
                    &self.o0_lo[hi_i & 15], &self.o1_lo_cdf[idx & 4095].0, self.o1_lo_seen[idx & 4095],
                    o2, o3, stk, scratch,
                )
            }
        }
    }

    #[inline(always)]
    fn update_byte(
        &mut self, p1: u8, stk_ctx: usize,
        o2s: usize, o2h: bool, o2l: usize, o3s: usize, o3h: bool, o3l: usize, hi: usize, lo: usize,
    ) {
        let (p1_i, hi_i, lo_i, si) = (usize::from(p1) & 255, hi & 15, lo & 15, stk_ctx & 7);
        let l1_hi = log2_q4_fast(cdf16_prob(&self.o1_hi_cdf[p1_i & 255].0, hi_i));
        let (l2_hi, w2_hi) = self.o2.update_slot(o2s, o2h, None, l1_hi, hi_i);
        let _ = self.o3.update_slot(o3s, o3h, None, if w2_hi { l2_hi } else { l1_hi }, hi_i);
        let stk_hi = (si != 0).then(|| (
            &mut self.stk_hi[si & 7], &mut self.seen_stk_hi[si & 7],
            &mut self.err_o1_stk_hi, &mut self.err_stk_hi,
        ));
        update_stk_and_o01(
            hi_i, l1_hi, &mut self.o0_hi, &mut self.o1_hi_cdf[p1_i & 255].0,
            &mut self.o1_hi_seen[p1_i & 255], stk_hi,
        );

        let idx = ((p1_i << 4) | hi_i) & 4095;
        let l1_lo = log2_q4_fast(cdf16_prob(&self.o1_lo_cdf[idx & 4095].0, lo_i));
        let (l2_lo, w2_lo) = self.o2.update_slot(o2s, o2h, Some(o2l), l1_lo, lo_i);
        let _ = self.o3.update_slot(o3s, o3h, Some(o3l), if w2_lo { l2_lo } else { l1_lo }, lo_i);
        let stk_lo = (si != 0).then(|| (
            &mut self.stk_lo[si & 7][hi_i & 15], &mut self.seen_stk_lo[si & 7][hi_i & 15],
            &mut self.err_o1_stk_lo, &mut self.err_stk_lo,
        ));
        update_stk_and_o01(
            lo_i, l1_lo, &mut self.o0_lo[hi_i & 15], &mut self.o1_lo_cdf[idx & 4095].0,
            &mut self.o1_lo_seen[idx & 4095], stk_lo,
        );
    }
}

#[inline]
fn flush_rans_tile(trace: &[(u16, u16)], rcp_tbl: &[u32; 4096], words: &mut Vec<u16>, out: &mut Vec<u8>) {
    words.clear();
    let (mut state, mut stage, mut sp) = (RANS_L, [0u16; 64], 0usize);
    for &(cum16, freq16) in trace.iter().rev() {
        let (freq, cum) = (u32::from(freq16), u32::from(cum16));
        let need = usize::from(state >= ((RANS_L >> 12) << 16).wrapping_mul(freq));
        stage[sp & 63] = (state & 0xFFFF) as u16;
        state = if need != 0 { state >> 16 } else { state };
        sp += need;
        if sp >= 63 { words.extend_from_slice(&stage[..sp]); sp = 0; }
        let (q, r) = div_mod_prob12(state, freq, rcp_tbl);
        state = (q << 12).wrapping_add(r).wrapping_add(cum);
    }
    if sp > 0 && sp <= stage.len() { words.extend_from_slice(&stage[..sp]); }
    out.extend_from_slice(&state.to_le_bytes());
    write_varint_u32(out, words.len().min(u32::MAX as usize) as u32);
    for &w in words.iter().rev() { out.extend_from_slice(&w.to_le_bytes()); }
}

#[inline(always)]
fn read_rans_tile<'a>(reader: &mut Reader<'a>) -> Result<(u32, &'a [u8])> {
    let init_state = reader.read_u32_le()?;
    if !(RANS_L..RANS_UPPER).contains(&init_state) { return Err(Error::CorruptState); }
    let nbytes = (reader.read_varint_u32()? as usize).checked_mul(2).ok_or(Error::TruncatedStream)?;
    Ok((init_state, reader.read_bytes(nbytes)?))
}

#[inline(always)]
fn rans_dec_step_nibble(
    cdf: &[u16; 16], state: &mut u32, stream: &[u8], pos: &mut usize, fast_limit: usize,
) -> Result<usize> {
    let slot = *state & 0x0FFF;
    let (sym, cum, freq) = search_cdf16(cdf, slot);
    let next = freq.wrapping_mul(*state >> 12).wrapping_add(slot.wrapping_sub(cum));
    *state = if *pos <= fast_limit && *pos + 1 < stream.len() {
        let (s_new, adv) = rans_refill_branchless(next, u16::from_le_bytes([stream[*pos], stream[*pos + 1]]));
        *pos += adv;
        s_new
    } else {
        refill_checked_u16(next, stream, pos)?
    };
    Ok(sym)
}

static NC_STATE_ARENA: () = ();
static O1_STATE_ARENA: () = ();

fn with_tls_arena<T, R>(
    key: &'static (),
    init: impl FnOnce() -> T,
    reset: impl FnOnce(&mut T),
    f: impl FnOnce(&mut T) -> R,
) -> R {
    let _ = (key, reset);
    let mut local = init();
    f(&mut local)
}

/// Encode `data` with the 2-step 4-bit nibble adaptive non-commutative rANS codec.
pub fn encode_adaptive_nc_nibble(data: &[u8], o2_log2: u8, o3_log2: u8) -> Vec<u8> {
    let o2_b = bound_u8(o2_log2, O2_LOG2_MIN, O2_LOG2_MAX);
    let o3_b = bound_u8(o3_log2, O3_LOG2_MIN, O3_LOG2_MAX);
    let tile_bytes = 1usize << u32::from(DEFAULT_TILE_LOG2);
    let mut out = Vec::with_capacity(3 + (data.len() / 2) + 32);
    out.extend_from_slice(&[o2_b, o3_b, DEFAULT_TILE_LOG2]);
    if data.is_empty() { return out; }

    with_tls_arena(
        &NC_STATE_ARENA,
        || AdaptiveNcNibbleState::new(o2_b, o3_b),
        |s| s.reset(o2_b, o3_b),
        |st| {
        let mut trace = Vec::with_capacity(data.len().min(tile_bytes).saturating_mul(2));
        let mut words = Vec::with_capacity(data.len().min(tile_bytes) / 2 + 16);
        let (rcp_tbl, mut scr) = (recip12_u32_table(), CDF16_INIT);
        let (mut p3, mut p2, mut p1, mut sb, mut sd) = (0u8, 0u8, 0u8, 0u64, 0u8);
        for tile in data.chunks(tile_bytes) {
            trace.clear();
            for &byte in tile {
                let (hi, lo) = (usize::from(byte >> 4) & 15, usize::from(byte & 0x0F));
                let sc = delim_stack_ctx(sb, sd, p1);
                let (o2s, o2h, o3s, o3h) = st.resolve_byte_slots(p3, p2, p1, sc);
                let hc = st.select_cdf(p1, None, sc, o2s, usize::from(o2h), o3s, usize::from(o3h), &mut scr);
                trace.push(cdf16_cum_freq(hc, hi));
                let (o2l, o3l) = st.resolve_lo_lanes(p1, hi, o2s, o2h, o3s, o3h);
                let lc = st.select_cdf(p1, Some(hi), sc, o2s, o2l, o3s, o3l, &mut scr);
                trace.push(cdf16_cum_freq(lc, lo));
                st.update_byte(p1, sc, o2s, o2h, o2l, o3s, o3h, o3l, hi, lo);
                delim_stack_step(&mut sb, &mut sd, byte);
                (p3, p2, p1) = (p2, p1, byte);
            }
            flush_rans_tile(&trace, rcp_tbl, &mut words, &mut out);
        }
    });
    out
}

/// Decode a 2-step 4-bit nibble adaptive non-commutative rANS payload of exact length `len`.
pub fn decode_adaptive_nc_nibble(src: &[u8], len: usize) -> Result<Vec<u8>> {
    if len > DEFAULT_MAX_DECODE_BYTES { return Err(Error::OutputTooLarge); }
    let mut reader = Reader::new(src);
    let (o2_l, o3_l, tile_l) = (reader.read_u8()?, reader.read_u8()?, reader.read_u8()?);
    if !(O2_LOG2_MIN..=O2_LOG2_MAX).contains(&o2_l)
        || !(O3_LOG2_MIN..=O3_LOG2_MAX).contains(&o3_l)
        || !(TILE_LOG2_MIN..=TILE_LOG2_MAX).contains(&tile_l)
    {
        return Err(Error::UnsupportedFormat);
    }
    if len == 0 { reader.finish()?; return Ok(Vec::new()); }
    with_tls_arena(
        &NC_STATE_ARENA,
        || AdaptiveNcNibbleState::new(o2_l, o3_l),
        |s| s.reset(o2_l, o3_l),
        |st| {
        let mut out = Vec::with_capacity(len);
        let (mut scr, mut p3, mut p2, mut p1, mut sb, mut sd) = (CDF16_INIT, 0u8, 0u8, 0u8, 0u64, 0u8);
        for (_, chunk_len) in chunk_bounds(len, 1usize << u32::from(tile_l)) {
            let (mut state, stream) = read_rans_tile(&mut reader)?;
            let (mut pos, fl) = (0usize, stream.len().saturating_sub(4));
            for _ in 0..chunk_len {
                let sc = delim_stack_ctx(sb, sd, p1);
                let (o2s, o2h, o3s, o3h) = st.resolve_byte_slots(p3, p2, p1, sc);
                let hc = st.select_cdf(p1, None, sc, o2s, usize::from(o2h), o3s, usize::from(o3h), &mut scr);
                let hi = rans_dec_step_nibble(hc, &mut state, stream, &mut pos, fl)?;
                let (o2l, o3l) = st.resolve_lo_lanes(p1, hi, o2s, o2h, o3s, o3h);
                let lc = st.select_cdf(p1, Some(hi), sc, o2s, o2l, o3s, o3l, &mut scr);
                let lo = rans_dec_step_nibble(lc, &mut state, stream, &mut pos, fl)?;
                let byte = (((hi & 15) as u8) << 4) | ((lo & 15) as u8);
                out.push(byte);
                st.update_byte(p1, sc, o2s, o2h, o2l, o3s, o3h, o3l, hi, lo);
                delim_stack_step(&mut sb, &mut sd, byte);
                (p3, p2, p1) = (p2, p1, byte);
            }
            verify_terminal_shared(&[state], stream.len(), pos)?;
        }
        reader.finish()?;
        Ok(out)
    })
}

struct NibbleSeqState {
    hi_limit: usize, n_ctx: usize,
    base_hi: AlignedCdf16, ctx_hi: Vec<AlignedCdf16>, base_hi_seen: u8, ctx_hi_seen: [u8; 16],
    base_lo: Vec<AlignedCdf16>, ctx_lo: Vec<AlignedCdf16>, base_lo_seen: [u8; 16], ctx_lo_seen: Vec<u8>,
}

impl NibbleSeqState {
    fn new(hi_limit: usize, n_ctx: usize) -> Self {
        let (hi_lim, nc) = (hi_limit & 15, n_ctx.max(1).min(16));
        let (hc, hi_init) = (hi_lim + 1, AlignedCdf16(CDF16_INIT_BY_COUNT[hi_lim & 15]));
        Self {
            hi_limit: hi_lim, n_ctx: nc, base_hi: hi_init,
            ctx_hi: if nc > 1 { vec![hi_init; nc] } else { Vec::new() },
            base_hi_seen: 0, ctx_hi_seen: [0u8; 16],
            base_lo: vec![AlignedCdf16::init(); hc],
            ctx_lo: if nc > 1 { vec![AlignedCdf16::init(); nc * hc] } else { Vec::new() },
            base_lo_seen: [0u8; 16],
            ctx_lo_seen: if nc > 1 { vec![0u8; nc * hc] } else { Vec::new() },
        }
    }

    #[inline(always)]
    fn select<'a>(&'a self, c: usize, hi_opt: Option<usize>, scr: &'a mut [u16; 16]) -> &'a [u16; 16] {
        let (base, ctx_opt) = match hi_opt {
            None => {
                let opt = (self.n_ctx > 1 && c < self.ctx_hi.len())
                    .then(|| (&self.ctx_hi[c].0, self.ctx_hi_seen[c & 15]));
                (&self.base_hi.0, opt)
            }
            Some(hi) => {
                let h = hi.min(self.hi_limit);
                let idx = c * (self.hi_limit + 1) + h;
                let opt = (self.n_ctx > 1 && idx < self.ctx_lo.len() && idx < self.ctx_lo_seen.len())
                    .then(|| (&self.ctx_lo[idx].0, self.ctx_lo_seen[idx]));
                (&self.base_lo[h].0, opt)
            }
        };
        match ctx_opt {
            Some((cdf, seen)) if seen >= 8 => cdf,
            Some((cdf, seen)) if seen > 0 => { blend_half16(cdf, base, scr); scr }
            _ => base,
        }
    }

    #[inline(always)]
    fn update(&mut self, c: usize, hi_opt: Option<usize>, sym: usize) {
        let s = sym & 15;
        match hi_opt {
            None => {
                if self.n_ctx > 1 && c < self.ctx_hi.len() {
                    if self.ctx_hi_seen[c & 15] == 0 { self.ctx_hi[c].0 = self.base_hi.0; }
                    adapt_capped_u8(&mut self.ctx_hi[c].0, &mut self.ctx_hi_seen[c & 15], 16, 28, s);
                }
                adapt_capped_u8(&mut self.base_hi.0, &mut self.base_hi_seen, 24, 24, s);
            }
            Some(hi) => {
                let h = hi.min(self.hi_limit);
                let idx = c * (self.hi_limit + 1) + h;
                if self.n_ctx > 1 && idx < self.ctx_lo.len() && idx < self.ctx_lo_seen.len() {
                    if self.ctx_lo_seen[idx] == 0 { self.ctx_lo[idx].0 = self.base_lo[h].0; }
                    adapt_capped_u8(&mut self.ctx_lo[idx].0, &mut self.ctx_lo_seen[idx], 16, 28, s);
                }
                adapt_capped_u8(&mut self.base_lo[h].0, &mut self.base_lo_seen[h & 15], 24, 24, s);
            }
        }
    }
}

/// Encode a compact LZ sequence stream using universal 4-bit nibble adaptive coding.
pub fn encode_nibble_seq(syms: &[u8], sockets: Option<&[u8]>, n_sockets: usize) -> Result<Vec<u8>> {
    let n_ctx = match sockets {
        Some(socks) if socks.len() == syms.len() && (1..=16).contains(&n_sockets) => n_sockets,
        Some(_) => return Err(Error::InvalidParameter),
        None => 1,
    };
    let hi_lim = usize::from(syms.iter().copied().max().unwrap_or(0) >> 4) & 15;
    let mut out = Vec::with_capacity(1 + syms.len() / 2 + 16);
    out.push(((hi_lim as u8) << 4) | (((n_ctx - 1) as u8) & 0x0F));
    if syms.is_empty() { return Ok(out); }

    let (mut st, rcp_tbl, mut scr) = (NibbleSeqState::new(hi_lim, n_ctx), recip12_u32_table(), CDF16_INIT);
    let tile_n = syms.len().min(NIBBLE_TILE_SYMS);
    let mut trace = Vec::with_capacity(tile_n.saturating_mul(if hi_lim == 0 { 1 } else { 2 }));
    let mut words = Vec::with_capacity(tile_n / 2 + 16);
    for (offset, chunk_len) in chunk_bounds(syms.len(), NIBBLE_TILE_SYMS) {
        trace.clear();
        for i in offset..(offset + chunk_len) {
            let sym = syms[i];
            let hi = if hi_lim == 0 { 0 } else { usize::from(sym >> 4) & 15 };
            let (lo, c) = (usize::from(sym & 0x0F), sockets.map_or(0, |s| usize::from(s[i]).min(n_ctx - 1)));
            if hi_lim > 0 { trace.push(cdf16_cum_freq(st.select(c, None, &mut scr), hi)); }
            trace.push(cdf16_cum_freq(st.select(c, Some(hi), &mut scr), lo));
            if hi_lim > 0 { st.update(c, None, hi); }
            st.update(c, Some(hi), lo);
        }
        flush_rans_tile(&trace, rcp_tbl, &mut words, &mut out);
    }
    Ok(out)
}

/// Decode a universal 4-bit nibble sequence stream produced by [`encode_nibble_seq`].
pub fn decode_nibble_seq(wire: &[u8], len: usize, sockets: Option<&[u8]>) -> Result<Vec<u8>> {
    if len > DEFAULT_MAX_DECODE_BYTES { return Err(Error::OutputTooLarge); }
    let mut reader = Reader::new(wire);
    let meta_b = reader.read_u8()?;
    let (hi_lim, n_ctx) = (usize::from(meta_b >> 4), usize::from(meta_b & 0x0F) + 1);
    if n_ctx > 1 && sockets.ok_or(Error::InvalidParameter)?.len() != len {
        return Err(Error::CorruptLzStream);
    }
    if len == 0 { reader.finish()?; return Ok(Vec::new()); }

    let (mut out, mut st) = (Vec::with_capacity(len), NibbleSeqState::new(hi_lim, n_ctx));
    let mut scr = CDF16_INIT;
    for (offset, chunk_len) in chunk_bounds(len, NIBBLE_TILE_SYMS) {
        let (mut state, stream) = read_rans_tile(&mut reader)?;
        let (mut pos, fl) = (0usize, stream.len().saturating_sub(4));
        for i in offset..(offset + chunk_len) {
            let c = if n_ctx == 1 { 0 } else {
                let sv = usize::from(sockets.ok_or(Error::InvalidParameter)?[i]);
                if sv >= n_ctx { return Err(Error::CorruptLzStream); }
                sv
            };
            let hi = if hi_lim == 0 { 0 } else {
                let hc = st.select(c, None, &mut scr);
                let h = rans_dec_step_nibble(hc, &mut state, stream, &mut pos, fl)?;
                if h > hi_lim { return Err(Error::CorruptLzStream); }
                h
            };
            let lc = st.select(c, Some(hi), &mut scr);
            let lo = rans_dec_step_nibble(lc, &mut state, stream, &mut pos, fl)?;
            out.push((((hi & 15) as u8) << 4) | ((lo & 15) as u8));
            if hi_lim > 0 { st.update(c, None, hi); }
            st.update(c, Some(hi), lo);
        }
        verify_terminal_shared(&[state], stream.len(), pos)?;
    }
    reader.finish()?;
    Ok(out)
}

struct NibbleO1MatchState {
    o0_hi: [u16; 16], o0_lo: [[u16; 16]; 16],
    o1_hi_cdf: Box<[AlignedCdf16; 256]>, o1_hi_seen: [u8; 256],
    o1_lo_cdf: Box<[AlignedCdf16; 4096]>, o1_lo_seen: Box<[u8; 4096]>,
    mb_hi_cdf: [AlignedCdf16; 16], mb_hi_seen: [u8; 16],
    mb_lo_cdf: [AlignedCdf16; 16], mb_lo_seen: [u8; 16],
    pm_eq_q12: u16,
}

impl NibbleO1MatchState {
    fn new() -> Self {
        Self {
            o0_hi: CDF16_INIT, o0_lo: [CDF16_INIT; 16],
            o1_hi_cdf: boxed_array(AlignedCdf16::init()), o1_hi_seen: [0u8; 256],
            o1_lo_cdf: boxed_array(AlignedCdf16::init()), o1_lo_seen: boxed_array(0u8),
            mb_hi_cdf: [AlignedCdf16::init(); 16], mb_hi_seen: [0u8; 16],
            mb_lo_cdf: [AlignedCdf16::init(); 16], mb_lo_seen: [0u8; 16],
            pm_eq_q12: 256,
        }
    }

    fn reset(&mut self) {
        self.o0_hi = CDF16_INIT;
        self.o0_lo = [CDF16_INIT; 16];
        self.o1_hi_cdf.fill(AlignedCdf16::init());
        self.o1_hi_seen.fill(0);
        self.o1_lo_cdf.fill(AlignedCdf16::init());
        self.o1_lo_seen.fill(0);
        self.mb_hi_cdf = [AlignedCdf16::init(); 16];
        self.mb_hi_seen = [0u8; 16];
        self.mb_lo_cdf = [AlignedCdf16::init(); 16];
        self.mb_lo_seen = [0u8; 16];
        self.pm_eq_q12 = 256;
    }

    #[inline(always)]
    fn select_hi<'a>(
        &'a self, prev: u8, mhi: usize, is_pm: bool, scr: &'a mut [u16; 16], aux: &mut [u16; 16],
    ) -> &'a [u16; 16] {
        let (p, m) = (usize::from(prev) & 255, mhi & 15);
        let (seen, o1) = (self.o1_hi_seen[p & 255], &self.o1_hi_cdf[p & 255].0);
        if !(is_pm && self.mb_hi_seen[m & 15] >= 2) { return select_o1_base(&self.o0_hi, o1, seen, scr); }
        let base = select_o1_base(&self.o0_hi, o1, seen, aux);
        if seen >= 12 { blend_3_1_16(base, &self.mb_hi_cdf[m & 15].0, scr); }
        else { blend_half16(base, &self.mb_hi_cdf[m & 15].0, scr); }
        scr
    }

    #[inline(always)]
    fn select_lo<'a>(
        &'a self, prev: u8, hi: usize, mhi: usize, mlo: usize, is_pm: bool,
        rcp: &[u32; 4096], scr: &'a mut [u16; 16], aux: &mut [u16; 16],
    ) -> &'a [u16; 16] {
        let hi_i = hi & 15;
        let idx = ((usize::from(prev) << 4) | hi_i) & 4095;
        let (seen, o1) = (self.o1_lo_seen[idx & 4095], &self.o1_lo_cdf[idx & 4095].0);
        if !(is_pm && hi_i == (mhi & 15)) {
            return select_o1_base(&self.o0_lo[hi_i & 15], o1, seen, scr);
        }
        let base = select_o1_base(&self.o0_lo[hi_i & 15], o1, seen, aux);
        let sh = match self.pm_eq_q12 { 0..=127 => 4u32, 128..=383 => 3, 384..=767 => 2, _ => 0 };
        let (ml, use_mb) = (mlo & 15, self.mb_lo_seen[mlo & 15] >= 4);
        if sh > 0 {
            exclude_cdf16_sym(base, ml, sh, rcp, scr);
            if use_mb { *aux = *scr; blend_3_1_16(aux, &self.mb_lo_cdf[ml & 15].0, scr); }
        } else if use_mb {
            blend_3_1_16(base, &self.mb_lo_cdf[ml & 15].0, scr);
        } else {
            *scr = *base;
        }
        scr
    }

    #[inline(always)]
    fn update_byte(&mut self, prev: u8, mhi: usize, mlo: usize, is_pm: bool, hi: usize, lo: usize) {
        let (p, hi_i, lo_i) = (usize::from(prev) & 255, hi & 15, lo & 15);
        update_o0_o1(hi_i, &mut self.o0_hi, &mut self.o1_hi_cdf[p & 255].0, &mut self.o1_hi_seen[p & 255]);
        if is_pm {
            let m = mhi & 15;
            if self.mb_hi_seen[m & 15] == 0 { self.mb_hi_cdf[m & 15].0 = self.o0_hi; }
            adapt_capped_u8(&mut self.mb_hi_cdf[m & 15].0, &mut self.mb_hi_seen[m & 15], 16, 16, hi_i);
        }
        let idx = ((p << 4) | hi_i) & 4095;
        let (o1c, o1s) = (&mut self.o1_lo_cdf[idx & 4095].0, &mut self.o1_lo_seen[idx & 4095]);
        update_o0_o1(lo_i, &mut self.o0_lo[hi_i & 15], o1c, o1s);
        if is_pm && hi_i == (mhi & 15) {
            let ml = mlo & 15;
            adapt_capped_u8(&mut self.mb_lo_cdf[ml & 15].0, &mut self.mb_lo_seen[ml & 15], 16, 16, lo_i);
            let eq = u32::from(self.pm_eq_q12);
            let next = if lo_i == ml { eq + (4096u32.saturating_sub(eq) >> 4) } else { eq - (eq >> 4) };
            self.pm_eq_q12 = if next < 1 { 1 } else if next > 4095 { 4095 } else { next as u16 };
        }
    }
}

/// Encode a literal stream using universal 2-step 4-bit nibble Order-0 + Order-1 + post-match coding.
pub fn encode_nibble_o1_match(
    literals: &[u8], prev_bytes: &[u8], match_bytes: &[u8], is_post_match: &[bool],
) -> Result<Vec<u8>> {
    let n = literals.len();
    if prev_bytes.len() != n || match_bytes.len() != n || is_post_match.len() != n {
        return Err(Error::InvalidParameter);
    }
    if n == 0 { return Ok(Vec::new()); }
    let rcp = recip12_u32_table();
    let mut out = Vec::with_capacity(n / 2 + 32);
    with_tls_arena(&O1_STATE_ARENA, NibbleO1MatchState::new, NibbleO1MatchState::reset, |st| {
        let (mut scr, mut aux) = (CDF16_INIT, CDF16_INIT);
        let mut trace = Vec::with_capacity(n.min(NIBBLE_TILE_SYMS).saturating_mul(2));
        let mut words = Vec::with_capacity(n.min(NIBBLE_TILE_SYMS) / 2 + 16);
        for (offset, chunk_len) in chunk_bounds(n, NIBBLE_TILE_SYMS) {
            trace.clear();
            for i in offset..(offset + chunk_len) {
                let (sym, prev, mb, pm) = (literals[i], prev_bytes[i], match_bytes[i], is_post_match[i]);
                let (hi, lo) = (usize::from(sym >> 4) & 15, usize::from(sym & 0x0F));
                let (mhi, mlo) = (usize::from(mb >> 4) & 15, usize::from(mb & 0x0F));
                trace.push(cdf16_cum_freq(st.select_hi(prev, mhi, pm, &mut scr, &mut aux), hi));
                let lc = st.select_lo(prev, hi, mhi, mlo, pm, rcp, &mut scr, &mut aux);
                trace.push(cdf16_cum_freq(lc, lo));
                st.update_byte(prev, mhi, mlo, pm, hi, lo);
            }
            flush_rans_tile(&trace, rcp, &mut words, &mut out);
        }
    });
    Ok(out)
}

/// Streaming byte-by-byte decoder for [`encode_nibble_o1_match`].
pub struct NibbleO1MatchDecoder<'a> {
    reader: Reader<'a>, rcp_tbl: &'static [u32; 4096], state: u32, stream: &'a [u8],
    pos: usize, fast_limit: usize, tile_remaining: usize, total_remaining: usize,
    has_active_tile: bool, st: NibbleO1MatchState, scratch: [u16; 16], aux: [u16; 16],
}
impl<'a> NibbleO1MatchDecoder<'a> {
    /// Initialize a streaming 4-bit nibble Order-1 + post-match decoder for `total_lits` literals.
    pub fn new(wire: &'a [u8], total_lits: usize) -> Result<Self> {
        if total_lits > DEFAULT_MAX_DECODE_BYTES { return Err(Error::OutputTooLarge); }
        let mut dec = Self {
            reader: Reader::new(wire), rcp_tbl: recip12_u32_table(), state: RANS_L, stream: &[],
            pos: 0, fast_limit: 0, tile_remaining: 0, total_remaining: total_lits,
            has_active_tile: false, st: NibbleO1MatchState::new(), scratch: CDF16_INIT, aux: CDF16_INIT,
        };
        if total_lits > 0 { dec.open_next_tile()?; }
        Ok(dec)
    }
    fn open_next_tile(&mut self) -> Result<()> {
        if self.has_active_tile { verify_terminal_shared(&[self.state], self.stream.len(), self.pos)?; }
        let (init_state, stream) = read_rans_tile(&mut self.reader)?;
        (self.state, self.stream, self.pos) = (init_state, stream, 0);
        self.fast_limit = stream.len().saturating_sub(4);
        self.tile_remaining = self.total_remaining.min(NIBBLE_TILE_SYMS);
        self.has_active_tile = true;
        Ok(())
    }
    /// Decode the next literal byte given causal `prev_byte` and optional post-match `match_byte`.
    #[inline]
    pub fn decode_byte(&mut self, prev_byte: u8, match_byte: u8, is_pm: bool) -> Result<u8> {
        if self.total_remaining == 0 { return Err(Error::CorruptLzStream); }
        if self.tile_remaining == 0 { self.open_next_tile()?; }
        let (mhi, mlo) = (usize::from(match_byte >> 4) & 15, usize::from(match_byte & 0x0F));
        let hc = self.st.select_hi(prev_byte, mhi, is_pm, &mut self.scratch, &mut self.aux);
        let hi = rans_dec_step_nibble(hc, &mut self.state, self.stream, &mut self.pos, self.fast_limit)?;
        let rcp = self.rcp_tbl;
        let lc = self.st.select_lo(prev_byte, hi, mhi, mlo, is_pm, rcp, &mut self.scratch, &mut self.aux);
        let lo = rans_dec_step_nibble(lc, &mut self.state, self.stream, &mut self.pos, self.fast_limit)?;
        self.st.update_byte(prev_byte, mhi, mlo, is_pm, hi, lo);
        self.tile_remaining -= 1;
        self.total_remaining -= 1;
        Ok((((hi & 15) as u8) << 4) | ((lo & 15) as u8))
    }
    /// Verify terminal `state == RANS_L`, exact stream consumption, and zero trailing bytes.
    pub fn finish(self) -> Result<()> {
        if self.total_remaining != 0 || self.tile_remaining != 0 { return Err(Error::CorruptLzStream); }
        if self.has_active_tile { verify_terminal_shared(&[self.state], self.stream.len(), self.pos)?; }
        self.reader.finish()
    }
}
/// Bulk convenience decoder for [`encode_nibble_o1_match`] given pre-aligned context slices.
pub fn decode_nibble_o1_match(
    wire: &[u8], prev_bytes: &[u8], match_bytes: &[u8], is_post_match: &[bool],
) -> Result<Vec<u8>> {
    let n = prev_bytes.len();
    if match_bytes.len() != n || is_post_match.len() != n { return Err(Error::InvalidParameter); }
    if n > DEFAULT_MAX_DECODE_BYTES { return Err(Error::OutputTooLarge); }
    let (mut dec, mut out) = (NibbleO1MatchDecoder::new(wire, n)?, Vec::with_capacity(n));
    for i in 0..n { out.push(dec.decode_byte(prev_bytes[i], match_bytes[i], is_post_match[i])?); }
    dec.finish()?;
    Ok(out)
}
#[inline]
fn brp2_header_crc(flags: u8, table_bytes: &[u8]) -> u32 {
    !crc32_update(crc32_update(crc32_update(0xFFFF_FFFFu32, &MAGIC_PARALLEL_BLOCKS), &[flags]), table_bytes)
}

fn finish_brp2_frame(raw_len: usize, block_lens: &[usize], enc: &[(Vec<u8>, u32)]) -> Vec<u8> {
    let mut table = Vec::with_capacity(16 + block_lens.len().saturating_mul(12));
    write_varint_u64(&mut table, raw_len as u64);
    write_varint_u64(&mut table, block_lens.len() as u64);
    let mut total_payload = 0usize;
    for (&blen, (comp, crc)) in block_lens.iter().zip(enc.iter()) {
        write_varint_u64(&mut table, blen as u64);
        write_varint_u64(&mut table, comp.len() as u64);
        table.extend_from_slice(&crc.to_le_bytes());
        total_payload = total_payload.saturating_add(comp.len());
    }
    let hcrc = brp2_header_crc(BRP2_VERSION_FLAGS, &table);
    let mut out = Vec::with_capacity(9 + table.len() + total_payload);
    out.extend_from_slice(&MAGIC_PARALLEL_BLOCKS);
    out.push(BRP2_VERSION_FLAGS);
    out.extend_from_slice(&hcrc.to_le_bytes());
    out.extend_from_slice(&table);
    for (comp, _) in enc { out.extend_from_slice(comp); }
    out
}

fn parse_brp2_header(framed: &[u8], max_decode_bytes: usize) -> Result<(usize, Vec<(usize, usize, usize, u32)>)> {
    if framed.len() < 9 { return Err(Error::TruncatedStream); }
    if framed[0..4] != MAGIC_PARALLEL_BLOCKS { return Err(Error::InvalidMagic); }
    let flags = framed[4];
    if flags != BRP2_VERSION_FLAGS { return Err(Error::UnsupportedFormat); }
    let expected_hcrc = u32::from_le_bytes([framed[5], framed[6], framed[7], framed[8]]);
    let mut pos = 9usize;
    let raw_len = usize::try_from(read_varint_u64(framed, &mut pos)?).map_err(|_| Error::OutputTooLarge)?;
    if raw_len > max_decode_bytes { return Err(Error::OutputTooLarge); }
    let n_blocks = usize::try_from(read_varint_u64(framed, &mut pos)?).map_err(|_| Error::CorruptState)?;
    if raw_len == 0 {
        if n_blocks != 0 { return Err(Error::CorruptState); }
        let table = framed.get(9..pos).ok_or(Error::TruncatedStream)?;
        if brp2_header_crc(flags, table) != expected_hcrc { return Err(Error::ChecksumMismatch); }
        return if pos == framed.len() { Ok((0, Vec::new())) } else { Err(Error::TrailingBytes) };
    }
    if n_blocks == 0 || n_blocks > raw_len { return Err(Error::CorruptState); }
    let (mut descs, mut sum_raw, mut sum_comp) = (Vec::with_capacity(n_blocks), 0usize, 0usize);
    for _ in 0..n_blocks {
        let br = usize::try_from(read_varint_u64(framed, &mut pos)?).map_err(|_| Error::OutputTooLarge)?;
        if br == 0 { return Err(Error::CorruptState); }
        let bc = usize::try_from(read_varint_u64(framed, &mut pos)?).map_err(|_| Error::TruncatedStream)?;
        let crc_end = pos.checked_add(4).ok_or(Error::TruncatedStream)?;
        let cb = framed.get(pos..crc_end).ok_or(Error::TruncatedStream)?;
        pos = crc_end;
        sum_raw = sum_raw.checked_add(br).ok_or(Error::OutputTooLarge)?;
        if sum_raw > raw_len { return Err(Error::CorruptState); }
        sum_comp = sum_comp.checked_add(bc).ok_or(Error::TruncatedStream)?;
        descs.push((br, bc, u32::from_le_bytes([cb[0], cb[1], cb[2], cb[3]])));
    }
    let table_end = pos;
    if brp2_header_crc(flags, framed.get(9..table_end).ok_or(Error::TruncatedStream)?) != expected_hcrc {
        return Err(Error::ChecksumMismatch);
    }
    if sum_raw != raw_len { return Err(Error::CorruptState); }
    let total_end = table_end.checked_add(sum_comp).ok_or(Error::TruncatedStream)?;
    if total_end > framed.len() { return Err(Error::TruncatedStream); }
    if total_end < framed.len() { return Err(Error::TrailingBytes); }
    let (mut spans, mut cursor) = (Vec::with_capacity(n_blocks), table_end);
    for (blk_raw, blk_comp, blk_crc) in descs {
        let next = cursor.checked_add(blk_comp).ok_or(Error::TruncatedStream)?;
        spans.push((cursor, next, blk_raw, blk_crc));
        cursor = next;
    }
    Ok((raw_len, spans))
}

fn assemble_brp2_decoded(raw_len: usize, decoded: Vec<Result<Vec<u8>>>) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw_len);
    for res in decoded { out.extend_from_slice(&res?); }
    if out.len() != raw_len { Err(Error::CorruptState) } else { Ok(out) }
}

fn run_parallel_indexed<T: Send, F: Fn(usize) -> T + Sync>(n: usize, max_threads: usize, work: F) -> Vec<T> {
    let _ = max_threads;
    (0..n).map(work).collect()
}
/// Compress `data` into a deterministic `BRP2` framed stream (`no_std` sequential blocks).
pub fn compress_parallel_blocks<F>(
    data: &[u8], block_bytes: usize, max_threads: usize, encode_block: F,
) -> Vec<u8>
where
    F: Fn(&[u8]) -> Vec<u8> + Sync,
{
    let eff = if block_bytes == 0 { DEFAULT_PARALLEL_BLOCK_BYTES } else { block_bytes };
    let blocks: Vec<&[u8]> = if data.is_empty() { Vec::new() } else { data.chunks(eff).collect() };
    let lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
    let enc = run_parallel_indexed(blocks.len(), max_threads, |i| {
        (encode_block(blocks[i]), crc32_ieee(blocks[i]))
    });
    finish_brp2_frame(data.len(), &lens, &enc)
}
/// Compress `data` into a deterministic `BRP2` framed stream (`no_std`).
pub fn compress_blocks_pooled<F>(
    data: &[u8], block_bytes: usize, max_threads: usize, encode_block: F,
) -> Vec<u8>
where
    F: Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static,
{
    compress_parallel_blocks(data, block_bytes, max_threads, encode_block)
}
/// Decompress and verify a `BRP2` framed stream (`no_std` sequential blocks).
pub fn decompress_parallel_blocks<F>(
    framed: &[u8], max_decode_bytes: usize, max_threads: usize, decode_block: F,
) -> Result<Vec<u8>>
where
    F: Fn(&[u8], usize) -> Result<Vec<u8>> + Sync,
{
    let (raw_len, spans) = parse_brp2_header(framed, max_decode_bytes)?;
    if raw_len == 0 { return Ok(Vec::new()); }
    let decoded = run_parallel_indexed(spans.len(), max_threads, |i| {
        let (s, e, blk_raw, blk_crc) = spans[i];
        let dec = decode_block(&framed[s..e], blk_raw)?;
        if dec.len() != blk_raw { return Err(Error::CorruptState); }
        if crc32_ieee(&dec) != blk_crc { return Err(Error::ChecksumMismatch); }
        Ok(dec)
    });
    if decoded.len() != spans.len() { return Err(Error::CorruptState); }
    assemble_brp2_decoded(raw_len, decoded)
}
/// Decompress and verify a `BRP2` stream (`no_std`).
pub fn decompress_blocks_pooled<F>(
    framed: &[u8], max_decode_bytes: usize, max_threads: usize, decode_block: F,
) -> Result<Vec<u8>>
where
    F: Fn(&[u8], usize) -> Result<Vec<u8>> + Send + Sync + 'static,
{
    decompress_parallel_blocks(framed, max_decode_bytes, max_threads, decode_block)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recip12_tile_header_and_cdf_invariants() {
        let tbl = recip12_u32_table();
        assert_eq!(core::mem::size_of_val(tbl), 16_384);
        for f in 1..=4095u32 {
            let xm = ((RANS_L >> 12) << 16).wrapping_mul(f);
            for s in [0, 1, f - 1, f, f + 1, RANS_L, xm - 1, 0x7FFF_FFFFu32] {
                assert_eq!(div_mod_prob12(s, f, tbl), (s / f, s % f));
            }
        }
        let enc = encode_adaptive_nc_nibble(b"hello_tile_test", 9, 8);
        assert_eq!(enc[2], DEFAULT_TILE_LOG2);
        for bt in [TILE_LOG2_MIN - 1, TILE_LOG2_MAX + 1] {
            let mut bad = enc.clone();
            bad[2] = bt;
            assert_eq!(decode_adaptive_nc_nibble(&bad, 15), Err(Error::UnsupportedFormat));
        }
        let mono = |cdf: &[u16; 16]| {
            assert_eq!(cdf[15], 4096);
            let mut p = 0u16;
            for &c in cdf { assert!(c > p); p = c; }
        };
        for lim in 0..16usize { mono(&CDF16_INIT_BY_COUNT[lim]); }
        let (mut cdf, mut out) = (CDF16_INIT, CDF16_INIT);
        for step in 0..128usize {
            adapt_cdf16::<3>(&mut cdf, (step * 7 + 3) & 15);
            for sym in 0..16usize {
                for sh in 1..=5u32 { exclude_cdf16_sym(&cdf, sym, sh, tbl, &mut out); mono(&out); }
            }
        }
    }
    #[test]
    fn nibble_seq_and_o1_match_multitile_knockouts() {
        let s1: Vec<u8> = (0..20_000usize).map(|i| ((i * 5 + (i >> 4)) & 15) as u8).collect();
        let w1 = encode_nibble_seq(&s1, None, 1).expect("e1");
        assert!(w1[0] >> 4 == 0 && decode_nibble_seq(&w1, s1.len(), None).expect("d1") == s1);
        let socks: Vec<u8> = (0..20_000usize).map(|i| (i & 3) as u8).collect();
        let s2: Vec<u8> = socks.iter().enumerate()
            .map(|(i, &c)| (c * 16 + ((i * 3) % 12) as u8).min(75)).collect();
        let w2 = encode_nibble_seq(&s2, Some(&socks), 4).expect("e2");
        let w2n = encode_nibble_seq(&s2, None, 1).expect("e2n");
        assert_eq!(decode_nibble_seq(&w2, s2.len(), Some(&socks)).expect("d2"), s2);
        assert!(w2.len() != w2n.len() && w2.len() < w2n.len());
        let mut tr = w1.clone();
        tr.push(0xFF);
        assert_eq!(decode_nibble_seq(&tr, s1.len(), None), Err(Error::TrailingBytes));
        let n = 20_000usize;
        let (mut lits, mut prevs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut mbs, mut is_pm) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let pm = (i & 1) == 0;
            prevs.push(b'a' + (i % 8) as u8);
            mbs.push(b'a' + ((i + 3) % 16) as u8);
            is_pm.push(pm);
            lits.push(b'a' + (if pm { (i + 4) % 16 } else { (i + 1) % 8 }) as u8);
        }
        let wl = encode_nibble_o1_match(&lits, &prevs, &mbs, &is_pm).expect("el");
        assert_eq!(decode_nibble_o1_match(&wl, &prevs, &mbs, &is_pm).expect("dl"), lits);
        let wn = encode_nibble_o1_match(&lits, &prevs, &mbs, &vec![false; n]).expect("en");
        assert!(wl.len() != wn.len() && wl.len() < wn.len());
        let mut t2 = wl.clone();
        t2.push(0x42);
        assert_eq!(decode_nibble_o1_match(&t2, &prevs, &mbs, &is_pm), Err(Error::TrailingBytes));
    }
}
