//! 4-way and 8-way interleaved SIMD-friendly rANS entropy coding (`BRN1`).

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::wire::{
    balanced_chunks, expect_magic, framed_header_crc, write_framed_header, write_varint_u32,
    Error, Reader, Result, DEFAULT_MAX_DECODE_BYTES,
};

/// Framed rANS stream magic header (`BRN1`).
pub const MAGIC_RANS: [u8; 4] = *b"BRN1";
/// Lower bound of the 32-bit rANS state (`2^15`), guaranteeing `2^15 <= x < 2^31`.
pub const RANS_L: u32 = 1 << 15;
pub(crate) const RANS_UPPER: u32 = 1 << 31;
pub(crate) const BATCH_STEPS: usize = 64;
pub(crate) const MAX_STAGE_WORDS: usize = BATCH_STEPS * 8;
pub(crate) const MAX_WIN_BYTES: usize = MAX_STAGE_WORDS * 2;

const FLAG_ORDER1: u8 = 0x10;
const FLAG_CTX_LOW15: u8 = 0x20;
const FLAG_EXTERNAL_MODEL: u8 = 0x40;
const NUM_CONTEXTS: usize = 16;

/// Probability normalization precision for rANS frequency tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleBits {
    #[doc = "12-bit precision (M = 4096): compact 16 KiB L1D-resident decode LUT."] Bits12 = 12,
    #[doc = "14-bit precision (M = 16384): higher modeling precision, 128 KiB LUT."] Bits14 = 14,
}

impl ScaleBits {
    /// Return the bit shift count (`12` or `14`).
    #[inline] #[must_use] pub const fn bits(self) -> u32 { self as u32 }
    /// Return the total normalized probability mass `1 << bits`.
    #[inline] #[must_use] pub const fn total(self) -> u32 { 1u32 << (self as u32) }
    #[inline]
    pub(crate) fn from_u8(raw: u8) -> Result<Self> {
        match raw { 12 => Ok(Self::Bits12), 14 => Ok(Self::Bits14), _ => Err(Error::UnsupportedFormat) }
    }
}

/// Previous-byte quantization map for order-1 context-modeled 8-way rANS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMap {
    #[doc = "Bucket by upper 4 bits (prev >> 4): effective for general bytes and pixels."] HighNibble,
    #[doc = "Bucket 0..=14 verbatim and >= 15 into bucket 15: effective for runs/levels."] Low15,
}

/// Quantize `prev` by upper 4 bits (`0..=15`) for `HighNibble` order-1 context bucketing.
/// @domain prev: 0..=255
#[inline]
const fn context_bucket_high_nibble(prev: u8) -> usize { (prev >> 4) as usize }

/// Quantize `prev` verbatim for `0..=14` and cap `>= 15` to `15` for `Low15` bucketing.
/// @domain prev: 0..=255
#[inline]
const fn context_bucket_low15(prev: u8) -> usize { if prev < 15 { prev as usize } else { 15 } }

/// Branchless single-lane 16-bit rANS refill step returning `(next_state, advance_bytes)`.
/// @domain next: 0..=2147483647, word: 0..=65535
#[inline(always)]
pub(crate) const fn rans_refill_branchless(next: u32, word: u16) -> (u32, usize) {
    let need = if next < 32_768 { 1u32 } else { 0u32 };
    let mask = 0u32.wrapping_sub(need);
    let refilled = (next << 16) | (word as u32);
    (next ^ ((refilled ^ next) & mask), (need as usize) << 1)
}

impl ContextMap {
    #[inline]
    const fn bucket(self, prev: u8) -> usize {
        match self {
            Self::HighNibble => context_bucket_high_nibble(prev),
            Self::Low15 => context_bucket_low15(prev),
        }
    }
}

/// Precomputed reciprocal-division encoder entry for a single byte symbol.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EncSym {
    pub(crate) freq: u32,
    pub(crate) cum: u32,
    pub(crate) x_max: u32,
    pub(crate) rcp: u64,
    pub(crate) rcp_shift: u32,
}

impl EncSym {
    #[inline]
    pub(crate) fn from_freq_cum(freq: u32, cum: u32, scale_bits: u32) -> Self {
        if freq == 0 { return Self::default(); }
        let x_max = ((RANS_L >> scale_bits) << 16) * freq;
        let l = if freq == 1 { 0u32 } else { (freq - 1).ilog2() + 1 };
        let rcp_shift = 31 + l;
        let rcp = (1u64 << rcp_shift).div_ceil(u64::from(freq));
        Self { freq, cum, x_max, rcp, rcp_shift }
    }
}

#[derive(Debug, Clone)]
enum DecodeLut { Lut12(Box<[u32; 4096]>), Lut14(Box<[u64; 16384]>) }

/// Word width abstraction for packed LUT entries (`u32` for 12-bit, `u64` for 14-bit).
pub(crate) trait LutWord: Copy { fn as_u64(self) -> u64; }
impl LutWord for u32 { #[inline(always)] fn as_u64(self) -> u64 { u64::from(self) } }
impl LutWord for u64 { #[inline(always)] fn as_u64(self) -> u64 { self } }

/// Zero-bounds-check O(1) packed LUT slot decoder shared across 12-bit and 14-bit tables.
#[inline(always)]
pub(crate) fn decode_packed_slot<W: LutWord, const N: usize, const BITS: u32>(
    lut: &[W; N], slot: u32,
) -> (u8, u32, u32) {
    let mask = (1u32 << BITS) - 1;
    let entry = lut[(slot & mask) as usize].as_u64();
    let sym = (entry & 0xFF) as u8;
    let bias = ((entry >> 8) & u64::from(mask)) as u32;
    let freq = (((entry >> (8 + BITS)) as u32) & mask) + 1;
    (sym, freq, bias)
}

/// Checked single-word 16-bit rANS refill for tail/slow paths.
#[inline(never)]
pub(crate) fn refill_checked_u16(next: u32, stream: &[u8], pos: &mut usize) -> Result<u32> {
    if next < RANS_L {
        let end = pos.checked_add(2).ok_or(Error::CorruptState)?;
        let Some(&[b0, b1]) = stream.get(*pos..end) else { return Err(Error::CorruptState); };
        *pos = end;
        Ok((next << 16) | u32::from(u16::from_le_bytes([b0, b1])))
    } else {
        Ok(next)
    }
}

/// Read one `(freq - 1)` LEB128 frequency entry and accumulate validated sum `<= total`.
#[inline]
pub(crate) fn read_checked_freq(reader: &mut Reader<'_>, total: u32, sum: &mut u32) -> Result<u16> {
    let f = reader.read_varint_u32()?.checked_add(1).ok_or(Error::CorruptProbabilityTable)?;
    *sum = sum.checked_add(f).ok_or(Error::CorruptProbabilityTable)?;
    if f > total || *sum > total { return Err(Error::CorruptProbabilityTable); }
    Ok(f as u16)
}

/// Normalized static rANS probability model with reciprocal encoder and O(1) LUT decoder.
#[derive(Debug, Clone)]
pub struct RansModel {
    scale: ScaleBits,
    freqs: [u16; 256],
    enc: [EncSym; 256],
    lut: DecodeLut,
    single_symbol: Option<u8>,
}

impl RansModel {
    /// Normalize raw symbol counts into an exact `ScaleBits` probability model.
    pub fn from_counts(counts: &[u64; 256], scale: ScaleBits) -> Result<Self> {
        Self::from_normalized(&normalize_counts(counts, scale)?, scale)
    }

    /// Build a model directly from a slice of bytes.
    pub fn from_bytes(data: &[u8], scale: ScaleBits) -> Result<Self> {
        let mut counts = [0u64; 256];
        for &b in data { counts[(b as usize) & 255] += 1; }
        Self::from_counts(&counts, scale)
    }

    /// Construct a validated model from pre-normalized frequencies summing to `scale.total()`.
    pub fn from_normalized(freqs: &[u16; 256], scale: ScaleBits) -> Result<Self> {
        let (total, mut sum, mut present, mut last_sym) = (scale.total(), 0u32, 0u32, 0u8);
        let mut enc = [EncSym::default(); 256];
        for sym in 0..256usize {
            let f = u32::from(freqs[sym]);
            if f > 0 {
                if sum.checked_add(f).map_or(true, |n| n > total) {
                    return Err(Error::CorruptProbabilityTable);
                }
                enc[sym] = EncSym::from_freq_cum(f, sum, scale.bits());
                sum += f;
                present += 1;
                last_sym = sym as u8;
            }
        }
        if sum != total || present == 0 { return Err(Error::CorruptProbabilityTable); }
        let lut = build_decode_lut(freqs, &enc, scale);
        Ok(Self { scale, freqs: *freqs, enc, lut, single_symbol: (present == 1).then_some(last_sym) })
    }

    /// Return the normalized frequency array (sums to `scale.total()`).
    #[inline] #[must_use] pub fn freqs(&self) -> &[u16; 256] { &self.freqs }
    /// Return the probability scale precision.
    #[inline] #[must_use] pub const fn scale(&self) -> ScaleBits { self.scale }
    #[inline] pub(crate) fn enc_table(&self) -> &[EncSym; 256] { &self.enc }
    #[inline] pub(crate) fn lut_12(&self) -> Option<&[u32; 4096]> {
        match &self.lut { DecodeLut::Lut12(t) => Some(t), DecodeLut::Lut14(_) => None }
    }
    #[inline] pub(crate) fn lut_14(&self) -> Option<&[u64; 16384]> {
        match &self.lut { DecodeLut::Lut14(t) => Some(t), DecodeLut::Lut12(_) => None }
    }

    pub(crate) fn write_bitmap(&self, out: &mut Vec<u8>) {
        let mut bitmap = [0u8; 32];
        for (sym, &f) in self.freqs.iter().enumerate() {
            if f > 0 { bitmap[sym >> 3] |= 1u8 << (sym & 7); }
        }
        out.extend_from_slice(&bitmap);
        for &f in &self.freqs { if f > 0 { write_varint_u32(out, u32::from(f - 1)); } }
    }

    pub(crate) fn read_bitmap(reader: &mut Reader<'_>, scale: ScaleBits) -> Result<Self> {
        let bitmap = reader.read_bytes(32)?;
        let (total, mut sum, mut freqs) = (scale.total(), 0u32, [0u16; 256]);
        for sym in 0..256usize {
            if (bitmap[sym >> 3] & (1u8 << (sym & 7))) != 0 {
                freqs[sym] = read_checked_freq(reader, total, &mut sum)?;
            }
        }
        if sum != total { return Err(Error::CorruptProbabilityTable); }
        Self::from_normalized(&freqs, scale)
    }
}

fn normalize_counts(counts: &[u64; 256], scale: ScaleBits) -> Result<[u16; 256]> {
    let total_count: u128 = counts.iter().map(|&c| u128::from(c)).sum();
    if total_count == 0 { return Err(Error::CorruptProbabilityTable); }
    let (target, t128) = (scale.total(), u128::from(scale.total()));
    let (mut freqs, mut rems, mut rem_len, mut sum) = ([0u32; 256], [(0u128, 0usize); 256], 0, 0u32);
    for (sym, &c) in counts.iter().enumerate() {
        if c == 0 { continue; }
        let numer = u128::from(c) * t128;
        let q = (numer / total_count) as u32;
        if q == 0 {
            freqs[sym] = 1;
            sum += 1;
        } else {
            freqs[sym] = q;
            sum += q;
            rems[rem_len] = (numer % total_count, sym);
            rem_len += 1;
        }
    }
    if sum < target {
        rems[..rem_len].sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let mut rem = (target - sum) as usize;
        while rem > 0 {
            let take = rem.min(rem_len);
            if take == 0 {
                if let Some((sym, _)) = counts.iter().enumerate().find(|(_, &c)| c > 0) {
                    freqs[sym] += rem as u32;
                }
                break;
            }
            for &(_, sym) in &rems[..take] { freqs[sym] += 1; }
            rem -= take;
        }
    } else if sum > target {
        for _ in 0..(sum - target) {
            let (mut best_s, mut best_f) = (0usize, 1u32);
            for (s, &f) in freqs.iter().enumerate() { if f > best_f { best_f = f; best_s = s; } }
            if best_f <= 1 { return Err(Error::CorruptProbabilityTable); }
            freqs[best_s] -= 1;
        }
    }
    Ok(core::array::from_fn(|sym| freqs[sym] as u16))
}

fn fill_lut<W: Copy + Default, const N: usize>(
    freqs: &[u16; 256], enc: &[EncSym; 256], pack: impl Fn(usize, u32, usize) -> W,
) -> Box<[W; N]> {
    let mut arr = Box::new([W::default(); N]);
    for sym in 0..256usize {
        let f = u32::from(freqs[sym]);
        if f == 0 { continue; }
        let start = enc[sym].cum as usize;
        for off in 0..(f as usize) { arr[start + off] = pack(sym, f, off); }
    }
    arr
}

fn build_decode_lut(freqs: &[u16; 256], enc: &[EncSym; 256], scale: ScaleBits) -> DecodeLut {
    match scale {
        ScaleBits::Bits12 => DecodeLut::Lut12(fill_lut(freqs, enc, |s, f, o| {
            (s as u32) | ((o as u32) << 8) | ((f - 1) << 20)
        })),
        ScaleBits::Bits14 => DecodeLut::Lut14(fill_lut(freqs, enc, |s, f, o| {
            (s as u64) | ((o as u64) << 8) | (u64::from(f - 1) << 22)
        })),
    }
}

/// Perform one branchless division-free rANS encode step into the shared staging buffer.
#[inline(always)]
pub(crate) fn rans_enc_step_scalar(
    state: u32, enc: &EncSym, scale_bits: u32, stage: &mut [u16; MAX_STAGE_WORDS], lp: &mut usize,
) -> u32 {
    if enc.freq == 0 || scale_bits == 0 || scale_bits > 16 || enc.rcp_shift >= 64 { return state; }
    let need = usize::from(state >= enc.x_max);
    stage[*lp & (MAX_STAGE_WORDS - 1)] = (state & 0xFFFF) as u16;
    let x = if need != 0 { state >> 16 } else { state };
    *lp = lp.wrapping_add(need);
    let q = (u64::from(x).wrapping_mul(enc.rcp) >> enc.rcp_shift) as u32;
    let r = x.wrapping_sub(q.wrapping_mul(enc.freq));
    q.wrapping_shl(scale_bits).wrapping_add(r).wrapping_add(enc.cum)
}

/// Emit terminal lane states, shared `u16` word count, and reversed `u16` bitstream.
pub(crate) fn finish_shared_stream<const LANES: usize>(
    out: &mut Vec<u8>, states: &[u32; LANES], words: &[u16],
) -> Result<()> {
    for &s in states { out.extend_from_slice(&s.to_le_bytes()); }
    write_varint_u32(out, u32::try_from(words.len()).map_err(|_| Error::InvalidParameter)?);
    out.reserve(words.len() * 2);
    for &w in words.iter().rev() { out.extend_from_slice(&w.to_le_bytes()); }
    Ok(())
}

/// Parse lane states and the single shared contiguous byte stream from `reader`.
pub(crate) fn parse_shared_stream<'a, const LANES: usize>(
    reader: &mut Reader<'a>,
) -> Result<([u32; LANES], &'a [u8])> {
    let mut states = [0u32; LANES];
    for s in &mut states {
        let v = reader.read_u32_le()?;
        if !(RANS_L..RANS_UPPER).contains(&v) { return Err(Error::CorruptState); }
        *s = v;
    }
    let nbytes = (reader.read_varint_u32()? as usize).checked_mul(2).ok_or(Error::TruncatedStream)?;
    let stream = reader.read_bytes(nbytes)?;
    reader.finish()?;
    Ok((states, stream))
}

/// Verify that all `LANES` states returned to `RANS_L` and every shared stream byte was consumed.
#[inline]
pub(crate) fn verify_terminal_shared<const LANES: usize>(
    states: &[u32; LANES], stream_len: usize, pos: usize,
) -> Result<()> {
    if pos != stream_len || states.iter().any(|&s| s != RANS_L) { Err(Error::CorruptState) } else { Ok(()) }
}

/// Core `LANES`-way LIFO rANS encoder parameterized by per-position encoder lookup `sym_enc`.
pub(crate) fn encode_lanes_by<'a, const LANES: usize, F: FnMut(usize) -> &'a EncSym>(
    out: &mut Vec<u8>, len: usize, scale_bits: u32, mut sym_enc: F,
) -> Result<()> {
    let mut states = [RANS_L; LANES];
    let (mut words, mut stage) = (Vec::with_capacity(len / 2 + 32), [0u16; MAX_STAGE_WORDS]);
    let (full_groups, rem) = (len / LANES, len % LANES);
    if rem > 0 {
        let (base, mut lp) = (full_groups * LANES, 0usize);
        for k in (0..rem).rev() {
            let enc = sym_enc(base + k);
            if enc.freq == 0 { return Err(Error::InvalidParameter); }
            states[k] = rans_enc_step_scalar(states[k], enc, scale_bits, &mut stage, &mut lp);
        }
        words.extend_from_slice(&stage[..lp]);
    }
    let mut g_end = full_groups;
    while g_end > 0 {
        let (g_start, mut lp) = (g_end.saturating_sub(BATCH_STEPS), 0usize);
        for g in (g_start..g_end).rev() {
            let base = g * LANES;
            for k in (0..LANES).rev() {
                let enc = sym_enc(base + k);
                if enc.freq == 0 { return Err(Error::InvalidParameter); }
                states[k] = rans_enc_step_scalar(states[k], enc, scale_bits, &mut stage, &mut lp);
            }
        }
        words.extend_from_slice(&stage[..lp]);
        g_end = g_start;
    }
    finish_shared_stream(out, &states, &words)
}

/// Monomorphized branchless `LANES`-way rANS decoder over one shared stream with per-index LUT.
#[inline(never)]
pub(crate) fn decode_lanes_by<'a, const LANES: usize, W, const N: usize, const BITS: u32, F>(
    mut lut_at: F, mut states: [u32; LANES], stream: &[u8], out: &mut [u8],
) -> Result<()>
where
    W: LutWord + 'a,
    F: FnMut(usize) -> &'a [W; N],
{
    let (mut pos, full_groups, mut g) = (0usize, out.len() / LANES, 0usize);
    while g + BATCH_STEPS <= full_groups && stream.len() - pos >= MAX_WIN_BYTES {
        let Some(win) = stream.get(pos..pos + MAX_WIN_BYTES)
            .and_then(|s| <&[u8; MAX_WIN_BYTES]>::try_from(s).ok())
        else { break; };
        let mut lp = 0usize;
        for step in 0..BATCH_STEPS {
            let (base, mut syms) = ((g + step) * LANES, [0u8; LANES]);
            for k in 0..LANES {
                let s = states[k];
                let (sym, freq, bias) = decode_packed_slot::<W, N, BITS>(lut_at(base + k), s);
                syms[k] = sym;
                states[k] = freq * (s >> BITS) + bias;
            }
            out[base..base + LANES].copy_from_slice(&syms);
            for state in &mut states {
                let off = lp & (MAX_WIN_BYTES - 2);
                let w = u16::from_le_bytes([win[off], win[off + 1]]);
                let (updated, adv) = rans_refill_branchless(*state, w);
                *state = updated;
                lp += adv;
            }
        }
        pos += lp;
        g += BATCH_STEPS;
    }
    let mut base = g * LANES;
    while base < out.len() {
        for k in 0..(out.len() - base).min(LANES) {
            let s = states[k];
            let (sym, freq, bias) = decode_packed_slot::<W, N, BITS>(lut_at(base + k), s);
            states[k] = refill_checked_u16(freq * (s >> BITS) + bias, stream, &mut pos)?;
            out[base + k] = sym;
        }
        base += LANES;
    }
    verify_terminal_shared(&states, stream.len(), pos)
}

fn encode_framed_lanes<const LANES: usize>(
    data: &[u8], scale: ScaleBits, external: Option<&RansModel>,
) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(data.len()).map_err(|_| Error::InvalidParameter)?;
    let meta = [(LANES as u8) | if external.is_some() { FLAG_EXTERNAL_MODEL } else { 0 }, scale as u8];
    let mut out = Vec::with_capacity(data.len() / 2 + if external.is_some() { 64 } else { 256 });
    write_framed_header(&mut out, &MAGIC_RANS, &meta, orig_len, data);
    if data.is_empty() { return Ok(out); }
    let owned;
    let model = match external {
        Some(ext) => ext,
        None => { owned = RansModel::from_bytes(data, scale)?; owned.write_bitmap(&mut out); &owned }
    };
    if model.single_symbol.is_some() {
        finish_shared_stream(&mut out, &[RANS_L; LANES], &[])?;
    } else {
        encode_lanes_by::<LANES, _>(&mut out, data.len(), model.scale.bits(), |i| {
            &model.enc[usize::from(data[i]) & 255]
        })?;
    }
    Ok(out)
}

fn decode_lanes_with_model<const LANES: usize>(
    reader: &mut Reader<'_>, meta: [u8; 2], model: &RansModel, orig_len: u32,
    expected_crc: u32, out: &mut Vec<u8>,
) -> Result<()> {
    let (states, stream) = parse_shared_stream::<LANES>(reader)?;
    out.clear();
    if let Some(only_sym) = model.single_symbol {
        verify_terminal_shared(&states, stream.len(), 0)?;
        out.resize(orig_len as usize, only_sym);
    } else {
        out.resize(orig_len as usize, 0u8);
        match &model.lut {
            DecodeLut::Lut12(t) => decode_lanes_by::<LANES, u32, 4096, 12, _>(|_| t, states, stream, out)?,
            DecodeLut::Lut14(t) => decode_lanes_by::<LANES, u64, 16384, 14, _>(|_| t, states, stream, out)?,
        }
    }
    if framed_header_crc(&MAGIC_RANS, &meta, orig_len, out) != expected_crc {
        out.clear();
        return Err(Error::ChecksumMismatch);
    }
    Ok(())
}

/// Encode `data` using 8-way interleaved rANS and a pre-agreed external [`RansModel`].
pub fn encode_8way_with_model(data: &[u8], model: &RansModel) -> Result<Vec<u8>> {
    encode_framed_lanes::<8>(data, model.scale, Some(model))
}
/// Encode `data` using 8-way interleaved SIMD-friendly rANS with 12-bit scale.
pub fn encode_8way(data: &[u8]) -> Result<Vec<u8>> { encode_framed_lanes::<8>(data, ScaleBits::Bits12, None) }
/// Encode `data` using 8-way interleaved rANS with caller-selected probability scale.
pub fn encode_8way_with_scale(data: &[u8], scale: ScaleBits) -> Result<Vec<u8>> {
    encode_framed_lanes::<8>(data, scale, None)
}
/// Encode `data` using 4-way interleaved SIMD-friendly rANS with 12-bit scale.
pub fn encode_4way(data: &[u8]) -> Result<Vec<u8>> { encode_framed_lanes::<4>(data, ScaleBits::Bits12, None) }
/// Encode `data` using 4-way interleaved rANS with caller-selected probability scale.
pub fn encode_4way_with_scale(data: &[u8], scale: ScaleBits) -> Result<Vec<u8>> {
    encode_framed_lanes::<4>(data, scale, None)
}
/// Decode an 8-way (or 4-way / order-1) `BRN1` stream into a new `Vec<u8>`.
pub fn decode_8way(input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decode_with_limit_into(input, DEFAULT_MAX_DECODE_BYTES, &mut out)?;
    Ok(out)
}
/// Decode a 4-way (or any valid `BRN1`) stream into a new `Vec<u8>`.
pub fn decode_4way(input: &[u8]) -> Result<Vec<u8>> { decode_8way(input) }

fn decode_brn1_inner(
    input: &[u8], external: Option<&RansModel>, max_out: usize, out: &mut Vec<u8>,
) -> Result<()> {
    let mut reader = Reader::new(input);
    expect_magic(&mut reader, &MAGIC_RANS)?;
    let (flags, scale) = (reader.read_u8()?, ScaleBits::from_u8(reader.read_u8()?)?);
    let (orig_u32, expected_crc) = (reader.read_u32_le()?, reader.read_u32_le()?);
    let orig_len = orig_u32 as usize;
    if orig_len > max_out { return Err(Error::OutputTooLarge); }
    let meta = [flags, scale as u8];

    if (flags & FLAG_EXTERNAL_MODEL) != 0 {
        let Some(model) = external else { return Err(Error::UnsupportedFormat); };
        if flags != (8u8 | FLAG_EXTERNAL_MODEL) || scale != model.scale {
            return Err(Error::UnsupportedFormat);
        }
        if orig_len == 0 {
            reader.finish()?;
            if framed_header_crc(&MAGIC_RANS, &meta, orig_u32, &[]) != expected_crc {
                return Err(Error::ChecksumMismatch);
            }
            out.clear();
            return Ok(());
        }
        return decode_lanes_with_model::<8>(&mut reader, meta, model, orig_u32, expected_crc, out);
    }

    let (lanes, is_o1, low15) = (flags & 0x0F, (flags & FLAG_ORDER1) != 0, (flags & FLAG_CTX_LOW15) != 0);
    let valid = (flags & !(0x0F | FLAG_ORDER1 | FLAG_CTX_LOW15)) == 0
        && (is_o1 || !low15)
        && if is_o1 { lanes == 8 && scale == ScaleBits::Bits12 } else { lanes == 4 || lanes == 8 };
    if !valid { return Err(Error::UnsupportedFormat); }
    if orig_len == 0 {
        reader.finish()?;
        if framed_header_crc(&MAGIC_RANS, &meta, orig_u32, &[]) != expected_crc {
            return Err(Error::ChecksumMismatch);
        }
        out.clear();
        return Ok(());
    }
    if is_o1 {
        let cmap = if low15 { ContextMap::Low15 } else { ContextMap::HighNibble };
        return decode_order1_8way_body(&mut reader, meta, cmap, orig_u32, expected_crc, out);
    }
    let model = RansModel::read_bitmap(&mut reader, scale)?;
    match lanes {
        4 => decode_lanes_with_model::<4>(&mut reader, meta, &model, orig_u32, expected_crc, out),
        8 => decode_lanes_with_model::<8>(&mut reader, meta, &model, orig_u32, expected_crc, out),
        _ => Err(Error::UnsupportedFormat),
    }
}

/// Decode a `BRN1` stream encoded with [`encode_8way_with_model`] (or inline `BRN1`).
pub fn decode_with_model_and_limit_into(
    input: &[u8], model: &RansModel, max_output_bytes: usize, out: &mut Vec<u8>,
) -> Result<()> {
    decode_brn1_inner(input, Some(model), max_output_bytes, out)
}

/// Decode a `BRN1` stream into a reusable `Vec<u8>` enforcing `max_output_bytes`.
pub fn decode_with_limit_into(input: &[u8], max_output_bytes: usize, out: &mut Vec<u8>) -> Result<()> {
    decode_brn1_inner(input, None, max_output_bytes, out)
}

/// Encode `data` using 8-way lane-parallel order-1 context-modeled rANS (16 contexts, 12-bit).
pub fn encode_order1_8way(data: &[u8], ctx_map: ContextMap) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(data.len()).map_err(|_| Error::InvalidParameter)?;
    let flags = 8u8 | FLAG_ORDER1 | if ctx_map == ContextMap::Low15 { FLAG_CTX_LOW15 } else { 0 };
    let meta = [flags, ScaleBits::Bits12 as u8];
    let mut out = Vec::with_capacity(data.len() / 2 + 1024);
    write_framed_header(&mut out, &MAGIC_RANS, &meta, orig_len, data);
    if data.is_empty() { return Ok(out); }

    let bounds = balanced_chunks::<8>(data.len());
    let mut counts = [[0u64; 256]; NUM_CONTEXTS];
    for &(start, len) in &bounds {
        let mut prev = 0u8;
        for &b in &data[start..start + len] {
            counts[ctx_map.bucket(prev)][usize::from(b) & 255] += 1;
            prev = b;
        }
    }
    let mut models: [Option<RansModel>; NUM_CONTEXTS] = core::array::from_fn(|_| None);
    let mut ctx_mask = 0u16;
    for ctx in 0..NUM_CONTEXTS {
        if counts[ctx].iter().any(|&c| c > 0) {
            ctx_mask |= 1u16 << ctx;
            models[ctx] = Some(RansModel::from_counts(&counts[ctx], ScaleBits::Bits12)?);
        }
    }
    out.extend_from_slice(&ctx_mask.to_le_bytes());
    for model in models.iter().flatten() { model.write_bitmap(&mut out); }

    let mut states = [RANS_L; 8];
    let (mut words, mut stage) = (Vec::with_capacity(data.len() / 2 + 32), [0u16; MAX_STAGE_WORDS]);
    let (min_len, rem) = (data.len() / 8, data.len() % 8);
    let step_at = |k: usize, j: usize, st: &mut [u32; 8],
        sg: &mut [u16; MAX_STAGE_WORDS], lp: &mut usize| -> Result<()> {
        let base = bounds[k].0;
        let prev = if j == 0 { 0u8 } else { data[base + j - 1] };
        let m = models[ctx_map.bucket(prev)].as_ref().ok_or(Error::InvalidParameter)?;
        st[k] = rans_enc_step_scalar(st[k], &m.enc[usize::from(data[base + j]) & 255], 12, sg, lp);
        Ok(())
    };
    if rem > 0 {
        let mut lp = 0usize;
        for k in (0..rem).rev() { step_at(k, min_len, &mut states, &mut stage, &mut lp)?; }
        words.extend_from_slice(&stage[..lp]);
    }
    let mut j_end = min_len;
    while j_end > 0 {
        let (j_start, mut lp) = (j_end.saturating_sub(BATCH_STEPS), 0usize);
        for j in (j_start..j_end).rev() {
            for k in (0..8usize).rev() { step_at(k, j, &mut states, &mut stage, &mut lp)?; }
        }
        words.extend_from_slice(&stage[..lp]);
        j_end = j_start;
    }
    finish_shared_stream(&mut out, &states, &words)?;
    Ok(out)
}

fn decode_order1_8way_body(
    reader: &mut Reader<'_>, meta: [u8; 2], ctx_map: ContextMap, orig_u32: u32,
    expected_crc: u32, out: &mut Vec<u8>,
) -> Result<()> {
    let orig_len = orig_u32 as usize;
    let mb = reader.read_bytes(2)?;
    let ctx_mask = u16::from_le_bytes([mb[0], mb[1]]);
    if ctx_mask == 0 { return Err(Error::CorruptProbabilityTable); }
    let mut luts: [Option<Box<[u32; 4096]>>; NUM_CONTEXTS] = core::array::from_fn(|_| None);
    for (ctx, slot) in luts.iter_mut().enumerate() {
        if (ctx_mask & (1u16 << ctx)) != 0 {
            if let DecodeLut::Lut12(t) = RansModel::read_bitmap(reader, ScaleBits::Bits12)?.lut {
                *slot = Some(t);
            }
        }
    }
    let (mut states, stream) = parse_shared_stream::<8>(reader)?;
    let bounds = balanced_chunks::<8>(orig_len);
    let (mut pos, mut prev, min_len, rem) = (0usize, [0u8; 8], orig_len / 8, orig_len % 8);
    out.clear();
    out.resize(orig_len, 0u8);
    for j in 0..=min_len {
        for k in 0..(if j < min_len { 8 } else { rem }) {
            let table = luts[ctx_map.bucket(prev[k])].as_ref().ok_or(Error::CorruptState)?;
            let s = states[k];
            let (sym, freq, bias) = decode_packed_slot::<u32, 4096, 12>(table, s);
            states[k] = refill_checked_u16(freq * (s >> 12) + bias, stream, &mut pos)?;
            out[bounds[k].0 + j] = sym;
            prev[k] = sym;
        }
    }
    verify_terminal_shared(&states, stream.len(), pos)?;
    if framed_header_crc(&MAGIC_RANS, &meta, orig_u32, out) != expected_crc {
        out.clear();
        return Err(Error::ChecksumMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn reciprocal_and_roundtrips() {
        for freq in [1u32, 2, 3, 5, 7, 17, 255, 256, 1023, 4095, 4096, 16383, 16384] {
            let enc = EncSym::from_freq_cum(freq, 0, 14);
            let probes = [0, 1, freq - 1, freq, freq + 1, RANS_L, RANS_L + freq, RANS_UPPER - 1];
            for x in probes { assert_eq!(((u64::from(x) * enc.rcp) >> enc.rcp_shift) as u32, x / freq); }
        }
        for &len in &[0usize, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 63, 64, 511, 512, 513, 4099] {
            let data: Vec<u8> = (0..len).map(|i| ((i * 37 + (i >> 3)) & 0x0F) as u8).collect();
            for sc in [ScaleBits::Bits12, ScaleBits::Bits14] {
                assert_eq!(decode_8way(&encode_8way_with_scale(&data, sc).expect("e8")).expect("d8"), data);
                assert_eq!(decode_4way(&encode_4way_with_scale(&data, sc).expect("e4")).expect("d4"), data);
            }
            for m in [ContextMap::HighNibble, ContextMap::Low15] {
                assert_eq!(decode_8way(&encode_order1_8way(&data, m).expect("o1")).expect("do1"), data);
            }
        }
        let cst = vec![0x42u8; 4096];
        let c8 = encode_8way(&cst).expect("enc const");
        assert!(c8.len() < 128 && decode_8way(&c8).expect("dec const") == cst);
    }
}
