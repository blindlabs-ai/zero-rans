//! Split-stream LZ77 + 8-way interleaved rANS compression (`BRL1`).

use alloc::vec;
use alloc::vec::Vec;

use crate::rans8::{decode_with_limit_into, encode_8way_with_scale, ScaleBits};
use crate::wire::{
    expect_magic, framed_header_crc, write_framed_header, write_substream, Error, Reader, Result,
    DEFAULT_MAX_DECODE_BYTES,
};

/// Framed split-stream LZ magic header (`BRL1`).
pub const MAGIC_LZ_SPLIT: [u8; 4] = *b"BRL1";

const MODE_STORED: u8 = 0;
const MODE_LZ_RANS8: u8 = 1;
const MIN_MATCH: usize = 4;
const MAX_OFFSET: usize = 65_535;
const HASH_SIZE: usize = 1 << 15;
const HASH_MASK: usize = HASH_SIZE - 1;
const WINDOW_RING: usize = 65_536;
const WINDOW_MASK: usize = WINDOW_RING - 1;
const NO_POS: u32 = u32::MAX;

/// Configuration for the split-stream LZ + rANS compressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LzConfig {
    #[doc = "Maximum hash-chain links inspected per position."] pub max_chain: usize,
    #[doc = "Check pos + 1 for a strictly longer match before emitting."] pub lazy_match: bool,
    #[doc = "Probability scale used for the 5 rans8 substreams."] pub scale: ScaleBits,
    #[doc = "Fall back to stored mode when LZ+rANS does not shrink payload."] pub allow_stored_fallback: bool,
}

impl Default for LzConfig {
    #[inline]
    fn default() -> Self {
        Self { max_chain: 12, lazy_match: true, scale: ScaleBits::Bits12, allow_stored_fallback: true }
    }
}

/// Knuth multiplicative 4-byte hash mapped into `0..HASH_SIZE`.
/// @domain v: 0..=4294967295
#[inline(always)]
const fn hash4_u32(v: u32) -> usize {
    (((v.wrapping_mul(2_654_435_761)) >> 17) & 0x7FFF) as usize
}

/// Saturating 4-bit LZ token nibble encoder (`0..=14` verbatim, `>= 15` -> `15`).
/// @domain len: 0..=65535
#[inline]
const fn token_nibble(len: usize) -> u8 { if len < 15 { len as u8 } else { 15 } }

#[inline]
fn hash4(data: &[u8], pos: usize) -> usize {
    let Some(w) = data.get(pos..pos + 4) else { return 0; };
    hash4_u32(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
}

/// Fast 8-byte-at-a-time match length computation (pure safe Rust).
#[inline]
pub(crate) fn match_len_at(data: &[u8], a: usize, b: usize, limit: usize) -> usize {
    let mut len = 0usize;
    while len + 8 <= limit {
        let (Some(sa), Some(sb)) = (data.get(a + len..a + len + 8), data.get(b + len..b + len + 8))
        else { break; };
        let va = u64::from_le_bytes([sa[0], sa[1], sa[2], sa[3], sa[4], sa[5], sa[6], sa[7]]);
        let vb = u64::from_le_bytes([sb[0], sb[1], sb[2], sb[3], sb[4], sb[5], sb[6], sb[7]]);
        let diff = va ^ vb;
        if diff != 0 { return len + (diff.trailing_zeros() as usize >> 3); }
        len += 8;
    }
    while len < limit && data[a + len] == data[b + len] { len += 1; }
    len
}

struct MatchFinder { head: Vec<u32>, prev: Vec<u32> }

impl MatchFinder {
    fn new() -> Self { Self { head: vec![NO_POS; HASH_SIZE], prev: vec![NO_POS; WINDOW_RING] } }

    #[inline]
    fn insert(&mut self, data: &[u8], pos: usize) -> u32 {
        let h = hash4(data, pos) & HASH_MASK;
        let old = self.head[h & HASH_MASK];
        self.head[h & HASH_MASK] = pos as u32;
        self.prev[pos & WINDOW_MASK] = old;
        old
    }

    fn find_best(&self, data: &[u8], pos: usize, mut cand: u32, max_chain: usize) -> (usize, usize) {
        let max_len = data.len() - pos;
        if max_len < MIN_MATCH { return (0, 0); }
        let (mut best_len, mut best_off, mut steps) = (MIN_MATCH - 1, 0usize, 0usize);
        while cand != NO_POS && steps < max_chain {
            let cpos = cand as usize;
            if pos <= cpos || pos - cpos > MAX_OFFSET { break; }
            if data[cpos + best_len] == data[pos + best_len]
                && data[cpos..cpos + 4] == data[pos..pos + 4]
            {
                let mlen = match_len_at(data, cpos, pos, max_len);
                if mlen > best_len {
                    (best_len, best_off) = (mlen, pos - cpos);
                    if best_len == max_len || best_len >= 258 { break; }
                }
            }
            cand = self.prev[cpos & WINDOW_MASK];
            steps += 1;
        }
        if best_len >= MIN_MATCH { (best_len, best_off) } else { (0, 0) }
    }
}

/// Append LZ4-style base-255 length extension bytes to `ext`.
#[inline]
pub(crate) fn push_ext_len(ext: &mut Vec<u8>, mut extra: usize) {
    while extra >= 255 { ext.push(255); extra -= 255; }
    ext.push(extra as u8);
}

/// Read LZ4-style base-255 length extension bytes with overflow and budget checks.
#[inline]
pub(crate) fn read_ext_len(ext: &[u8], pos: &mut usize, max_add: usize) -> Result<usize> {
    let mut total = 0usize;
    loop {
        let b = *ext.get(*pos).ok_or(Error::CorruptLzStream)?;
        *pos += 1;
        total = total.checked_add(usize::from(b)).ok_or(Error::CorruptLzStream)?;
        if total > max_add { return Err(Error::OutputTooLarge); }
        if b < 255 { return Ok(total); }
    }
}

fn push_packet(streams: &mut [Vec<u8>; 5], lits: &[u8], match_pair: Option<(usize, usize)>) {
    let lit_n = token_nibble(lits.len());
    let mat_n = match_pair.map_or(0u8, |(mlen, _)| token_nibble(mlen - (MIN_MATCH - 1)));
    streams[0].push((lit_n << 4) | mat_n);
    if lit_n == 15 { push_ext_len(&mut streams[4], lits.len() - 15); }
    streams[1].extend_from_slice(lits);
    if let Some((mlen, offset)) = match_pair {
        if mat_n == 15 { push_ext_len(&mut streams[4], mlen - (MIN_MATCH - 1 + 15)); }
        let off_m1 = (offset - 1) as u16;
        streams[2].push((off_m1 & 0xFF) as u8);
        streams[3].push((off_m1 >> 8) as u8);
    }
}

fn parse_lz_split(data: &[u8], cfg: LzConfig) -> [Vec<u8>; 5] {
    let n = data.len();
    let mut s = [
        Vec::with_capacity(n / 8 + 16), Vec::with_capacity(n / 2 + 16),
        Vec::with_capacity(n / 8 + 16), Vec::with_capacity(n / 8 + 16),
        Vec::with_capacity(n / 32 + 16),
    ];
    if n == 0 { return s; }
    let (max_chain, mut finder, mut pos, mut anchor) = (cfg.max_chain.max(1), MatchFinder::new(), 0, 0);
    if n >= MIN_MATCH {
        let limit = n - MIN_MATCH;
        while pos <= limit {
            let cand = finder.insert(data, pos);
            let (mut best_len, mut best_off) = finder.find_best(data, pos, cand, max_chain);
            if best_len >= MIN_MATCH {
                if cfg.lazy_match && pos < limit && best_len < 128 {
                    let nc = finder.insert(data, pos + 1);
                    let (nl, no) = finder.find_best(data, pos + 1, nc, max_chain);
                    if nl > best_len + 1 { pos += 1; best_len = nl; best_off = no; }
                }
                push_packet(&mut s, &data[anchor..pos], Some((best_len, best_off)));
                let end_match = pos + best_len;
                for cur in (pos + 1)..end_match.min(limit + 1) { let _ = finder.insert(data, cur); }
                pos = end_match;
                anchor = pos;
            } else {
                pos += 1;
            }
        }
    }
    if anchor < n { push_packet(&mut s, &data[anchor..], None); }
    s
}

/// Compress `data` using split-stream LZ77 + 8-way interleaved rANS.
pub fn compress_lz_split(data: &[u8]) -> Result<Vec<u8>> {
    compress_lz_split_with_config(data, LzConfig::default())
}

/// Compress `data` using split-stream LZ77 + 8-way interleaved rANS with custom [`LzConfig`].
pub fn compress_lz_split_with_config(data: &[u8], cfg: LzConfig) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(data.len()).map_err(|_| Error::InvalidParameter)?;
    if data.is_empty() {
        let mut out = Vec::with_capacity(13);
        write_framed_header(&mut out, &MAGIC_LZ_SPLIT, &[MODE_STORED], 0, &[]);
        return Ok(out);
    }
    let streams = parse_lz_split(data, cfg);
    let (mut comp, mut est) = (Vec::with_capacity(5), 32usize);
    for lane in &streams {
        let c = encode_8way_with_scale(lane, cfg.scale)?;
        est += c.len();
        comp.push(c);
    }
    if cfg.allow_stored_fallback && est >= 13 + data.len() {
        let mut out = Vec::with_capacity(13 + data.len());
        write_framed_header(&mut out, &MAGIC_LZ_SPLIT, &[MODE_STORED], orig_len, data);
        out.extend_from_slice(data);
        return Ok(out);
    }
    let mut out = Vec::with_capacity(est);
    write_framed_header(&mut out, &MAGIC_LZ_SPLIT, &[MODE_LZ_RANS8], orig_len, data);
    for c in &comp { write_substream(&mut out, c)?; }
    Ok(out)
}

/// Copy `match_len` bytes from `out[out.len() - offset..]` supporting overlapping periodic matches.
#[inline]
pub(crate) fn copy_lz_match(out: &mut Vec<u8>, offset: usize, match_len: usize) -> Result<()> {
    if offset == 0 || offset > out.len() { return Err(Error::CorruptLzStream); }
    if offset >= match_len {
        let start = out.len() - offset;
        out.extend_from_within(start..start + match_len);
    } else if offset == 1 {
        let b = out[out.len() - 1];
        out.resize(out.len() + match_len, b);
    } else {
        let (mut rem, mut span) = (match_len, offset);
        while rem > 0 {
            let step = rem.min(span);
            let start = out.len() - span;
            out.extend_from_within(start..start + step);
            rem -= step;
            span += step;
        }
    }
    Ok(())
}

#[inline]
fn decode_nibble_len(nibble: usize, base: usize, budget: usize, ext: &[u8], ep: &mut usize) -> Result<usize> {
    let mut len = nibble + base;
    if nibble == 15 {
        let extra = read_ext_len(ext, ep, budget.saturating_sub(15 + base))?;
        len = len.checked_add(extra).ok_or(Error::OutputTooLarge)?;
    }
    if len > budget { Err(Error::OutputTooLarge) } else { Ok(len) }
}

/// Decompress a `BRL1` split-stream LZ + 8-way rANS frame.
pub fn decompress_lz_split(input: &[u8]) -> Result<Vec<u8>> {
    decompress_lz_split_with_limit(input, DEFAULT_MAX_DECODE_BYTES)
}

/// Decompress a `BRL1` frame enforcing a caller-specified maximum output length.
pub fn decompress_lz_split_with_limit(input: &[u8], max_output_bytes: usize) -> Result<Vec<u8>> {
    let mut reader = Reader::new(input);
    expect_magic(&mut reader, &MAGIC_LZ_SPLIT)?;
    let (mode, orig_u32, expected_crc) = (reader.read_u8()?, reader.read_u32_le()?, reader.read_u32_le()?);
    let orig_len = orig_u32 as usize;
    if orig_len > max_output_bytes { return Err(Error::OutputTooLarge); }
    let meta = [mode];
    if orig_len == 0 || mode == MODE_STORED {
        if orig_len == 0 && mode != MODE_STORED { return Err(Error::UnsupportedFormat); }
        let raw = if orig_len == 0 { &[][..] } else { reader.read_bytes(orig_len)? };
        reader.finish()?;
        if framed_header_crc(&MAGIC_LZ_SPLIT, &meta, orig_u32, raw) != expected_crc {
            return Err(Error::ChecksumMismatch);
        }
        return Ok(raw.to_vec());
    }
    if mode != MODE_LZ_RANS8 { return Err(Error::UnsupportedFormat); }
    let mut bufs: [Vec<u8>; 5] = Default::default();
    for buf in &mut bufs { decode_with_limit_into(reader.read_substream()?, orig_len, buf)?; }
    reader.finish()?;
    let [tokens, literals, offset_lo, offset_hi, ext_lens] = bufs;
    if offset_lo.len() != offset_hi.len() { return Err(Error::CorruptLzStream); }

    let (mut out, mut lit_p, mut off_p, mut ext_p) = (Vec::with_capacity(orig_len), 0usize, 0usize, 0usize);
    for &tok in &tokens {
        let bud = orig_len - out.len();
        let lit_len = decode_nibble_len(usize::from(tok >> 4), 0, bud, &ext_lens, &mut ext_p)?;
        let lit_end = lit_p.checked_add(lit_len).ok_or(Error::CorruptLzStream)?;
        out.extend_from_slice(literals.get(lit_p..lit_end).ok_or(Error::CorruptLzStream)?);
        lit_p = lit_end;
        let mn = usize::from(tok & 0x0F);
        if mn > 0 {
            let mlen = decode_nibble_len(mn, MIN_MATCH - 1, orig_len - out.len(), &ext_lens, &mut ext_p)?;
            let lo = *offset_lo.get(off_p).ok_or(Error::CorruptLzStream)?;
            let hi = *offset_hi.get(off_p).ok_or(Error::CorruptLzStream)?;
            off_p += 1;
            copy_lz_match(&mut out, ((usize::from(hi) << 8) | usize::from(lo)) + 1, mlen)?;
        }
    }
    let ok = out.len() == orig_len && lit_p == literals.len()
        && off_p == offset_lo.len() && ext_p == ext_lens.len();
    if !ok { return Err(Error::CorruptLzStream); }
    if framed_header_crc(&MAGIC_LZ_SPLIT, &meta, orig_u32, &out) != expected_crc {
        return Err(Error::ChecksumMismatch);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz_split_roundtrip_repetitive_overlapping_and_short() {
        let mut data = Vec::new();
        for i in 0..400usize {
            data.extend_from_slice(b"ABCDABCDABCD_frame_delta_00000000_");
            data.push((i & 0xFF) as u8);
        }
        data.extend(core::iter::repeat(0x7Au8).take(512));
        for _ in 0..200 { data.extend_from_slice(b"XYZ"); }
        let comp = compress_lz_split(&data).expect("compress");
        assert!(comp.len() * 4 < data.len());
        assert_eq!(decompress_lz_split(&comp).expect("decompress"), data);
        for len in 0..=20usize {
            let s: Vec<u8> = (0..len).map(|i| (i * 13) as u8).collect();
            assert_eq!(decompress_lz_split(&compress_lz_split(&s).expect("c")).expect("d"), s);
        }
    }
}
