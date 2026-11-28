//! Socket-plugged MDL-clustered 8-way interleaved rANS (`BRS1`), exact base-`S`
//! mixed-radix span packing (`RadixSpan`), hybrid unsigned integer coding (`BRH1`),
//! and block-agnostic sparse run-level coding (`BRR1`).

use alloc::vec;
use alloc::vec::Vec;

use crate::rans8::{
    decode_lanes_by, decode_with_limit_into, encode_8way_with_scale, encode_lanes_by,
    encode_order1_8way, parse_shared_stream, read_checked_freq, ContextMap, RansModel, ScaleBits,
};
use crate::residual::{zigzag_decode_u16, zigzag_encode_i16};
use crate::wire::{
    crc32_update, expect_magic, framed_header_crc, write_framed_header, write_substream,
    write_varint_u32, Error, Reader, Result, DEFAULT_MAX_DECODE_BYTES,
};

/// Framed socket-plugged rANS magic header (`BRS1`).
pub const MAGIC_SOCKET_RANS: [u8; 4] = *b"BRS1";
/// Framed socket-plugged hybrid unsigned integer magic header (`BRH1`).
pub const MAGIC_HYBRID_UINT: [u8; 4] = *b"BRH1";
/// Framed block-agnostic sparse run-level magic header (`BRR1`).
pub const MAGIC_SPARSE_RUNS: [u8; 4] = *b"BRR1";
/// Maximum active rANS probability clusters emitted in a single `BRS1` frame.
pub const MAX_SOCKET_CLUSTERS: usize = 32;

const BRS1_MODE_RAW: u8 = 0;
const BRS1_MODE_CONST: u8 = 1;
const BRS1_MODE_RADIX_SPAN: u8 = 2;
const BRS1_MODE_RANS8_SINGLE: u8 = 3;
const BRS1_MODE_RANS8_ORDER1_HI: u8 = 4;
const BRS1_MODE_RANS8_ORDER1_LOW15: u8 = 5;
const BRS1_MODE_SOCKET_CLUSTERED: u8 = 6;
const MODEL_TAG_SINGLE: u8 = 0;
const MODEL_TAG_SPARSE: u8 = 1;
const MODEL_TAG_BITMAP: u8 = 2;
const MAX_SPARSE_MODEL_SYMS: usize = 32;

/// Summary of MDL socket clustering produced by [`ClusteredSocketRans::fit`].
#[derive(Debug, Clone)]
pub struct ClusteredSocketRans {
    #[doc = "Number of caller-declared input sockets (1..=256)."] pub n_sockets: usize,
    #[doc = "Number of merged active clusters (1..=32)."] pub n_clusters: usize,
    #[doc = "Probability precision used for all cluster tables."] pub scale: ScaleBits,
    #[doc = "Mapping from socket index 0..256 to cluster index."] pub socket_to_cluster: [u8; 256],
    #[doc = "Normalized rANS models for each active cluster."] pub models: Vec<RansModel>,
}

/// Bijective signed-to-unsigned 32-bit zigzag mapping (`0 -> 0, -1 -> 1, 1 -> 2, ...`).
/// @domain v: -2147483648..=2147483647
#[inline] #[must_use]
pub const fn zigzag_encode_i32(v: i32) -> u32 { ((v << 1) ^ (v >> 31)) as u32 }

/// Inverse of [`zigzag_encode_i32`], exact across all `2^32` `u32` values.
/// @domain z: 0..=4294967295
#[inline] #[must_use]
pub const fn zigzag_decode_u32(z: u32) -> i32 { ((z >> 1) as i32) ^ -((z & 1) as i32) }

/// Pack a bounded-span byte slice (`span = max - min + 1 <= 128`) using exact base-`S` limbs.
#[must_use]
pub fn pack_radix_span(syms: &[u8]) -> Option<Vec<u8>> {
    let orig_len = u32::try_from(syms.len()).ok()?;
    if syms.is_empty() { return Some(vec![0u8; 3]); }
    let (mut min_v, mut max_v) = (255u8, 0u8);
    for &b in syms { min_v = min_v.min(b); max_v = max_v.max(b); }
    let span = usize::from(max_v - min_v) + 1;
    if span > 128 { return None; }
    let mut out = Vec::with_capacity(syms.len() + 8);
    out.extend_from_slice(&[min_v, (span - 1) as u8]);
    write_varint_u32(&mut out, orig_len);
    if span == 1 { return Some(out); }

    let (s, full_limbs, rem) = (span as u64, syms.len() / 8, syms.len() % 8);
    let b_radix = s.pow(8);
    let mut tail_limb = 0u64;
    for k in (0..rem).rev() { tail_limb = tail_limb * s + u64::from(syms[full_limbs * 8 + k] - min_v); }
    let (mut x, mut rev) = (b_radix + tail_limb, Vec::with_capacity(syms.len() + 8));
    for chunk in (0..full_limbs).rev() {
        let mut limb = 0u64;
        for &b in syms[chunk * 8..chunk * 8 + 8].iter().rev() { limb = limb * s + u64::from(b - min_v); }
        while x >= 256 { rev.push((x & 0xFF) as u8); x >>= 8; }
        x = x * b_radix + limb;
    }
    while x > 0 { rev.push((x & 0xFF) as u8); x >>= 8; }
    out.extend(rev.iter().rev().copied());
    Some(out)
}

/// Unpack a [`pack_radix_span`] wire buffer with fail-closed validation and `max_out` limit.
pub fn unpack_radix_span(wire: &[u8], max_out: usize) -> Result<Vec<u8>> {
    let mut r = Reader::new(wire);
    let (min_v, span_m1, orig_len) = (r.read_u8()?, r.read_u8()?, r.read_varint_u32()? as usize);
    if orig_len > max_out { return Err(Error::OutputTooLarge); }
    if span_m1 >= 128 || u16::from(min_v) + u16::from(span_m1) > 255 { return Err(Error::CorruptState); }
    let span = usize::from(span_m1) + 1;
    if orig_len == 0 || span == 1 {
        r.finish()?;
        return Ok(if orig_len == 0 { Vec::new() } else { vec![min_v; orig_len] });
    }
    let payload = r.read_rest();
    if payload.is_empty() || payload[0] == 0 { return Err(Error::CorruptState); }
    let (s, full_limbs, rem) = (span as u64, orig_len / 8, orig_len % 8);
    let (b_radix, mut pos, mut x) = (s.pow(8), 0usize, 0u64);
    let refill = |x: &mut u64, pos: &mut usize| -> Result<()> {
        while *x < b_radix {
            *x = (*x << 8) | u64::from(*payload.get(*pos).ok_or(Error::TruncatedStream)?);
            *pos += 1;
        }
        Ok(())
    };
    refill(&mut x, &mut pos)?;
    let mut out = Vec::with_capacity(orig_len);
    for _ in 0..full_limbs {
        let mut limb = x % b_radix;
        x /= b_radix;
        for _ in 0..8usize { out.push(min_v + (limb % s) as u8); limb /= s; }
        refill(&mut x, &mut pos)?;
    }
    if pos != payload.len() || x < b_radix { return Err(Error::CorruptState); }
    let mut tail = x - b_radix;
    if tail >= s.pow(rem as u32) { return Err(Error::CorruptState); }
    for _ in 0..rem { out.push(min_v + (tail % s) as u8); tail /= s; }
    Ok(out)
}

/// Compactly serialize a normalized [`RansModel`] (single-symbol, sparse, or 32 B bitmap).
pub(crate) fn write_compact_model(model: &RansModel, out: &mut Vec<u8>) {
    let freqs = model.freqs();
    let (mut nnz, mut last_sym) = (0usize, 0u8);
    for (sym, &f) in freqs.iter().enumerate() { if f > 0 { nnz += 1; last_sym = sym as u8; } }
    if nnz == 1 {
        out.extend_from_slice(&[MODEL_TAG_SINGLE, last_sym]);
    } else if nnz <= MAX_SPARSE_MODEL_SYMS {
        out.extend_from_slice(&[MODEL_TAG_SPARSE, (nnz - 1) as u8]);
        for (sym, &f) in freqs.iter().enumerate() {
            if f > 0 { out.push(sym as u8); write_varint_u32(out, u32::from(f - 1)); }
        }
    } else {
        out.push(MODEL_TAG_BITMAP);
        model.write_bitmap(out);
    }
}

/// Deserialize and validate a compact [`RansModel`] emitted by [`write_compact_model`].
pub(crate) fn read_compact_model(reader: &mut Reader<'_>, scale: ScaleBits) -> Result<RansModel> {
    let (tag, total, mut freqs) = (reader.read_u8()?, scale.total(), [0u16; 256]);
    match tag {
        MODEL_TAG_SINGLE => {
            freqs[usize::from(reader.read_u8()?) & 255] = total as u16;
            RansModel::from_normalized(&freqs, scale)
        }
        MODEL_TAG_SPARSE => {
            let count = usize::from(reader.read_u8()?) + 1;
            if count > MAX_SPARSE_MODEL_SYMS { return Err(Error::CorruptProbabilityTable); }
            let (mut sum, mut prev) = (0u32, None::<u8>);
            for _ in 0..count {
                let sym = reader.read_u8()?;
                if prev.map_or(false, |p| sym <= p) { return Err(Error::CorruptProbabilityTable); }
                prev = Some(sym);
                freqs[usize::from(sym) & 255] = read_checked_freq(reader, total, &mut sum)?;
            }
            if sum != total { return Err(Error::CorruptProbabilityTable); }
            RansModel::from_normalized(&freqs, scale)
        }
        MODEL_TAG_BITMAP => RansModel::read_bitmap(reader, scale),
        _ => Err(Error::CorruptProbabilityTable),
    }
}

struct HistCluster {
    counts: [u64; 256],
    nz_syms: Vec<u8>,
    total: u64,
    entropy_bits: f64,
    members: Vec<usize>,
}

#[inline]
fn x_log2_x(x: u64) -> f64 { if x <= 1 { 0.0 } else { let f = x as f64; f * crate::wire::log2_f64(f) } }

#[inline]
fn calc_entropy_bits(counts: &[u64; 256], nz: &[u8], total: u64) -> f64 {
    if total <= 1 { return 0.0; }
    let sum_cl: f64 = nz.iter().map(|&s| x_log2_x(counts[usize::from(s) & 255])).sum();
    (x_log2_x(total) - sum_cl).max(0.0)
}

#[inline]
fn est_table_header_bits(nnz: usize) -> f64 {
    if nnz <= 1 { 16.0 }
    else if nnz <= MAX_SPARSE_MODEL_SYMS { (2.0 + 2.35 * (nnz as f64)) * 8.0 }
    else { (33.0 + 1.45 * (nnz as f64)) * 8.0 }
}

impl HistCluster {
    fn from_counts(counts: [u64; 256], member: usize) -> Self {
        let (mut total, mut nz_syms) = (0u64, Vec::with_capacity(32));
        for (sym, &c) in counts.iter().enumerate() {
            if c > 0 { total += c; nz_syms.push(sym as u8); }
        }
        let entropy_bits = calc_entropy_bits(&counts, &nz_syms, total);
        Self { counts, nz_syms, total, entropy_bits, members: vec![member] }
    }

    fn merge_from(&mut self, other: &Self) {
        for &sym in &other.nz_syms {
            let idx = usize::from(sym) & 255;
            if self.counts[idx] == 0 { self.nz_syms.push(sym); }
            self.counts[idx] += other.counts[idx];
        }
        self.nz_syms.sort_unstable();
        self.total += other.total;
        self.entropy_bits = calc_entropy_bits(&self.counts, &self.nz_syms, self.total);
        self.members.extend_from_slice(&other.members);
    }
}

fn pair_mdl_delta(a: &HistCluster, b: &HistCluster) -> f64 {
    let (mut sum_cl, mut nnz_u, mut ia, mut ib) = (0.0f64, 0usize, 0usize, 0usize);
    while ia < a.nz_syms.len() || ib < b.nz_syms.len() {
        let c = match (a.nz_syms.get(ia), b.nz_syms.get(ib)) {
            (Some(&sa), Some(&sb)) if sa == sb => {
                ia += 1; ib += 1;
                a.counts[usize::from(sa) & 255] + b.counts[usize::from(sb) & 255]
            }
            (Some(&sa), Some(&sb)) if sa < sb => { ia += 1; a.counts[usize::from(sa) & 255] }
            (Some(&sa), None) => { ia += 1; a.counts[usize::from(sa) & 255] }
            (_, Some(&sb)) => { ib += 1; b.counts[usize::from(sb) & 255] }
            (None, None) => break,
        };
        sum_cl += x_log2_x(c);
        nnz_u += 1;
    }
    let merged_h = (x_log2_x(a.total + b.total) - sum_cl).max(0.0);
    let jsd = (merged_h - a.entropy_bits - b.entropy_bits).max(0.0);
    let saved = est_table_header_bits(a.nz_syms.len())
        + est_table_header_bits(b.nz_syms.len())
        - est_table_header_bits(nnz_u);
    jsd - saved
}

impl ClusteredSocketRans {
    /// Fit an MDL-contracted `M* <= 32` cluster model set from `(syms, sockets)` (`n_sockets <= 256`).
    pub fn fit(syms: &[u8], sockets: &[u8], n_sockets: usize, scale: ScaleBits) -> Result<Self> {
        if syms.is_empty() || syms.len() != sockets.len() || !(1..=256).contains(&n_sockets) {
            return Err(Error::InvalidParameter);
        }
        let (mut hists, mut totals) = (vec![[0u64; 256]; n_sockets], vec![0u64; n_sockets]);
        for (&sym, &sock) in syms.iter().zip(sockets.iter()) {
            let s = usize::from(sock);
            if s >= n_sockets { return Err(Error::InvalidParameter); }
            hists[s][usize::from(sym) & 255] += 1;
            totals[s] += 1;
        }
        let mut active: Vec<HistCluster> = (0..n_sockets)
            .filter(|&s| totals[s] > 0)
            .map(|s| HistCluster::from_counts(hists[s], s))
            .collect();
        const PRE_CAP: usize = 64;
        if active.len() > PRE_CAP {
            active.sort_unstable_by(|a, b| b.total.cmp(&a.total));
            for item in active.split_off(PRE_CAP) {
                let (mut bk, mut bd) = (0usize, f64::INFINITY);
                for (k, seed) in active.iter().enumerate() {
                    let d = pair_mdl_delta(seed, &item);
                    if d < bd { bd = d; bk = k; }
                }
                active[bk].merge_from(&item);
            }
        }
        let k_init = active.len();
        let mut dist = vec![vec![f64::INFINITY; k_init]; k_init];
        for i in 0..k_init {
            for j in (i + 1)..k_init {
                let d = pair_mdl_delta(&active[i], &active[j]);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        while active.len() > 1 {
            let (m, mut bi, mut bj, mut bnet) = (active.len(), 0usize, 1usize, f64::INFINITY);
            for (i, row) in dist.iter().enumerate().take(m) {
                for (j, &d) in row.iter().enumerate().take(m).skip(i + 1) {
                    if d < bnet { (bnet, bi, bj) = (d, i, j); }
                }
            }
            if m <= MAX_SOCKET_CLUSTERS && bnet >= 0.0 { break; }
            let merged = active.swap_remove(bj);
            let last = active.len();
            active[bi].merge_from(&merged);
            if bj < last {
                dist.swap(bj, last);
                for row in &mut dist[..last] { row[bj] = row[last]; }
            }
            dist.pop();
            for k in 0..active.len() {
                if k != bi {
                    let d = pair_mdl_delta(&active[bi], &active[k]);
                    dist[bi][k] = d;
                    dist[k][bi] = d;
                }
            }
        }
        let (mut map, mut models) = ([0u8; 256], Vec::with_capacity(active.len()));
        for (cid, cluster) in active.iter().enumerate() {
            for &s in &cluster.members { map[s & 255] = cid as u8; }
            models.push(RansModel::from_counts(&cluster.counts, scale)?);
        }
        Ok(Self { n_sockets, n_clusters: active.len(), scale, socket_to_cluster: map, models })
    }
}

/// Serialize `socket_to_cluster[0..n_sockets]` compactly.
pub(crate) fn write_socket_map(out: &mut Vec<u8>, n_sock: usize, n_cl: usize, map: &[u8; 256]) {
    out.extend_from_slice(&[(n_sock - 1) as u8, (n_cl - 1) as u8]);
    if n_cl == 1 { return; }
    if n_cl <= 16 {
        for pair in map[..n_sock].chunks(2) {
            out.push((pair[0] & 0x0F) | ((pair.get(1).copied().unwrap_or(0) & 0x0F) << 4));
        }
    } else {
        out.extend_from_slice(&map[..n_sock]);
    }
}

/// Deserialize and validate `socket_to_cluster[0..n_sockets]`.
pub(crate) fn read_socket_map(reader: &mut Reader<'_>) -> Result<(usize, usize, [u8; 256])> {
    let (n_sock, n_cl) = (usize::from(reader.read_u8()?) + 1, usize::from(reader.read_u8()?) + 1);
    if n_cl > MAX_SOCKET_CLUSTERS { return Err(Error::CorruptProbabilityTable); }
    let mut map = [0u8; 256];
    if n_cl == 1 { return Ok((n_sock, n_cl, map)); }
    if n_cl <= 16 {
        let packed_len = n_sock.div_ceil(2);
        let packed = reader.read_bytes(packed_len)?;
        for s in 0..n_sock {
            let b = packed[s >> 1];
            let c = if (s & 1) == 0 { b & 0x0F } else { b >> 4 };
            if usize::from(c) >= n_cl { return Err(Error::CorruptProbabilityTable); }
            map[s & 255] = c;
        }
        if (n_sock & 1) == 1 && (packed[packed_len - 1] >> 4) != 0 {
            return Err(Error::CorruptProbabilityTable);
        }
    } else {
        for (s, &c) in reader.read_bytes(n_sock)?.iter().enumerate() {
            if usize::from(c) >= n_cl { return Err(Error::CorruptProbabilityTable); }
            map[s & 255] = c;
        }
    }
    Ok((n_sock, n_cl, map))
}

/// Encode `syms` under caller-supplied causal `sockets` (`n_sockets <= 256`) into `BRS1`.
pub fn encode_socket_8way(
    syms: &[u8], sockets: &[u8], n_sockets: usize, scale: ScaleBits,
) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(syms.len()).map_err(|_| Error::InvalidParameter)?;
    if syms.len() != sockets.len() || !(1..=256).contains(&n_sockets) {
        return Err(Error::InvalidParameter);
    }
    if syms.is_empty() {
        let mut out = Vec::with_capacity(14);
        write_framed_header(&mut out, &MAGIC_SOCKET_RANS, &[BRS1_MODE_RAW, 0], 0, &[]);
        return Ok(out);
    }
    let cl = ClusteredSocketRans::fit(syms, sockets, n_sockets, scale)?;
    let mut out = Vec::with_capacity(syms.len() / 2 + 256);
    let meta = [BRS1_MODE_SOCKET_CLUSTERED, scale as u8];
    write_framed_header(&mut out, &MAGIC_SOCKET_RANS, &meta, orig_len, syms);
    write_socket_map(&mut out, cl.n_sockets, cl.n_clusters, &cl.socket_to_cluster);
    for model in &cl.models { write_compact_model(model, &mut out); }
    encode_lanes_by::<8, _>(&mut out, syms.len(), cl.scale.bits(), |idx| {
        let cid = usize::from(cl.socket_to_cluster[usize::from(sockets[idx]) & 255]);
        &cl.models[cid % cl.models.len()].enc_table()[usize::from(syms[idx]) & 255]
    })?;
    Ok(out)
}

/// Decode a `BRS1` framed stream with caller-supplied `sockets` and explicit output limit.
pub fn decode_socket_8way_with_limit(wire: &[u8], sockets: &[u8], max_out: usize) -> Result<Vec<u8>> {
    decode_substream_socket_auto(wire, Some(sockets), max_out)
}

/// Decode a `BRS1` framed stream using [`DEFAULT_MAX_DECODE_BYTES`].
pub fn decode_socket_8way(wire: &[u8], sockets: &[u8]) -> Result<Vec<u8>> {
    decode_socket_8way_with_limit(wire, sockets, DEFAULT_MAX_DECODE_BYTES)
}

fn try_brs1(best: &mut Vec<u8>, meta: [u8; 2], orig_len: u32, syms: &[u8], payload: &[u8]) {
    if 14 + payload.len() < best.len() {
        let mut cand = Vec::with_capacity(14 + payload.len());
        write_framed_header(&mut cand, &MAGIC_SOCKET_RANS, &meta, orig_len, syms);
        cand.extend_from_slice(payload);
        *best = cand;
    }
}

/// Auto-compete `Const`, `Raw`, `RadixSpan`, `Rans8`, `Order1`, and `ClusteredSocket8Way` into `BRS1`.
pub fn encode_substream_socket_auto(
    syms: &[u8], sockets: Option<&[u8]>, n_sockets: usize, scale: ScaleBits,
) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(syms.len()).map_err(|_| Error::InvalidParameter)?;
    if syms.is_empty() || syms.iter().all(|&b| b == syms[0]) {
        let mut out = Vec::with_capacity(14);
        let meta = if syms.is_empty() { [BRS1_MODE_RAW, 0] } else { [BRS1_MODE_CONST, syms[0]] };
        write_framed_header(&mut out, &MAGIC_SOCKET_RANS, &meta, orig_len, syms);
        return Ok(out);
    }
    let mut best = Vec::with_capacity(14 + syms.len());
    write_framed_header(&mut best, &MAGIC_SOCKET_RANS, &[BRS1_MODE_RAW, 0], orig_len, syms);
    best.extend_from_slice(syms);

    if let Some(radix) = pack_radix_span(syms) {
        try_brs1(&mut best, [BRS1_MODE_RADIX_SPAN, 0], orig_len, syms, &radix);
    }
    if syms.len() >= 16 {
        let r8 = encode_8way_with_scale(syms, scale)?;
        try_brs1(&mut best, [BRS1_MODE_RANS8_SINGLE, scale as u8], orig_len, syms, &r8);
    }
    if syms.len() >= 128 {
        let b12 = ScaleBits::Bits12 as u8;
        let hi = encode_order1_8way(syms, ContextMap::HighNibble)?;
        try_brs1(&mut best, [BRS1_MODE_RANS8_ORDER1_HI, b12], orig_len, syms, &hi);
        let lo = encode_order1_8way(syms, ContextMap::Low15)?;
        try_brs1(&mut best, [BRS1_MODE_RANS8_ORDER1_LOW15, b12], orig_len, syms, &lo);
    }
    if let Some(sock) = sockets {
        if !(1..=256).contains(&n_sockets) || sock.len() != syms.len() {
            return Err(Error::InvalidParameter);
        }
        let cs = encode_socket_8way(syms, sock, n_sockets, scale)?;
        if cs.len() < best.len() { best = cs; }
    }
    Ok(best)
}

/// Decode any `BRS1` frame produced by [`encode_substream_socket_auto`] or [`encode_socket_8way`].
pub fn decode_substream_socket_auto(wire: &[u8], sockets: Option<&[u8]>, max_out: usize) -> Result<Vec<u8>> {
    let mut reader = Reader::new(wire);
    expect_magic(&mut reader, &MAGIC_SOCKET_RANS)?;
    let (mode, aux) = (reader.read_u8()?, reader.read_u8()?);
    let (orig_u32, expected_crc) = (reader.read_u32_le()?, reader.read_u32_le()?);
    let orig_len = orig_u32 as usize;
    if orig_len > max_out { return Err(Error::OutputTooLarge); }
    let meta = [mode, aux];
    if orig_len == 0 {
        if mode != BRS1_MODE_RAW || aux != 0 { return Err(Error::UnsupportedFormat); }
        reader.finish()?;
        if framed_header_crc(&MAGIC_SOCKET_RANS, &meta, orig_u32, &[]) != expected_crc {
            return Err(Error::ChecksumMismatch);
        }
        return Ok(Vec::new());
    }
    let out = match mode {
        BRS1_MODE_RAW => {
            if aux != 0 { return Err(Error::UnsupportedFormat); }
            let raw = reader.read_bytes(orig_len)?;
            reader.finish()?;
            raw.to_vec()
        }
        BRS1_MODE_CONST => { reader.finish()?; vec![aux; orig_len] }
        BRS1_MODE_RADIX_SPAN => {
            if aux != 0 { return Err(Error::UnsupportedFormat); }
            let dec = unpack_radix_span(reader.read_rest(), orig_len)?;
            if dec.len() != orig_len { return Err(Error::CorruptState); }
            dec
        }
        BRS1_MODE_RANS8_SINGLE | BRS1_MODE_RANS8_ORDER1_HI | BRS1_MODE_RANS8_ORDER1_LOW15 => {
            let scale = ScaleBits::from_u8(aux)?;
            if mode != BRS1_MODE_RANS8_SINGLE && scale != ScaleBits::Bits12 {
                return Err(Error::UnsupportedFormat);
            }
            let payload = reader.read_rest();
            if payload.len() < 6 || payload[5] != scale as u8 { return Err(Error::UnsupportedFormat); }
            let mut dec = Vec::new();
            decode_with_limit_into(payload, orig_len, &mut dec)?;
            if dec.len() != orig_len { return Err(Error::CorruptState); }
            dec
        }
        BRS1_MODE_SOCKET_CLUSTERED => {
            let scale = ScaleBits::from_u8(aux)?;
            let sock = sockets.ok_or(Error::InvalidParameter)?;
            if sock.len() != orig_len { return Err(Error::InvalidParameter); }
            let (n_sock, n_cl, map) = read_socket_map(&mut reader)?;
            if sock.iter().any(|&s| usize::from(s) >= n_sock) { return Err(Error::InvalidParameter); }
            let mut models = Vec::with_capacity(n_cl);
            for _ in 0..n_cl { models.push(read_compact_model(&mut reader, scale)?); }
            let (states, stream) = parse_shared_stream::<8>(&mut reader)?;
            let mut dec = vec![0u8; orig_len];
            match scale {
                ScaleBits::Bits12 => {
                    let mut luts = Vec::with_capacity(n_cl);
                    for m in &models { luts.push(m.lut_12().ok_or(Error::CorruptProbabilityTable)?); }
                    let n_l = luts.len();
                    decode_lanes_by::<8, u32, 4096, 12, _>(
                        |i| luts[usize::from(map[usize::from(sock[i]) & 255]) % n_l],
                        states, stream, &mut dec,
                    )?;
                }
                ScaleBits::Bits14 => {
                    let mut luts = Vec::with_capacity(n_cl);
                    for m in &models { luts.push(m.lut_14().ok_or(Error::CorruptProbabilityTable)?); }
                    let n_l = luts.len();
                    decode_lanes_by::<8, u64, 16384, 14, _>(
                        |i| luts[usize::from(map[usize::from(sock[i]) & 255]) % n_l],
                        states, stream, &mut dec,
                    )?;
                }
            }
            dec
        }
        _ => return Err(Error::UnsupportedFormat),
    };
    if framed_header_crc(&MAGIC_SOCKET_RANS, &meta, orig_u32, &out) != expected_crc {
        return Err(Error::ChecksumMismatch);
    }
    Ok(out)
}

#[inline]
const fn hybrid_uint_split(v: u32) -> (u8, u32, u32) {
    if v < 16 { return (v as u8, 0, 0); }
    let msb = 31 - v.leading_zeros();
    let nbits = msb - 3;
    ((16 + ((msb - 4) << 3) + ((v >> nbits) & 7)) as u8, v & ((1u32 << nbits) - 1), nbits)
}

fn crc_hybrid_u32(orig_len: u32, vals: &[u32]) -> u32 {
    let mut s = crc32_update(crc32_update(0xFFFF_FFFFu32, &MAGIC_HYBRID_UINT), &[4u8, 0u8]);
    s = crc32_update(s, &orig_len.to_le_bytes());
    for &v in vals { s = crc32_update(s, &v.to_le_bytes()); }
    !s
}

/// Encode `vals: &[u32]` using `split_exp = 4` hybrid unsigned integer coding (`BRH1`).
pub fn encode_hybrid_uint_socket(vals: &[u32], sockets: &[u8], n_sockets: usize) -> Result<Vec<u8>> {
    let orig_len = u32::try_from(vals.len()).map_err(|_| Error::InvalidParameter)?;
    if vals.len() != sockets.len() || !(1..=256).contains(&n_sockets) {
        return Err(Error::InvalidParameter);
    }
    let (mut tokens, mut extra_bytes) = (Vec::with_capacity(vals.len()), Vec::with_capacity(vals.len()));
    let (mut bit_buf, mut bit_cnt, mut total_bits) = (0u64, 0u32, 0u64);
    for &v in vals {
        let (tok, extra, nbits) = hybrid_uint_split(v);
        tokens.push(tok);
        if nbits > 0 {
            bit_buf |= u64::from(extra) << bit_cnt;
            bit_cnt += nbits;
            total_bits += u64::from(nbits);
            while bit_cnt >= 8 { extra_bytes.push((bit_buf & 0xFF) as u8); bit_buf >>= 8; bit_cnt -= 8; }
        }
    }
    if bit_cnt > 0 { extra_bytes.push((bit_buf & 0xFF) as u8); }
    let tok_wire = encode_substream_socket_auto(&tokens, Some(sockets), n_sockets, ScaleBits::Bits14)?;
    let extra_bits_u32 = u32::try_from(total_bits).map_err(|_| Error::InvalidParameter)?;
    let crc = crc_hybrid_u32(orig_len, vals);
    let mut out = Vec::with_capacity(24 + tok_wire.len() + extra_bytes.len());
    out.extend_from_slice(&MAGIC_HYBRID_UINT);
    out.extend_from_slice(&[4u8, 0u8]);
    out.extend_from_slice(&orig_len.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    write_varint_u32(&mut out, extra_bits_u32);
    write_substream(&mut out, &tok_wire)?;
    write_substream(&mut out, &extra_bytes)?;
    Ok(out)
}

/// Decode a `BRH1` hybrid unsigned integer frame with caller-supplied causal `sockets`.
pub fn decode_hybrid_uint_socket(wire: &[u8], sockets: &[u8], max_out: usize) -> Result<Vec<u32>> {
    let mut r = Reader::new(wire);
    expect_magic(&mut r, &MAGIC_HYBRID_UINT)?;
    if r.read_u8()? != 4 || r.read_u8()? != 0 { return Err(Error::UnsupportedFormat); }
    let (orig_u32, expected_crc) = (r.read_u32_le()?, r.read_u32_le()?);
    let orig_len = orig_u32 as usize;
    if orig_len > max_out { return Err(Error::OutputTooLarge); }
    if sockets.len() != orig_len { return Err(Error::InvalidParameter); }
    let declared_bits = u64::from(r.read_varint_u32()?);
    let (tok_wire, extra) = (r.read_substream()?, r.read_substream()?);
    r.finish()?;
    let tokens = decode_substream_socket_auto(tok_wire, Some(sockets), orig_len)?;
    if tokens.len() != orig_len || extra.len() != declared_bits.div_ceil(8) as usize {
        return Err(Error::CorruptState);
    }
    let mut out = Vec::with_capacity(orig_len);
    let (mut bpos, mut bit_buf, mut bit_cnt, mut consumed) = (0usize, 0u64, 0u32, 0u64);
    for &tok in &tokens {
        let val = if tok < 16 {
            u32::from(tok)
        } else if tok <= 239 {
            let rel = u32::from(tok - 16);
            let (nbits, mant) = ((rel >> 3) + 1, rel & 7);
            while bit_cnt < nbits {
                bit_buf |= u64::from(*extra.get(bpos).ok_or(Error::TruncatedStream)?) << bit_cnt;
                bpos += 1;
                bit_cnt += 8;
            }
            let ex = (bit_buf & ((1u64 << nbits) - 1)) as u32;
            bit_buf >>= nbits;
            bit_cnt -= nbits;
            consumed += u64::from(nbits);
            ((8 | mant) << nbits) | ex
        } else {
            return Err(Error::CorruptState);
        };
        out.push(val);
    }
    if consumed != declared_bits || bpos != extra.len() || bit_buf != 0 { return Err(Error::CorruptState); }
    if crc_hybrid_u32(orig_u32, &out) != expected_crc { return Err(Error::ChecksumMismatch); }
    Ok(out)
}

fn split_u16_esc<T: Copy>(vals: &[T], to_u16: impl Fn(T) -> u16) -> (Vec<u8>, Vec<u8>) {
    let (mut lo, mut esc) = (Vec::with_capacity(vals.len()), Vec::new());
    for &v in vals {
        let u = to_u16(v);
        if u < 255 { lo.push(u as u8); } else { lo.push(255); esc.extend_from_slice(&u.to_le_bytes()); }
    }
    (lo, esc)
}

fn join_u16_esc(lo: &[u8], esc: &[u8]) -> Result<Vec<u16>> {
    let (mut out, mut pos) = (Vec::with_capacity(lo.len()), 0usize);
    for &b in lo {
        let v = if b < 255 {
            u16::from(b)
        } else {
            let Some(&[b0, b1]) = esc.get(pos..pos + 2) else { return Err(Error::CorruptResidualBlock); };
            pos += 2;
            let val = u16::from_le_bytes([b0, b1]);
            if val < 255 { return Err(Error::CorruptResidualBlock); }
            val
        };
        out.push(v);
    }
    if pos != esc.len() { Err(Error::CorruptResidualBlock) } else { Ok(out) }
}

fn crc_sparse_runs(meta: &[u8; 4], num_blocks: u32, nnz: &[u16], gaps: &[u16], levels: &[i16]) -> u32 {
    let mut s = crc32_update(crc32_update(0xFFFF_FFFFu32, &MAGIC_SPARSE_RUNS), meta);
    s = crc32_update(s, &num_blocks.to_le_bytes());
    for &n in nnz { s = crc32_update(s, &n.to_le_bytes()); }
    for &g in gaps { s = crc32_update(s, &g.to_le_bytes()); }
    for &lv in levels { s = crc32_update(s, &lv.to_le_bytes()); }
    !s
}

/// Encode block-agnostic sparse run-level triples `(block_nnz, gaps, levels)` into `BRR1`.
pub fn encode_sparse_runs_8way(block_nnz: &[u16], gaps: &[u16], levels: &[i16]) -> Result<Vec<u8>> {
    if gaps.len() != levels.len() { return Err(Error::InvalidParameter); }
    let mut sum_nnz = 0usize;
    for &n in block_nnz { sum_nnz = sum_nnz.checked_add(usize::from(n)).ok_or(Error::InvalidParameter)?; }
    if sum_nnz != gaps.len() { return Err(Error::InvalidParameter); }
    let num_b = u32::try_from(block_nnz.len()).map_err(|_| Error::InvalidParameter)?;
    let tot_n = u32::try_from(gaps.len()).map_err(|_| Error::InvalidParameter)?;
    let (nnz_lo, nnz_esc) = split_u16_esc(block_nnz, |v| v);
    let (gap_lo, gap_esc) = split_u16_esc(gaps, |v| v);
    let (lvl_lo, lvl_esc) = split_u16_esc(levels, zigzag_encode_i16);

    let (mut wires, mut cap) = (Vec::with_capacity(6), 32usize);
    for s in [&nnz_lo, &nnz_esc, &gap_lo, &gap_esc, &lvl_lo, &lvl_esc] {
        let w = encode_substream_socket_auto(s, None, 1, ScaleBits::Bits12)?;
        cap += w.len();
        wires.push(w);
    }
    let meta = tot_n.to_le_bytes();
    let crc = crc_sparse_runs(&meta, num_b, block_nnz, gaps, levels);
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(&MAGIC_SPARSE_RUNS);
    out.extend_from_slice(&meta);
    out.extend_from_slice(&num_b.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    for w in &wires { write_substream(&mut out, w)?; }
    Ok(out)
}

/// Decode a `BRR1` sparse run-level frame, enforcing `max_blocks` and `max_nnz` ceilings.
pub fn decode_sparse_runs_8way(
    wire: &[u8], max_blocks: usize, max_nnz: usize,
) -> Result<(Vec<u16>, Vec<u16>, Vec<i16>)> {
    let mut r = Reader::new(wire);
    expect_magic(&mut r, &MAGIC_SPARSE_RUNS)?;
    let mb = r.read_bytes(4)?;
    let meta = [mb[0], mb[1], mb[2], mb[3]];
    let total_nnz = u32::from_le_bytes(meta) as usize;
    let (num_b_u32, expected_crc) = (r.read_u32_le()?, r.read_u32_le()?);
    let num_b = num_b_u32 as usize;
    if num_b > max_blocks || total_nnz > max_nnz { return Err(Error::OutputTooLarge); }
    let (nb2, tn2) = (num_b.saturating_mul(2), total_nnz.saturating_mul(2));
    let caps = [num_b, nb2, total_nnz, tn2, total_nnz, tn2];
    let mut dec = Vec::with_capacity(6);
    for cap in caps { dec.push(decode_substream_socket_auto(r.read_substream()?, None, cap)?); }
    r.finish()?;
    if dec[0].len() != num_b || dec[2].len() != total_nnz || dec[4].len() != total_nnz {
        return Err(Error::CorruptResidualBlock);
    }
    let block_nnz = join_u16_esc(&dec[0], &dec[1])?;
    let mut sum_nnz = 0usize;
    for &n in &block_nnz {
        sum_nnz = sum_nnz.checked_add(usize::from(n)).ok_or(Error::CorruptResidualBlock)?;
    }
    if sum_nnz != total_nnz { return Err(Error::CorruptResidualBlock); }
    let gaps = join_u16_esc(&dec[2], &dec[3])?;
    let levels: Vec<i16> = join_u16_esc(&dec[4], &dec[5])?.into_iter().map(zigzag_decode_u16).collect();
    if crc_sparse_runs(&meta, num_b_u32, &block_nnz, &gaps, &levels) != expected_crc {
        return Err(Error::ChecksumMismatch);
    }
    Ok((block_nnz, gaps, levels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brs1_rejects_mismatched_outer_aux_scale_even_with_valid_crc() {
        let data: Vec<u8> = (0..256).map(|i| ((i * 7) & 31) as u8).collect();
        let e14 = encode_8way_with_scale(&data, ScaleBits::Bits14).expect("e14");
        let o1 = encode_order1_8way(&data, ContextMap::HighNibble).expect("o1");
        for (meta, inner) in [
            ([BRS1_MODE_RANS8_SINGLE, ScaleBits::Bits12 as u8], e14),
            ([BRS1_MODE_RANS8_ORDER1_HI, ScaleBits::Bits14 as u8], o1),
        ] {
            let mut forged = Vec::new();
            write_framed_header(&mut forged, &MAGIC_SOCKET_RANS, &meta, data.len() as u32, &data);
            forged.extend_from_slice(&inner);
            let res = decode_substream_socket_auto(&forged, None, data.len());
            assert_eq!(res, Err(Error::UnsupportedFormat));
        }
    }
}
