//! Shared wire framing, varint encoding, error types, and IEEE 802.3 CRC-32.

use alloc::vec::Vec;
use core::fmt;

/// Pure-`core` `log2(x)` for positive finite `f64` (`< 1e-12` error via `atanh` series).
#[inline]
pub(crate) fn log2_f64(x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() { return 0.0; }
    let bits = x.to_bits();
    let exp = (((bits >> 52) & 0x7FF) as i32) - 1023;
    let m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let poly = 1.0 + z2 * (1.0 / 3.0 + z2 * (1.0 / 5.0 + z2 * (1.0 / 7.0
        + z2 * (1.0 / 9.0 + z2 * (1.0 / 11.0)))));
    (exp as f64) + (2.0 * z * poly) * core::f64::consts::LOG2_E
}

/// Pure-`core` `log10(x)` for positive finite `f64`.
#[inline]
pub(crate) fn log10_f64(x: f64) -> f64 { log2_f64(x) * core::f64::consts::LOG10_2 }

/// Pure-`core` round-to-nearest for bounded `f64` values.
#[inline]
pub(crate) fn round_f64(x: f64) -> f64 {
    if x >= 0.0 { (x + 0.5) as i64 as f64 } else { (x - 0.5) as i64 as f64 }
}

/// Default upper bound (256 MiB) on decoded output size to prevent allocation bombs.
pub const DEFAULT_MAX_DECODE_BYTES: usize = 256 * 1024 * 1024;

/// Single bit-step of IEEE 802.3 reversed CRC-32 polynomial division.
/// @domain crc: 0..=4294967295
#[inline]
const fn crc32_bit_step(crc: u32) -> u32 {
    if (crc & 1) != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 }
}

const fn build_crc32_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0usize;
    while i < 256 {
        let (mut crc, mut bit) = (i as u32, 0usize);
        while bit < 8 { crc = crc32_bit_step(crc); bit += 1; }
        t[0][i] = crc;
        i += 1;
    }
    let mut lane = 1usize;
    while lane < 8 {
        let mut j = 0usize;
        while j < 256 {
            let prev = t[lane - 1][j];
            t[lane][j] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            j += 1;
        }
        lane += 1;
    }
    t
}

static CRC32_TABLES: [[u32; 256]; 8] = build_crc32_tables();

/// Update a running inverted IEEE 802.3 CRC-32 accumulator (slice-by-8, zero unsafe).
#[inline]
pub(crate) fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    let mut chunks = data.chunks_exact(8);
    for c in chunks.by_ref() {
        let lo = crc ^ u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let hi = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
        let t = &CRC32_TABLES;
        crc = t[7][(lo & 255) as usize] ^ t[6][((lo >> 8) & 255) as usize]
            ^ t[5][((lo >> 16) & 255) as usize] ^ t[4][((lo >> 24) & 255) as usize]
            ^ t[3][(hi & 255) as usize] ^ t[2][((hi >> 8) & 255) as usize]
            ^ t[1][((hi >> 16) & 255) as usize] ^ t[0][((hi >> 24) & 255) as usize];
    }
    for &b in chunks.remainder() {
        crc = (crc >> 8) ^ CRC32_TABLES[0][((crc ^ u32::from(b)) & 255) as usize];
    }
    crc
}

/// Compute standard IEEE 802.3 CRC-32 (slice-by-8, zero unsafe).
#[must_use]
pub fn crc32_ieee(data: &[u8]) -> u32 { !crc32_update(0xFFFF_FFFFu32, data) }

/// Compute IEEE 802.3 CRC-32 over `[magic(4), meta..., orig_len_le(4)] || data`.
#[inline] #[must_use]
pub(crate) fn framed_header_crc(magic: &[u8; 4], meta: &[u8], orig_len: u32, data: &[u8]) -> u32 {
    let s = crc32_update(crc32_update(0xFFFF_FFFFu32, magic), meta);
    !crc32_update(crc32_update(s, &orig_len.to_le_bytes()), data)
}

/// Write `[magic(4), meta..., orig_len_le(4), crc32_le(4)]` with header-chained CRC-32.
#[inline]
pub(crate) fn write_framed_header(
    out: &mut Vec<u8>, magic: &[u8; 4], meta: &[u8], orig_len: u32, data: &[u8],
) {
    let crc = framed_header_crc(magic, meta, orig_len, data);
    out.extend_from_slice(magic);
    out.extend_from_slice(meta);
    out.extend_from_slice(&orig_len.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
}

/// Errors returned by `zero-rans` encoders and decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    #[doc = "Stream magic header bytes did not match expected format."] InvalidMagic,
    #[doc = "Stream version or mode flag is unsupported."] UnsupportedFormat,
    #[doc = "Input ended unexpectedly before all required fields were read."] TruncatedStream,
    #[doc = "Trailing unconsumed bytes remained at the end of a framed stream."] TrailingBytes,
    #[doc = "Probability table counts or normalized sums failed validation."] CorruptProbabilityTable,
    #[doc = "rANS state failed range or terminal-state normalization verification."] CorruptState,
    #[doc = "Caller supplied an invalid configuration parameter or buffer length."] InvalidParameter,
    #[doc = "Decoded payload failed IEEE CRC-32 integrity verification."] ChecksumMismatch,
    #[doc = "Declared or decoded output exceeds the configured safety limit."] OutputTooLarge,
    #[doc = "LZ back-reference offset or match length was out of bounds."] CorruptLzStream,
    #[doc = "Video residual run/level stream violated block bounds or coding."] CorruptResidualBlock,
    #[doc = "Video frame dimensions failed even-4:2:0 or maximum safety validation."] InvalidDimensions,
    #[doc = "Inter-frame reference CRC-32 did not match decoder state."] ReferenceMismatch,
    #[doc = "Framed video packet (BVF1) failed syntax, header, or substream validation."] CorruptVideoFrame,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidMagic => "invalid stream magic header",
            Self::UnsupportedFormat => "unsupported format version or flags",
            Self::TruncatedStream => "unexpected end of compressed stream",
            Self::TrailingBytes => "unexpected trailing bytes in compressed stream",
            Self::CorruptProbabilityTable => "corrupt rANS probability table",
            Self::CorruptState => "corrupt rANS state refill or terminal state",
            Self::InvalidParameter => "invalid encoder or decoder parameter",
            Self::ChecksumMismatch => "decoded payload CRC-32 checksum mismatch",
            Self::OutputTooLarge => "declared output length exceeds decode limit",
            Self::CorruptLzStream => "corrupt split-stream LZ packet or offset",
            Self::CorruptResidualBlock => "corrupt transform residual block stream",
            Self::InvalidDimensions => "invalid or unsupported YUV420 frame dimensions",
            Self::ReferenceMismatch => "inter-frame reference digest mismatch",
            Self::CorruptVideoFrame => "corrupt BVF1 framed video packet",
        })
    }
}

impl core::error::Error for Error {}

/// Result alias for `zero-rans` fallible operations.
pub type Result<T> = core::result::Result<T, Error>;

#[inline]
fn read_varint_bounded<const MAX_B: usize, const TOP: u8>(data: &[u8], pos: &mut usize) -> Result<u64> {
    let (mut acc, mut step) = (0u64, 0usize);
    while step < MAX_B {
        let b = *data.get(*pos).ok_or(Error::TruncatedStream)?;
        *pos = pos.checked_add(1).ok_or(Error::TruncatedStream)?;
        if (step == MAX_B - 1 && (b & TOP) != 0) || (step > 0 && b == 0) {
            return Err(Error::CorruptState);
        }
        acc |= u64::from(b & 0x7F) << ((step as u32).wrapping_mul(7) & 63);
        if (b & 0x80) == 0 { return Ok(acc); }
        step = step.wrapping_add(1);
    }
    Err(Error::CorruptState)
}

/// Read canonical unsigned LEB128 `u64` (rejecting overlong or overflowing encodings).
#[inline]
pub(crate) fn read_varint_u64(data: &[u8], pos: &mut usize) -> Result<u64> {
    read_varint_bounded::<10, 0xFE>(data, pos)
}
/// Append canonical unsigned LEB128 `u64` to `out`.
#[inline]
pub(crate) fn write_varint_u64(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        out.push(if value == 0 { byte } else { byte | 0x80 });
        if value == 0 { break; }
    }
}
/// Append canonical unsigned LEB128 `u32` to `out`.
#[inline]
pub(crate) fn write_varint_u32(out: &mut Vec<u8>, value: u32) { write_varint_u64(out, u64::from(value)); }
/// Append a length-prefixed substream slice to `out`.
#[inline]
pub(crate) fn write_substream(out: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    write_varint_u32(out, u32::try_from(payload.len()).map_err(|_| Error::InvalidParameter)?);
    out.extend_from_slice(payload);
    Ok(())
}
/// Partition `0..len` into contiguous `(start, chunk_len)` slices of at most `max_chunk` bytes.
#[inline]
pub(crate) fn chunk_bounds(len: usize, max_chunk: usize) -> Vec<(usize, usize)> {
    let (step, mut start) = (max_chunk.max(1), 0usize);
    let mut out = Vec::with_capacity(len.div_ceil(step));
    while start < len {
        let take = (len - start).min(step);
        out.push((start, take));
        start += take;
    }
    out
}
/// Partition `0..len` into `N` balanced contiguous `(start, chunk_len)` slices.
#[inline]
pub(crate) fn balanced_chunks<const N: usize>(len: usize) -> [(usize, usize); N] {
    let (base, rem, mut cursor) = (len / N, len % N, 0usize);
    core::array::from_fn(|k| {
        let take = base + usize::from(k < rem);
        let pair = (cursor, take);
        cursor += take;
        pair
    })
}

/// Cursor helper for fail-closed reading of framed binary streams.
pub(crate) struct Reader<'a> { data: &'a [u8], pos: usize }
impl<'a> Reader<'a> {
    #[inline] pub(crate) fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    #[inline] pub(crate) fn read_u8(&mut self) -> Result<u8> {
        let b = *self.data.get(self.pos).ok_or(Error::TruncatedStream)?;
        self.pos += 1;
        Ok(b)
    }
    #[inline] pub(crate) fn read_u32_le(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    #[inline] pub(crate) fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).ok_or(Error::TruncatedStream)?;
        let slice = self.data.get(self.pos..end).ok_or(Error::TruncatedStream)?;
        self.pos = end;
        Ok(slice)
    }
    #[inline] pub(crate) fn read_varint_u32(&mut self) -> Result<u32> {
        read_varint_bounded::<5, 0xF0>(self.data, &mut self.pos).map(|v| v as u32)
    }
    #[inline] pub(crate) fn read_substream(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint_u32()? as usize;
        self.read_bytes(len)
    }
    #[inline] pub(crate) fn finish(&self) -> Result<()> {
        if self.pos == self.data.len() { Ok(()) } else { Err(Error::TrailingBytes) }
    }
    #[inline] pub(crate) fn read_rest(&mut self) -> &'a [u8] {
        let slice = &self.data[self.pos..];
        self.pos = self.data.len();
        slice
    }
}

/// Validate a 4-byte magic prefix at the start of `reader`.
#[inline]
pub(crate) fn expect_magic(reader: &mut Reader<'_>, expected: &[u8; 4]) -> Result<()> {
    if reader.read_bytes(4)? == expected { Ok(()) } else { Err(Error::InvalidMagic) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn crc32_and_varint_roundtrips() {
        assert_eq!((crc32_ieee(b"123456789"), crc32_ieee(b"")), (0xCBF4_3926, 0));
        for &v in &[0u32, 1, 127, 128, 16_383, 16_384, 1 << 21, u32::MAX] {
            let mut b = Vec::new();
            write_varint_u32(&mut b, v);
            let mut r = Reader::new(&b);
            assert_eq!((r.read_varint_u32(), r.finish()), (Ok(v), Ok(())));
        }
        for &v in &[0u64, 127, 128, u32::MAX as u64, u64::MAX] {
            let (mut b, mut p) = (Vec::new(), 0usize);
            write_varint_u64(&mut b, v);
            assert_eq!((read_varint_u64(&b, &mut p), p), (Ok(v), b.len()));
        }
        assert_eq!(chunk_bounds(10, 4), vec![(0, 4), (4, 4), (8, 2)]);
    }
}
