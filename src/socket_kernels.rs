//! Pure-integer 8-lane kernels for 2D prediction, gradient sockets, mod-256 transforms,
//! and non-commutative hashing/mixing compiled cleanly by `zero-cc` (`0` spills, `divs=0->0`).

/// Branchless 8-lane Paeth predictor from left (`w`), top (`n`), and top-left (`nw`).
/// @domain w: 0..=255, n: 0..=255, nw: 0..=255
#[inline]
pub fn paeth_pred8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let (d_n, d_w) = (n[k] - nw[k], w[k] - nw[k]);
        let (pa, pb) = (d_n.max(-d_n), d_w.max(-d_w));
        let d_c = d_w + d_n;
        let pc = d_c.max(-d_c);
        let (m_ab, m_ac, m_bc) = ((pa - pb - 1) >> 31, (pa - pc - 1) >> 31, (pb - pc - 1) >> 31);
        let n_or_nw = nw[k] ^ ((n[k] ^ nw[k]) & m_bc);
        out[k] = n_or_nw ^ ((w[k] ^ n_or_nw) & (m_ab & m_ac));
    }
}

/// Branchless 8-lane 2D `(dh, dv)` gradient quantizer producing a 6-bit context socket (`0..=63`).
/// @domain w: 0..=255, n: 0..=255, nw: 0..=255
#[inline]
pub fn grad_socket8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let dh = w[k] - nw[k];
        let qh = -((-24 - dh) >> 31) - ((-7 - dh) >> 31) - ((-1 - dh) >> 31) - ((-dh) >> 31)
            - ((2 - dh) >> 31) - ((8 - dh) >> 31) - ((24 - dh) >> 31);
        let dv = n[k] - nw[k];
        let qv = -((-24 - dv) >> 31) - ((-7 - dv) >> 31) - ((-1 - dv) >> 31) - ((-dv) >> 31)
            - ((2 - dv) >> 31) - ((8 - dv) >> 31) - ((24 - dv) >> 31);
        out[k] = (qh << 3) | qv;
    }
}

/// Branchless 8-lane modulo-256 residual subtraction `(cur[k] - ref_b[k]) & 255`.
/// @domain cur: 0..=255, ref_b: 0..=255
#[inline]
pub fn sub_mod256_8(cur: &[i32; 8], ref_b: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize { out[k] = (cur[k] - ref_b[k]) & 255; }
}

/// Branchless 8-lane modulo-256 residual reconstruction `(res[k] + ref_b[k]) & 255`.
/// @domain res: 0..=255, ref_b: 0..=255
#[inline]
pub fn add_mod256_8(res: &[i32; 8], ref_b: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize { out[k] = (res[k] + ref_b[k]) & 255; }
}

/// Branchless 8-lane signed `i8` (`-128..=127`) to unsigned `u8` (`0..=255`) zigzag bijection.
/// @domain err: -128..=127
#[inline]
pub fn zigzag_u8_8(err: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize { out[k] = ((err[k] << 1) ^ (err[k] >> 7)) & 255; }
}

/// Branchless 8-lane non-commutative Order-2 context hash mapped uniformly into `0..=255`.
/// @domain p2: 0..=255, p1: 0..=255
#[inline]
pub fn nc_hash_o2_8(p2: &[i32; 8], p1: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize { out[k] = (p2[k] * 137 + (p2[k] >> 3) + p1[k] * 3) & 255; }
}

/// Branchless 8-lane non-commutative 4-byte matchfinder hash mapped into `0..=131071`.
/// @domain b0: 0..=255, b1: 0..=255, b2: 0..=255, b3: 0..=255
#[inline]
pub fn nc_hash4_8(b0: &[i32; 8], b1: &[i32; 8], b2: &[i32; 8], b3: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let x = b0[k] * 131 + b1[k] * 4099 + b2[k] * 65537 + b3[k] * 1_048_583;
        out[k] = (x + (x >> 11) + (x >> 5)) & 131_071;
    }
}

/// Branchless 8-lane 3-model Bayesian probability mixer (`12-bit`, `1..=4095`).
/// @domain p_o0: 1..=4095, p_o1: 1..=4095, p_o2: 1..=4095, w_o2: 0..=24
#[inline]
pub fn nc_mix3_prob12_8(
    p_o0: &[i32; 8], p_o1: &[i32; 8], p_o2: &[i32; 8], w_o2: &[i32; 8], out: &mut [i32; 8],
) {
    for k in 0..8usize {
        let base = (p_o0[k] + p_o1[k] * 3 + 2) >> 2;
        let w = w_o2[k];
        let m = (w * p_o2[k] + (32 - w) * base + 16) >> 5;
        out[k] = m.max(1).min(4095);
    }
}

/// Scalar Paeth predictor bit-identical to [`paeth_pred8`].
/// @domain w: 0..=255, n: 0..=255, nw: 0..=255
#[inline]
#[must_use]
pub const fn paeth_pred_u8(w: u8, n: u8, nw: u8) -> u8 {
    let (d_n, d_w) = ((n as i32) - (nw as i32), (w as i32) - (nw as i32));
    let pa = if d_n < 0 { -d_n } else { d_n };
    let pb = if d_w < 0 { -d_w } else { d_w };
    let d_c = d_w + d_n;
    let pc = if d_c < 0 { -d_c } else { d_c };
    if pa <= pb && pa <= pc { w } else if pb <= pc { n } else { nw }
}

/// Scalar 6-bit `(dh, dv)` gradient socket quantizer bit-identical to [`grad_socket8`].
/// @domain w: 0..=255, n: 0..=255, nw: 0..=255
#[inline]
#[must_use]
pub const fn grad_socket_u8(w: u8, n: u8, nw: u8) -> u8 {
    let dh = (w as i32) - (nw as i32);
    let qh = -((-24 - dh) >> 31) - ((-7 - dh) >> 31) - ((-1 - dh) >> 31) - ((-dh) >> 31)
        - ((2 - dh) >> 31) - ((8 - dh) >> 31) - ((24 - dh) >> 31);
    let dv = (n as i32) - (nw as i32);
    let qv = -((-24 - dv) >> 31) - ((-7 - dv) >> 31) - ((-1 - dv) >> 31) - ((-dv) >> 31)
        - ((2 - dv) >> 31) - ((8 - dv) >> 31) - ((24 - dv) >> 31);
    ((qh << 3) | qv) as u8
}

/// Scalar non-commutative Order-2 context hash bit-identical to [`nc_hash_o2_8`].
/// @domain p2: 0..=255, p1: 0..=255
#[inline]
#[must_use]
pub const fn nc_hash_o2_u8(p2: u8, p1: u8) -> u8 {
    let a = p2 as i32;
    ((a * 137 + (a >> 3) + (p1 as i32) * 3) & 255) as u8
}

/// Scalar 17-bit non-commutative 4-byte hash bit-identical to [`nc_hash4_8`].
/// @domain b0: 0..=255, b1: 0..=255, b2: 0..=255, b3: 0..=255
#[inline]
#[must_use]
pub const fn nc_hash4_u8(b0: u8, b1: u8, b2: u8, b3: u8) -> usize {
    let x = (b0 as i32) * 131 + (b1 as i32) * 4099 + (b2 as i32) * 65537 + (b3 as i32) * 1_048_583;
    ((x + (x >> 11) + (x >> 5)) & 131_071) as usize
}

/// Callback type for 8-lane 3-input kernels (`paeth_pred8`, `grad_socket8`).
pub type SocketPred3Fn = fn(&[i32; 8], &[i32; 8], &[i32; 8], &mut [i32; 8]);
/// Callback type for 8-lane 2-input modulo-256 kernels (`sub_mod256_8`, `add_mod256_8`).
pub type SocketMod256Fn = fn(&[i32; 8], &[i32; 8], &mut [i32; 8]);
/// Callback type for 8-lane 1-input zigzag kernel (`zigzag_u8_8`).
pub type SocketZigzagFn = fn(&[i32; 8], &mut [i32; 8]);
/// Slice-batched callback for 2-input 8-lane kernels.
pub type SocketBatch2Fn = fn(&[[i32; 8]], &[[i32; 8]], &mut [[i32; 8]]);
/// Slice-batched callback for 3-input 8-lane kernels.
pub type SocketBatch3Fn = fn(&[[i32; 8]], &[[i32; 8]], &[[i32; 8]], &mut [[i32; 8]]);
/// Slice-batched callback for 4-input 8-lane kernels.
pub type SocketBatch4Fn = fn(&[[i32; 8]], &[[i32; 8]], &[[i32; 8]], &[[i32; 8]], &mut [[i32; 8]]);

/// Optional per-block accelerator table for 8-lane pure-integer socket kernels.
#[derive(Clone, Copy)]
pub struct SocketAccel {
    /// Accelerated `paeth_pred8`.
    pub paeth_pred8: SocketPred3Fn,
    /// Accelerated `grad_socket8`.
    pub grad_socket8: SocketPred3Fn,
    /// Accelerated `sub_mod256_8`.
    pub sub_mod256_8: SocketMod256Fn,
    /// Accelerated `add_mod256_8`.
    pub add_mod256_8: SocketMod256Fn,
    /// Accelerated `zigzag_u8_8`.
    pub zigzag_u8_8: SocketZigzagFn,
}

/// Optional slice-batched accelerator table (`zero_cc_*_batch` AVX2/SIMD128 kernels).
#[derive(Clone, Copy)]
pub struct SocketBatchAccel {
    /// Batched `paeth_pred8`.
    pub paeth_pred8_batch: SocketBatch3Fn,
    /// Batched `grad_socket8`.
    pub grad_socket8_batch: SocketBatch3Fn,
    /// Batched `sub_mod256_8`.
    pub sub_mod256_8_batch: SocketBatch2Fn,
    /// Batched `add_mod256_8`.
    pub add_mod256_8_batch: SocketBatch2Fn,
    /// Batched `nc_hash_o2_8`.
    pub nc_hash_o2_8_batch: SocketBatch2Fn,
    /// Batched `nc_hash4_8`.
    pub nc_hash4_8_batch: SocketBatch4Fn,
}

#[cfg(all(feature = "zero-cc-native", target_arch = "x86_64", unix))]
include!(concat!(env!("OUT_DIR"), "/zcc_native.rs"));

/// Register `zero-cc` compiled socket accelerators (`no_std` stub; use `zero-cc-native` feature).
pub fn register_socket_accel(accel: SocketAccel) -> bool { let _ = accel; false }

/// Register slice-batched `zero-cc` socket/hash accelerators (`no_std` stub).
pub fn register_socket_batch_accel(accel: SocketBatchAccel) -> bool { let _ = accel; false }

/// Returns `true` if a slice-batched `zero-cc` accelerator is active (`zero-cc-native` on x86_64).
#[inline]
#[must_use]
pub fn has_socket_batch_accel() -> bool {
    cfg!(all(feature = "zero-cc-native", target_arch = "x86_64", unix))
}

macro_rules! def_scalar_dispatches {
    ($(($doc:expr, $run:ident, $field:ident, ($($arg:ident),+))),+ $(,)?) => {$(
        #[doc = $doc]
        #[inline]
        pub fn $run($($arg: &[i32; 8]),+, out: &mut [i32; 8]) {
            #[cfg(all(feature = "zero-cc-native", target_arch = "x86_64", unix))]
            { zcc_native::$field($($arg),+, out); return; }
            #[cfg(not(all(feature = "zero-cc-native", target_arch = "x86_64", unix)))]
            $field($($arg),+, out);
        }
    )+};
}

macro_rules! def_batch_dispatches {
    ($(($doc:expr, $run:ident, $field:ident, $scalar:ident, ($($arg:ident),+))),+ $(,)?) => {$(
        #[doc = $doc]
        #[inline]
        pub fn $run($($arg: &[[i32; 8]]),+, out: &mut [[i32; 8]]) {
            let mut len = out.len();
            $(len = len.min($arg.len());)+
            if len == 0 { return; }
            #[cfg(all(feature = "zero-cc-native", target_arch = "x86_64", unix))]
            { zcc_native::$field($($arg),+, out, len); return; }
            #[cfg(not(all(feature = "zero-cc-native", target_arch = "x86_64", unix)))]
            for i in 0..len { $scalar($(&$arg[i]),+, &mut out[i]); }
        }
    )+};
}

def_scalar_dispatches!(
    ("Dispatch `paeth_pred8`.", run_paeth_pred8, paeth_pred8, (w, n, nw)),
    ("Dispatch `grad_socket8`.", run_grad_socket8, grad_socket8, (w, n, nw)),
    ("Dispatch `sub_mod256_8`.", run_sub_mod256_8, sub_mod256_8, (cur, ref_b)),
    ("Dispatch `add_mod256_8`.", run_add_mod256_8, add_mod256_8, (res, ref_b)),
    ("Dispatch `zigzag_u8_8`.", run_zigzag_u8_8, zigzag_u8_8, (err)),
);

def_batch_dispatches!(
    ("Batch `paeth_pred8`.", run_paeth_pred8_batch, paeth_pred8_batch, paeth_pred8, (w, n, nw)),
    ("Batch `grad_socket8`.", run_grad_socket8_batch, grad_socket8_batch, grad_socket8, (w, n, nw)),
    ("Batch `sub_mod256_8`.", run_sub_mod256_8_batch, sub_mod256_8_batch, sub_mod256_8, (cur, ref_b)),
    ("Batch `add_mod256_8`.", run_add_mod256_8_batch, add_mod256_8_batch, add_mod256_8, (res, ref_b)),
    ("Batch `nc_hash_o2_8`.", run_nc_hash_o2_8_batch, nc_hash_o2_8_batch, nc_hash_o2_8, (p2, p1)),
    ("Batch `nc_hash4_8`.", run_nc_hash4_8_batch, nc_hash4_8_batch, nc_hash4_8, (b0, b1, b2, b3)),
);
