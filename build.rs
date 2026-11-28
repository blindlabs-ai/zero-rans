use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const KERNELS: [&str; 8] = [
    "paeth_pred8",
    "grad_socket8",
    "sub_mod256_8",
    "add_mod256_8",
    "zigzag_u8_8",
    "nc_hash_o2_8",
    "nc_hash4_8",
    "nc_mix3_prob12_8",
];

const ZCC_NATIVE_BINDINGS: &str = r#"mod zcc_native {
    extern "C" {
        fn zero_cc_paeth_pred8(w: *const i32, n: *const i32, nw: *const i32, out: *mut i32);
        fn zero_cc_grad_socket8(w: *const i32, n: *const i32, nw: *const i32, out: *mut i32);
        fn zero_cc_sub_mod256_8(cur: *const i32, ref_b: *const i32, out: *mut i32);
        fn zero_cc_add_mod256_8(res: *const i32, ref_b: *const i32, out: *mut i32);
        fn zero_cc_zigzag_u8_8(err: *const i32, out: *mut i32);
        fn zero_cc_paeth_pred8_batch(
            w: *const i32, n: *const i32, nw: *const i32, out: *mut i32, count: usize,
        );
        fn zero_cc_grad_socket8_batch(
            w: *const i32, n: *const i32, nw: *const i32, out: *mut i32, count: usize,
        );
        fn zero_cc_sub_mod256_8_batch(
            cur: *const i32, ref_b: *const i32, out: *mut i32, count: usize,
        );
        fn zero_cc_add_mod256_8_batch(
            res: *const i32, ref_b: *const i32, out: *mut i32, count: usize,
        );
        fn zero_cc_nc_hash_o2_8_batch(
            p2: *const i32, p1: *const i32, out: *mut i32, count: usize,
        );
        fn zero_cc_nc_hash4_8_batch(
            b0: *const i32, b1: *const i32, b2: *const i32, b3: *const i32, out: *mut i32, n: usize,
        );
    }

    #[inline]
    pub fn paeth_pred8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
        unsafe { zero_cc_paeth_pred8(w.as_ptr(), n.as_ptr(), nw.as_ptr(), out.as_mut_ptr()) };
    }
    #[inline]
    pub fn grad_socket8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
        unsafe { zero_cc_grad_socket8(w.as_ptr(), n.as_ptr(), nw.as_ptr(), out.as_mut_ptr()) };
    }
    #[inline]
    pub fn sub_mod256_8(cur: &[i32; 8], ref_b: &[i32; 8], out: &mut [i32; 8]) {
        unsafe { zero_cc_sub_mod256_8(cur.as_ptr(), ref_b.as_ptr(), out.as_mut_ptr()) };
    }
    #[inline]
    pub fn add_mod256_8(res: &[i32; 8], ref_b: &[i32; 8], out: &mut [i32; 8]) {
        unsafe { zero_cc_add_mod256_8(res.as_ptr(), ref_b.as_ptr(), out.as_mut_ptr()) };
    }
    #[inline]
    pub fn zigzag_u8_8(err: &[i32; 8], out: &mut [i32; 8]) {
        unsafe { zero_cc_zigzag_u8_8(err.as_ptr(), out.as_mut_ptr()) };
    }
    #[inline]
    pub fn paeth_pred8_batch(
        w: &[[i32; 8]], n: &[[i32; 8]], nw: &[[i32; 8]], out: &mut [[i32; 8]], len: usize,
    ) {
        unsafe {
            zero_cc_paeth_pred8_batch(
                w.as_ptr().cast(), n.as_ptr().cast(), nw.as_ptr().cast(),
                out.as_mut_ptr().cast(), len,
            );
        }
    }
    #[inline]
    pub fn grad_socket8_batch(
        w: &[[i32; 8]], n: &[[i32; 8]], nw: &[[i32; 8]], out: &mut [[i32; 8]], len: usize,
    ) {
        unsafe {
            zero_cc_grad_socket8_batch(
                w.as_ptr().cast(), n.as_ptr().cast(), nw.as_ptr().cast(),
                out.as_mut_ptr().cast(), len,
            );
        }
    }
    #[inline]
    pub fn sub_mod256_8_batch(
        cur: &[[i32; 8]], ref_b: &[[i32; 8]], out: &mut [[i32; 8]], len: usize,
    ) {
        unsafe {
            zero_cc_sub_mod256_8_batch(
                cur.as_ptr().cast(), ref_b.as_ptr().cast(), out.as_mut_ptr().cast(), len,
            );
        }
    }
    #[inline]
    pub fn add_mod256_8_batch(
        res: &[[i32; 8]], ref_b: &[[i32; 8]], out: &mut [[i32; 8]], len: usize,
    ) {
        unsafe {
            zero_cc_add_mod256_8_batch(
                res.as_ptr().cast(), ref_b.as_ptr().cast(), out.as_mut_ptr().cast(), len,
            );
        }
    }
    #[inline]
    pub fn nc_hash_o2_8_batch(
        p2: &[[i32; 8]], p1: &[[i32; 8]], out: &mut [[i32; 8]], len: usize,
    ) {
        unsafe {
            zero_cc_nc_hash_o2_8_batch(
                p2.as_ptr().cast(), p1.as_ptr().cast(), out.as_mut_ptr().cast(), len,
            );
        }
    }
    #[inline]
    pub fn nc_hash4_8_batch(
        b0: &[[i32; 8]], b1: &[[i32; 8]], b2: &[[i32; 8]], b3: &[[i32; 8]],
        out: &mut [[i32; 8]], len: usize,
    ) {
        unsafe {
            zero_cc_nc_hash4_8_batch(
                b0.as_ptr().cast(), b1.as_ptr().cast(), b2.as_ptr().cast(), b3.as_ptr().cast(),
                out.as_mut_ptr().cast(), len,
            );
        }
    }
}
"#;

fn resolve_zero_cc_bin() -> PathBuf {
    if let Some(val) = env::var_os("ZERO_CC_BIN") {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    if let Some(home) = env::var_os("HOME") {
        let candidate = PathBuf::from(home).join("zero-cc/target/release/zero-cc");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("zero-cc")
}

fn main() {
    println!("cargo:rerun-if-changed=src/socket_kernels.rs");
    println!("cargo:rerun-if-env-changed=ZERO_CC_BIN");

    let enabled = env::var_os("CARGO_FEATURE_ZERO_CC_NATIVE").is_some();
    let is_x86_64 = env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64");
    let is_unix = env::var_os("CARGO_CFG_UNIX").is_some();
    if !(enabled && is_x86_64 && is_unix) {
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    fs::write(out_dir.join("zcc_native.rs"), ZCC_NATIVE_BINDINGS)
        .expect("failed to write zcc_native.rs");

    let zcc = resolve_zero_cc_bin();
    let mut objs = Vec::with_capacity(KERNELS.len());
    for k in KERNELS {
        let obj = out_dir.join(format!("{k}.o"));
        let status = Command::new(&zcc)
            .args(["src/socket_kernels.rs", "--kernel", k, "--emit-elf"])
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| {
                panic!("failed to execute zero-cc at {}: {e}", zcc.display())
            });
        assert!(status.success(), "zero-cc failed compiling kernel `{k}`");
        objs.push(obj);
    }

    let lib = out_dir.join("libzero_rans_zcc.a");
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&lib);
    for obj in &objs {
        ar.arg(obj);
    }
    let ar_status = ar.status().expect("failed to execute `ar`");
    assert!(ar_status.success(), "`ar rcs` failed creating {}", lib.display());

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=zero_rans_zcc");
}
