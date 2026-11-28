//! Declarative AMD Zen 1..5 / EPYC 7001..9005 microarchitecture specification parser,
//! zero-unsafe Linux host detection, and closed-form cache/SIMD tuning plan derivation.

use alloc::string::String;
use alloc::vec::Vec;

/// Embedded AMD Zen 1..5 / EPYC 7001..9005 hardware specification (`specs/amd_zen_epyc.spec`).
pub const EMBEDDED_AMD_ZEN_SPEC: &str = include_str!("../specs/amd_zen_epyc.spec");

/// Architectural SIMD execution tier declared in `specs/amd_zen_epyc.spec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTier {
    #[doc = "Portable scalar fallback (no x86 AVX2/AVX-512 profile matched)."] Scalar = 0,
    #[doc = "Double-pumped 128-bit pipes implementing 256-bit AVX2 (Zen 1)."] Avx2DoublePump128 = 1,
    #[doc = "Native 256-bit AVX2 execution pipes (Zen 2, Zen 3)."] Avx2Native256 = 2,
    #[doc = "Double-pumped 256-bit pipes implementing 512-bit AVX-512 (Zen 4/4c)."] Avx512DoublePump256 = 3,
    #[doc = "Full native 512-bit AVX-512 execution pipes (Zen 5)."] Avx512Native512 = 4,
}

const SIMD_TIERS: [(&str, SimdTier); 5] = [
    ("scalar", SimdTier::Scalar), ("avx2_128x2", SimdTier::Avx2DoublePump128),
    ("avx2", SimdTier::Avx2Native256), ("avx512_256x2", SimdTier::Avx512DoublePump256),
    ("avx512_512", SimdTier::Avx512Native512),
];
impl SimdTier {
    /// Parse a declarative `simd` value from `amd_zen_epyc.spec`.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        let t = token.trim();
        SIMD_TIERS.iter().find(|&&(k, _)| k == t).map(|&(_, v)| v)
    }
    /// Canonical declarative name matching `specs/amd_zen_epyc.spec`.
    #[must_use]
    pub const fn as_str(self) -> &'static str { SIMD_TIERS[(self as usize) % SIMD_TIERS.len()].0 }
}

/// Hardware cache, topology, and SIMD specification for a CPU microarchitecture profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuProfile {
    #[doc = "Profile identifier (for example zen2_rome or intel_sapphire_rapids)."] pub name: String,
    #[doc = "CPUID vendor string (AuthenticAMD, GenuineIntel, or generic)."] pub vendor: String,
    #[doc = "CPUID family number (for example 6, 23, 25, 26)."] pub family: u32,
    #[doc = "Matching CPUID model numbers for this family."] pub models: Vec<u32>,
    #[doc = "Per-core L1 data cache size in KiB (32 on Zen 1..4, 48 on Zen 5 / Ice Lake+)."] pub l1d_kib: usize,
    #[doc = "Per-core L1 data cache associativity (8 or 12)."] pub l1d_ways: usize,
    #[doc = "Per-core private L2 cache size in KiB (512..2048)."] pub l2_kib: usize,
    #[doc = "Base L3 cache size per CCX in MiB."] pub l3_ccx_mib: usize,
    #[doc = "Total L3 cache per CCD with 3D V-Cache in MiB (0 or 96)."] pub l3_3dv_mib: usize,
    #[doc = "Physical cores per CCX (4 or 8)."] pub ccx_cores: usize,
    #[doc = "Cache coherency line size in bytes (64)."] pub line_bytes: usize,
    #[doc = "SIMD execution tier."] pub simd: SimdTier,
    #[doc = "Architectural vector register count (16 or 32)."] pub vec_regs: usize,
    #[doc = "Architectural vector register width in bits (256 or 512)."] pub vec_bits: usize,
    #[doc = "Native single-cycle vector pipe width in bits."] pub native_pipe_bits: usize,
    #[doc = "Approximate PAUSE instruction latency in cycles (64 Zen, 140 Xeon)."] pub pause_cycles: usize,
    #[doc = "Detected logical hardware thread count (>= 1)."] pub logical_cpus: usize,
    #[doc = "Detected physical core estimate (>= 1)."] pub physical_cores: usize,
}

/// Closed-form cache-hierarchy and thread tuning plan derived from a [`CpuProfile`].
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZenTuningPlan {
    #[doc = "L1D-resident tile budget in bytes (half L1D rounded to 4096 B)."] pub l1_tile_bytes: usize,
    #[doc = "L2-resident independent block length (128 KiB or 256 KiB)."] pub l2_block_bytes: usize,
    #[doc = "Log2 Order-2 non-commutative table slots (9 or 10)."] pub nc_o2_log2: u8,
    #[doc = "Log2 Order-3 non-commutative table slots (8 or 9)."] pub nc_o3_log2: u8,
    #[doc = "L2-resident 4-byte matchfinder table log2 buckets (15 or 16)."] pub l2_hash4_bits: u32,
    #[doc = "L2-resident 6-byte matchfinder table log2 buckets (14 or 15)."] pub l2_hash6_bits: u32,
    #[doc = "Minimum payload bytes before parallel fan-out."] pub parallel_threshold_bytes: usize,
    #[doc = "Capped worker thread count targeting physical cores (1..=64)."] pub max_threads: usize,
    #[doc = "PAUSE spin iterations (~15 us) before Condvar park."] pub spin_iters: usize,
}

impl ZenTuningPlan {
    /// Return the calibrated `PAUSE` spin-wait iteration count (~15 us before park).
    #[must_use]
    pub const fn spin_iters(&self) -> usize { self.spin_iters }
}

impl CpuProfile {
    /// Construct a conservative generic profile when no Zen/EPYC or Xeon family matches.
    #[must_use]
    pub fn generic(logical_cpus: usize, physical_cores: usize) -> Self {
        let logi = logical_cpus.max(1);
        Self {
            name: String::from("generic_host"), vendor: String::from("generic"),
            family: 0, models: Vec::new(),
            l1d_kib: 32, l1d_ways: 8, l2_kib: 512, l3_ccx_mib: 16, l3_3dv_mib: 0,
            ccx_cores: 4, line_bytes: 64, simd: SimdTier::Scalar, vec_regs: 16,
            vec_bits: 128, native_pipe_bits: 128, pause_cycles: 64, logical_cpus: logi,
            physical_cores: physical_cores.max(1).min(logi),
        }
    }
    /// Calibrated `PAUSE` spin iterations targeting ~15 us before Condvar park.
    #[must_use]
    pub fn spin_iters(&self) -> usize { (30_000 / self.pause_cycles.max(16)).max(64) }
    /// Derive closed-form L1/L2/NC-table/parallelism tuning parameters from this CPU profile.
    #[must_use]
    pub fn tuning_plan(&self) -> ZenTuningPlan {
        let l1_tile_bytes = (self.l1d_kib.saturating_mul(512) / 4096).max(1).saturating_mul(4096);
        let lg = self.l2_kib >= 1024;
        let l2_block_bytes = if lg { 262_144 } else { 131_072 };
        let phys = if self.physical_cores > 0 { self.physical_cores } else { (self.logical_cpus / 2).max(1) };
        ZenTuningPlan {
            l1_tile_bytes, l2_block_bytes,
            nc_o2_log2: if lg { 10 } else { 9 }, nc_o3_log2: if lg { 9 } else { 8 },
            l2_hash4_bits: if lg { 16 } else { 15 }, l2_hash6_bits: if lg { 15 } else { 14 },
            parallel_threshold_bytes: l2_block_bytes.saturating_mul(2),
            max_threads: phys.max(1).min(64),
            spin_iters: self.spin_iters(),
        }
    }
}
#[inline]
fn parse_into<T: core::str::FromStr>(dst: &mut T, s: &str) { if let Ok(v) = s.parse() { *dst = v; } }
/// Parse declarative CPU profiles from `amd_zen_epyc.spec`-formatted text.
#[must_use]
pub fn parse_cpu_specs(spec_text: &str) -> Vec<CpuProfile> {
    let (mut profiles, mut cur) = (Vec::new(), None::<CpuProfile>);
    for line in spec_text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') { continue; }
        if line.starts_with('[') && line.ends_with(']') && line.len() >= 3 {
            if let Some(prev) = cur.take() { profiles.push(prev); }
            let mut p = CpuProfile::generic(1, 1);
            p.name = String::from(line[1..line.len() - 1].trim());
            cur = Some(p);
        } else if let (Some((k, v)), Some(p)) = (line.split_once('='), cur.as_mut()) {
            let (k, v) = (k.trim(), v.trim());
            match k {
                "vendor" => p.vendor = String::from(v),
                "family" => parse_into(&mut p.family, v),
                "models" => p.models = v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
                "l1d_kib" => parse_into(&mut p.l1d_kib, v), "l1d_ways" => parse_into(&mut p.l1d_ways, v),
                "l2_kib" => parse_into(&mut p.l2_kib, v), "l3_ccx_mib" => parse_into(&mut p.l3_ccx_mib, v),
                "l3_3dv_mib" => parse_into(&mut p.l3_3dv_mib, v),
                "ccx_cores" => parse_into(&mut p.ccx_cores, v),
                "line_bytes" => parse_into(&mut p.line_bytes, v),
                "simd" => if let Some(t) = SimdTier::parse(v) { p.simd = t; },
                "vec_regs" => parse_into(&mut p.vec_regs, v), "vec_bits" => parse_into(&mut p.vec_bits, v),
                "native_pipe_bits" => parse_into(&mut p.native_pipe_bits, v),
                "pause_cycles" => parse_into(&mut p.pause_cycles, v),
                _ => {}
            }
        }
    }
    if let Some(last) = cur { profiles.push(last); }
    profiles
}
/// Look up a built-in AMD Zen/EPYC profile by exact section name (for example `"zen2_rome"`).
#[must_use]
pub fn profile_by_name(name: &str) -> Option<CpuProfile> {
    parse_cpu_specs(EMBEDDED_AMD_ZEN_SPEC).into_iter().find(|p| p.name == name)
}
#[derive(Default)]
struct ProcCpuInfo {
    vendor_id: String, family: u32, model: u32, cpu_cores: usize,
    socket_count: usize, has_avx2: bool, has_avx512f: bool,
}

fn parse_proc_cpuinfo(text: &str) -> ProcCpuInfo {
    let (mut info, mut phys_ids, mut primary) = (ProcCpuInfo::default(), Vec::<u32>::new(), false);
    for line in text.lines().map(str::trim) {
        if line.is_empty() { if !info.vendor_id.is_empty() { primary = true; } continue; }
        let Some((k, v)) = line.split_once(':') else { continue; };
        let (k, v) = (k.trim(), v.trim());
        if k == "physical id" {
            if let Ok(pid) = v.parse::<u32>() { if !phys_ids.contains(&pid) { phys_ids.push(pid); } }
        }
        if primary { continue; }
        match k {
            "vendor_id" => info.vendor_id = String::from(v),
            "cpu family" => parse_into(&mut info.family, v),
            "model" => parse_into(&mut info.model, v),
            "cpu cores" => parse_into(&mut info.cpu_cores, v),
            "flags" => for fl in v.split_whitespace() {
                if fl == "avx2" { info.has_avx2 = true; }
                else if fl == "avx512f" { info.has_avx512f = true; }
            },
            _ => {}
        }
    }
    info.socket_count = phys_ids.len().max(1);
    info
}
fn match_profile_from_cpuinfo(specs: &[CpuProfile], info: &ProcCpuInfo, logical: usize) -> CpuProfile {
    let logi = logical.max(1);
    let phys = if info.cpu_cores > 0 { info.cpu_cores.saturating_mul(info.socket_count) } else { logi / 2 }
        .max(1).min(logi);
    if info.vendor_id == "AuthenticAMD" || info.vendor_id == "GenuineIntel" {
        let fb = match (info.vendor_id.as_str(), info.family) {
            ("AuthenticAMD", 26) => "zen5_turin",
            ("AuthenticAMD", 25) if info.has_avx512f => "zen4_genoa",
            ("AuthenticAMD", 25) => "zen3_milan",
            ("AuthenticAMD", 23) => "zen2_rome",
            ("GenuineIntel", 6) if info.has_avx512f => "intel_icelake_sp",
            ("GenuineIntel", 6) if info.has_avx2 => "intel_skylake_sp",
            _ => "",
        };
        let found = specs
            .iter()
            .find(|p| p.vendor == info.vendor_id && p.family == info.family && p.models.contains(&info.model))
            .or_else(|| specs.iter().find(|p| p.name == fb));
        if let Some(p) = found {
            let mut m = p.clone();
            (m.family, m.logical_cpus, m.physical_cores) = (info.family, logi, phys);
            return m;
        }
    }
    let mut g = CpuProfile::generic(logi, phys);
    g.family = info.family;
    if !info.vendor_id.is_empty() { g.vendor = info.vendor_id.clone(); }
    if info.has_avx512f {
        (g.simd, g.vec_regs, g.vec_bits, g.native_pipe_bits) = (SimdTier::Avx512DoublePump256, 32, 512, 256);
    } else if info.has_avx2 {
        (g.simd, g.vec_regs, g.vec_bits, g.native_pipe_bits) = (SimdTier::Avx2Native256, 16, 256, 256);
    }
    g
}
/// Match a [`CpuProfile`] from raw `/proc/cpuinfo` text and logical CPU count (`no_std`).
#[must_use]
pub fn select_profile_from_cpuinfo(specs: &[CpuProfile], text: &str, logical: usize) -> CpuProfile {
    match_profile_from_cpuinfo(specs, &parse_proc_cpuinfo(text), logical)
}

static DEFAULT_HOST_PROFILE: CpuProfile = CpuProfile {
    name: String::new(), vendor: String::new(), family: 23, models: Vec::new(),
    l1d_kib: 32, l1d_ways: 8, l2_kib: 512, l3_ccx_mib: 16, l3_3dv_mib: 0,
    ccx_cores: 4, line_bytes: 64, simd: SimdTier::Avx2Native256, vec_regs: 16,
    vec_bits: 256, native_pipe_bits: 256, pause_cycles: 64, logical_cpus: 1, physical_cores: 1,
};

/// Return the static default [`CpuProfile`] (`no_std`, zero OS/filesystem dependencies).
#[must_use]
pub fn detect_host_profile() -> &'static CpuProfile { &DEFAULT_HOST_PROFILE }
/// Return the cached closed-form [`ZenTuningPlan`] for the running host CPU.
#[must_use] pub fn host_tuning_plan() -> ZenTuningPlan { detect_host_profile().tuning_plan() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_embedded_zen_epyc_profiles_and_host_plan() {
        let all = parse_cpu_specs(EMBEDDED_AMD_ZEN_SPEC);
        assert_eq!(all.len(), 10);
        let z = |n| profile_by_name(n).expect(n);
        let (z1, z2, z3) = (z("zen1_naples"), z("zen2_rome"), z("zen3_milan"));
        let (z4, z4c, z5) = (z("zen4_genoa"), z("zen4c_bergamo"), z("zen5_turin"));
        let (skx, icx) = (z("intel_skylake_sp"), z("intel_icelake_sp"));
        let (spr, gnr) = (z("intel_sapphire_rapids"), z("intel_granite_rapids"));
        assert_eq!((z1.vendor.as_str(), z1.family, z1.l1d_kib, z1.l2_kib, z1.simd, z1.pause_cycles),
            ("AuthenticAMD", 23, 32, 512, SimdTier::Avx2DoublePump128, 64));
        assert!(z2.models.contains(&113) && z2.l3_ccx_mib == 16 && z2.simd == SimdTier::Avx2Native256);
        assert_eq!((z3.l3_ccx_mib, z3.l3_3dv_mib, z3.ccx_cores), (32, 96, 8));
        assert_eq!((z4.l2_kib, z4.simd, z4.vec_regs, z4c.l3_ccx_mib, z4c.l2_kib),
            (1024, SimdTier::Avx512DoublePump256, 32, 16, 1024));
        assert_eq!((z5.family, z5.l1d_kib, z5.l1d_ways, z5.l2_kib, z5.simd, z5.native_pipe_bits),
            (26, 48, 12, 1024, SimdTier::Avx512Native512, 512));
        assert_eq!((skx.vendor.as_str(), skx.family, skx.l2_kib, skx.pause_cycles),
            ("GenuineIntel", 6, 1024, 140));
        assert_eq!((icx.l1d_kib, icx.l2_kib, icx.simd), (48, 1280, SimdTier::Avx512Native512));
        assert_eq!((spr.l2_kib, spr.l3_ccx_mib, gnr.l3_ccx_mib), (2048, 60, 64));
        assert_ne!(z2.spin_iters(), spr.spin_iters());
        let spr_info = parse_proc_cpuinfo(
            "vendor_id : GenuineIntel\ncpu family : 6\nmodel : 143\ncpu cores : 32\nflags : avx2 avx512f\n",
        );
        let matched_spr = match_profile_from_cpuinfo(&all, &spr_info, 64);
        assert_eq!((matched_spr.name.as_str(), matched_spr.physical_cores), ("intel_sapphire_rapids", 32));
        let (p1, p4, p5) = (z1.tuning_plan(), z4.tuning_plan(), z5.tuning_plan());
        assert_eq!((p1.l1_tile_bytes, p1.l2_block_bytes, p1.nc_o2_log2, p1.nc_o3_log2,
            p1.l2_hash4_bits, p1.l2_hash6_bits, p1.parallel_threshold_bytes),
            (16_384, 131_072, 9, 8, 15, 14, 262_144));
        assert_eq!((p4.l1_tile_bytes, p4.l2_block_bytes, p4.nc_o2_log2, p4.nc_o3_log2,
            p4.parallel_threshold_bytes), (16_384, 262_144, 10, 9, 524_288));
        assert_eq!((p5.l1_tile_bytes, p5.l2_block_bytes, p5.nc_o2_log2, p5.nc_o3_log2,
            p5.l2_hash4_bits, p5.l2_hash6_bits, p5.parallel_threshold_bytes),
            (24_576, 262_144, 10, 9, 16, 15, 524_288));
        assert_ne!(p1, p5);
        let (host, hp) = (detect_host_profile(), host_tuning_plan());
        assert!(host.l1d_kib >= 16 && host.l2_kib >= 128 && hp.l1_tile_bytes % 4096 == 0);
        assert!(hp.parallel_threshold_bytes == hp.l2_block_bytes * 2 && (1..=64).contains(&hp.max_threads));
        assert!(hp.spin_iters() >= 64);
    }
}
