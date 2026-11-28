# AMD Zen 1..5 / EPYC 7001..9005 & Intel Xeon Server Hardware Specification
# Pure 7-bit ASCII declarative table for zero-rans cache tiling and SIMD dispatch.
# Fields:
#   vendor           : CPUID vendor string (AuthenticAMD or GenuineIntel)
#   family           : CPUID family (decimal)
#   models           : comma-separated CPUID model numbers (decimal)
#   l1d_kib          : per-core L1 data cache size in KiB
#   l1d_ways         : L1 data cache set associativity
#   l2_kib           : per-core private L2 cache size in KiB
#   l3_ccx_mib       : base L3 cache size per CCX in MiB
#   l3_3dv_mib       : total L3 per CCD with 3D V-Cache in MiB (0 if none)
#   ccx_cores        : physical cores per CCX
#   line_bytes       : cache line size in bytes
#   simd             : avx2_128x2 | avx2 | avx512_256x2 | avx512_512
#   vec_regs         : architectural vector register count (16 or 32)
#   vec_bits         : architectural vector width in bits (256 or 512)
#   native_pipe_bits : native single-cycle vector execution pipe width in bits
#   pause_cycles     : approximate PAUSE instruction latency in cycles

[zen1_naples]
vendor = AuthenticAMD
family = 23
models = 1,8,17,24
l1d_kib = 32
l1d_ways = 8
l2_kib = 512
l3_ccx_mib = 8
l3_3dv_mib = 0
ccx_cores = 4
line_bytes = 64
simd = avx2_128x2
vec_regs = 16
vec_bits = 256
native_pipe_bits = 128
pause_cycles = 64

[zen2_rome]
vendor = AuthenticAMD
family = 23
models = 49,71,96,104,113,144
l1d_kib = 32
l1d_ways = 8
l2_kib = 512
l3_ccx_mib = 16
l3_3dv_mib = 0
ccx_cores = 4
line_bytes = 64
simd = avx2
vec_regs = 16
vec_bits = 256
native_pipe_bits = 256
pause_cycles = 64

[zen3_milan]
vendor = AuthenticAMD
family = 25
models = 1,8,33,64,68,80
l1d_kib = 32
l1d_ways = 8
l2_kib = 512
l3_ccx_mib = 32
l3_3dv_mib = 96
ccx_cores = 8
line_bytes = 64
simd = avx2
vec_regs = 16
vec_bits = 256
native_pipe_bits = 256
pause_cycles = 64

[zen4_genoa]
vendor = AuthenticAMD
family = 25
models = 17,97,116,120
l1d_kib = 32
l1d_ways = 8
l2_kib = 1024
l3_ccx_mib = 32
l3_3dv_mib = 96
ccx_cores = 8
line_bytes = 64
simd = avx512_256x2
vec_regs = 32
vec_bits = 512
native_pipe_bits = 256
pause_cycles = 64

[zen4c_bergamo]
vendor = AuthenticAMD
family = 25
models = 160
l1d_kib = 32
l1d_ways = 8
l2_kib = 1024
l3_ccx_mib = 16
l3_3dv_mib = 0
ccx_cores = 8
line_bytes = 64
simd = avx512_256x2
vec_regs = 32
vec_bits = 512
native_pipe_bits = 256
pause_cycles = 64

[zen5_turin]
vendor = AuthenticAMD
family = 26
models = 2,17,32,36,68
l1d_kib = 48
l1d_ways = 12
l2_kib = 1024
l3_ccx_mib = 32
l3_3dv_mib = 96
ccx_cores = 8
line_bytes = 64
simd = avx512_512
vec_regs = 32
vec_bits = 512
native_pipe_bits = 512
pause_cycles = 64

[intel_skylake_sp]
vendor = GenuineIntel
family = 6
models = 85
l1d_kib = 32
l1d_ways = 8
l2_kib = 1024
l3_ccx_mib = 32
l3_3dv_mib = 0
ccx_cores = 8
line_bytes = 64
simd = avx2
vec_regs = 16
vec_bits = 256
native_pipe_bits = 256
pause_cycles = 140

[intel_icelake_sp]
vendor = GenuineIntel
family = 6
models = 106,108
l1d_kib = 48
l1d_ways = 12
l2_kib = 1280
l3_ccx_mib = 48
l3_3dv_mib = 0
ccx_cores = 8
line_bytes = 64
simd = avx512_512
vec_regs = 32
vec_bits = 512
native_pipe_bits = 512
pause_cycles = 140

[intel_sapphire_rapids]
vendor = GenuineIntel
family = 6
models = 143,207
l1d_kib = 48
l1d_ways = 12
l2_kib = 2048
l3_ccx_mib = 60
l3_3dv_mib = 0
ccx_cores = 8
line_bytes = 64
simd = avx512_512
vec_regs = 32
vec_bits = 512
native_pipe_bits = 512
pause_cycles = 140

[intel_granite_rapids]
vendor = GenuineIntel
family = 6
models = 173,174
l1d_kib = 48
l1d_ways = 12
l2_kib = 2048
l3_ccx_mib = 64
l3_3dv_mib = 0
ccx_cores = 8
line_bytes = 64
simd = avx512_512
vec_regs = 32
vec_bits = 512
native_pipe_bits = 512
pause_cycles = 140
