#![forbid(unsafe_code)]

use std::io::Write;
use std::process::{Command, Stdio};

use zero_rans::{
    compress, decompress, encode_8way_with_scale, encode_order1_8way, nc_hash4_8, nc_hash4_u8,
    nc_hash_o2_8, nc_hash_o2_u8, nc_mix3_prob12_8, nc_nibble, ContextMap, ScaleBits,
};

struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

fn zstd19_len(data: &[u8]) -> Option<usize> {
    let mut child = Command::new("zstd")
        .args(["-19", "-c", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(data).ok()?;
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout.len())
}

/// Corpus NC-1 (`65,536 B`): Pure non-commutative Order-2 trigram process (`AB -> X`, `BA -> Y`).
///
/// Order-0 is near-uniform and Order-1 (`prev1`) sees identical marginal next-byte distributions
/// after every symbol, whereas non-commutative Order-2 `(prev2, prev1)` predicts the third byte
/// deterministically (`AB` and `BA` map to distinct targets).
fn synth_nc_trigram_corpus(len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(0xA1B2_C3D4_0001);
    let mut out = Vec::with_capacity(len);
    let alphabet: [u8; 8] = *b"ABCDEFGH";
    while out.len() + 3 <= len {
        let a_idx = rng.next_usize(8);
        let mut b_idx = rng.next_usize(7);
        if b_idx >= a_idx {
            b_idx += 1;
        }
        let a = alphabet[a_idx];
        let b = alphabet[b_idx];
        // Strictly non-commutative target: f(a, b) != f(b, a) for all distinct (a, b).
        let target = if a_idx < b_idx {
            b'a' + (((a_idx * 3 + b_idx * 5) & 15) as u8)
        } else {
            b'q' + (((a_idx * 7 + b_idx * 2) & 7) as u8)
        };
        out.push(a);
        out.push(b);
        out.push(target);
    }
    while out.len() < len {
        out.push(b'A');
    }
    out
}

/// Corpus NC-2 (`65,536 B`): Non-Abelian dihedral group `D_8` Cayley walk (`r^4 = s^2 = 1`, `s*r = r^3*s`).
///
/// Emits `(gen_t, state_t)` pairs where `state_t = state_{t-1} * gen_t`. Because `D_8` is
/// non-Abelian (`r*s != s*r`), `(state_{t-1}, gen_t)` uniquely determines `state_t`, while
/// Order-0 and Order-1 (`prev1 = gen_t`) remain completely uniform across all 8 group elements.
fn synth_nc_dihedral_d8_walk(len: usize) -> Vec<u8> {
    // Encode D_8 elements as `(rot in 0..4, flip in 0..2)` -> `0..8`.
    let mul_d8 = |lhs: u8, rhs: u8| -> u8 {
        let (r1, f1) = (lhs & 3, (lhs >> 2) & 1);
        let (r2, f2) = (rhs & 3, (rhs >> 2) & 1);
        let r = if f1 == 0 {
            (r1 + r2) & 3
        } else {
            (r1 + 4 - r2) & 3
        };
        let f = f1 ^ f2;
        (f << 2) | r
    };
    // Verify D_8 non-commutativity (`r * s != s * r`) right in the generator!
    let gen_r = 1u8;
    let gen_s = 4u8;
    assert_ne!(mul_d8(gen_r, gen_s), mul_d8(gen_s, gen_r));

    let gens: [u8; 4] = [1, 3, 4, 5]; // r, r^3, s, s*r
    let mut rng = SplitMix64::new(0xD800_CA1E_0002);
    let mut state = 0u8;
    let mut out = Vec::with_capacity(len);
    while out.len() + 2 <= len {
        let g = gens[rng.next_usize(gens.len())];
        state = mul_d8(state, g);
        out.push(b'g' + g);
        out.push(b'0' + state);
    }
    while out.len() < len {
        out.push(b'0');
    }
    out
}

/// Corpus NC-3 (`65,536 B`): Deep nested Dyck/AST delimiter language (`{`, `(`, `[`) with random
/// inner identifier payloads that break finite-order LZ/Markov matching across brackets.
fn synth_nc_dyck_ast(len: usize) -> Vec<u8> {
    let opens: [u8; 3] = [b'{', b'(', b'['];
    let closes: [u8; 3] = [b'}', b')', b']'];
    let payload_alpha: &[u8] = b"abcdefghijklmnopqrstuvwxyz012345";
    let mut rng = SplitMix64::new(0xD1C4_A570_0003);
    let mut stack: Vec<usize> = Vec::with_capacity(16);
    let mut out = Vec::with_capacity(len);

    while out.len() + 16 < len {
        let do_push = stack.is_empty() || (stack.len() < 10 && rng.next_usize(10) < 5);
        if do_push {
            let kind = rng.next_usize(3);
            out.push(opens[kind]);
            stack.push(kind);
            let inner = 2 + rng.next_usize(4);
            for _ in 0..inner {
                out.push(payload_alpha[rng.next_usize(payload_alpha.len())]);
            }
        } else if let Some(kind) = stack.pop() {
            out.push(closes[kind]);
        }
    }
    while let Some(kind) = stack.pop() {
        if out.len() < len {
            out.push(closes[kind]);
        }
    }
    while out.len() < len {
        out.push(b' ');
    }
    out
}

/// Corpus NC-4 (`65,536 B`): Non-commutative stack bytecode (`PUSH rA; PUSH rB; SUB/DIV; TAG`)
/// where operand order `(rA, rB)` vs `(rB, rA)` deterministically selects the result tag byte.
fn synth_nc_stack_bytecode(len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(0xB17E_C0DE_0004);
    let mut out = Vec::with_capacity(len);
    while out.len() + 4 <= len {
        let ra = rng.next_usize(8) as u8;
        let rb = rng.next_usize(8) as u8;
        let op = if ra > rb {
            0x40u8 // SUB_POS
        } else if ra < rb {
            0x41u8 // SUB_NEG
        } else {
            0x42u8 // EQ_ZERO
        };
        let dst = ((ra.wrapping_mul(5).wrapping_add(rb.wrapping_mul(3))) & 7) | 0x80;
        out.push(0x10 | ra);
        out.push(0x20 | rb);
        out.push(op);
        out.push(dst);
    }
    while out.len() < len {
        out.push(0x00);
    }
    out
}

/// Emit a concrete byte block exercising one of 5 non-commuting `BRL2` codec state-machine events.
fn append_codec_event(buf: &mut Vec<u8>, ev: usize, salt: u8) {
    match ev {
        // E0: fresh literal byte (`lit_run: 0 -> 1`, breaks zero-lit repcode shift).
        0 => buf.push(b'!' + (salt % 90)),
        // E1: short repeat match (length 6 at offset 12, tests rep0 / active_rep_offsets).
        1 => {
            if buf.len() >= 12 {
                let start = buf.len() - 12;
                for k in 0..6usize {
                    let b = buf[start + k];
                    buf.push(b);
                }
            } else {
                buf.extend_from_slice(b"rep0__");
            }
        }
        // E2: second-distance repeat match (offset 24, swaps MTF `reps[0] <-> reps[1]`).
        2 => {
            if buf.len() >= 24 {
                let start = buf.len() - 24;
                for k in 0..8usize {
                    let b = buf[start + k];
                    buf.push(b);
                }
            } else {
                buf.extend_from_slice(b"rep1_swap");
            }
        }
        // E3: `>= 256 B` match triggering `fast_jump` + partial hash-chain warmup.
        3 => {
            if buf.len() >= 32 {
                let start = buf.len() - 32;
                for k in 0..272usize {
                    let b = buf[start + (k & 31)];
                    buf.push(b);
                }
            } else {
                for k in 0..272usize {
                    buf.push(b'A' + ((k as u8) & 15));
                }
            }
        }
        // E4: nested non-commutative delimiter frame (`{a(b[c])}`).
        _ => {
            let c = b'a' + (salt % 26);
            buf.extend_from_slice(&[b'{', c, b'(', c.wrapping_add(1), b'[', b']', b')', b'}']);
        }
    }
}

#[test]
fn commutator_and_ordered_twise_codec_fuzz() {
    // 1. Exhaustive uniformity & non-commutativity law on `nc_hash_o2_u8` / `nc_hash_o2_8`.
    let mut buckets = [0u32; 256];
    let mut non_comm_pairs = 0usize;
    let mut total_pairs = 0usize;
    for a in 0..=255u8 {
        for b in 0..=255u8 {
            let h = nc_hash_o2_u8(a, b);
            buckets[usize::from(h)] += 1;
            if a < b {
                total_pairs += 1;
                if h != nc_hash_o2_u8(b, a) {
                    non_comm_pairs += 1;
                }
            }
        }
    }
    for (idx, &cnt) in buckets.iter().enumerate() {
        assert_eq!(cnt, 256, "nc_hash_o2 bucket {idx} must be exact 256/256 uniform");
    }
    assert!(
        non_comm_pairs * 1000 > total_pairs * 995,
        "nc_hash_o2 must separate >99.5% of unordered pairs: {non_comm_pairs}/{total_pairs}"
    );

    // 2. Bit-exact 8-lane SIMD kernel parity (`nc_hash_o2_8`, `nc_hash4_8`, `nc_mix3_prob12_8`)
    //    plus counterfactual non-commutativity and mixer weight knockouts (`assert_ne!`).
    let mut rng = SplitMix64::new(0xC044_07A7_0001);
    for _ in 0..256 {
        let (mut p2, mut p1, mut b2, mut b3) = ([0i32; 8], [0i32; 8], [0i32; 8], [0i32; 8]);
        let (mut po0, mut po1, mut po2, mut w2) = ([0i32; 8], [0i32; 8], [0i32; 8], [0i32; 8]);
        for k in 0..8usize {
            p2[k] = (rng.next_u64() & 255) as i32;
            p1[k] = (rng.next_u64() & 255) as i32;
            b2[k] = (rng.next_u64() & 255) as i32;
            b3[k] = (rng.next_u64() & 255) as i32;
            po0[k] = 1 + (rng.next_usize(4095) as i32);
            po1[k] = 1 + (rng.next_usize(4095) as i32);
            po2[k] = 1 + (rng.next_usize(4095) as i32);
            w2[k] = rng.next_usize(25) as i32;
        }
        let (mut o_h2, mut o_h4, mut o_mix) = ([0i32; 8], [0i32; 8], [0i32; 8]);
        nc_hash_o2_8(&p2, &p1, &mut o_h2);
        nc_hash4_8(&p2, &p1, &b2, &b3, &mut o_h4);
        nc_mix3_prob12_8(&po0, &po1, &po2, &w2, &mut o_mix);
        for k in 0..8usize {
            assert_eq!(o_h2[k], i32::from(nc_hash_o2_u8(p2[k] as u8, p1[k] as u8)));
            assert_eq!(
                o_h4[k] as usize,
                nc_hash4_u8(p2[k] as u8, p1[k] as u8, b2[k] as u8, b3[k] as u8)
            );
            assert!((1..=4095).contains(&o_mix[k]));
        }
    }
    assert_ne!(
        nc_hash4_u8(b'A', b'B', b'C', b'D'),
        nc_hash4_u8(b'B', b'A', b'C', b'D'),
        "nc_hash4 must be strictly non-commutative"
    );
    assert_ne!(
        nc_hash4_u8(b'A', b'B', b'C', b'D'),
        nc_hash4_u8(b'D', b'C', b'B', b'A'),
        "nc_hash4 must distinguish byte reversals"
    );
    let mut mix_live = [0i32; 8];
    let mut mix_zero = [0i32; 8];
    nc_mix3_prob12_8(&[1000; 8], &[1200; 8], &[3900; 8], &[24; 8], &mut mix_live);
    nc_mix3_prob12_8(&[1000; 8], &[1200; 8], &[3900; 8], &[0; 8], &mut mix_zero);
    assert_ne!(mix_live, mix_zero, "nc_mix3_prob12_8 weight knockout must change output");

    // 3. Exhaustive ordered `t = 3` covering array over all `5^3 = 125` ordered event triples!
    let mut salt = 0u8;
    for e1 in 0..5usize {
        for e2 in 0..5usize {
            for e3 in 0..5usize {
                let mut payload = b"seed_prefix_0123456789abcdef_SEED_PREFIX_0123456789abcdef"
                    .to_vec();
                append_codec_event(&mut payload, e1, salt);
                append_codec_event(&mut payload, e2, salt.wrapping_add(7));
                append_codec_event(&mut payload, e3, salt.wrapping_add(19));
                // Repeat the ordered triple once to exercise both cold and warm adaptive states.
                append_codec_event(&mut payload, e1, salt.wrapping_add(3));
                append_codec_event(&mut payload, e2, salt.wrapping_add(11));
                append_codec_event(&mut payload, e3, salt.wrapping_add(23));
                salt = salt.wrapping_add(1);

                let wire = compress(&payload).expect("compress ordered t=3 tuple");
                let dec = decompress(&wire).expect("decompress ordered t=3 tuple");
                assert_eq!(dec, payload, "roundtrip mismatch on ordered tuple ({e1},{e2},{e3})");

                // Also test exact reversal (non-commutative order flip).
                let mut rev = payload.clone();
                rev.reverse();
                let wire_r = compress(&rev).expect("compress reversed tuple");
                let dec_r = decompress(&wire_r).expect("decompress reversed tuple");
                assert_eq!(dec_r, rev, "roundtrip mismatch on reversed tuple ({e1},{e2},{e3})");
            }
        }
    }

    // 4. Fail-closed single-bit tamper check across an NC-compressed frame.
    let nc_sample = synth_nc_trigram_corpus(4096);
    let wire = compress(&nc_sample).expect("compress nc_sample");
    assert_eq!(decompress(&wire).expect("decompress nc_sample"), nc_sample);
    for flip_idx in [0usize, 4, 5, 10, wire.len() / 2, wire.len() - 1] {
        let mut tampered = wire.clone();
        tampered[flip_idx] ^= 0x01;
        assert!(
            decompress(&tampered).is_err(),
            "single-bit flip at offset {flip_idx} must fail closed"
        );
    }
}

#[test]
fn non_commutative_corpora_knockout_and_zstd19_dominance() {
    let corpora: [(&str, Vec<u8>); 4] = [
        ("NC-1 Trigram (AB->X / BA->Y)", synth_nc_trigram_corpus(65_536)),
        ("NC-2 Non-Abelian D_8 Walk", synth_nc_dihedral_d8_walk(65_536)),
        ("NC-3 Nested Dyck/AST Stack", synth_nc_dyck_ast(65_536)),
        ("NC-4 Stack Bytecode (SUB/DIV)", synth_nc_stack_bytecode(65_536)),
    ];

    let mut rng = SplitMix64::new(0x5A0F_F1E0_9999);
    for (name, data) in corpora {
        let wire = compress(&data).expect("compress nc corpus");
        let dec = decompress(&wire).expect("decompress nc corpus");
        assert_eq!(dec, data, "{name}: bit-exact roundtrip failed");

        // Counterfactual knockout 1: destroying sequential order via Fisher-Yates shuffle
        // must strictly worsen compression (proving order/non-commutativity sensitivity!).
        let mut shuffled = data.clone();
        for i in (1..shuffled.len()).rev() {
            let j = rng.next_usize(i + 1);
            shuffled.swap(i, j);
        }
        let wire_shuf = compress(&shuffled).expect("compress shuffled");
        assert_ne!(wire.len(), wire_shuf.len(), "{name}: shuffle knockout must change size");
        assert!(
            wire.len() < wire_shuf.len(),
            "{name}: ordered corpus ({} B) must strictly beat shuffled ({} B)",
            wire.len(),
            wire_shuf.len()
        );

        // Counterfactual knockout 2: must strictly beat pure Order-0 and Order-1 rANS.
        let o0 = encode_8way_with_scale(&data, ScaleBits::Bits14).expect("o0");
        let o1 = encode_order1_8way(&data, ContextMap::Low15).expect("o1");
        assert!(
            wire.len() < o0.len() && wire.len() < o1.len(),
            "{name}: BRL2 ({} B) must strictly beat Order-0 ({} B) and Order-1 ({} B)",
            wire.len(),
            o0.len(),
            o1.len()
        );

        if let Some(z19) = zstd19_len(&data) {
            eprintln!(
                "{name:32} | raw={} B | zero-rans={} B ({:.2}x) | zstd-19={} B ({:+.1}%) | o1={} B | shuf={} B",
                data.len(),
                wire.len(),
                data.len() as f64 / wire.len() as f64,
                z19,
                100.0 * (wire.len() as f64 - z19 as f64) / z19 as f64,
                o1.len(),
                wire_shuf.len()
            );
            assert!(
                wire.len() < z19,
                "{name}: zero-rans ({} B) must beat zstd -19 ({} B)",
                wire.len(),
                z19
            );
        }
    }
}

#[test]
fn nc_nibble_cross_profile_and_knockout_laws() {
    let corpora: [(&str, Vec<u8>); 4] = [
        ("NC-1 Trigram (AB->X / BA->Y)", synth_nc_trigram_corpus(65_536)),
        ("NC-2 Non-Abelian D_8 Walk", synth_nc_dihedral_d8_walk(65_536)),
        ("NC-3 Nested Dyck/AST Stack", synth_nc_dyck_ast(65_536)),
        ("NC-4 Stack Bytecode (SUB/DIV)", synth_nc_stack_bytecode(65_536)),
    ];

    let mut rng = SplitMix64::new(0x4E49_BB1E_2026);
    for (name, data) in &corpora {
        // Cross-profile bit-exactness: Zen 1-3 (o2=9, o3=8) and Zen 4-5 (o2=10, o3=9).
        let enc_z2 = nc_nibble::encode_adaptive_nc_nibble(data, 9, 8);
        let dec_z2 = nc_nibble::decode_adaptive_nc_nibble(&enc_z2, data.len()).expect("dec z2");
        assert_eq!(&dec_z2, data, "{name}: Zen 1-3 nibble roundtrip mismatch");

        let enc_z4 = nc_nibble::encode_adaptive_nc_nibble(data, 10, 9);
        let dec_z4 = nc_nibble::decode_adaptive_nc_nibble(&enc_z4, data.len()).expect("dec z4");
        assert_eq!(&dec_z4, data, "{name}: Zen 4-5 nibble roundtrip mismatch");

        // Header-clamped bounds: values outside 6..=12 / 6..=11 clamp safely on encode.
        let enc_lo = nc_nibble::encode_adaptive_nc_nibble(&data[..2048], 2, 2);
        assert_eq!((enc_lo[0], enc_lo[1]), (6, 6));
        assert_eq!(
            nc_nibble::decode_adaptive_nc_nibble(&enc_lo, 2048).expect("dec lo"),
            &data[..2048]
        );
        let enc_hi = nc_nibble::encode_adaptive_nc_nibble(&data[..2048], 20, 20);
        assert_eq!((enc_hi[0], enc_hi[1]), (12, 11));
        assert_eq!(
            nc_nibble::decode_adaptive_nc_nibble(&enc_hi, 2048).expect("dec hi"),
            &data[..2048]
        );

        // Counterfactual knockout: Fisher-Yates shuffle must strictly worsen nibble size.
        let mut shuffled = data.clone();
        for i in (1..shuffled.len()).rev() {
            let j = rng.next_usize(i + 1);
            shuffled.swap(i, j);
        }
        let enc_shuf = nc_nibble::encode_adaptive_nc_nibble(&shuffled, 9, 8);
        assert_ne!(enc_z2.len(), enc_shuf.len());
        assert!(
            enc_z2.len() < enc_shuf.len(),
            "{name}: nibble ordered ({} B) must beat shuffled ({} B)",
            enc_z2.len(),
            enc_shuf.len()
        );

        // Single-bit header/body tamper fail-closed check.
        let mut bad_hdr = enc_z2.clone();
        bad_hdr[0] = 5;
        assert!(nc_nibble::decode_adaptive_nc_nibble(&bad_hdr, data.len()).is_err());
        let mut bad_tail = enc_z2.clone();
        bad_tail.push(0x42);
        assert!(nc_nibble::decode_adaptive_nc_nibble(&bad_tail, data.len()).is_err());

        if let Some(z19) = zstd19_len(data) {
            assert!(
                enc_z2.len() < z19 && enc_z4.len() < z19,
                "{name}: nibble z2={} B / z4={} B must beat zstd -19 ({} B)",
                enc_z2.len(),
                enc_z4.len(),
                z19
            );
        }
    }
}
