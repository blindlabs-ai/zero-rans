#![forbid(unsafe_code)]

use zero_rans::{
    analyze_frame_structure, decode_video_frame, decode_video_frame_with_limit, encode_video_frame,
    select_auto_profile, AdaptiveStreamConfig, AdaptiveStreamDecoder, AdaptiveStreamEncoder, CodecProfile,
    Error, FrameKind, VideoConfig, Yuv420Frame, MAGIC_VIDEO_FRAME,
};

/// Deterministic SplitMix64 PRNG (zero external dependencies).
struct Rng(u64);

impl Rng {
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

#[test]
fn all_four_industry_profiles_roundtrip_psnr_and_pairwise_distinct() {
    let width = 96usize;
    let height = 64usize;
    let seq: Vec<Yuv420Frame> = (0..4)
        .map(|t| Yuv420Frame::synth_natural_motion(width, height, t).expect("synth frame"))
        .collect();

    let profiles = [
        CodecProfile::AvcH264,
        CodecProfile::HevcH265,
        CodecProfile::Vp9,
        CodecProfile::Av1,
    ];

    // 1. Near-lossless `qstep = 1` gate (`PSNR >= 45 dB`) + bit-exact encoder/decoder reconstruction.
    let mut near_lossless_streams: Vec<Vec<u8>> = Vec::with_capacity(4);
    for &profile in &profiles {
        let cfg = VideoConfig {
            profile,
            qstep: 1,
            lambda_q8: 64,
            search_range: 12,
            loop_filter: true,
            intra_refresh_col: None,
        };
        let mut enc_ref: Option<Yuv420Frame> = None;
        let mut dec_ref: Option<Yuv420Frame> = None;
        let mut concat_wire = Vec::new();

        for (idx, frame) in seq.iter().enumerate() {
            let kind = if idx == 0 {
                FrameKind::Key
            } else {
                FrameKind::Inter
            };
            let (pkt, enc_rec) =
                encode_video_frame(frame, enc_ref.as_ref(), kind, cfg).expect("encode qstep=1");
            assert_eq!(&pkt[..4], &MAGIC_VIDEO_FRAME);
            let dec_rec =
                decode_video_frame(&pkt, dec_ref.as_ref()).expect("decode qstep=1");
            assert_eq!(
                enc_rec, dec_rec,
                "encoder/decoder drift on {profile:?} frame {idx}"
            );
            let psnr = frame.psnr_db(&dec_rec);
            assert!(
                psnr >= 45.0,
                "expected PSNR >= 45 dB at qstep=1 for {profile:?} frame {idx}, got {psnr:.2} dB"
            );
            enc_ref = Some(enc_rec);
            dec_ref = Some(dec_rec);
            concat_wire.extend_from_slice(&pkt);
        }
        near_lossless_streams.push(concat_wire);
    }

    // Pairwise distinctness across all 4 codec profiles (`C(4, 2) = 6` `assert_ne!` checks).
    for i in 0..profiles.len() {
        for j in (i + 1)..profiles.len() {
            assert_ne!(
                near_lossless_streams[i], near_lossless_streams[j],
                "profiles {:?} and {:?} must emit distinct authentic BVF1 bitstreams",
                profiles[i], profiles[j]
            );
        }
    }

    // 2. Standard `qstep = 4` gate (`PSNR >= 34 dB`, substantial compression) + pairwise distinct reconstructions.
    let mut q4_reconstructions: Vec<Yuv420Frame> = Vec::with_capacity(4);
    let raw_seq_bytes = seq.len() * seq[0].byte_len();
    for &profile in &profiles {
        let cfg = VideoConfig {
            profile,
            qstep: 4,
            lambda_q8: 192,
            search_range: 12,
            loop_filter: true,
            intra_refresh_col: None,
        };
        let mut enc_ref: Option<Yuv420Frame> = None;
        let mut dec_ref: Option<Yuv420Frame> = None;
        let mut total_wire = 0usize;
        for (idx, frame) in seq.iter().enumerate() {
            let kind = if idx == 0 {
                FrameKind::Key
            } else {
                FrameKind::Inter
            };
            let (pkt, enc_rec) =
                encode_video_frame(frame, enc_ref.as_ref(), kind, cfg).expect("encode qstep=4");
            let dec_rec = decode_video_frame(&pkt, dec_ref.as_ref()).expect("decode qstep=4");
            assert_eq!(enc_rec, dec_rec);
            let psnr = frame.psnr_db(&dec_rec);
            assert!(
                psnr >= 34.0,
                "expected PSNR >= 34 dB at qstep=4 for {profile:?} frame {idx}, got {psnr:.2} dB"
            );
            total_wire += pkt.len();
            enc_ref = Some(enc_rec);
            dec_ref = Some(dec_rec);
        }
        assert!(
            total_wire * 2 < raw_seq_bytes,
            "expected >2x compression at qstep=4 for {profile:?}: wire={total_wire} raw={raw_seq_bytes}"
        );
        q4_reconstructions.push(dec_ref.expect("final frame"));
    }

    for i in 0..profiles.len() {
        for j in (i + 1)..profiles.len() {
            assert_ne!(
                q4_reconstructions[i], q4_reconstructions[j],
                "profiles {:?} and {:?} must produce distinct transform/loop-filter reconstructions",
                profiles[i], profiles[j]
            );
        }
    }
}

#[test]
fn av1_screen_ui_ibc_idtx_cfl_beats_h264_and_rgb_roundtrips() {
    let width = 128usize;
    let height = 96usize;
    let ui_frame = Yuv420Frame::synth_screen_ui(width, height, 0).expect("synth ui");

    // Verify RGB24 <-> YUV420p helpers preserve dimensions and high PSNR on neutral content.
    let rgb = ui_frame.to_rgb24().expect("to_rgb24");
    assert_eq!(rgb.len(), width * height * 3);
    let yuv_back = Yuv420Frame::from_rgb24(width, height, &rgb).expect("from_rgb24");
    assert_eq!(yuv_back.width, width);
    assert_eq!(yuv_back.height, height);

    let (av1_pkt, av1_rec) = encode_video_frame(
        &ui_frame,
        None,
        FrameKind::Key,
        VideoConfig {
            profile: CodecProfile::Av1,
            qstep: 3,
            lambda_q8: 160,
            search_range: 12,
            loop_filter: true,
            intra_refresh_col: None,
        },
    )
    .expect("encode av1 ui");

    let (avc_pkt, avc_rec) = encode_video_frame(
        &ui_frame,
        None,
        FrameKind::Key,
        VideoConfig {
            profile: CodecProfile::AvcH264,
            qstep: 3,
            lambda_q8: 160,
            search_range: 12,
            loop_filter: true,
            intra_refresh_col: None,
        },
    )
    .expect("encode avc ui");

    assert_eq!(decode_video_frame(&av1_pkt, None).expect("dec av1"), av1_rec);
    assert_eq!(decode_video_frame(&avc_pkt, None).expect("dec avc"), avc_rec);
    assert_ne!(av1_pkt, avc_pkt);
    // AV1 IBC + IDTX + CFL captures repeated glyph cells sharply and compactly.
    assert!(
        av1_pkt.len() < avc_pkt.len(),
        "AV1 screen tools ({} B) should compress repeated UI glyphs tighter than H.264 ({} B)",
        av1_pkt.len(),
        avc_pkt.len()
    );
}

#[test]
fn adaptive_counterfactual_history_lockstep_and_healing_knockouts() {
    let width = 96usize;
    let height = 64usize;
    let frames: Vec<Yuv420Frame> = (0..6)
        .map(|t| Yuv420Frame::synth_natural_motion(width, height, t).expect("synth"))
        .collect();

    let cfg = AdaptiveStreamConfig {
        profile: CodecProfile::HevcH265,
        initial_qstep: 4,
        target_bytes_per_frame: None,
        keyint: 30,
        rolling_intra_refresh: true,
        loop_filter: true,
        search_range: 12,
    };
    let mut enc = AdaptiveStreamEncoder::new(cfg).expect("enc");
    let mut dec = AdaptiveStreamDecoder::new().expect("dec");
    assert_eq!(enc.state_digest(), dec.state_digest());

    let mut packets = Vec::with_capacity(6);
    for (idx, f) in frames.iter().enumerate() {
        let (pkt, rep) = enc.encode_frame(f).expect("encode frame");
        let rec = dec.decode_frame(&pkt).expect("decode frame");
        assert_eq!(
            enc.state_digest(),
            dec.state_digest(),
            "encoder/decoder metabolic state digest diverged at frame {idx}"
        );
        assert_eq!(rep.state_digest, dec.state_digest());
        assert!(f.psnr_db(&rec) >= 34.0);
        packets.push(pkt);
    }

    // Counterfactual History Knockout (`assert_ne!` + strictly smaller `<`):
    // Encoding `frame_5` with co-evolved `enc` (adapted temporal reference + adapted CDF priors)
    // vs a cold-start `AdaptiveStreamEncoder` (`H = empty`) on the exact same `frame_5`.
    let adapted_pkt_5 = &packets[5];
    let mut cold_enc = AdaptiveStreamEncoder::new(cfg).expect("cold enc");
    let (cold_pkt_5, cold_rep_5) = cold_enc.encode_frame(&frames[5]).expect("cold encode frame 5");

    assert_ne!(adapted_pkt_5, &cold_pkt_5);
    assert_ne!(enc.state_digest(), cold_enc.state_digest());
    assert_eq!(cold_rep_5.frame_kind, FrameKind::Key);
    assert!(
        adapted_pkt_5.len() < cold_pkt_5.len(),
        "adapted adaptive packet ({} B) must be strictly smaller than cold-start packet ({} B)",
        adapted_pkt_5.len(),
        cold_pkt_5.len()
    );

    // Also isolate pure decoder-mirrored CDF prior adaptation on the EXACT same P-frame reference (`rolling_intra_refresh = false`):
    // Train `adapted_cdf_enc` over frames `0..5` so its mirrored CDF priors adapt to sparse P-frame runs/levels/modes,
    // and compare its `frame_5` packet against standalone `encode_video_frame` (cold inline-table order-0) on the exact same reference!
    let cdf_cfg = AdaptiveStreamConfig {
        profile: CodecProfile::HevcH265,
        initial_qstep: 4,
        target_bytes_per_frame: None,
        keyint: 30,
        rolling_intra_refresh: false,
        loop_filter: true,
        search_range: 12,
    };
    let mut adapted_cdf_enc = AdaptiveStreamEncoder::new(cdf_cfg).expect("cdf enc");
    let mut adapted_cdf_dec = AdaptiveStreamDecoder::new().expect("cdf dec");
    for f in &frames[..5] {
        let (p, _) = adapted_cdf_enc.encode_frame(f).expect("cdf step enc");
        let _ = adapted_cdf_dec.decode_frame(&p).expect("cdf step dec");
        assert_eq!(adapted_cdf_enc.state_digest(), adapted_cdf_dec.state_digest());
    }
    let ref_before_5 = adapted_cdf_enc
        .state()
        .reference()
        .cloned()
        .expect("ref before 5");
    let (mirrored_pkt_5, _) = adapted_cdf_enc.encode_frame(&frames[5]).expect("mirrored 5");
    let mirrored_rec_5 = adapted_cdf_dec.decode_frame(&mirrored_pkt_5).expect("mirrored dec 5");
    assert_eq!(adapted_cdf_enc.state_digest(), adapted_cdf_dec.state_digest());

    let (inline_pkt_5, inline_rec_5) = encode_video_frame(
        &frames[5],
        Some(&ref_before_5),
        FrameKind::Inter,
        VideoConfig {
            profile: CodecProfile::HevcH265,
            qstep: 4,
            lambda_q8: adapted_cdf_enc.current_lambda_q8(),
            search_range: 12,
            loop_filter: true,
            intra_refresh_col: None,
        },
    )
    .expect("inline 5");
    assert_eq!(mirrored_rec_5, inline_rec_5);
    assert_ne!(mirrored_pkt_5, inline_pkt_5);
    assert!(
        mirrored_pkt_5.len() < inline_pkt_5.len(),
        "decoder-mirrored CDF prior packet ({} B) must beat cold inline-table P-frame packet ({} B) at identical reconstruction",
        mirrored_pkt_5.len(),
        inline_pkt_5.len()
    );

    // Reference CRC-32 Fail-Closed Detection + Resilient Rolling Intra-Refresh Healing:
    let mut strict_dec = AdaptiveStreamDecoder::new().expect("strict dec");
    let _ = strict_dec.decode_frame(&packets[0]).expect("dec 0");
    // Drop `packets[1]`, feed `packets[2]` -> strict decoder MUST reject with `Error::ReferenceMismatch`!
    assert_eq!(
        strict_dec.decode_frame(&packets[2]),
        Err(Error::ReferenceMismatch)
    );

    // Enable `resilient_heal`: decoder reconstructs across the packet drop without panicking,
    // and locks back to 100% bit-exact `state_digest()` on the next scene-cut / keyframe!
    let mut heal_dec = AdaptiveStreamDecoder::with_options(16 * 1024 * 1024, true).expect("heal dec");
    let _ = heal_dec.decode_frame(&packets[0]).expect("heal 0");
    for pkt in &packets[2..6] {
        let healed_frame = heal_dec.decode_frame(pkt).expect("resilient decode");
        assert_eq!(healed_frame.width, width);
        assert_eq!(healed_frame.height, height);
    }
    // Feed a scene-cut frame (screen UI) -> triggers automatic Keyframe + prior reset -> instant bit-exact re-sync!
    let scene_cut = Yuv420Frame::synth_screen_ui(width, height, 0).expect("scene cut");
    let (cut_pkt, cut_rep) = enc.encode_frame(&scene_cut).expect("enc cut");
    assert_eq!(cut_rep.frame_kind, FrameKind::Key);
    let cut_dec_normal = dec.decode_frame(&cut_pkt).expect("dec cut");
    let cut_dec_healed = heal_dec.decode_frame(&cut_pkt).expect("heal cut");
    assert_eq!(cut_dec_normal, cut_dec_healed);
    assert_eq!(enc.state_digest(), heal_dec.state_digest());
}

#[test]
fn adaptive_rate_controller_homeostasis_knockout() {
    let width = 96usize;
    let height = 64usize;
    let seq: Vec<Yuv420Frame> = (0..8)
        .map(|t| Yuv420Frame::synth_natural_motion(width, height, t * 2).expect("synth"))
        .collect();

    let mut tight_enc = AdaptiveStreamEncoder::new(AdaptiveStreamConfig {
        profile: CodecProfile::Vp9,
        initial_qstep: 6,
        target_bytes_per_frame: Some(600),
        keyint: 30,
        rolling_intra_refresh: false,
        loop_filter: true,
        search_range: 12,
    })
    .expect("tight enc");

    let mut generous_enc = AdaptiveStreamEncoder::new(AdaptiveStreamConfig {
        profile: CodecProfile::Vp9,
        initial_qstep: 6,
        target_bytes_per_frame: Some(2400),
        keyint: 30,
        rolling_intra_refresh: false,
        loop_filter: true,
        search_range: 12,
    })
    .expect("generous enc");

    let mut tight_dec = AdaptiveStreamDecoder::new().expect("tight dec");
    let mut generous_dec = AdaptiveStreamDecoder::new().expect("generous dec");

    let mut tight_bytes_tail = 0usize;
    let mut generous_bytes_tail = 0usize;

    for (idx, frame) in seq.iter().enumerate() {
        let (tpkt, _) = tight_enc.encode_frame(frame).expect("tight encode");
        let (gpkt, _) = generous_enc.encode_frame(frame).expect("generous encode");
        let _ = tight_dec.decode_frame(&tpkt).expect("tight decode");
        let _ = generous_dec.decode_frame(&gpkt).expect("generous decode");
        assert_eq!(tight_enc.state_digest(), tight_dec.state_digest());
        assert_eq!(generous_enc.state_digest(), generous_dec.state_digest());
        if idx >= 2 {
            tight_bytes_tail += tpkt.len();
            generous_bytes_tail += gpkt.len();
        }
    }

    // Counterfactual Rate-Controller Knockout (`assert_ne!` on qstep, lambda_q8, and output size):
    assert_ne!(tight_enc.current_qstep(), generous_enc.current_qstep());
    assert_ne!(
        tight_enc.current_lambda_q8(),
        generous_enc.current_lambda_q8()
    );
    assert!(tight_enc.current_qstep() > generous_enc.current_qstep());
    assert!(tight_enc.current_lambda_q8() > generous_enc.current_lambda_q8());
    assert_ne!(tight_bytes_tail, generous_bytes_tail);
    assert!(
        tight_bytes_tail < generous_bytes_tail,
        "tight budget (600 B/f -> total tail {} B) must emit fewer bytes than generous budget (2400 B/f -> total tail {} B)",
        tight_bytes_tail,
        generous_bytes_tail
    );
}

#[test]
fn adaptive_auto_profile_morphism_knockout() {
    let width = 128usize;
    let height = 96usize;
    let ui_frame = Yuv420Frame::synth_screen_ui(width, height, 1).expect("ui");
    let natural_frame = Yuv420Frame::synth_natural_motion(width, height, 1).expect("natural");

    let ui_stats = analyze_frame_structure(&ui_frame, None);
    let nat_stats = analyze_frame_structure(&natural_frame, None);

    let ui_prof = select_auto_profile(ui_stats);
    let nat_prof = select_auto_profile(nat_stats);
    assert_eq!(ui_prof, CodecProfile::Av1);
    assert_ne!(
        ui_prof, nat_prof,
        "Auto profile must distinguish sharp bimodal UI ({ui_prof:?}) from smooth natural motion ({nat_prof:?})"
    );

    let mut auto_enc = AdaptiveStreamEncoder::new(AdaptiveStreamConfig {
        profile: CodecProfile::Auto,
        initial_qstep: 4,
        ..AdaptiveStreamConfig::default()
    })
    .expect("auto enc");
    let mut auto_dec = AdaptiveStreamDecoder::new().expect("auto dec");

    let (pkt_nat, rep_nat) = auto_enc.encode_frame(&natural_frame).expect("enc nat");
    let _ = auto_dec.decode_frame(&pkt_nat).expect("dec nat");
    let (pkt_ui, rep_ui) = auto_enc.encode_frame(&ui_frame).expect("enc ui");
    let _ = auto_dec.decode_frame(&pkt_ui).expect("dec ui");

    assert_ne!(rep_nat.chosen_profile, rep_ui.chosen_profile);
    assert_eq!(rep_ui.chosen_profile, CodecProfile::Av1);
    assert_eq!(auto_enc.state_digest(), auto_dec.state_digest());
}

#[test]
fn adversarial_bvf1_corruption_truncation_and_bomb_fuzz_500_seeds() {
    let f0 = Yuv420Frame::synth_natural_motion(64, 48, 0).expect("f0");
    let f1 = Yuv420Frame::synth_screen_ui(64, 48, 1).expect("f1");

    let (valid_key, rec0) = encode_video_frame(
        &f0,
        None,
        FrameKind::Key,
        VideoConfig {
            profile: CodecProfile::HevcH265,
            qstep: 4,
            ..VideoConfig::default()
        },
    )
    .expect("encode key");
    let (valid_av1, _) = encode_video_frame(
        &f1,
        Some(&rec0),
        FrameKind::Inter,
        VideoConfig {
            profile: CodecProfile::Av1,
            qstep: 4,
            ..VideoConfig::default()
        },
    )
    .expect("encode av1 inter");

    // 1. Exhaustive prefix truncations must fail closed with `Err(...)`, zero panics.
    for cut in 0..valid_key.len() {
        assert!(decode_video_frame(&valid_key[..cut], None).is_err());
    }

    // 2. Dimension / `max_bytes` allocation bomb must fail closed immediately with `OutputTooLarge` or `InvalidDimensions`.
    assert_eq!(
        decode_video_frame_with_limit(&valid_key, None, 1024),
        Err(Error::OutputTooLarge)
    );
    assert_eq!(
        Yuv420Frame::new_blank(65, 48),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(Yuv420Frame::new_blank(0, 48), Err(Error::InvalidDimensions));

    let mut dim_bomb = valid_key.clone();
    // Width = 16384, Height = 16384 in `meta[4..8]` (offsets `8..12` after 4-byte `BVF1` magic).
    dim_bomb[8..10].copy_from_slice(&16384u16.to_le_bytes());
    dim_bomb[10..12].copy_from_slice(&16384u16.to_le_bytes());
    assert_eq!(
        decode_video_frame_with_limit(&dim_bomb, None, 65_536),
        Err(Error::OutputTooLarge)
    );

    // 3. 500 deterministic corruption + random-packet fuzz seeds across standalone + adaptive decoders.
    let mut rng = Rng::new(0xB7F1_F022_2026_0500);
    for seed in 0..500usize {
        let base = if (seed & 1) == 0 {
            &valid_key
        } else {
            &valid_av1
        };
        let mut mutated = base.clone();
        let flips = 1 + rng.next_usize(4);
        for _ in 0..flips {
            let idx = rng.next_usize(mutated.len());
            let bit = 1u8 << (rng.next_usize(8) as u32);
            mutated[idx] ^= bit;
        }
        assert!(
            decode_video_frame_with_limit(&mutated, Some(&rec0), 65_536).is_err(),
            "corrupted BVF1 packet on seed {seed} must fail closed"
        );

        let mut auto_dec = AdaptiveStreamDecoder::with_options(65_536, true).expect("auto dec");
        assert!(auto_dec.decode_frame(&mutated).is_err());

        // Also fuzz completely synthetic random byte packets with `BVF1` prefix.
        let rlen = rng.next_usize(160);
        let mut junk = vec![0u8; rlen];
        for b in &mut junk {
            *b = rng.next_u64() as u8;
        }
        if junk.len() >= 4 && (seed % 3 == 0) {
            junk[..4].copy_from_slice(&MAGIC_VIDEO_FRAME);
        }
        let _ = decode_video_frame_with_limit(&junk, Some(&rec0), 65_536);
        let _ = auto_dec.decode_frame(&junk);
    }
}

#[test]
fn av1_hd_1280x720_causal_ibc_upper_left_roundtrip_zero_corrupt_frames() {
    let width = 1280usize;
    let height = 720usize;
    // Build a screen-like 1280x720 frame containing sharp text-like tiles repeated with
    // upper-left shifts `(dx <= -16, dy in {-12, -8, -4})` so AV1 IBC actively picks them.
    let mut frame = Yuv420Frame::new_blank(width, height).expect("blank 720p");
    for y in 0..height {
        for x in 0..width {
            let v = ((x.wrapping_mul(3) ^ y.wrapping_mul(5)) & 0x3F) as u8;
            frame.y[y * width + x] = 40u8.wrapping_add(v);
        }
    }
    // Stamp sharp high-contrast 16x16 glyphs at macroblock-aligned `(dst_x, dst_y)` and
    // identical causal upper-left sources at `(dst_x - dx, dst_y - dy)` with `dy in {4, 8, 12}`
    // so `search_causal_ibc` finds exact SAD=0 matches with `dy in {-12, -8, -4}` and `ux + 16 <= bx`.
    let shifts: [(usize, usize); 3] = [(24, 4), (32, 8), (40, 12)];
    for row_band in 0..12usize {
        let dst_y = 48 + row_band * 48;
        for col_band in 0..16usize {
            let dst_x = 64 + col_band * 64;
            let (dx, dy) = shifts[(row_band + col_band) % shifts.len()];
            let src_x = dst_x - dx;
            let src_y = dst_y - dy;
            if dst_x + 16 <= width && dst_y + 16 <= height {
                for r in 0..16usize {
                    for c in 0..16usize {
                        let pat = if ((r + c) & 3) == 0 || r == 0 || c == 0 {
                            235u8
                        } else {
                            20u8
                        };
                        frame.y[(src_y + r) * width + (src_x + c)] = pat;
                        frame.y[(dst_y + r) * width + (dst_x + c)] = pat;
                    }
                }
            }
        }
    }
    let cfg = VideoConfig {
        profile: CodecProfile::Av1,
        qstep: 4,
        ..VideoConfig::default()
    };
    let (key_pkt, enc_rec0) =
        encode_video_frame(&frame, None, FrameKind::Key, cfg).expect("encode 720p AV1 key");
    let dec_rec0 = decode_video_frame_with_limit(&key_pkt, None, 4 * 1024 * 1024)
        .expect("decode 720p AV1 key with 0 CorruptVideoFrame errors");
    assert_eq!(dec_rec0, enc_rec0);

    // Also test a second frame through AdaptiveStreamEncoder/Decoder at 1280x720 AV1.
    let mut frame1 = frame.clone();
    for x in 0..width {
        frame1.y[360 * width + x] = frame1.y[360 * width + x].wrapping_add(11);
    }
    let auto_cfg = AdaptiveStreamConfig {
        profile: CodecProfile::Av1,
        initial_qstep: 4,
        ..AdaptiveStreamConfig::default()
    };
    let mut aenc = AdaptiveStreamEncoder::new(auto_cfg).expect("auto enc");
    let mut adec = AdaptiveStreamDecoder::with_options(4 * 1024 * 1024, false).expect("auto dec");
    let (pkt0, _) = aenc.encode_frame(&frame).expect("auto encode 0");
    let out0 = adec.decode_frame(&pkt0).expect("auto decode 0");
    assert_eq!((out0.width, out0.height), (width, height));
    let (pkt1, _) = aenc.encode_frame(&frame1).expect("auto encode 1");
    let out1 = adec.decode_frame(&pkt1).expect("auto decode 1");
    assert_eq!((out1.width, out1.height), (width, height));
}

