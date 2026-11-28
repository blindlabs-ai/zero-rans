//! AdaptiveStream Video Codec Engine: Decoder-Mirrored Empirical CDF Homeostasis,
//! Online Lagrangian Rate-Distortion Dual Ascent, Structural Basis Morphism (`Auto`),
//! and Self-Healing Rolling Intra-Refresh Wave.

use alloc::vec::Vec;

use crate::rans8::{RansModel, ScaleBits};
use crate::residual::ResidualPriors;
use crate::video::{
    analyze_frame_structure, decode_video_frame_internal, encode_video_frame_internal,
    select_auto_profile, CodecProfile, FrameKind, FramePriors, FrameStructuralStats,
    FrameSymbolObs, VideoConfig, Yuv420Frame,
};
use crate::wire::{crc32_update, Result, DEFAULT_MAX_DECODE_BYTES};

const PRIOR_TOTAL_12: u16 = 4096;

/// Exponential moving average frequency step `(3 * prior + obs) >> 2` bounded in `1..=4096`.
/// @domain prior: 1..=4096, obs: 1..=4096
#[inline]
pub const fn ema_freq_step(prior: u16, obs: u16) -> u16 {
    let b = (3 * (prior as u32) + (obs as u32) + 2) >> 2;
    if b < 1 { 1 } else if b > 4096 { 4096 } else { b as u16 }
}
/// Online Lagrangian dual-ascent step on `lambda_q8` (`16..=8192`) driven by byte-budget error.
/// @domain lambda_q8: 16..=8192, actual_bytes: 0..=16777216, target_bytes: 64..=16777216
#[inline]
pub const fn rd_dual_step(lambda_q8: u16, actual_bytes: u32, target_bytes: u32) -> u16 {
    let t = if target_bytes < 64 { 64 } else { target_bytes } as i64;
    let base = lambda_q8 as i64;
    let raw = (base * ((actual_bytes as i64) - t)) / (t * 2);
    let delta = if raw > base { base } else if raw < -(base >> 1) { -(base >> 1) } else { raw };
    let next = base + delta;
    if next < 16 { 16 } else if next > 8192 { 8192 } else { next as u16 }
}
/// Homeostatic quantizer step (`qstep in 1..=64`) coupled to byte-budget ratio and spatial variance.
/// @domain cur_qstep: 1..=64, actual_bytes: 0..=16777216, target_bytes: 64..=16777216, spatial_var: 0..=65535
#[inline]
pub const fn homeostatic_qstep(cur_qstep: u8, actual_bytes: u32, target_bytes: u32, spatial_var: u16) -> u8 {
    let t = if target_bytes < 64 { 64 } else { target_bytes } as i64;
    let (err, q) = ((actual_bytes as i64) - t, cur_qstep as i64);
    let next = q + ((err * (q + 4)) / (t * 2)) + if spatial_var > 1200 && err > 0 { 1 } else { 0 };
    if next < 1 { 1 } else if next > 64 { 64 } else { next as u8 }
}
/// Integer divergence proxy `|p - q|` contribution per symbol bin (`0..=4096`).
/// @domain p: 1..=4096, q: 1..=4096
#[inline]
pub const fn l1_prior_divergence_bin(p: u16, q: u16) -> u32 {
    if p >= q { (p - q) as u32 } else { (q - p) as u32 }
}

/// Self-maintaining `12-bit` normalized symbol distribution (`sum == 4096`, `freqs[s] >= 1`).
#[derive(Debug, Clone)]
pub struct AdaptivePrior256 {
    freqs: [u16; 256],
    model: RansModel,
}

impl AdaptivePrior256 {
    /// Uniform `16/4096` cold-start prior across all `256` byte values.
    pub fn new_uniform() -> Result<Self> {
        let freqs = [16u16; 256];
        Ok(Self { model: RansModel::from_normalized(&freqs, ScaleBits::Bits12)?, freqs })
    }
    /// Return current normalized frequencies (`sum == 4096`).
    #[inline] #[must_use]
    pub fn freqs(&self) -> &[u16; 256] { &self.freqs }
    /// Return compiled `12-bit` [`RansModel`] for zero-header mirrored encoding/decoding.
    #[inline] #[must_use]
    pub fn model(&self) -> &RansModel { &self.model }
    /// Measure L1 distribution shift (`0..=8192`) between empirical `obs_bytes` and current prior.
    #[must_use]
    pub fn divergence_against(&self, obs_bytes: &[u8]) -> u32 {
        if obs_bytes.is_empty() { return 0; }
        let target = empirical_laplace_4096(obs_bytes);
        self.freqs.iter().zip(&target).map(|(&p, &q)| l1_prior_divergence_bin(p, q)).sum()
    }
    /// Update prior via integer EMA against `obs_bytes` and rescale to exact sum `4096`.
    pub fn update_ema(&mut self, obs_bytes: &[u8]) -> Result<()> {
        if obs_bytes.is_empty() { return Ok(()); }
        let (target, mut blended) = (empirical_laplace_4096(obs_bytes), [1u16; 256]);
        for (slot, (&p, &q)) in blended.iter_mut().zip(self.freqs.iter().zip(&target)) {
            *slot = ema_freq_step(p, q);
        }
        rescale_in_place_4096(&mut blended);
        self.model = RansModel::from_normalized(&blended, ScaleBits::Bits12)?;
        self.freqs = blended;
        Ok(())
    }
}
fn empirical_laplace_4096(obs_bytes: &[u8]) -> [u16; 256] {
    let mut counts = [1u64; 256];
    for &b in obs_bytes { counts[usize::from(b) & 255] += 16; }
    let (total, mut out) = (counts.iter().sum::<u64>().max(1), [1u16; 256]);
    for (slot, &c) in out.iter_mut().zip(&counts) {
        *slot = ((c * u64::from(PRIOR_TOTAL_12)) / total).max(1).min(u64::from(PRIOR_TOTAL_12)) as u16;
    }
    rescale_in_place_4096(&mut out);
    out
}
fn rescale_in_place_4096(freqs: &mut [u16; 256]) {
    let mut sum = 0i32;
    for slot in freqs.iter_mut() { *slot = (*slot).max(1); sum += i32::from(*slot); }
    let mut diff = i32::from(PRIOR_TOTAL_12) - sum;
    while diff != 0 {
        let (mut best_idx, mut best_val) = (0usize, if diff > 0 { 0u16 } else { 1u16 });
        for (i, &v) in freqs.iter().enumerate() { if v > best_val { (best_val, best_idx) = (v, i); } }
        if diff < 0 && best_val <= 1 { break; }
        if diff > 0 { freqs[best_idx] += 1; diff -= 1; } else { freqs[best_idx] -= 1; diff += 1; }
    }
}

/// Configuration for [`AdaptiveStreamEncoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveStreamConfig {
    /// Preferred codec profile (`CodecProfile::Auto` enables live structural basis morphism).
    pub profile: CodecProfile,
    /// Initial quantizer step (`1..=64`, or `1..=255` for `Compact`).
    pub initial_qstep: u8,
    /// Optional target compressed byte budget per frame (`Some(B)` activates dual-ascent RC).
    pub target_bytes_per_frame: Option<u32>,
    /// Maximum keyframe interval (`0` = only scene-cut / first-frame keyframes).
    pub keyint: u32,
    /// Enable rolling intra-refresh macroblock column wave on P-frames.
    pub rolling_intra_refresh: bool,
    /// Enable codec-specific in-loop restoration filter (`Deblock`, `SAO`, `VP9-LF`, `CDEF`).
    pub loop_filter: bool,
    /// Integer motion estimation search radius (`1..=32`).
    pub search_range: u8,
}

impl Default for AdaptiveStreamConfig {
    #[inline]
    fn default() -> Self {
        Self {
            profile: CodecProfile::Auto, initial_qstep: 4, target_bytes_per_frame: None,
            keyint: 30, rolling_intra_refresh: true, loop_filter: true, search_range: 12,
        }
    }
}

/// Per-frame telemetry emitted by [`AdaptiveStreamEncoder::encode_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveStreamReport {
    /// Concrete codec profile used on the wire (`AvcH264`, `HevcH265`, `Vp9`, `Av1`, `Compact`).
    pub chosen_profile: CodecProfile,
    /// Frame prediction type (`Key` or `Inter`).
    pub frame_kind: FrameKind,
    /// Quantization step used for this frame.
    pub qstep: u8,
    /// Lagrangian multiplier (`Q8`) used for this frame.
    pub lambda_q8: u16,
    /// Encoded `BVF1` packet byte length.
    pub packet_bytes: usize,
    /// Post-evolution mirrored metabolic state CRC-32 (guaranteed equal on decoder).
    pub state_digest: u32,
    /// Measured input structural statistics.
    pub structure: FrameStructuralStats,
    /// Measured prior L1 innovation divergence (`0..=8192`) prior to adaptation.
    pub prior_divergence: u32,
}

fn uniform_priors() -> Result<[AdaptivePrior256; 5]> {
    let p = AdaptivePrior256::new_uniform()?;
    Ok([p.clone(), p.clone(), p.clone(), p.clone(), p])
}

/// Deterministic self-producing probability & reference state shared by encoder and decoder.
#[derive(Debug, Clone)]
pub struct RateControlState {
    frames_processed: u64,
    epoch_frames: u32,
    refresh_cursor: u16,
    priors: [AdaptivePrior256; 5],
    reference: Option<Yuv420Frame>,
}

impl RateControlState {
    /// Create a cold-start uniform metabolic state (`frames_processed == 0`).
    pub fn new() -> Result<Self> {
        Ok(Self {
            frames_processed: 0, epoch_frames: 0, refresh_cursor: 0,
            priors: uniform_priors()?, reference: None,
        })
    }
    /// Reset adaptive priors and epoch counter to cold-start (invoked on Keyframes).
    pub fn reset_priors(&mut self) -> Result<()> {
        self.priors = uniform_priors()?;
        (self.epoch_frames, self.refresh_cursor) = (0, 0);
        Ok(())
    }
    /// Return total number of frames evolved through this metabolic state.
    #[inline] #[must_use]
    pub const fn frames_processed(&self) -> u64 { self.frames_processed }
    /// Return current reconstructed reference frame, if any.
    #[inline] #[must_use]
    pub fn reference(&self) -> Option<&Yuv420Frame> { self.reference.as_ref() }
    /// Return the current adaptive rANS model for macroblock modes.
    #[inline] #[must_use]
    pub fn mode_prior_model(&self) -> &RansModel { self.priors[0].model() }
    /// Compute deterministic IEEE CRC-32 digest over all 5 mirrored CDFs, counters, and reference.
    #[must_use]
    pub fn state_digest(&self) -> u32 {
        let mut crc = crc32_update(0xFFFF_FFFFu32, &self.epoch_frames.to_le_bytes());
        crc = crc32_update(crc, &self.refresh_cursor.to_le_bytes());
        for prior in &self.priors {
            for &f in prior.freqs() { crc = crc32_update(crc, &f.to_le_bytes()); }
        }
        let ref_crc = self.reference.as_ref().map_or(0u32, Yuv420Frame::digest_crc32);
        !crc32_update(crc, &ref_crc.to_le_bytes())
    }
    pub(crate) fn priors(&self) -> FramePriors<'_> {
        FramePriors {
            mode: self.priors[0].model(), side: self.priors[1].model(),
            res: ResidualPriors {
                cbp: self.priors[2].model(), run: self.priors[3].model(), level_lo: self.priors[4].model(),
            },
        }
    }
    pub(crate) fn evolve(&mut self, rec_frame: Yuv420Frame, obs: &FrameSymbolObs) -> Result<()> {
        let mb_cols = rec_frame.width.div_ceil(16).max(1) as u16;
        self.priors[0].update_ema(&obs.modes)?;
        self.priors[1].update_ema(&obs.side)?;
        let mut merged = Vec::new();
        let pairs = [
            (&obs.res4.cbp[..], &obs.res8.cbp[..]),
            (&obs.res4.runs[..], &obs.res8.runs[..]),
            (&obs.res4.levels_lo[..], &obs.res8.levels_lo[..]),
        ];
        for (prior, (a, b)) in self.priors[2..].iter_mut().zip(pairs) {
            merged.clear();
            merged.extend_from_slice(a);
            merged.extend_from_slice(b);
            prior.update_ema(&merged)?;
        }
        self.refresh_cursor = (self.refresh_cursor + 1) % mb_cols;
        self.epoch_frames = self.epoch_frames.saturating_add(1);
        self.frames_processed += 1;
        self.reference = Some(rec_frame);
        Ok(())
    }
}

/// Stateful adaptive video encoder with mirrored CDF homeostasis, dual-ascent RC,
/// scene-cut detection, rolling intra-refresh, and `Auto` codec morphism.
#[derive(Debug, Clone)]
pub struct AdaptiveStreamEncoder {
    cfg: AdaptiveStreamConfig,
    state: RateControlState,
    qstep: u8,
    lambda_q8: u16,
    since_keyframe: u32,
}

impl AdaptiveStreamEncoder {
    /// Construct a cold-start [`AdaptiveStreamEncoder`] from `cfg`.
    pub fn new(cfg: AdaptiveStreamConfig) -> Result<Self> {
        let max_q = if cfg.profile == CodecProfile::Compact { 255 } else { 64 };
        let qstep = cfg.initial_qstep.max(1).min(max_q);
        Ok(Self {
            cfg, state: RateControlState::new()?, qstep,
            lambda_q8: (u16::from(qstep) * 32).max(32), since_keyframe: 0,
        })
    }
    /// Inspect the shared encoder [`RateControlState`].
    #[inline] #[must_use]
    pub fn state(&self) -> &RateControlState { &self.state }
    /// Return the current post-frame metabolic state digest (`CRC-32`).
    #[inline] #[must_use]
    pub fn state_digest(&self) -> u32 { self.state.state_digest() }
    /// Return current homeostatic quantizer step.
    #[inline] #[must_use]
    pub const fn current_qstep(&self) -> u8 { self.qstep }
    /// Return current Lagrangian dual variable `lambda_q8`.
    #[inline] #[must_use]
    pub const fn current_lambda_q8(&self) -> u16 { self.lambda_q8 }

    /// Encode the next [`Yuv420Frame`] in the sequence, evolving mirrored priors and RC state.
    pub fn encode_frame(&mut self, frame: &Yuv420Frame) -> Result<(Vec<u8>, AdaptiveStreamReport)> {
        let structure = analyze_frame_structure(frame, self.state.reference());
        let dim_diff = self.state.reference()
            .is_some_and(|r| (r.width, r.height) != (frame.width, frame.height));
        let need_key = self.state.reference().is_none() || dim_diff || structure.temporal_sad_avg >= 68
            || (self.cfg.keyint > 0 && self.since_keyframe >= self.cfg.keyint);
        let kind = if need_key {
            self.state.reset_priors()?;
            self.since_keyframe = 0;
            FrameKind::Key
        } else { FrameKind::Inter };
        let chosen_profile = if self.cfg.profile == CodecProfile::Auto {
            select_auto_profile(structure)
        } else { self.cfg.profile };
        let is_refresh = kind == FrameKind::Inter && self.cfg.rolling_intra_refresh;
        let used_qstep = if chosen_profile == CodecProfile::Compact && kind == FrameKind::Key {
            (self.qstep / 2).max(1)
        } else { self.qstep };
        let used_lambda = self.lambda_q8;
        let mb_cols = frame.width.div_ceil(16).max(1) as u16;
        let vcfg = VideoConfig {
            profile: chosen_profile, qstep: used_qstep, lambda_q8: used_lambda,
            search_range: self.cfg.search_range, loop_filter: self.cfg.loop_filter,
            intra_refresh_col: is_refresh.then_some(self.state.refresh_cursor % mb_cols),
        };
        let ref_arg = (kind == FrameKind::Inter).then(|| self.state.reference()).flatten();
        let (packet, rec_frame, obs) = encode_video_frame_internal(
            frame, ref_arg, kind, vcfg, (!is_refresh).then(|| self.state.priors()),
        )?;
        let (packet_bytes, prior_divergence) =
            (packet.len(), self.state.priors[0].divergence_against(&obs.modes));
        self.state.evolve(rec_frame, &obs)?;
        self.since_keyframe += 1;
        if let Some(target) = self.cfg.target_bytes_per_frame {
            let actual = u32::try_from(packet_bytes).unwrap_or(u32::MAX);
            self.lambda_q8 = rd_dual_step(self.lambda_q8, actual, target);
            if chosen_profile == CodecProfile::Compact {
                if kind == FrameKind::Inter {
                    let (t, q) = (i64::from(target.max(64)), i64::from(self.qstep));
                    self.qstep = (q + ((i64::from(actual) - t) * (q + 4)) / (t * 2)).max(2).min(255) as u8;
                }
            } else {
                self.qstep = homeostatic_qstep(self.qstep, actual, target, structure.spatial_var_avg);
            }
        }
        Ok((packet, AdaptiveStreamReport {
            chosen_profile, frame_kind: kind, qstep: used_qstep, lambda_q8: used_lambda,
            packet_bytes, state_digest: self.state.state_digest(), structure, prior_divergence,
        }))
    }
}

/// Stateful adaptive video decoder evolving [`RateControlState`] in lockstep with encoder.
#[derive(Debug, Clone)]
pub struct AdaptiveStreamDecoder {
    state: RateControlState,
    max_decode_bytes: usize,
    resilient_heal: bool,
}

impl AdaptiveStreamDecoder {
    /// Construct a cold-start [`AdaptiveStreamDecoder`].
    pub fn new() -> Result<Self> { Self::with_options(DEFAULT_MAX_DECODE_BYTES, false) }
    /// Construct a decoder with explicit allocation ceiling and optional packet-loss healing.
    pub fn with_options(max_decode_bytes: usize, resilient_heal: bool) -> Result<Self> {
        Ok(Self { state: RateControlState::new()?, max_decode_bytes, resilient_heal })
    }
    /// Enable or disable packet-drop resilient reconstruction over the rolling intra-refresh wave.
    pub fn set_resilient_heal(&mut self, enabled: bool) { self.resilient_heal = enabled; }
    /// Inspect the shared decoder [`RateControlState`].
    #[inline] #[must_use]
    pub fn state(&self) -> &RateControlState { &self.state }
    /// Return the current post-frame metabolic state digest (`CRC-32`).
    #[inline] #[must_use]
    pub fn state_digest(&self) -> u32 { self.state.state_digest() }
    /// Decode a `BVF1` packet and co-evolve the mirrored [`RateControlState`].
    pub fn decode_frame(&mut self, packet: &[u8]) -> Result<Yuv420Frame> {
        let is_key = packet.get(4).is_some_and(|&pk| (pk >> 4) == 0);
        let mut trial = self.state.clone();
        if is_key { trial.reset_priors()?; }
        let ref_arg = (!is_key).then(|| trial.reference()).flatten();
        let (frame, obs) = decode_video_frame_internal(
            packet, ref_arg, self.max_decode_bytes, self.resilient_heal, Some(trial.priors()),
        )?;
        trial.evolve(frame.clone(), &obs)?;
        self.state = trial;
        Ok(frame)
    }
}
