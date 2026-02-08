//! MatFormer stabilization knobs (C2): width-rescale, suffix gate, and in-place distillation.
//!
//! Goal: make heterogeneous MatFormer training less fragile without adding bandwidth.
//! - Width-rescale: correct the static scale mismatch when slicing FFN width.
//! - Suffix gate: make tier-0 explicitly "core + correction" by ramping suffix contribution.
//! - Residual scaling + norm tier gain: calibrate activation statistics per tier.
//! - Distillation: couple small-tier objective to full-model output distribution.

use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MatformerStabilizationConfig {
    /// If true, rescale MLP output for MatFormer (tier>0) to correct for the
    /// initialization mismatch when down_proj fan-in changes with width.
    #[serde(default)]
    pub width_rescale_mlp_output: bool,

    /// Power applied to the width ratio (full/active). Default 0.5 (sqrt).
    #[serde(default = "default_width_rescale_power")]
    pub width_rescale_power: f64,

    /// Optional tier-0 suffix gate schedule (progressive growth).
    #[serde(default)]
    pub suffix_gate: Option<SuffixGateConfig>,

    /// Per-tier MLP/attn residual scaling: α = (active/full)^power.
    /// Tier-0 (active==full) always gets α=1.0 (no change).
    #[serde(default)]
    pub residual_scale: Option<ResidualScaleConfig>,

    /// Per-tier RMSNorm gain scalar: gain = (active/full)^power.
    /// Tier-0 (active==full) always gets gain=1.0 (no change).
    #[serde(default)]
    pub norm_tier_gain: Option<NormTierGainConfig>,

    /// In-place distillation: small tiers match tier-0's output distribution.
    /// Only active for tier > 0 when teacher logits are available.
    #[serde(default)]
    pub distillation: Option<DistillationConfig>,
}

fn default_width_rescale_power() -> f64 {
    0.5
}

impl Default for MatformerStabilizationConfig {
    fn default() -> Self {
        Self {
            width_rescale_mlp_output: false,
            width_rescale_power: default_width_rescale_power(),
            suffix_gate: None,
            residual_scale: None,
            norm_tier_gain: None,
            distillation: None,
        }
    }
}

/// Per-tier residual scaling for MLP and/or attention outputs.
///
/// When a smaller tier uses fewer FFN neurons, the MLP output has a different
/// magnitude than the full-width tier-0. This scales the residual contribution
/// by `(active_width / full_width)^power` to calibrate activation statistics.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ResidualScaleConfig {
    /// Exponent applied to (active/full) ratio. Default 0.5 (sqrt).
    #[serde(default = "default_residual_scale_power")]
    pub power: f64,

    /// Apply residual scaling to MLP output. Default true.
    #[serde(default = "default_true")]
    pub apply_to_mlp: bool,

    /// Apply residual scaling to attention output. Default false
    /// (attention is full-width, no slicing).
    #[serde(default)]
    pub apply_to_attn: bool,
}

fn default_residual_scale_power() -> f64 {
    0.5
}

fn default_true() -> bool {
    true
}

impl Default for ResidualScaleConfig {
    fn default() -> Self {
        Self {
            power: default_residual_scale_power(),
            apply_to_mlp: true,
            apply_to_attn: false,
        }
    }
}

/// Per-tier RMSNorm gain scalar.
///
/// Different FFN widths produce different activation magnitudes flowing through
/// shared RMSNorm layers. A per-tier gain `(active/full)^power` gently
/// calibrates the norm output without adding learnable parameters outside DisTrO.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct NormTierGainConfig {
    /// Exponent applied to (active/full) ratio. Default 0.25 (gentler than residual scaling).
    #[serde(default = "default_norm_tier_gain_power")]
    pub power: f64,
}

fn default_norm_tier_gain_power() -> f64 {
    0.25
}

impl Default for NormTierGainConfig {
    fn default() -> Self {
        Self {
            power: default_norm_tier_gain_power(),
        }
    }
}

/// In-place distillation: student (tier > 0) matches teacher (tier-0) output.
///
/// The student loss becomes: `(1 - β) * CE(student, labels) + β * KL(student/T, teacher/T) * T²`
/// where β ramps from 0 to `beta_max` over `warmup_steps` starting at `start_step`.
/// Teacher logits are compressed to top-k per token (~1MB per batch).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DistillationConfig {
    /// Number of top logits to transmit per token. Default 32.
    #[serde(default = "default_top_k")]
    pub top_k: u16,

    /// Temperature for KL divergence softening. Default 2.0.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum KD loss weight. Default 0.5.
    #[serde(default = "default_beta_max")]
    pub beta_max: f64,

    /// Step at which distillation begins (teacher is garbage early). Default 0.
    #[serde(default)]
    pub start_step: u32,

    /// Steps to ramp β from 0 to beta_max. Default 100.
    #[serde(default = "default_distill_warmup_steps")]
    pub warmup_steps: u32,

    /// Schedule shape for the β ramp.
    #[serde(default)]
    pub schedule: SuffixGateSchedule,
}

fn default_top_k() -> u16 {
    32
}

fn default_temperature() -> f32 {
    2.0
}

fn default_beta_max() -> f64 {
    0.5
}

fn default_distill_warmup_steps() -> u32 {
    100
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            temperature: default_temperature(),
            beta_max: default_beta_max(),
            start_step: 0,
            warmup_steps: default_distill_warmup_steps(),
            schedule: Default::default(),
        }
    }
}

/// Compute the distillation weight β at a given step.
///
/// Returns a value in `[0, beta_max]`. Before `start_step`, returns 0.
/// Ramps linearly or via cosine from 0 to `beta_max` over `warmup_steps`.
pub fn distillation_beta(cfg: &DistillationConfig, step: u32) -> f64 {
    if cfg.warmup_steps == 0 {
        return cfg.beta_max;
    }
    if step <= cfg.start_step {
        return 0.0;
    }
    let t = (step - cfg.start_step) as f64 / (cfg.warmup_steps as f64);
    let t = t.clamp(0.0, 1.0);
    let ramp = match cfg.schedule {
        SuffixGateSchedule::Linear => t,
        SuffixGateSchedule::Cosine => 0.5 - 0.5 * (std::f64::consts::PI * t).cos(),
    };
    ramp * cfg.beta_max
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SuffixGateSchedule {
    Linear,
    Cosine,
}

impl Default for SuffixGateSchedule {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SuffixGateConfig {
    /// Which MatFormer tier's prefix defines the "core" width. E.g. 2 means core width is
    /// intermediate_size / 4, matching a tier-2 client.
    #[serde(default = "default_gate_tier")]
    pub gate_tier: u8,

    /// Step at which the suffix starts ramping in.
    #[serde(default)]
    pub start_step: u32,

    /// Steps to ramp beta from 0 to 1.
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: u32,

    #[serde(default)]
    pub schedule: SuffixGateSchedule,
}

fn default_gate_tier() -> u8 {
    2
}

fn default_warmup_steps() -> u32 {
    100
}

impl Default for SuffixGateConfig {
    fn default() -> Self {
        Self {
            gate_tier: default_gate_tier(),
            start_step: 0,
            warmup_steps: default_warmup_steps(),
            schedule: Default::default(),
        }
    }
}

static CURRENT_STEP: AtomicU32 = AtomicU32::new(0);

pub(crate) fn set_current_step(step: u32) {
    CURRENT_STEP.store(step, Ordering::Relaxed);
}

pub(crate) fn current_step() -> u32 {
    CURRENT_STEP.load(Ordering::Relaxed)
}

pub(crate) fn suffix_beta(cfg: &SuffixGateConfig, step: u32) -> f64 {
    if cfg.warmup_steps == 0 {
        return 1.0;
    }
    if step <= cfg.start_step {
        return 0.0;
    }

    let t = (step - cfg.start_step) as f64 / (cfg.warmup_steps as f64);
    let t = t.clamp(0.0, 1.0);

    match cfg.schedule {
        SuffixGateSchedule::Linear => t,
        SuffixGateSchedule::Cosine => 0.5 - 0.5 * (std::f64::consts::PI * t).cos(),
    }
}

pub(crate) fn gate_prefix_len(full_intermediate_size: i64, gate_tier: u8) -> i64 {
    let divisor = 1_i64
        .checked_shl(gate_tier as u32)
        .unwrap_or_else(|| panic!("gate_tier too large: {}", gate_tier));
    assert!(
        divisor > 0,
        "invalid gate_tier {}, divisor computed as 0",
        gate_tier
    );
    assert!(
        full_intermediate_size % divisor == 0,
        "intermediate_size {} must be divisible by 2^gate_tier ({}), got divisor {}",
        full_intermediate_size,
        gate_tier,
        divisor
    );
    let prefix = full_intermediate_size / divisor;
    assert!(
        prefix > 0,
        "gate_tier {} makes prefix_len 0 for intermediate_size {}",
        gate_tier,
        full_intermediate_size
    );
    prefix
}

/// Compute residual/norm scaling factor: `(active / full) ^ power`.
///
/// - Tier-0 (active == full): returns 1.0 (identity).
/// - Tier-1 (active = full/2, power=0.5): returns ~0.707.
/// - Tier-2 (active = full/4, power=0.5): returns 0.5.
pub(crate) fn residual_scale_factor(
    full_intermediate_size: i64,
    active_intermediate_size: i64,
    power: f64,
) -> f64 {
    assert!(
        full_intermediate_size > 0 && active_intermediate_size > 0,
        "invalid intermediate sizes: full={}, active={}",
        full_intermediate_size,
        active_intermediate_size
    );
    let ratio = (active_intermediate_size as f64) / (full_intermediate_size as f64);
    ratio.powf(power)
}

pub(crate) fn width_rescale_factor(
    full_intermediate_size: i64,
    active_intermediate_size: i64,
    power: f64,
) -> f64 {
    assert!(
        full_intermediate_size > 0 && active_intermediate_size > 0,
        "invalid intermediate sizes: full={}, active={}",
        full_intermediate_size,
        active_intermediate_size
    );
    let ratio = (full_intermediate_size as f64) / (active_intermediate_size as f64);
    ratio.powf(power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_prefix_len() {
        assert_eq!(gate_prefix_len(3072, 0), 3072);
        assert_eq!(gate_prefix_len(3072, 1), 1536);
        assert_eq!(gate_prefix_len(3072, 2), 768);
    }

    #[test]
    fn test_suffix_beta_linear() {
        let cfg = SuffixGateConfig {
            gate_tier: 2,
            start_step: 10,
            warmup_steps: 10,
            schedule: SuffixGateSchedule::Linear,
        };
        assert_eq!(suffix_beta(&cfg, 0), 0.0);
        assert_eq!(suffix_beta(&cfg, 10), 0.0);
        assert!((suffix_beta(&cfg, 15) - 0.5).abs() < 1e-9);
        assert_eq!(suffix_beta(&cfg, 20), 1.0);
        assert_eq!(suffix_beta(&cfg, 100), 1.0);
    }

    #[test]
    fn test_suffix_beta_cosine_endpoints() {
        let cfg = SuffixGateConfig {
            gate_tier: 2,
            start_step: 10,
            warmup_steps: 10,
            schedule: SuffixGateSchedule::Cosine,
        };
        assert_eq!(suffix_beta(&cfg, 10), 0.0);
        assert!((suffix_beta(&cfg, 20) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mlp_width_rescale_factor_default_sqrt() {
        // full/active = 4 -> sqrt = 2
        assert!((width_rescale_factor(3072, 768, 0.5) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_residual_scale_factor_tier0_identity() {
        // Tier-0: active == full → always 1.0
        assert!((residual_scale_factor(3072, 3072, 0.5) - 1.0).abs() < 1e-12);
        assert!((residual_scale_factor(3072, 3072, 0.25) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_residual_scale_factor_tier1_sqrt() {
        // Tier-1: active = full/2, power=0.5 → sqrt(0.5) ≈ 0.7071
        let factor = residual_scale_factor(3072, 1536, 0.5);
        assert!((factor - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn test_residual_scale_factor_tier2_sqrt() {
        // Tier-2: active = full/4, power=0.5 → sqrt(0.25) = 0.5
        assert!((residual_scale_factor(3072, 768, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_residual_scale_factor_quarter_power() {
        // Tier-1: active = full/2, power=0.25 → (0.5)^0.25 ≈ 0.8409
        let factor = residual_scale_factor(3072, 1536, 0.25);
        assert!((factor - 0.5_f64.powf(0.25)).abs() < 1e-12);
    }

    #[test]
    fn test_distillation_beta_before_start() {
        let cfg = DistillationConfig {
            start_step: 50,
            warmup_steps: 100,
            beta_max: 0.5,
            ..Default::default()
        };
        assert_eq!(distillation_beta(&cfg, 0), 0.0);
        assert_eq!(distillation_beta(&cfg, 50), 0.0);
    }

    #[test]
    fn test_distillation_beta_linear_ramp() {
        let cfg = DistillationConfig {
            start_step: 0,
            warmup_steps: 100,
            beta_max: 0.5,
            schedule: SuffixGateSchedule::Linear,
            ..Default::default()
        };
        assert!((distillation_beta(&cfg, 50) - 0.25).abs() < 1e-9);
        assert!((distillation_beta(&cfg, 100) - 0.5).abs() < 1e-9);
        // Past warmup: clamped to beta_max
        assert!((distillation_beta(&cfg, 200) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_distillation_beta_zero_warmup() {
        let cfg = DistillationConfig {
            start_step: 0,
            warmup_steps: 0,
            beta_max: 0.3,
            ..Default::default()
        };
        // Zero warmup means immediate beta_max
        assert!((distillation_beta(&cfg, 0) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_distillation_beta_cosine_endpoints() {
        let cfg = DistillationConfig {
            start_step: 10,
            warmup_steps: 10,
            beta_max: 0.5,
            schedule: SuffixGateSchedule::Cosine,
            ..Default::default()
        };
        assert_eq!(distillation_beta(&cfg, 10), 0.0);
        assert!((distillation_beta(&cfg, 20) - 0.5).abs() < 1e-12);
    }
}
