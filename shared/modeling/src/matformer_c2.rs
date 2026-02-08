//! MatFormer stabilization knobs (C2): width-rescale + suffix gate.
//!
//! Goal: make heterogeneous MatFormer training less fragile without adding bandwidth.
//! - Width-rescale: correct the static scale mismatch when slicing FFN width.
//! - Suffix gate: make tier-0 explicitly "core + correction" by ramping suffix contribution.

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
        }
    }
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
}
