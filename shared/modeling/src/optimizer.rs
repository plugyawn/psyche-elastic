use crate::{CausalLM, Distro};
use psyche_core::OptimizerDefinition;
use serde::{Deserialize, Serialize};
use tch::COptimizer;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistroApplyMode {
    #[default]
    Sign,
    Raw,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistroAggregateMode {
    #[default]
    Legacy,
    DilocoLite,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistroValueMode {
    #[default]
    Auto,
    Sign,
    Raw,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistroRawNormMode {
    #[default]
    Off,
    MatchPreL2,
    MatchSignEquivalent,
    MatchSignEquivalentNnz,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistroRawMissingSidecarPolicy {
    #[default]
    WarnOff,
    Fail,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DistroRawConfig {
    pub enabled: bool,
    pub norm_mode: DistroRawNormMode,
    pub scale_multiplier: f64,
    pub scale_max: f64,
    pub abs_clip_mult: f64,
    pub sign_equiv_mult: f64,
    pub missing_sidecar_policy: DistroRawMissingSidecarPolicy,
}

impl Default for DistroRawConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            norm_mode: DistroRawNormMode::Off,
            scale_multiplier: 1.0,
            // Match-sign-equivalent mode often needs >1e7 early in training.
            // Keep this high and rely on abs_clip_mult for per-tensor safety.
            scale_max: 1.0e9,
            abs_clip_mult: 8.0,
            sign_equiv_mult: 1.0,
            missing_sidecar_policy: DistroRawMissingSidecarPolicy::WarnOff,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DistroDilocoLiteConfig {
    /// EMA factor for server-side outer momentum.
    pub outer_momentum: f64,
    /// Multiplier on the aggregated direction before optimizer step.
    pub outer_lr_multiplier: f64,
    /// Trust-region target as a ratio of parameter L2 norm.
    pub trust_region_target: f64,
    /// Clamp factor for up/down-scaling in trust-region normalization.
    pub trust_region_max_scale: f64,
    /// Max per-peer weight as a multiple of mean peer weight.
    pub tier_weight_cap: f64,
}

impl Default for DistroDilocoLiteConfig {
    fn default() -> Self {
        Self {
            outer_momentum: 0.9,
            outer_lr_multiplier: 1.0,
            trust_region_target: 0.02,
            trust_region_max_scale: 1.0,
            tier_weight_cap: 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DistroSignErrorFeedbackConfig {
    /// Enable apply-side error feedback for sign compression.
    pub enabled: bool,
    /// Residual decay factor in [0, 1]. 1.0 keeps full residual memory.
    pub decay: f64,
}

impl Default for DistroSignErrorFeedbackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            decay: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DistroTierProxConfig {
    /// Proximal strength used to downweight tier>0 peer updates at apply time.
    pub mu: f64,
    /// If true, only scale MatFormer FFN prefix-aligned tensors.
    pub prefix_only: bool,
}

impl Default for DistroTierProxConfig {
    fn default() -> Self {
        Self {
            mu: 0.0,
            prefix_only: true,
        }
    }
}

pub enum Optimizer {
    Torch {
        optimizer: COptimizer,
        clip_grad_norm: Option<f32>,
    },
    Distro {
        optimizer: Box<Distro>,
        clip_grad_norm: Option<f32>,
        quantize_1bit: bool,
    },
    Null,
}

impl Optimizer {
    pub fn new(
        definition: OptimizerDefinition,
        model: &dyn CausalLM,
        distro_apply_mode: DistroApplyMode,
        distro_aggregate_mode: DistroAggregateMode,
        distro_value_mode: DistroValueMode,
        distro_raw_config: DistroRawConfig,
        distro_diloco_lite_config: DistroDilocoLiteConfig,
        distro_sign_error_feedback_config: DistroSignErrorFeedbackConfig,
        distro_tier_prox_config: DistroTierProxConfig,
    ) -> Self {
        match definition {
            OptimizerDefinition::AdamW {
                betas,
                weight_decay,
                eps,
                clip_grad_norm,
            } => Self::Torch {
                optimizer: {
                    let mut adamw = COptimizer::adamw(
                        1.0e-1,
                        betas[0] as f64,
                        betas[1] as f64,
                        weight_decay as f64,
                        eps as f64,
                        false,
                    )
                    .unwrap();
                    for var in model.variables() {
                        let tensor = var.logical_tensor();
                        adamw.add_parameters(&tensor, 0).unwrap();
                    }
                    adamw
                },
                clip_grad_norm,
            },
            OptimizerDefinition::Distro {
                clip_grad_norm,
                weight_decay,
                compression_decay,
                compression_topk,
                compression_chunk,
                quantize_1bit,
            } => Self::Distro {
                optimizer: Distro::new(
                    model,
                    compression_decay as f64,
                    compression_chunk as i64,
                    compression_topk as i64,
                    weight_decay.unwrap_or(0.0) as f64,
                    distro_apply_mode,
                    distro_aggregate_mode,
                    distro_value_mode,
                    distro_raw_config,
                    distro_diloco_lite_config,
                    distro_sign_error_feedback_config,
                    distro_tier_prox_config,
                )
                .into(),
                clip_grad_norm,
                quantize_1bit,
            },
            OptimizerDefinition::Dummy => Self::Null,
        }
    }
}
