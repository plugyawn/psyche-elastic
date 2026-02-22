use crate::cli::MatformerLoadStrategy;
use crate::{IntegrationTestLogMarker, WandBInfo, fetch_data::DataFetcher};
use psyche_coordinator::{
    Coordinator, HealthChecks,
    model::{self, HttpLLMTrainingDataLocation, LLMTrainingDataLocation},
};
use psyche_core::{
    Barrier, CancellableBarrier, NodeIdentity, OptimizerDefinition, Shuffle, TokenSize, sha256,
};
use psyche_data_provider::{
    DataProvider, DataProviderTcpClient, DummyDataProvider, LocalDataProvider, LocalDataSplit,
    PreprocessedDataProvider, Split, WeightedDataProvider, download_dataset_repo_async,
    download_model_repo_async,
    http::{FileURLs, HttpDataProvider},
};
use psyche_metrics::ClientMetrics;
use psyche_modeling::{
    AttentionImplementation, AutoConfig, AutoTokenizerError, CausalLM, CommunicatorId,
    DataParallel, DeepseekConfig, DeepseekForCausalLM, Devices, DistroAggregateMode,
    DistroApplyMode, DistroDilocoLiteConfig, DistroRawConfig, DistroValueMode, DummyModel,
    LlamaConfig, LlamaForCausalLM, LocalTrainer, ModelConfig, ModelLoadError, NanoGPTConfig,
    NanoGPTForCausalLM, ParallelModels, PretrainedSource, Trainer, auto_tokenizer,
};
use psyche_network::{AuthenticatableIdentity, BlobTicket};
use psyche_watcher::{ModelSchemaInfo, OpportunisticData};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tch::{Device, Kind, Tensor};
use thiserror::Error;
use tokenizers::{ModelWrapper, Tokenizer, models::wordlevel::WordLevel};
use tokio::{
    io,
    sync::{mpsc::UnboundedSender, oneshot},
    task::{JoinError, JoinHandle},
};
use tracing::{debug, error, info, warn};

use super::{
    CheckpointConfig, FinishedBroadcast,
    cooldown::{
        CooldownStepMetadata, HeldoutEvalConfig, HeldoutEvaluator, MatformerCheckpointInfo,
    },
    evals::ModelTaskRunner,
    stats::StatsLogger,
    steps::StepStateMachine,
    train::TrainingStepMetadata,
    types::DistroBroadcastAndPayload,
    warmup::WarmupStepMetadata,
    witness::WitnessStepMetadata,
};
use iroh_blobs::api::Tag;

pub struct RunInitConfig<T: NodeIdentity, A: AuthenticatableIdentity> {
    // identity for connecting to the data server
    pub identity: T,
    pub network_identity: A,
    pub private_key: A::PrivateKey,

    // p2p model parameters sharing config
    pub max_concurrent_parameter_requests: usize,

    // model & dataload
    pub device: Devices,
    pub matformer_tier: u8,
    pub matformer_load_strategy: MatformerLoadStrategy,
    pub matformer_helper_fraction: f32,
    pub matformer_helper_rotation_interval: u64,
    pub hub_read_token: Option<String>,
    pub hub_max_concurrent_downloads: usize,
    pub data_parallelism: usize,
    pub tensor_parallelism: usize,
    pub micro_batch_size: usize,
    /// Experimental: during the first N training steps, fetch the same canonical batch for every
    /// trainer (client-side data aliasing for early gradient alignment).
    pub same_batch_warmup_steps: u32,
    /// Experimental: after warmup, inject same-batch "anchor" steps every N steps.
    pub same_batch_anchor_every_steps: u32,
    /// First step eligible for periodic same-batch anchors.
    pub same_batch_anchor_start_step: u32,
    /// Experimental: number of local inner updates per coordinator step on smaller tiers.
    pub matformer_local_inner_steps: u32,
    pub optim_stats_every_n_steps: Option<u32>,
    pub grad_accum_in_fp32: bool,
    pub log_memory_usage: bool,

    // evaluation
    pub eval_task_max_docs: Option<usize>,
    pub eval_tasks: Vec<psyche_eval::Task>,
    pub prompt_task: bool,
    pub heldout_eval_config: Option<HeldoutEvalConfig>,

    // logging
    pub wandb_info: Option<WandBInfo>,

    // debugging
    pub write_gradients_dir: Option<PathBuf>,

    // checkpointing
    pub checkpoint_config: Option<CheckpointConfig>,

    // configurable dummy training time (in seconds) for this client - relevant just for testing
    pub dummy_training_delay_secs: Option<u64>,

    pub sidecar_port: Option<u16>,

    /// Tier-0 suffix gate schedule (progressive growth). Applied as a runtime config override.
    pub suffix_gate_config: Option<psyche_modeling::SuffixGateConfig>,

    /// Distillation config: if set, tier-0 produces teacher logits, tier>0 consumes them.
    pub distillation_config: Option<psyche_modeling::DistillationConfig>,

    /// DisTrO apply mode (`sign` default or `raw`).
    pub distro_apply_mode: DistroApplyMode,

    /// DisTrO aggregation mode (`legacy` default or `diloco-lite`).
    pub distro_aggregate_mode: DistroAggregateMode,

    /// DisTrO value mode (`auto`, `sign`, `raw`).
    pub distro_value_mode: DistroValueMode,

    /// Experimental: apply updates from only one trainer slot per step (drop other trainers).
    pub distro_apply_only_trainer_index: Option<u16>,

    /// Guarded raw-v2 config.
    pub distro_raw_config: DistroRawConfig,

    /// DiLoCo-lite aggregation config.
    pub distro_diloco_lite_config: DistroDilocoLiteConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_apply_matformer_checkpoint_tier_overrides_auto() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        let (uses_sliced, effective_tier) = apply_matformer_checkpoint_tier_overrides(
            &config,
            &MatformerLoadStrategy::Auto,
            false,
            2,
        );
        assert!(uses_sliced);
        assert_eq!(effective_tier, 0);
    }

    #[test]
    fn test_apply_matformer_checkpoint_tier_overrides_universal() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        let (uses_sliced, effective_tier) = apply_matformer_checkpoint_tier_overrides(
            &config,
            &MatformerLoadStrategy::Universal,
            false,
            2,
        );
        assert!(uses_sliced);
        assert_eq!(effective_tier, 2);
    }

    #[test]
    fn test_validate_no_double_slicing_universal() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        let err = validate_no_double_slicing(&config, 1, &MatformerLoadStrategy::Universal, true)
            .unwrap_err();
        assert!(matches!(err, InitRunError::DoubleSlicingDetected { .. }));
    }

    #[test]
    fn test_validate_no_double_slicing_auto() {
        let config = json!({
            "intermediate_size": 1024,
            "matformer_tier": 1
        });
        assert!(
            validate_no_double_slicing(&config, 1, &MatformerLoadStrategy::Auto, true,).is_ok()
        );
    }
}

/// Print MatFormer configuration summary with ASCII art header.
/// Called after model loading to provide clear visibility into tier configuration.
fn print_matformer_summary(
    checkpoint_path: &str,
    cli_tier: u8,
    effective_tier: u8,
    uses_sliced_checkpoint: bool,
    load_strategy: &MatformerLoadStrategy,
    helper_fraction: f32,
    intermediate_size: u64,
    active_intermediate_size: u64,
) {
    let helper_status = if helper_fraction > 0.0 {
        if uses_sliced_checkpoint {
            "Auto-disabled (sliced checkpoint)"
        } else {
            "Enabled"
        }
    } else {
        "Disabled"
    };

    let tier_match = if cli_tier == effective_tier {
        "✓"
    } else {
        "≠"
    };
    let capacity_pct = (active_intermediate_size * 100) / intermediate_size;

    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║                      MATFORMER CONFIGURATION                     ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Checkpoint:       {:<45} ║",
        truncate_path(checkpoint_path, 45)
    );
    eprintln!(
        "║  Load strategy:    {:<45} ║",
        format!("{:?}", load_strategy)
    );
    eprintln!(
        "║  Sliced checkpoint: {:<44} ║",
        if uses_sliced_checkpoint { "Yes" } else { "No" }
    );
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!("║  CLI tier:         {:<45} ║", cli_tier);
    eprintln!(
        "║  Effective tier:   {:<45} ║",
        format!("{} {}", effective_tier, tier_match)
    );
    eprintln!("║  Helper mode:      {:<45} ║", helper_status);
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  FFN width:        {:<45} ║",
        format!(
            "{} / {} ({}%)",
            active_intermediate_size, intermediate_size, capacity_pct
        )
    );
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");

    // Validation warnings
    if cli_tier != effective_tier {
        eprintln!();
        eprintln!(
            "[WARNING] CLI tier ({}) differs from effective tier ({})",
            cli_tier, effective_tier
        );
        if uses_sliced_checkpoint {
            eprintln!(
                "          Sliced checkpoint detected - using tier 0 to avoid double-slicing."
            );
            eprintln!("          This is expected behavior for pre-truncated checkpoints.");
        }
    }

    if helper_fraction > 0.0 && uses_sliced_checkpoint {
        eprintln!();
        eprintln!("[INFO] Helper mode requested but auto-disabled for sliced checkpoint.");
        eprintln!("       Sliced checkpoints have no suffix neurons to sample from.");
    }

    if helper_fraction > 0.0 && cli_tier > 0 && !uses_sliced_checkpoint {
        eprintln!();
        eprintln!("[WARNING] Helper mode with tier > 0 on full checkpoint.");
        eprintln!("          This is an advanced configuration. Ensure gradients align correctly.");
    }

    eprintln!();
}

/// Truncate a path string to fit within max_len, adding "..." prefix if needed.
fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - (max_len - 3)..])
    }
}

async fn resolve_matformer_local_repo_path(
    base: &Path,
    tier: u8,
    strategy: MatformerLoadStrategy,
) -> Result<(Option<PathBuf>, bool), InitRunError> {
    let base_exists = tokio::fs::try_exists(base).await?;
    if tier == 0 {
        return Ok((base_exists.then(|| base.to_path_buf()), false));
    }

    let tier_path = base
        .file_name()
        .and_then(|n| n.to_str())
        .map(|name| base.with_file_name(format!("{name}-tier{tier}")))
        .unwrap_or_else(|| base.to_path_buf());
    let tier_exists = tokio::fs::try_exists(&tier_path).await?;

    match strategy {
        MatformerLoadStrategy::Universal => Ok((base_exists.then(|| base.to_path_buf()), false)),
        MatformerLoadStrategy::Auto => {
            if tier_exists {
                Ok((Some(tier_path), true))
            } else if base_exists {
                Ok((Some(base.to_path_buf()), false))
            } else {
                Ok((None, false))
            }
        }
        MatformerLoadStrategy::Sliced => {
            if tier_exists {
                Ok((Some(tier_path), true))
            } else if base_exists {
                Err(InitRunError::MissingMatformerTierCheckpoint {
                    path: tier_path.to_string_lossy().to_string(),
                    tier,
                })
            } else {
                Ok((None, false))
            }
        }
    }
}

/// Infer matformer tier from checkpoint config.json.
/// Returns (inferred_tier, base_intermediate_size) if determinable.
fn infer_tier_from_checkpoint_config(config: &serde_json::Value) -> (Option<u8>, Option<u64>) {
    let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_u64());

    // Check for explicit tier stored in checkpoint
    if let Some(tier) = config.get("matformer_tier").and_then(|v| v.as_u64()) {
        let base_size = config
            .get("matformer_base_intermediate_size")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                // Infer base size from tier if not explicitly stored
                intermediate_size.and_then(|size| size.checked_shl(tier as u32))
            });
        return (Some(tier as u8), base_size);
    }

    // No explicit tier - check if base size stored (can infer tier)
    if let (Some(base), Some(current)) = (
        config
            .get("matformer_base_intermediate_size")
            .and_then(|v| v.as_u64()),
        intermediate_size,
    ) {
        if base > 0 && current > 0 && base >= current {
            let ratio = base / current;
            if ratio.is_power_of_two() {
                return (Some(ratio.trailing_zeros() as u8), Some(base));
            }
        }
    }

    (None, intermediate_size)
}

fn apply_matformer_checkpoint_tier_overrides(
    checkpoint_config: &serde_json::Value,
    load_strategy: &MatformerLoadStrategy,
    uses_sliced_checkpoint: bool,
    matformer_tier_for_loading: u8,
) -> (bool, u8) {
    let (checkpoint_tier, _) = infer_tier_from_checkpoint_config(checkpoint_config);
    let uses_sliced_checkpoint =
        uses_sliced_checkpoint || checkpoint_tier.map(|tier| tier > 0).unwrap_or(false);
    let mut matformer_tier_for_loading = matformer_tier_for_loading;

    if uses_sliced_checkpoint
        && matches!(
            load_strategy,
            MatformerLoadStrategy::Auto | MatformerLoadStrategy::Sliced
        )
    {
        matformer_tier_for_loading = 0;
    }

    (uses_sliced_checkpoint, matformer_tier_for_loading)
}

/// Validate that the requested tier configuration won't cause double-slicing.
/// Returns an error if a sliced checkpoint would be further sliced.
fn validate_no_double_slicing(
    checkpoint_config: &serde_json::Value,
    cli_tier: u8,
    load_strategy: &MatformerLoadStrategy,
    uses_sliced_checkpoint: bool,
) -> Result<(), InitRunError> {
    let (checkpoint_tier, _) = infer_tier_from_checkpoint_config(checkpoint_config);

    // If we detected this is a sliced checkpoint (via naming or explicit tier)
    let is_sliced = uses_sliced_checkpoint || checkpoint_tier.map(|t| t > 0).unwrap_or(false);

    if is_sliced && cli_tier > 0 {
        // Check if user is trying to further slice an already-sliced checkpoint
        match load_strategy {
            MatformerLoadStrategy::Universal => {
                // Universal mode bypasses sliced detection - this is dangerous
                return Err(InitRunError::DoubleSlicingDetected {
                    checkpoint_tier: checkpoint_tier.unwrap_or(0),
                    cli_tier,
                    hint: "Use --matformer-load-strategy auto or specify --matformer-tier 0"
                        .to_string(),
                });
            }
            MatformerLoadStrategy::Auto | MatformerLoadStrategy::Sliced => {
                // These modes correctly set effective_tier=0 for sliced checkpoints
                // Double-slicing is prevented by the effective_tier logic
            }
        }
    }

    Ok(())
}

fn schema_hash_for_config(
    architecture: model::LLMArchitecture,
    config: &serde_json::Value,
    parameter_names: &[String],
) -> [u8; 32] {
    let arch = format!("{architecture:?}");
    let mut names = parameter_names.to_vec();
    names.sort();
    let config_json = serde_json::to_string(config).expect("config value is always serializable");
    let mut data = Vec::with_capacity(arch.len() + config_json.len() + names.len() * 32);
    data.extend_from_slice(arch.as_bytes());
    data.push(0);
    data.extend_from_slice(config_json.as_bytes());
    data.push(0);
    for name in names {
        data.extend_from_slice(name.as_bytes());
        data.push(0);
    }
    sha256(&data)
}

fn canonicalize_config_for_schema(
    mut config: serde_json::Value,
    matformer_tier: u8,
    uses_sliced_checkpoint: bool,
) -> serde_json::Value {
    if let Some(obj) = config.as_object_mut() {
        // Tier-sliced checkpoints may include MatFormer metadata fields that are
        // absent from universal checkpoints. Normalize them away so all tiers
        // compute the same canonical schema hash.
        obj.remove("matformer_base_intermediate_size");
        // Some NanoGPT checkpoints carry extra MatFormer metadata that does not
        // affect parameter shapes/names, and may be missing from tier slices.
        // Strip it for canonical hashing to avoid schema mismatches.
        obj.remove("matformer_mixer_rank");
        obj.remove("matformer_mixer_tier");
        obj.remove("matformer_mixer_alpha");
        obj.remove("matformer_mixer_loss_scale");

        if uses_sliced_checkpoint && matformer_tier > 0 {
            if let Some(value) = obj.get_mut("intermediate_size") {
                if let Some(base) = value.as_u64() {
                    if let Some(scaled) = base.checked_shl(matformer_tier as u32) {
                        *value = serde_json::Value::from(scaled);
                    }
                }
            }
        }
        // ALWAYS set matformer_tier to 0 for canonical comparison
        // This ensures full checkpoints (no field) and sliced checkpoints (field=0)
        // produce the same hash for heterogeneous tier training
        obj.insert("matformer_tier".to_string(), serde_json::Value::from(0));
    }
    config
}

#[derive(Debug, Error)]
pub enum InitRunError {
    #[error("No model provided in Coordinator state, nothing to do.")]
    NoModel,

    #[error("Model is Ephemeral, it's impossible to join this run.")]
    ModelIsEphemeral,

    #[error("failed to read local model info: {0}")]
    LocalModelLoad(#[from] io::Error),

    #[error("failed to read HF model info: {0}")]
    HfModelLoad(#[from] hf_hub::api::tokio::ApiError),

    #[error("model loading thread crashed")]
    ModelLoadingThreadCrashed(JoinError),

    #[error("failed to load model: {0}")]
    ModelLoad(#[from] ModelLoadError),

    #[error("Couldn't load tokenizer: {0}")]
    TokenizerLoad(#[from] AutoTokenizerError),

    // TODO refactor data provider for real errors
    #[error("Couldn't initialize data provider: {0}")]
    DataProviderConnect(anyhow::Error),

    #[error("wandb setup thread crashed")]
    WandbThreadCrashed(JoinError),

    #[error("wandb failed to create run: {0}")]
    WandbLoad(#[from] wandb::ApiError),

    #[error("could not parse config: {0}")]
    FailedToParseConfig(#[from] serde_json::Error),

    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[cfg(feature = "python")]
    #[error("Python distributed error: {0}")]
    PythonDistributedError(#[from] psyche_modeling::PythonDistributedCausalLMError),

    #[cfg(feature = "python")]
    #[error("Python model error: {0}")]
    PythonModelError(#[from] psyche_modeling::PythonCausalLMError),

    #[cfg(feature = "python")]
    #[error("Python distributed trainer error: {0}")]
    PythonDistributedTrainerError(#[from] psyche_modeling::PythonDistributedTrainerError),

    #[error("Required MatFormer tier checkpoint for tier {tier} not found at {path}")]
    MissingMatformerTierCheckpoint { path: String, tier: u8 },

    #[error(
        "Double-slicing detected: checkpoint is tier {checkpoint_tier}, CLI requests tier {cli_tier}. {hint}"
    )]
    DoubleSlicingDetected {
        checkpoint_tier: u8,
        cli_tier: u8,
        hint: String,
    },

    #[error("Invalid distillation config: {reason}")]
    InvalidDistillationConfig { reason: String },

    #[error("Invalid suffix gate config: {reason}")]
    InvalidSuffixGateConfig { reason: String },

    #[error("Invalid DisTrO config: {reason}")]
    InvalidDistroConfig { reason: String },
}

enum RawLoadedModelType {
    ParallelNativeModels(Vec<Box<dyn CausalLM>>),
    #[cfg(feature = "python")]
    Python(psyche_modeling::PythonCausalLM),
    #[cfg(feature = "python")]
    PythonDistributed(psyche_modeling::PythonDistributedCausalLM),
}

struct RawLoadedModel {
    models: RawLoadedModelType,
    tokenizer: Arc<Tokenizer>,
    model_task_runner: ModelTaskRunner,
    checkpoint_extra_files: Vec<PathBuf>,
    parameter_names: Arc<Vec<String>>,
    matformer_effective_tier: u8,
    matformer_base_intermediate_size: Option<u64>,
}

type OneshotModelParameterSender = oneshot::Sender<HashMap<String, Tensor>>;
type OneShotModelConfigSender = oneshot::Sender<(String, Tokenizer)>;

pub struct RunInitConfigAndIO<T: NodeIdentity, A: AuthenticatableIdentity> {
    pub init_config: RunInitConfig<T, A>,

    pub tx_health_check: UnboundedSender<HealthChecks<T>>,
    pub tx_witness: UnboundedSender<OpportunisticData>,
    pub tx_checkpoint: UnboundedSender<model::HubRepo>,
    pub tx_schema: UnboundedSender<ModelSchemaInfo>,
    pub tx_model: UnboundedSender<HashMap<String, Tensor>>,
    pub tx_parameters_req: UnboundedSender<(Vec<String>, OneshotModelParameterSender)>,
    pub tx_config: UnboundedSender<(String, String)>,
    pub tx_distro_result: UnboundedSender<DistroBroadcastAndPayload>,
    pub tx_request_download: UnboundedSender<(BlobTicket, Tag)>,
    pub tx_request_model_config: UnboundedSender<OneShotModelConfigSender>,
    pub tx_broadcast_finished: UnboundedSender<FinishedBroadcast>,
    pub tx_teacher_logits: UnboundedSender<psyche_network::TransmittableTeacherLogits>,

    pub metrics: Arc<ClientMetrics>,
}

struct LoadedCheckpoint {
    source: PretrainedSource<AutoConfig>,
    tokenizer: Arc<Tokenizer>,
    checkpoint_extra_files: Vec<PathBuf>,
    uses_sliced_checkpoint: bool,
    matformer_tier_for_loading: u8,
    matformer_checkpoint_path: Option<String>,
}

impl<T: NodeIdentity, A: AuthenticatableIdentity + 'static> RunInitConfigAndIO<T, A> {
    /// Call this on first warmup - when we need to enter the run, we have to load the model, connect to the data server, etc
    pub async fn init_run(
        self,
        state: Coordinator<T>,
    ) -> Result<StepStateMachine<T, A>, InitRunError> {
        let Self {
            init_config,
            tx_witness,
            tx_health_check,
            tx_checkpoint,
            tx_schema,
            tx_model,
            tx_config,
            tx_parameters_req,
            tx_distro_result,
            tx_request_download,
            tx_request_model_config,
            tx_broadcast_finished,
            tx_teacher_logits,
            metrics,
        } = self;

        tch::manual_seed(1337);

        // Check device availability early
        if !init_config.device.is_probably_available() {
            return Err(InitRunError::ModelLoad(
                psyche_modeling::ModelLoadError::UnavailbleDevice(init_config.device),
            ));
        }

        let model::Model::LLM(llm) = state.model;

        if let Some(cfg) = init_config.suffix_gate_config.as_ref() {
            if cfg.gate_tier > 60 {
                return Err(InitRunError::InvalidSuffixGateConfig {
                    reason: format!("suffix_gate gate_tier is too large (got {})", cfg.gate_tier),
                });
            }
            if init_config.tensor_parallelism > 1 && init_config.matformer_tier == 0 {
                warn!(
                    tensor_parallelism = init_config.tensor_parallelism,
                    "Suffix gate is currently ignored for tier-0 under tensor parallelism; run with --tensor-parallelism 1 if you need suffix gating."
                );
            }
        }

        if let Some(cfg) = init_config.distillation_config.as_ref() {
            if !cfg.beta_max.is_finite() || cfg.beta_max < 0.0 {
                return Err(InitRunError::InvalidDistillationConfig {
                    reason: format!("beta_max must be finite and >= 0 (got {})", cfg.beta_max),
                });
            }
            if !cfg.temperature.is_finite() || cfg.temperature <= 0.0 {
                return Err(InitRunError::InvalidDistillationConfig {
                    reason: format!(
                        "temperature must be finite and > 0 (got {})",
                        cfg.temperature
                    ),
                });
            }
            if cfg.top_k == 0 {
                return Err(InitRunError::InvalidDistillationConfig {
                    reason: "top_k must be >= 1".to_string(),
                });
            }
            if let Some(n) = cfg.logits_to_keep {
                if n < 2 {
                    return Err(InitRunError::InvalidDistillationConfig {
                        reason: format!("logits_to_keep must be >= 2 (got {n})"),
                    });
                }
                if (n as u64) > (llm.max_seq_len as u64) {
                    return Err(InitRunError::InvalidDistillationConfig {
                        reason: format!(
                            "logits_to_keep ({n}) exceeds max_seq_len ({})",
                            llm.max_seq_len
                        ),
                    });
                }
            }
            if let Some(min_q) = cfg.min_teacher_topk_mass {
                if !min_q.is_finite() || !(0.0..=1.0).contains(&min_q) {
                    return Err(InitRunError::InvalidDistillationConfig {
                        reason: format!(
                            "min_teacher_topk_mass must be finite and in [0,1] (got {min_q})"
                        ),
                    });
                }
            }
            if !cfg.kd_q_topk_mass_floor.is_finite()
                || cfg.kd_q_topk_mass_floor < 0.0
                || cfg.kd_q_topk_mass_floor > 1.0
            {
                return Err(InitRunError::InvalidDistillationConfig {
                    reason: format!(
                        "kd_q_topk_mass_floor must be finite and in [0,1] (got {})",
                        cfg.kd_q_topk_mass_floor
                    ),
                });
            }
        }

        if init_config.distro_raw_config.enabled
            && init_config.distro_apply_mode != DistroApplyMode::Raw
        {
            return Err(InitRunError::InvalidDistroConfig {
                reason: "raw-v2 is enabled but distro_apply_mode != raw".to_string(),
            });
        }
        if !init_config.distro_raw_config.scale_multiplier.is_finite()
            || init_config.distro_raw_config.scale_multiplier < 0.0
        {
            return Err(InitRunError::InvalidDistroConfig {
                reason: format!(
                    "distro_raw_scale_multiplier must be finite and >= 0 (got {})",
                    init_config.distro_raw_config.scale_multiplier
                ),
            });
        }
        if !init_config.distro_raw_config.scale_max.is_finite()
            || init_config.distro_raw_config.scale_max <= 0.0
        {
            return Err(InitRunError::InvalidDistroConfig {
                reason: format!(
                    "distro_raw_scale_max must be finite and > 0 (got {})",
                    init_config.distro_raw_config.scale_max
                ),
            });
        }
        if !init_config.distro_raw_config.abs_clip_mult.is_finite()
            || init_config.distro_raw_config.abs_clip_mult < 0.0
        {
            return Err(InitRunError::InvalidDistroConfig {
                reason: format!(
                    "distro_raw_abs_clip_mult must be finite and >= 0 (got {})",
                    init_config.distro_raw_config.abs_clip_mult
                ),
            });
        }
        if !init_config.distro_raw_config.sign_equiv_mult.is_finite()
            || init_config.distro_raw_config.sign_equiv_mult <= 0.0
        {
            return Err(InitRunError::InvalidDistroConfig {
                reason: format!(
                    "distro_raw_sign_equiv_mult must be finite and > 0 (got {})",
                    init_config.distro_raw_config.sign_equiv_mult
                ),
            });
        }
        if init_config.distro_raw_config.enabled
            && matches!(
                init_config.distro_raw_config.norm_mode,
                psyche_modeling::DistroRawNormMode::MatchSignEquivalent
                    | psyche_modeling::DistroRawNormMode::MatchSignEquivalentNnz
            )
            && init_config.distro_raw_config.scale_max < 1.0e6
        {
            warn!(
                "distro_raw_scale_max={} is likely too low for match-sign-equivalent modes; expect heavy clamp and sign-mismatched update magnitudes",
                init_config.distro_raw_config.scale_max
            );
        }
        if init_config.distro_raw_config.norm_mode != psyche_modeling::DistroRawNormMode::Off
            && !init_config.distro_raw_config.enabled
        {
            warn!(
                "distro_raw_norm_mode={} has no effect unless distro_raw_v2_enabled=true",
                match init_config.distro_raw_config.norm_mode {
                    psyche_modeling::DistroRawNormMode::Off => "off",
                    psyche_modeling::DistroRawNormMode::MatchPreL2 => "match-pre-l2",
                    psyche_modeling::DistroRawNormMode::MatchSignEquivalent => {
                        "match-sign-equivalent"
                    }
                    psyche_modeling::DistroRawNormMode::MatchSignEquivalentNnz => {
                        "match-sign-equivalent-nnz"
                    }
                }
            );
        }

        match &llm.optimizer {
            OptimizerDefinition::Distro { quantize_1bit, .. } => {
                if init_config.distro_value_mode == DistroValueMode::Raw && *quantize_1bit {
                    warn!(
                        "distro_value_mode=raw overrides optimizer quantize_1bit=true; transmitting non-quantized sparse values"
                    );
                }
                if init_config.distro_value_mode == DistroValueMode::Sign && !*quantize_1bit {
                    info!(
                        "distro_value_mode=sign overrides optimizer quantize_1bit=false; transmitting sign-quantized sparse values"
                    );
                }
            }
            _ => {
                if init_config.distro_apply_mode != DistroApplyMode::Sign
                    || init_config.distro_value_mode != DistroValueMode::Auto
                    || init_config.distro_raw_config.enabled
                {
                    warn!(
                        "DisTrO flags are set but optimizer is not DisTrO; these options will be ignored"
                    );
                }
            }
        }

        let hub_read_token = init_config.hub_read_token.clone();
        let hub_max_concurrent_downloads = init_config.hub_max_concurrent_downloads;
        let heldout_eval_config = init_config.heldout_eval_config;
        let data_future = async {
            debug!("Setting up data provider from {:?}", llm.data_location);
            let (data_provider, heldout_evaluator) = match llm.data_location {
                LLMTrainingDataLocation::Server(data_server) => (
                    DataProvider::Server(
                        DataProviderTcpClient::connect(
                            (&data_server).into(),
                            init_config.network_identity,
                            init_config.private_key,
                        )
                        .await?,
                    ),
                    if heldout_eval_config.is_some() {
                        return Err(anyhow::anyhow!(
                            "heldout eval is not supported for server-backed datasets"
                        ));
                    } else {
                        None
                    },
                ),
                LLMTrainingDataLocation::Local(url) => {
                    let url: String = (&url).into();
                    let dir = if std::fs::exists(&url).unwrap_or_default() {
                        PathBuf::from(url)
                    } else {
                        download_dataset_repo_async(
                            url.clone(),
                            None,
                            None,
                            hub_read_token,
                            Some(hub_max_concurrent_downloads),
                            false,
                        )
                        .await?
                        .first()
                        .ok_or(anyhow::anyhow!("No files downloaded for {url}"))?
                        .parent()
                        .unwrap()
                        .into()
                    };
                    let train_split = if heldout_eval_config.is_some() {
                        LocalDataSplit::Train
                    } else {
                        LocalDataSplit::All
                    };
                    let train_provider = LocalDataProvider::new_from_directory_with_split(
                        dir.clone(),
                        TokenSize::TwoBytes,
                        llm.max_seq_len as usize,
                        Shuffle::DontShuffle,
                        train_split,
                    )?;
                    let heldout = if let Some(cfg) = heldout_eval_config {
                        Some(Arc::new(tokio::sync::Mutex::new(
                            HeldoutEvaluator::new_local(
                                LocalDataProvider::new_from_directory_with_split(
                                    dir,
                                    TokenSize::TwoBytes,
                                    llm.max_seq_len as usize,
                                    Shuffle::DontShuffle,
                                    LocalDataSplit::Validation,
                                )?,
                                cfg,
                            ),
                        )))
                    } else {
                        None
                    };
                    (DataProvider::Local(train_provider), heldout)
                }
                LLMTrainingDataLocation::Dummy => (
                    DataProvider::Dummy(DummyDataProvider::new(
                        TokenSize::TwoBytes,
                        llm.max_seq_len as usize,
                        u64::MAX,
                    )),
                    if heldout_eval_config.is_some() {
                        return Err(anyhow::anyhow!(
                            "heldout eval is not supported for dummy datasets"
                        ));
                    } else {
                        None
                    },
                ),
                LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
                    location,
                    token_size_in_bytes,
                    shuffle,
                }) => {
                    let file_urls = FileURLs::from_location(&location).await?;
                    (
                        DataProvider::Http(HttpDataProvider::new(
                            file_urls,
                            token_size_in_bytes,
                            llm.max_seq_len,
                            shuffle,
                        )?),
                        if heldout_eval_config.is_some() {
                            return Err(anyhow::anyhow!(
                                "heldout eval is not supported for HTTP datasets"
                            ));
                        } else {
                            None
                        },
                    )
                }
                LLMTrainingDataLocation::WeightedHttp(config_url) => (
                    DataProvider::WeightedHttp(
                        WeightedDataProvider::<HttpDataProvider>::from_config_url(
                            &String::from(&config_url),
                            llm.max_seq_len,
                        )
                        .await?,
                    ),
                    if heldout_eval_config.is_some() {
                        return Err(anyhow::anyhow!(
                            "heldout eval is not supported for weighted HTTP datasets"
                        ));
                    } else {
                        None
                    },
                ),
                LLMTrainingDataLocation::Preprocessed(url) => {
                    let url: String = (&url).into();
                    let dir = if std::fs::exists(&url).unwrap_or_default() {
                        PathBuf::from(url)
                    } else {
                        download_dataset_repo_async(
                            url.clone(),
                            None,
                            None,
                            hub_read_token,
                            Some(hub_max_concurrent_downloads),
                            false,
                        )
                        .await?
                        .first()
                        .ok_or(anyhow::anyhow!("No files downloaded for {url}"))?
                        .parent()
                        .unwrap()
                        .into()
                    };
                    let train_provider = PreprocessedDataProvider::new_from_directory(
                        dir.clone(),
                        llm.max_seq_len as usize,
                        Shuffle::DontShuffle,
                        Some(Split::Train),
                        None,
                    )?;
                    let heldout = if let Some(cfg) = heldout_eval_config {
                        Some(Arc::new(tokio::sync::Mutex::new(
                            HeldoutEvaluator::new_preprocessed(
                                PreprocessedDataProvider::new_from_directory(
                                    dir,
                                    llm.max_seq_len as usize,
                                    Shuffle::DontShuffle,
                                    Some(Split::Validation),
                                    None,
                                )?,
                                cfg,
                            ),
                        )))
                    } else {
                        None
                    };
                    (DataProvider::Preprocessed(train_provider), heldout)
                }
            };
            Ok((data_provider, heldout_evaluator))
        };

        // Avoid moving these out of `init_config` into spawned tasks; we still need the originals
        // later when wiring up the training state machine.
        let suffix_gate_config_for_model = init_config.suffix_gate_config.clone();
        let distillation_config_for_model = init_config.distillation_config.clone();

        let model_future: JoinHandle<Result<RawLoadedModel, InitRunError>> = match &llm.architecture
        {
            model::LLMArchitecture::HfLlama
            | model::LLMArchitecture::HfDeepseek
            | model::LLMArchitecture::HfNanoGPT
            | model::LLMArchitecture::HfAuto => match &llm.checkpoint {
                model::Checkpoint::Dummy(_) => tokio::spawn(async move {
                    let tokenizer = Arc::new(Tokenizer::new(ModelWrapper::WordLevel(
                        WordLevel::builder().build().unwrap(),
                    )));

                    let mut parameter_names = LlamaConfig::dummy().get_parameter_names();
                    parameter_names.sort();
                    let parameter_names = Arc::new(parameter_names);

                    let model = RawLoadedModel {
                        models: RawLoadedModelType::ParallelNativeModels(
                            (0..(init_config.data_parallelism * init_config.tensor_parallelism))
                                .map(|_| {
                                    if let Some(training_delay) =
                                        init_config.dummy_training_delay_secs
                                    {
                                        Box::new(DummyModel::new(training_delay))
                                            as Box<dyn CausalLM>
                                    } else {
                                        Box::new(DummyModel::default()) as Box<dyn CausalLM>
                                    }
                                })
                                .collect(),
                        ),
                        tokenizer: tokenizer.clone(),
                        checkpoint_extra_files: vec![],
                        model_task_runner: ModelTaskRunner::new(
                            vec![],
                            false,
                            tokenizer.clone(),
                            None,
                            0,
                        ),
                        parameter_names: parameter_names.clone(),
                        matformer_effective_tier: 0,
                        matformer_base_intermediate_size: None,
                    };
                    #[allow(clippy::arc_with_non_send_sync)]
                    let config = &PretrainedSource::ConfigAndTensors(
                        AutoConfig::Llama(LlamaConfig::dummy()),
                        Arc::new(psyche_modeling::get_dummy_parameters()),
                    )
                    .serialize_config()?;
                    let tokenizer = tokenizer.to_string(false).unwrap();
                    info!("Config Uploaded: {}", config);
                    tx_config.send((config.to_string(), tokenizer)).unwrap();
                    Ok(model)
                }),
                model::Checkpoint::Hub(_) | model::Checkpoint::P2P(_) => tokio::spawn(async move {
                    let suffix_gate_config_for_model = suffix_gate_config_for_model;
                    let distillation_config_for_model = distillation_config_for_model;
                    let checkpoint = llm.checkpoint;
                    let loaded_checkpoint = match checkpoint {
                        model::Checkpoint::Hub(hub_repo) => {
                            let repo_id: String = (&hub_repo.repo_id).into();
                            let potential_local_path = PathBuf::from(repo_id.clone());
                            let revision = hub_repo.revision.map(|bytes| (&bytes).into());

                            let (maybe_local_path, sliced_checkpoint) =
                                resolve_matformer_local_repo_path(
                                    &potential_local_path,
                                    init_config.matformer_tier,
                                    init_config.matformer_load_strategy,
                                )
                                .await?;

                            let (repo_files, checkpoint_path, effective_tier_for_load) = if revision
                                .is_none()
                                && maybe_local_path
                                    .as_ref()
                                    .map(|p| p.exists())
                                    .unwrap_or(false)
                            {
                                let repo_root = maybe_local_path.unwrap();
                                let mut ret = Vec::new();
                                let mut read_dir = tokio::fs::read_dir(repo_root.clone()).await?;
                                while let Some(dir_entry) = read_dir.next_entry().await? {
                                    ret.push(dir_entry.path())
                                }
                                let effective_tier_for_load = if sliced_checkpoint {
                                    0
                                } else {
                                    init_config.matformer_tier
                                };
                                (
                                    ret,
                                    Some(repo_root.to_string_lossy().to_string()),
                                    effective_tier_for_load,
                                )
                            } else {
                                info!(
                                    "Downloading {}, revision: {:?} (if needed)",
                                    hub_repo.repo_id, revision
                                );
                                let repo_files = download_model_repo_async(
                                    &repo_id,
                                    revision,
                                    None,
                                    init_config.hub_read_token,
                                    Some(init_config.hub_max_concurrent_downloads),
                                    false,
                                )
                                .await?;
                                (
                                    repo_files,
                                    Some(repo_id.clone()),
                                    init_config.matformer_tier,
                                )
                            };
                            let checkpoint_extra_files = repo_files
                                .iter()
                                .filter(|file| {
                                    file.ends_with("config.json")
                                        || file.ends_with("tokenizer.json")
                                        || file.ends_with("tokenizer_config.json")
                                        || file.ends_with("special_tokens_map.json")
                                        || file.ends_with("generation_config.json")
                                        || file.ends_with(".py")
                                })
                                .cloned()
                                .collect();
                            let tokenizer = Arc::new(auto_tokenizer(&repo_files)?);
                            LoadedCheckpoint {
                                source: PretrainedSource::<AutoConfig>::RepoFiles(repo_files),
                                tokenizer,
                                checkpoint_extra_files,
                                uses_sliced_checkpoint: sliced_checkpoint,
                                matformer_tier_for_loading: effective_tier_for_load,
                                matformer_checkpoint_path: checkpoint_path,
                            }
                        }
                        model::Checkpoint::P2P(_) => {
                            let (tx_model_config_response, rx_model_config_response) =
                                oneshot::channel();
                            info!("Checkpoint is p2p, requesting model config over network");

                            tx_request_model_config
                                .send(tx_model_config_response)
                                .unwrap();

                            let (model_config, tokenizer) = rx_model_config_response.await.unwrap();
                            debug!("Got p2p info, model_config: {}", model_config);

                            let model_config = match llm.architecture {
                                model::LLMArchitecture::HfLlama => {
                                    AutoConfig::Llama(serde_json::from_str(&model_config)?)
                                }
                                model::LLMArchitecture::HfDeepseek => {
                                    AutoConfig::Deepseek(serde_json::from_str(&model_config)?)
                                }
                                model::LLMArchitecture::HfNanoGPT => {
                                    AutoConfig::NanoGPT(serde_json::from_str(&model_config)?)
                                }
                                model::LLMArchitecture::HfAuto => {
                                    #[cfg(feature = "python")]
                                    {
                                        AutoConfig::Auto(serde_json::from_str::<
                                            psyche_modeling::PythonModelConfig,
                                        >(
                                            &model_config
                                        )?)
                                    }

                                    #[cfg(not(feature = "python"))]
                                    {
                                        return Err(InitRunError::UnsupportedArchitecture(
                                            "HfAuto".to_string(),
                                        ));
                                    }
                                }
                            };
                            let parameter_names = model_config.get_parameter_names();
                            info!(
                                "Requesting {} parameters over p2p network",
                                parameter_names.len()
                            );

                            let (tx_params_response, rx_params_response) = oneshot::channel();
                            tx_parameters_req
                                .send((parameter_names, tx_params_response))
                                .unwrap();
                            #[allow(clippy::arc_with_non_send_sync)]
                            let parameters = Arc::new(rx_params_response.await.unwrap());

                            LoadedCheckpoint {
                                source: PretrainedSource::<AutoConfig>::ConfigAndTensors(
                                    model_config,
                                    parameters,
                                ),
                                tokenizer: Arc::new(tokenizer),
                                checkpoint_extra_files: vec![],
                                uses_sliced_checkpoint: false,
                                matformer_tier_for_loading: init_config.matformer_tier,
                                matformer_checkpoint_path: Some("p2p".to_string()),
                            }
                        }
                        _ => unreachable!(),
                    };

                    let source = loaded_checkpoint.source;
                    let tokenizer = loaded_checkpoint.tokenizer;
                    let checkpoint_extra_files = loaded_checkpoint.checkpoint_extra_files;
                    let mut uses_sliced_checkpoint = loaded_checkpoint.uses_sliced_checkpoint;
                    let mut matformer_tier_for_loading =
                        loaded_checkpoint.matformer_tier_for_loading;
                    let matformer_checkpoint_path = loaded_checkpoint.matformer_checkpoint_path;

                    info!("Loading model...");

                    let model_task_runner = ModelTaskRunner::new(
                        init_config.eval_tasks,
                        init_config.prompt_task,
                        tokenizer.clone(),
                        init_config.eval_task_max_docs,
                        // if doing python fsdp we only have one effective dp rank for inference
                        if init_config.data_parallelism > 1
                            && llm.architecture == model::LLMArchitecture::HfAuto
                        {
                            1
                        } else {
                            init_config.data_parallelism
                        },
                    );

                    let serialized_config = source.serialize_config()?;
                    let mut config_json: serde_json::Value =
                        serde_json::from_str(&serialized_config)?;
                    (uses_sliced_checkpoint, matformer_tier_for_loading) =
                        apply_matformer_checkpoint_tier_overrides(
                            &config_json,
                            &init_config.matformer_load_strategy,
                            uses_sliced_checkpoint,
                            matformer_tier_for_loading,
                        );
                    // Validate no double-slicing before model load
                    validate_no_double_slicing(
                        &config_json,
                        init_config.matformer_tier,
                        &init_config.matformer_load_strategy,
                        uses_sliced_checkpoint,
                    )?;

                    // Apply runtime MatFormer C2 knobs into the reported config (schema hash + UI).
                    // This keeps schema hashes stable and run configs self-describing when
                    // suffix gate and/or distillation are enabled via CLI.
                    if let Some(obj) = config_json.as_object_mut() {
                        let stab = obj
                            .entry("matformer_stabilization".to_string())
                            .or_insert_with(|| serde_json::Value::Object(Default::default()));
                        if let Some(stab_obj) = stab.as_object_mut() {
                            if let Some(cfg) = suffix_gate_config_for_model.as_ref() {
                                let value = serde_json::to_value(cfg)
                                    .expect("SuffixGateConfig is always serializable");
                                stab_obj.insert("suffix_gate".to_string(), value);
                            }
                            if let Some(cfg) = distillation_config_for_model.as_ref() {
                                let value = serde_json::to_value(cfg)
                                    .expect("DistillationConfig is always serializable");
                                stab_obj.insert("distillation".to_string(), value);
                            }
                        }
                    }
                    let serialized_config = serde_json::to_string(&config_json)
                        .expect("config_json is always serializable");
                    let mut parameter_names = match llm.architecture {
                        model::LLMArchitecture::HfLlama => {
                            let config: LlamaConfig = serde_json::from_str(&serialized_config)?;
                            config.get_parameter_names()
                        }
                        model::LLMArchitecture::HfDeepseek => {
                            let config: DeepseekConfig = serde_json::from_str(&serialized_config)?;
                            config.get_parameter_names()
                        }
                        model::LLMArchitecture::HfNanoGPT => {
                            let config: NanoGPTConfig = serde_json::from_str(&serialized_config)?;
                            config.get_parameter_names()
                        }
                        model::LLMArchitecture::HfAuto => {
                            #[cfg(feature = "python")]
                            {
                                let config: psyche_modeling::PythonModelConfig =
                                    serde_json::from_str(&serialized_config)?;
                                config.get_parameter_names()
                            }

                            #[cfg(not(feature = "python"))]
                            {
                                return Err(InitRunError::UnsupportedArchitecture(
                                    "HfAuto".to_string(),
                                ));
                            }
                        }
                    };
                    parameter_names.sort();
                    let parameter_names = Arc::new(parameter_names);
                    let attn_implementation: Option<AttentionImplementation> = match llm.data_type {
                        model::LLMTrainingDataType::Finetuning => {
                            #[cfg(feature = "parallelism")]
                            {
                                // use varlen backend if available
                                Some(AttentionImplementation::FlashAttention2)
                            }

                            #[cfg(not(feature = "parallelism"))]
                            None
                        }
                        model::LLMTrainingDataType::Pretraining => None,
                    };

                    let raw_loaded_model_type: RawLoadedModelType = if llm.architecture
                        == model::LLMArchitecture::HfAuto
                    {
                        #[cfg(feature = "python")]
                        {
                            let dp = init_config.data_parallelism;
                            let tp = init_config.tensor_parallelism;

                            tokio::task::spawn_blocking(move || {
                                if tp != 1 || dp != 1 {
                                    psyche_modeling::PythonDistributedCausalLM::new(
                                        "hf-auto".to_string(),
                                        source.try_into()?,
                                        tch::Device::cuda_if_available(),
                                        attn_implementation.unwrap_or_default(),
                                        psyche_modeling::ParallelismConfig { dp, tp },
                                        Some(llm.max_seq_len as usize),
                                        init_config.sidecar_port,
                                        None,
                                    )
                                    .map(RawLoadedModelType::PythonDistributed)
                                    .map_err(InitRunError::PythonDistributedError)
                                } else {
                                    let device =
                                        init_config.device.device_for_rank(0).ok_or_else(|| {
                                            ModelLoadError::NoDeviceForRank(0, init_config.device)
                                        })?;
                                    psyche_modeling::PythonCausalLM::new(
                                        "hf-auto",
                                        &source.try_into()?,
                                        device,
                                        attn_implementation.unwrap_or_default(),
                                        None,
                                        Some(llm.max_seq_len as usize),
                                    )
                                    .map(RawLoadedModelType::Python)
                                    .map_err(InitRunError::PythonModelError)
                                }
                            })
                            .await
                            .map_err(InitRunError::ModelLoadingThreadCrashed)??
                        }

                        #[cfg(not(feature = "python"))]
                        {
                            return Err(InitRunError::UnsupportedArchitecture(
                                "HfAuto".to_string(),
                            ));
                        }
                    } else {
                        let mut futures: Vec<
                            JoinHandle<Result<Box<dyn CausalLM>, ModelLoadError>>,
                        > = Vec::with_capacity(
                            init_config.data_parallelism * init_config.tensor_parallelism,
                        );
                        let devices = init_config.device.clone();
                        let matformer_tier = matformer_tier_for_loading;
                        let helper_fraction = init_config.matformer_helper_fraction;
                        let helper_rotation_interval =
                            init_config.matformer_helper_rotation_interval;
                        let suffix_gate_config = suffix_gate_config_for_model.clone();
                        let distillation_config = distillation_config_for_model.clone();

                        for dp in 0..init_config.data_parallelism {
                            let communicator_id: Option<CommunicatorId> =
                                match init_config.tensor_parallelism {
                                    0 | 1 => None,
                                    #[cfg(feature = "parallelism")]
                                    _ => Some(tch::CStore::new().into()),
                                    #[cfg(not(feature = "parallelism"))]
                                    _ => unimplemented!(),
                                };
                            for tp in 0..init_config.tensor_parallelism {
                                let tensor_parallelism_world =
                                    communicator_id.as_ref().map(|communicator_id| {
                                        (
                                            communicator_id.clone(),
                                            tp,
                                            init_config.tensor_parallelism,
                                        )
                                    });
                                let source = source.clone();
                                let rank = dp * init_config.tensor_parallelism + tp;
                                let devices = devices.clone();
                                let device = devices.device_for_rank(rank);
                                let suffix_gate_config = suffix_gate_config.clone();
                                let distillation_config = distillation_config.clone();
                                futures.push(tokio::task::spawn_blocking(move || {
                                        let device = device.ok_or_else(|| {
                                            ModelLoadError::NoDeviceForRank(rank, devices)
                                        })?;
                                        let kind = match device {
                                            // MPS has incomplete BF16 support; prefer FP32 for correctness.
                                            Device::Mps => Kind::Float,
                                            _ => Kind::BFloat16,
                                        };
                                        match llm.architecture {
                                            model::LLMArchitecture::HfLlama => {
                                                LlamaForCausalLM::from_pretrained_with_config_overrides(
                                                    &source.try_into()?,
                                                    Some(kind),
                                                    attn_implementation,
                                                    Some(device),
                                                    tensor_parallelism_world,
                                                    Some(llm.max_seq_len as usize),
                                                    |config| {
                                                        config.matformer_tier = matformer_tier;
                                                        config.matformer_helper_fraction = helper_fraction;
                                                        config.matformer_helper_rotation_interval =
                                                            helper_rotation_interval;
                                                        config.matformer_stabilization.suffix_gate =
                                                            suffix_gate_config;
                                                        config.matformer_stabilization.distillation =
                                                            distillation_config;
                                                    },
                                                )
                                                .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                            }
                                            model::LLMArchitecture::HfDeepseek => {
                                                DeepseekForCausalLM::from_pretrained(
                                                    &source.try_into()?,
                                                    Some(kind),
                                                    attn_implementation,
                                                    Some(device),
                                                    tensor_parallelism_world,
                                                    Some(llm.max_seq_len as usize),
                                                )
                                                .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                            }
                                            model::LLMArchitecture::HfNanoGPT => {
                                                NanoGPTForCausalLM::from_pretrained_with_config_overrides(
                                                    &source.try_into()?,
                                                    Some(kind),
                                                    attn_implementation,
                                                    Some(device),
                                                    tensor_parallelism_world,
                                                    Some(llm.max_seq_len as usize),
                                                    |config| {
                                                        config.matformer_tier = matformer_tier;
                                                        config.matformer_helper_fraction = helper_fraction;
                                                        config.matformer_helper_rotation_interval =
                                                            helper_rotation_interval;
                                                        config.matformer_stabilization.suffix_gate =
                                                            suffix_gate_config;
                                                        config.matformer_stabilization.distillation =
                                                            distillation_config;
                                                    },
                                                )
                                                .map(|x| Box::new(x) as Box<dyn CausalLM>)
                                            }
                                            model::LLMArchitecture::HfAuto => unreachable!(),
                                        }
                                    }));
                            }
                        }

                        let mut models: Vec<Box<dyn CausalLM>> = Vec::new();
                        for future in futures {
                            let model = future
                                .await
                                .map_err(InitRunError::ModelLoadingThreadCrashed)??;
                            models.push(model);
                        }

                        RawLoadedModelType::ParallelNativeModels(models)
                    };

                    debug!("Config uploaded: {}", serialized_config);
                    let serialized_tokenizer = tokenizer.to_string(false).unwrap();
                    tx_config
                        .send((serialized_config.clone(), serialized_tokenizer))
                        .unwrap();

                    let hidden_size = config_json.get("hidden_size").and_then(|v| v.as_u64());
                    let intermediate_size = config_json
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64());
                    let active_intermediate_size = intermediate_size.and_then(|h| {
                        let divisor = 1_u64.checked_shl(matformer_tier_for_loading as u32)?;
                        Some(h / divisor)
                    });
                    // Compute base intermediate size (before any tier slicing)
                    // For sliced checkpoints, scale up by CLI tier to get original base
                    let base_intermediate_size: Option<u64> = intermediate_size.and_then(|size| {
                        if uses_sliced_checkpoint && init_config.matformer_tier > 0 {
                            // Sliced checkpoint: scale up to get base
                            size.checked_shl(init_config.matformer_tier as u32)
                        } else {
                            // Full checkpoint: intermediate_size IS the base
                            Some(size)
                        }
                    });
                    let num_hidden_layers = config_json
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64());
                    let vocab_size = config_json.get("vocab_size").and_then(|v| v.as_u64());

                    let canonical_config = canonicalize_config_for_schema(
                        config_json.clone(),
                        init_config.matformer_tier,
                        uses_sliced_checkpoint,
                    );
                    let schema_info = ModelSchemaInfo {
                        schema_hash_local: schema_hash_for_config(
                            llm.architecture,
                            &config_json,
                            &parameter_names,
                        ),
                        schema_hash_canonical: schema_hash_for_config(
                            llm.architecture,
                            &canonical_config,
                            &parameter_names,
                        ),
                        parameter_count: u32::try_from(parameter_names.len()).unwrap_or(u32::MAX),
                        matformer_tier: init_config.matformer_tier,
                        uses_sliced_checkpoint,
                    };
                    if tx_schema.send(schema_info).is_err() {
                        warn!("Failed to send schema hash info; channel closed");
                    }

                    match (hidden_size, intermediate_size, active_intermediate_size) {
                        (
                            Some(hidden_size),
                            Some(intermediate_size),
                            Some(active_intermediate_size),
                        ) => {
                            info!(
                                integration_test_log_marker = %IntegrationTestLogMarker::LoadedModel,
                                checkpoint = %llm.checkpoint,
                                gpus = init_config.data_parallelism * init_config.tensor_parallelism,
                                dp = init_config.data_parallelism,
                                tp = init_config.tensor_parallelism,
                                matformer_tier = init_config.matformer_tier,
                                matformer_effective_tier = matformer_tier_for_loading,
                                matformer_load_strategy = ?init_config.matformer_load_strategy,
                                matformer_uses_sliced_checkpoint = uses_sliced_checkpoint,
                                matformer_checkpoint_path = matformer_checkpoint_path.as_deref().unwrap_or("unknown"),
                                hidden_size = hidden_size,
                                intermediate_size = intermediate_size,
                                intermediate_size_active = active_intermediate_size,
                                num_hidden_layers = num_hidden_layers.unwrap_or_default(),
                                vocab_size = vocab_size.unwrap_or_default(),
                                "loaded_model",
                            );

                            // Print MatFormer configuration summary
                            print_matformer_summary(
                                matformer_checkpoint_path.as_deref().unwrap_or("unknown"),
                                init_config.matformer_tier,
                                matformer_tier_for_loading,
                                uses_sliced_checkpoint,
                                &init_config.matformer_load_strategy,
                                init_config.matformer_helper_fraction,
                                intermediate_size,
                                active_intermediate_size,
                            );
                        }
                        _ => {
                            info!(
                                integration_test_log_marker = %IntegrationTestLogMarker::LoadedModel,
                                checkpoint = %llm.checkpoint,
                                gpus = init_config.data_parallelism * init_config.tensor_parallelism,
                                dp = init_config.data_parallelism,
                                tp = init_config.tensor_parallelism,
                                matformer_tier = init_config.matformer_tier,
                                "loaded_model",
                            );
                        }
                    }

                    Ok(RawLoadedModel {
                        models: raw_loaded_model_type,
                        tokenizer,
                        model_task_runner,
                        checkpoint_extra_files,
                        parameter_names,
                        matformer_effective_tier: matformer_tier_for_loading,
                        matformer_base_intermediate_size: base_intermediate_size,
                    })
                }),
                model::Checkpoint::Ephemeral => return Err(InitRunError::ModelIsEphemeral),
            },
        };

        // Extract step logging config before wandb_info is consumed
        let step_logging_enabled = init_config
            .wandb_info
            .as_ref()
            .map(|info| info.step_logging)
            .unwrap_or(false);
        let system_metrics_enabled = init_config
            .wandb_info
            .as_ref()
            .map(|info| info.system_metrics)
            .unwrap_or(false);
        let system_metrics_interval_secs = init_config
            .wandb_info
            .as_ref()
            .map(|info| info.system_metrics_interval_secs)
            .unwrap_or(10);

        let wandb_future: JoinHandle<Result<Option<wandb::Run>, wandb::ApiError>> = tokio::spawn({
            let run_id = String::from(&state.run_id);
            async move {
                match init_config.wandb_info {
                    Some(wandb_info) => {
                        let wandb =
                            wandb::WandB::new(wandb::BackendOptions::new(wandb_info.api_key));
                        let mut run_info = wandb::RunInfo::new(wandb_info.project)
                            .name(wandb_info.run)
                            .config((
                                (
                                    "global_batch_size_start",
                                    state.config.global_batch_size_start,
                                ),
                                ("global_batch_size_end", state.config.global_batch_size_end),
                                (
                                    "global_batch_size_warmup_tokens",
                                    state.config.global_batch_size_warmup_tokens,
                                ),
                                ("total_steps", state.config.total_steps),
                                ("run_id", run_id),
                            ));
                        if let Some(entity) = wandb_info.entity {
                            run_info = run_info.entity(entity);
                        }
                        if let Some(group) = wandb_info.group {
                            run_info = run_info.group(group);
                        }
                        match wandb.new_run(run_info.build()?).await {
                            Ok(run) => Ok(Some(run)),
                            Err(e) => {
                                error!(
                                    "[init_run] Could not connect to wandb. Will continue training without it."
                                );
                                debug!("[init_run] wandb error: {:?}", e);
                                Ok(None)
                            }
                        }
                    }
                    None => {
                        info!(
                            "[init_run] No wandb info provided. Will continue training without it."
                        );
                        Ok(None)
                    }
                }
            }
        });

        let (data, models, wandb_run) = tokio::join!(data_future, model_future, wandb_future);
        let RawLoadedModel {
            models,
            tokenizer,
            checkpoint_extra_files,
            model_task_runner,
            parameter_names,
            matformer_effective_tier,
            matformer_base_intermediate_size,
        } = models.map_err(InitRunError::ModelLoadingThreadCrashed)??;

        // TODO add data fetching for verifying, too..
        let (data_provider, heldout_evaluator) = data.map_err(InitRunError::DataProviderConnect)?;
        let data_fetcher = DataFetcher::<T, A>::new(
            data_provider,
            init_config.data_parallelism * 2,
            init_config.same_batch_warmup_steps,
            init_config.same_batch_anchor_every_steps,
            init_config.same_batch_anchor_start_step,
        );

        let trainers: Vec<Trainer> = match models {
            RawLoadedModelType::ParallelNativeModels(models) => {
                let mut tp_models: Vec<Vec<Box<dyn CausalLM>>> = Vec::new();
                for model in models {
                    if tp_models
                        .last()
                        .map(|x| x.len() == init_config.tensor_parallelism)
                        .unwrap_or(true)
                    {
                        tp_models.push(Vec::with_capacity(init_config.tensor_parallelism));
                    }
                    tp_models.last_mut().unwrap().push(model);
                }

                let data_parallel: Option<Vec<(CommunicatorId, Arc<dyn Barrier>)>> =
                    if init_config.data_parallelism > 1 {
                        #[cfg(feature = "parallelism")]
                        {
                            Some(
                                (0..init_config.tensor_parallelism)
                                    .map(|_| {
                                        (
                                            tch::CStore::new().into(),
                                            Arc::new(CancellableBarrier::new(
                                                init_config.tensor_parallelism,
                                            ))
                                                as Arc<dyn Barrier>,
                                        )
                                    })
                                    .collect(),
                            )
                        }

                        #[cfg(not(feature = "parallelism"))]
                        {
                            unimplemented!()
                        }
                    } else {
                        None
                    };

                tp_models
                    .into_iter()
                    .enumerate()
                    .map(|(dp, models)| {
                        let data_parallel = data_parallel.as_ref().map(|data_parallel| {
                            data_parallel
                                .iter()
                                .map(|(id, barrier)| DataParallel {
                                    id: id.clone(),
                                    barrier: barrier.clone(),
                                    rank: dp,
                                    world_size: init_config.data_parallelism,
                                })
                                .collect()
                        });
                        let barrier =
                            Arc::new(CancellableBarrier::new(init_config.tensor_parallelism))
                                as Arc<dyn Barrier>;
                        LocalTrainer::new(
                            ParallelModels {
                                models,
                                barrier,
                                data_parallel,
                            },
                            llm.lr_schedule,
                            llm.optimizer,
                            init_config.distro_apply_mode,
                            init_config.distro_aggregate_mode,
                            init_config.distro_value_mode,
                            init_config.distro_raw_config,
                            init_config.distro_diloco_lite_config,
                            init_config.micro_batch_size,
                            init_config.optim_stats_every_n_steps,
                            init_config.grad_accum_in_fp32,
                        )
                        .into()
                    })
                    .collect()
            }
            #[cfg(feature = "python")]
            RawLoadedModelType::Python(model) => {
                vec![
                    psyche_modeling::LocalTrainer::new(
                        ParallelModels {
                            models: vec![Box::new(model) as Box<dyn CausalLM>],
                            barrier: Arc::new(psyche_modeling::NopBarrier) as Arc<dyn Barrier>,
                            data_parallel: None,
                        },
                        llm.lr_schedule,
                        llm.optimizer,
                        init_config.distro_apply_mode,
                        init_config.distro_aggregate_mode,
                        init_config.distro_value_mode,
                        init_config.distro_raw_config,
                        init_config.distro_diloco_lite_config,
                        init_config.micro_batch_size,
                        init_config.optim_stats_every_n_steps,
                        init_config.grad_accum_in_fp32,
                    )
                    .into(),
                ]
            }
            #[cfg(feature = "python")]
            RawLoadedModelType::PythonDistributed(model) => {
                vec![
                    psyche_modeling::PythonDistributedTrainer::new(
                        model,
                        llm.lr_schedule,
                        llm.optimizer,
                        init_config.distro_apply_mode,
                        init_config.distro_aggregate_mode,
                        init_config.distro_value_mode,
                        init_config.distro_raw_config,
                        init_config.distro_diloco_lite_config,
                        init_config.micro_batch_size,
                        init_config.optim_stats_every_n_steps,
                        init_config.grad_accum_in_fp32,
                    )?
                    .into(),
                ]
            }
        };

        let wandb_run = wandb_run.map_err(InitRunError::WandbThreadCrashed)??;
        // Wrap in Arc for sharing between StatsLogger and TrainingStepMetadata
        let wandb_run = wandb_run.map(Arc::new);

        let stats_logger = StatsLogger::new(
            tokenizer,
            model_task_runner.clone(),
            llm.lr_schedule,
            wandb_run.clone(),
            metrics,
        );

        // Start system metrics logging if enabled
        let _system_metrics_task = if system_metrics_enabled {
            stats_logger.start_system_metrics_logging(system_metrics_interval_secs)
        } else {
            None
        };

        let warmup = WarmupStepMetadata {
            model_task_runner: model_task_runner.clone(),
        };

        let training = TrainingStepMetadata {
            data_fetcher,
            identity: init_config.identity,
            write_gradients_dir: init_config.write_gradients_dir,
            tx_health_check,
            tx_distro_result,
            parameter_names: parameter_names.clone(),
            matformer_tier: init_config.matformer_tier,
            distro_apply_only_trainer_index: init_config.distro_apply_only_trainer_index,

            model_task_runner: model_task_runner.clone(),
            log_memory_usage: init_config.log_memory_usage,
            step_logging_enabled,
            wandb_run,
            tx_teacher_logits,
            distillation_config: init_config.distillation_config,
            distro_value_mode: init_config.distro_value_mode,
            matformer_local_inner_steps: init_config.matformer_local_inner_steps,
        };
        let wandb_run_for_cooldown = training.wandb_run.clone();

        let witness = WitnessStepMetadata {
            model_task_runner: model_task_runner.clone(),
            identity: init_config.identity,
            tx_witness: tx_witness.clone(),
        };

        let cooldown = CooldownStepMetadata::new(
            tx_checkpoint,
            tx_model,
            init_config.checkpoint_config,
            checkpoint_extra_files,
            MatformerCheckpointInfo {
                effective_tier: matformer_effective_tier,
                base_intermediate_size: matformer_base_intermediate_size,
            },
            heldout_evaluator,
            wandb_run_for_cooldown,
            model_task_runner,
        );

        Ok(StepStateMachine::new(
            init_config.identity,
            warmup,
            training,
            witness,
            cooldown,
            trainers,
            state,
            tx_request_download,
            tx_witness,
            tx_broadcast_finished,
            stats_logger,
        ))
    }
}
