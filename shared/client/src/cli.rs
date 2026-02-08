use crate::{CheckpointConfig, HubUploadInfo, WandBInfo};

use anyhow::{anyhow, bail, Result};
use clap::Args;
use clap::ValueEnum;
use psyche_eval::tasktype_from_name;
use psyche_modeling::Devices;
use psyche_network::{DiscoveryMode, RelayKind, SecretKey};
use psyche_network_fault_injection::{FaultConfig, FaultConfigBuilder};
use psyche_tui::LogOutput;
use std::{path::PathBuf, time::Duration};

pub fn read_identity_secret_key(
    identity_secret_key_path: Option<&PathBuf>,
) -> Result<Option<SecretKey>> {
    let raw_identity_secret_key = std::env::var("RAW_IDENTITY_SECRET_KEY").ok();
    let bytes: [u8; 32] = match (raw_identity_secret_key, identity_secret_key_path) {
        (None, None) => return Ok(None),
        (Some(raw), None) => {
            let vals = hex::decode(raw)?;
            let l = vals.len();
            vals.try_into().map_err(|_| {
                anyhow!(
                    "invalid raw identity secret key, expected 32 bytes, got {}",
                    l
                )
            })?
        }

        (None, Some(key_file)) => std::fs::read(key_file)?
            .try_into()
            .map_err(|_| anyhow!("key file {key_file:?} was not 32 bytes long."))?,

        _ => unreachable!(),
    };
    Ok(Some(SecretKey::from_bytes(&bytes)))
}

pub fn print_identity_keys(key: Option<&PathBuf>) -> Result<()> {
    let key = read_identity_secret_key(key)?.ok_or_else(|| {
        anyhow!("Use --identity-secret-key-path or use `RAW_IDENTITY_SECRET_KEY` env variable")
    })?;
    println!("Public key: {}", key.public());
    println!("Secret key: {}", hex::encode(key.to_bytes()));
    Ok(())
}

fn parse_trim_quotes(s: &str) -> Result<String, String> {
    Ok(s.trim_matches('"').to_string())
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to the clients secret key. Create a new random one running `openssl rand 32 > secret.key`. If not provided a random one will be generated.
    #[clap(short, long, env)]
    pub identity_secret_key_path: Option<PathBuf>,

    /// Sets the port for the client's P2P network participation. If not provided, a random port will be chosen.
    #[clap(long, env)]
    pub bind_p2p_port: Option<u16>,

    /// Sets the network interface for the client's P2P network participation. If not provided, will bind to all interfaces.
    #[clap(long, env)]
    pub bind_p2p_interface: Option<String>,

    /// What relays to use - public n0 or the private Psyche ones
    #[clap(long, env, default_value = "psyche")]
    pub iroh_relay: RelayKind,

    /// What discovery to use - public n0 or local
    #[clap(long, env, default_value = "n0")]
    pub iroh_discovery: DiscoveryMode,

    /// Sets clients logs interface
    /// tui: Enables a terminal-based graphical interface for monitoring analytics.
    /// console: standard logs
    /// json: standard logs with json format
    #[clap(
        long,
        env,
        default_value_t = LogOutput::TUI,
        value_enum,
        ignore_case = true
    )]
    pub logs: LogOutput,

    /// An auth header string for an opentelemetry endpoint. Used for both logging and metrics.
    #[clap(long, env)]
    pub oltp_auth_header: Option<String>,

    /// A URL for sending opentelemetry metrics. probably ends in /v1/metrics
    #[clap(long, env)]
    pub oltp_metrics_url: Option<String>,

    /// A URL for sending opentelemetry traces. probably ends in /v1/traces
    #[clap(long, env)]
    pub oltp_tracing_url: Option<String>,

    /// A URL for sending opentelemetry logs. probably ends in /v1/logs
    #[clap(long, env)]
    pub oltp_logs_url: Option<String>,

    /// how often to report metrics thru opentelemetry
    #[clap(long, env,
    default_value = "60.0",
    value_parser = parse_duration_from_seconds)]
    pub oltp_report_interval: Duration,

    /// If present, output some metrics & stats via this TCP port in JSON format. Useful for debugging or local integration.
    #[clap(long, env)]
    pub metrics_local_port: Option<u16>,

    /// A unique identifier for the training run. This ID allows the client to join a specific active run.
    #[clap(long, env, value_parser = parse_trim_quotes)]
    pub run_id: String,

    #[clap(long, default_value_t = 1, env)]
    pub data_parallelism: usize,

    #[clap(long, default_value_t = 1, env)]
    pub tensor_parallelism: usize,

    #[clap(long, env, default_value_t = 1)]
    pub micro_batch_size: usize,

    /// Experimental: for the first N training steps, fetch the same canonical batch for every
    /// trainer to improve early gradient alignment across tiers.
    ///
    /// This intentionally breaks the "assigned BatchId -> actual data" provenance and is meant
    /// only for off-chain debugging/ablations.
    #[clap(long, env, default_value_t = 0)]
    pub same_batch_warmup_steps: u32,

    /// Experimental: periodic same-batch calibration schedule.
    ///
    /// When enabled, clients fetch a canonical same batch at sparse intervals for
    /// alignment diagnostics/guardrails.
    #[clap(long, env, default_value_t = 0)]
    pub same_batch_calibration_every_steps: u32,

    /// First step eligible for periodic same-batch calibration.
    #[clap(long, env, default_value_t = 1)]
    pub same_batch_calibration_start_step: u32,

    /// If true, calibration steps still run forward/backward, but distributed apply is skipped.
    #[clap(long, env, default_value_t = true)]
    pub same_batch_calibration_no_apply: bool,

    /// Experimental: number of local inner training updates per coordinator step on
    /// smaller MatFormer tiers (`tier > 0`).
    ///
    /// The global LR schedule remains keyed to the outer coordinator step (not inner-step count).
    #[clap(long, env, default_value_t = 1)]
    pub matformer_local_inner_steps: u32,

    /// Proximal strength for downweighting tier>0 peer updates during apply.
    #[clap(long, env, default_value_t = 0.0)]
    pub matformer_prox_mu: f64,

    /// If true, apply proximal downweighting only on MatFormer FFN prefix tensors.
    #[clap(long, env, default_value_t = true)]
    pub matformer_prox_prefix_only: bool,

    /// If provided, every shared gradient this client sees will be written to this directory.
    #[clap(long, env)]
    pub write_gradients_dir: Option<PathBuf>,

    #[clap(long, env)]
    pub eval_tasks: Option<String>,

    #[clap(long, default_value_t = 42, env)]
    pub eval_seed: u64,

    #[clap(long, env)]
    pub eval_task_max_docs: Option<usize>,

    /// If > 0, run held-out validation loss evaluation during cooldown/checkpointing.
    /// This uses validation data only (where supported) and logs `heldout_eval`.
    #[clap(long, env, default_value_t = 0)]
    pub heldout_eval_batches: usize,

    /// Batch size (number of sequences) per held-out eval batch.
    #[clap(long, env, default_value_t = 16)]
    pub heldout_eval_batch_size: usize,

    // enable the execution of the model prompting task
    #[clap(long, env)]
    pub prompt_task: bool,

    /// If provided, every model parameters update will be save in this directory after each epoch.
    #[clap(long, env)]
    pub checkpoint_dir: Option<PathBuf>,

    /// Path to the Hugging Face repository containing model data and configuration.
    #[clap(long, env)]
    pub hub_repo: Option<String>,

    #[clap(long, env, default_value_t = 3)]
    pub hub_max_concurrent_downloads: usize,

    #[clap(long, env)]
    pub wandb_project: Option<String>,

    #[clap(long, env)]
    pub wandb_run: Option<String>,

    #[clap(long, env)]
    pub wandb_group: Option<String>,

    #[clap(long, env)]
    pub wandb_entity: Option<String>,

    /// Enable WandB step-level logging (every training step, not just rounds)
    #[clap(long, env, default_value_t = false)]
    pub wandb_step_logging: bool,

    /// Enable WandB system metrics logging (GPU usage, memory, temperature)
    #[clap(long, env, default_value_t = false)]
    pub wandb_system_metrics: bool,

    /// WandB system metrics logging interval in seconds
    #[clap(long, env, default_value_t = 10)]
    pub wandb_system_metrics_interval_secs: u64,

    /// Inject network latency for stress testing. Format: "base_ms" or "base_ms-jitter_ms"
    /// Example: "50" = 50ms fixed, "50-20" = 50ms +/- 20ms uniform jitter
    #[clap(long, env)]
    pub fault_latency_ms: Option<String>,

    /// Packet loss probability for stress testing (0.0 to 1.0)
    /// Example: 0.1 = 10% packet loss
    #[clap(long, env)]
    pub fault_packet_loss: Option<f64>,

    /// Bandwidth limit in bytes per second for stress testing
    #[clap(long, env)]
    pub fault_bandwidth_limit: Option<u64>,

    /// Random seed for fault injection (for reproducibility)
    #[clap(long, env)]
    pub fault_seed: Option<u64>,

    #[clap(long, env)]
    pub write_log: Option<PathBuf>,

    #[clap(long, env)]
    pub optim_stats_steps: Option<u32>,

    #[clap(long, default_value_t = false, env)]
    pub grad_accum_in_fp32: bool,

    #[clap(long, env)]
    pub dummy_training_delay_secs: Option<u64>,

    #[clap(long, default_value_t = 4, env)]
    pub max_concurrent_parameter_requests: usize,

    #[clap(long, default_value_t = 4, env)]
    pub max_concurrent_downloads: usize,

    #[arg(
        long,
        help = "Device(s) to use: auto, cpu, mps, cuda, cuda:N, cuda:X,Y,Z",
        default_value = "auto"
    )]
    pub device: Devices,

    /// MatFormer tier this client can train at (`0` = largest, higher = smaller).
    ///
    /// Currently used for capability handshakes / heterogeneous assignment in centralized testnets.
    #[clap(long, env, default_value_t = 0)]
    pub matformer_tier: u8,

    /// How to load MatFormer weights:
    /// - auto (default): load a tier-sliced checkpoint if present, otherwise load the universal checkpoint and slice at runtime.
    /// - universal: always load the universal checkpoint and slice at runtime (current behavior).
    /// - sliced: require a tier-sliced checkpoint for tiers > 0 (errors if missing); tier 0 loads universal.
    #[clap(long, env, value_enum, default_value_t = MatformerLoadStrategy::Auto)]
    pub matformer_load_strategy: MatformerLoadStrategy,

    /// Fraction of suffix neurons to help train (0.0-1.0).
    /// When > 0, this tier will also train a random sample of neurons
    /// outside its normal range, helping larger tiers converge faster.
    /// Only applies to tiers > 0.
    #[clap(long, env, default_value_t = 0.0)]
    pub matformer_helper_fraction: f32,

    /// How many rounds to keep helper indices fixed before rotating.
    /// Higher values allow DisTrO delta to accumulate; lower values
    /// give faster coverage of all suffix neurons.
    /// Recommended: 8-32 rounds.
    #[clap(long, env, default_value_t = 16)]
    pub matformer_helper_rotation_interval: u64,

    /// DisTrO apply mode: `sign` (default, original behavior) or `raw`.
    ///
    /// `raw` preserves decoded gradient magnitude at apply-time and rescales to avoid lr^2 shrinkage.
    #[clap(long, env, value_enum, default_value_t = DistroApplyMode::Sign)]
    pub distro_apply_mode: DistroApplyMode,

    /// DisTrO aggregation mode.
    /// - legacy: existing sparse aggregation path
    /// - diloco-lite: weighted aggregation + outer momentum + trust-region scaling
    #[clap(long, env, value_enum, default_value_t = DistroAggregateMode::Legacy)]
    pub distro_aggregate_mode: DistroAggregateMode,

    /// DisTrO transmitted value mode.
    /// - auto: preserves existing behavior
    /// - sign: force sign-valued sparse payloads
    /// - raw: force raw sparse payloads (disables 1-bit sign quantization)
    #[clap(long, env, value_enum, default_value_t = DistroValueMode::Auto)]
    pub distro_value_mode: DistroValueMode,

    /// Enable apply-side sign error-feedback memory (default off).
    #[clap(long, env, default_value_t = false)]
    pub distro_error_feedback: bool,

    /// Apply-side sign error-feedback residual decay in [0,1].
    #[clap(long, env, default_value_t = 1.0)]
    pub distro_ef_decay: f64,

    /// Experimental: apply updates from only one trainer "slot" per step.
    ///
    /// The slot index is computed from the round's trainer set in stable order
    /// (the coordinator's `epoch_state.clients` order, filtered to trainers for that round).
    ///
    /// This is useful for "L+S minus S (no redistribution)" controls: keep 2+ trainers
    /// for batch partitioning/witnessing, but apply only trainer slot 0 (or another index).
    #[clap(long, env)]
    pub distro_apply_only_trainer_index: Option<u16>,

    /// Enable guarded raw-v2 normalization path for DisTrO.
    #[clap(long, env, default_value_t = false)]
    pub distro_raw_v2_enabled: bool,

    /// Raw-v2 normalization mode.
    #[clap(long, env, value_enum, default_value_t = DistroRawNormMode::Off)]
    pub distro_raw_norm_mode: DistroRawNormMode,

    /// Raw-v2 extra scale multiplier after norm matching.
    #[clap(long, env, default_value_t = 1.0)]
    pub distro_raw_scale_multiplier: f64,

    /// Raw-v2 maximum allowed scale factor.
    ///
    /// For `match-sign-equivalent`, early training often needs very large scale factors.
    /// Keep this high and use `distro_raw_abs_clip_mult` as the primary safety bound.
    #[clap(long, env, default_value_t = 1.0e9)]
    pub distro_raw_scale_max: f64,

    /// Raw-v2 per-parameter abs-clip multiplier relative to matched RMS.
    #[clap(long, env, default_value_t = 8.0)]
    pub distro_raw_abs_clip_mult: f64,

    /// Raw-v2 sign-equivalent target multiplier used by `match-sign-equivalent`.
    ///
    /// Target norm is:
    /// - `sign_equiv_mult * sqrt(numel_effective)` for `match-sign-equivalent`
    /// - `sign_equiv_mult * sqrt(nnz_effective)` for `match-sign-equivalent-nnz`
    #[clap(long, env, default_value_t = 1.0)]
    pub distro_raw_sign_equiv_mult: f64,

    /// Policy when raw-v2 sidecars are missing or invalid.
    #[clap(
        long,
        env,
        value_enum,
        default_value_t = DistroRawMissingSidecarPolicy::WarnOff
    )]
    pub distro_raw_missing_sidecar_policy: DistroRawMissingSidecarPolicy,

    /// DiLoCo-lite outer momentum (EMA beta) on the server-side aggregation path.
    #[clap(long, env, default_value_t = 0.9)]
    pub distro_diloco_outer_momentum: f64,

    /// DiLoCo-lite multiplier on the aggregated direction before optimizer step.
    #[clap(long, env, default_value_t = 1.0)]
    pub distro_diloco_outer_lr_multiplier: f64,

    /// DiLoCo-lite trust-region target expressed as update-norm / parameter-norm.
    #[clap(long, env, default_value_t = 0.02)]
    pub distro_diloco_trust_region_target: f64,

    /// DiLoCo-lite clamp factor for trust-region scaling.
    #[clap(long, env, default_value_t = 1.0)]
    pub distro_diloco_trust_region_max_scale: f64,

    /// DiLoCo-lite cap on per-peer aggregation weight as multiple of mean peer weight.
    #[clap(long, env, default_value_t = 2.0)]
    pub distro_diloco_tier_weight_cap: f64,

    /// Tier-0 suffix gate warmup steps (0 = disabled).
    ///
    /// When enabled, tier-0 scales its MLP suffix channels by a beta that ramps
    /// from 0 -> 1 over `warmup_steps`, starting at `start_step`. This makes tier-0
    /// behave like a smaller MatFormer tier early (prefix-only), then smoothly
    /// grow to full width.
    #[clap(long, env, default_value_t = 0)]
    pub matformer_suffix_gate_warmup_steps: u32,

    /// Tier-0 suffix gate start step (used only if warmup steps > 0).
    #[clap(long, env, default_value_t = 0)]
    pub matformer_suffix_gate_start_step: u32,

    /// Which MatFormer tier's prefix defines the "core" width for the suffix gate.
    ///
    /// Example: for tiers 0 and 1, set this to 1 (core = intermediate/2).
    #[clap(long, env, default_value_t = 1)]
    pub matformer_suffix_gate_tier: u8,

    /// Suffix gate schedule shape for ramping beta (linear or cosine).
    #[clap(long, env, value_enum, default_value_t = MatformerGateSchedule::Linear)]
    pub matformer_suffix_gate_schedule: MatformerGateSchedule,

    /// If true, additionally scale the scheduled suffix beta by a learned per-layer scalar
    /// `sigmoid(logit)` (in (0,1)). This gives tier-0 a knob to suppress/enable suffix
    /// capacity automatically.
    #[clap(long, env, default_value_t = false)]
    pub matformer_suffix_gate_learnable: bool,

    /// Initial logit for the learned suffix gate scalar (higher => closer to 1.0).
    #[clap(long, env, default_value_t = 6.0)]
    pub matformer_suffix_gate_learnable_init_logit: f64,

    /// Convenience: configure a coupled "prefix-only warmup" + "ramp" schedule for both:
    /// - tier-0 suffix gate (extra FFN capacity), and
    /// - in-place distillation (tier>0 matches tier-0).
    ///
    /// When enabled, the client overrides:
    /// - `matformer_suffix_gate_start_step` and `matformer_distillation_start_step` = `phase_a_steps`
    /// - `matformer_suffix_gate_warmup_steps` and `matformer_distillation_warmup_steps` = `ramp_steps`
    ///
    /// Set `phase_a_steps` > 0 to keep tier-0 and tier>0 functionally similar early.
    #[clap(long, env, default_value_t = 0)]
    pub matformer_synergy_phase_a_steps: u32,

    /// Coupled ramp length used by `--matformer-synergy-phase-a-steps`.
    /// Must be > 0 when synergy is enabled.
    #[clap(long, env, default_value_t = 0)]
    pub matformer_synergy_ramp_steps: u32,

    /// Log a one-time memory snapshot (RSS and, if available, GPU memory) when training starts.
    #[clap(long, env, default_value_t = false)]
    pub log_memory_usage: bool,

    #[clap(long, env)]
    pub sidecar_port: Option<u16>,

    #[clap(long, default_value_t = true, env)]
    pub delete_old_steps: bool,

    #[clap(long, default_value_t = 3, env)]
    pub keep_steps: u32,

    /// Enable MatFormer distillation: tier-0 broadcasts teacher logits, tier>0 uses them.
    /// Set beta-max > 0 to enable. All distillation args are optional with sane defaults.
    #[clap(long, env, default_value_t = 0.0)]
    pub matformer_distillation_beta_max: f64,

    /// Number of top logits to transmit per token for distillation.
    #[clap(long, env, default_value_t = 64)]
    pub matformer_distillation_top_k: u16,

    /// Temperature for KL divergence softening in distillation.
    #[clap(long, env, default_value_t = 1.0)]
    pub matformer_distillation_temperature: f32,

    /// Step at which distillation begins (teacher is garbage early).
    #[clap(long, env, default_value_t = 0)]
    pub matformer_distillation_start_step: u32,

    /// Steps to ramp distillation β from 0 to beta_max.
    #[clap(long, env, default_value_t = 100)]
    pub matformer_distillation_warmup_steps: u32,

    /// Distillation objective combine mode.
    /// `mix` = (1-β)CE + βKD, `add` = CE + βKD.
    #[clap(long, env, value_enum, default_value_t = MatformerDistillationCombineMode::Add)]
    pub matformer_distillation_combine_mode: MatformerDistillationCombineMode,

    /// If set, distill only on the last N token positions (teacher transmits only that suffix).
    ///
    /// This is the main knob for WAN bandwidth and student-side KD memory.
    /// Must be >= 2.
    #[clap(long, env)]
    pub matformer_distillation_logits_to_keep: Option<u16>,

    /// Optional confidence gate: if mean teacher top-k mass is below this threshold,
    /// disable KD for that step (β_eff = 0).
    #[clap(long, env)]
    pub matformer_distillation_min_teacher_topk_mass: Option<f64>,

    /// Alias for KD confidence gating threshold (maps to min teacher top-k mass).
    #[clap(long, env)]
    pub matformer_distill_confidence_threshold: Option<f64>,

    /// Optional KD scaling to avoid "KD vanishes when the teacher is high-entropy".
    ///
    /// If > 0, KD is multiplied by:
    /// `1 / clamp(mean_q_topk_mass, floor, 1.0)`.
    ///
    /// Set to 0 to disable.
    #[clap(long, env, default_value_t = 0.05)]
    pub matformer_distillation_kd_q_topk_mass_floor: f64,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum MatformerLoadStrategy {
    Auto,
    Universal,
    Sliced,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum MatformerGateSchedule {
    Linear,
    Cosine,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum MatformerDistillationCombineMode {
    Mix,
    Add,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DistroApplyMode {
    Sign,
    Raw,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum DistroAggregateMode {
    Legacy,
    DilocoLite,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DistroValueMode {
    Auto,
    Sign,
    Raw,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum DistroRawNormMode {
    Off,
    MatchPreL2,
    MatchSignEquivalent,
    MatchSignEquivalentNnz,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum DistroRawMissingSidecarPolicy {
    WarnOff,
    Fail,
}

impl TrainArgs {
    pub fn wandb_info(&self, run_name: String) -> Result<Option<WandBInfo>> {
        let wandb_info = match std::env::var("WANDB_API_KEY") {
            Ok(wandb_api_key) => Some(WandBInfo {
                project: self.wandb_project.clone().unwrap_or("psyche".to_string()),
                run: self.wandb_run.clone().unwrap_or(run_name),
                entity: self.wandb_entity.clone(),
                api_key: wandb_api_key,
                group: self.wandb_group.clone(),
                step_logging: self.wandb_step_logging,
                system_metrics: self.wandb_system_metrics,
                system_metrics_interval_secs: self.wandb_system_metrics_interval_secs,
            }),
            Err(_) => {
                match self.wandb_entity.is_some()
                    || self.wandb_run.is_some()
                    || self.wandb_project.is_some()
                    || self.wandb_group.is_some()
                {
                    true => bail!(
                        "WANDB_API_KEY environment variable must be set for wandb integration"
                    ),
                    false => None,
                }
            }
        };
        Ok(wandb_info)
    }

    pub fn checkpoint_config(&self) -> Result<Option<CheckpointConfig>> {
        let hub_read_token = std::env::var("HF_TOKEN").ok();
        let checkpoint_upload_info = match (
            &hub_read_token,
            self.hub_repo.clone(),
            self.checkpoint_dir.clone(),
            self.delete_old_steps,
            self.keep_steps,
        ) {
            (Some(token), Some(repo), Some(dir), delete_old_steps, keep_steps) => {
                if keep_steps == 0 {
                    bail!("keep_steps must be >= 1 for hub repository uploads (got {keep_steps})")
                }
                Some(CheckpointConfig {
                    checkpoint_dir: dir,
                    hub_upload: Some(HubUploadInfo {
                        hub_repo: repo,
                        hub_token: token.to_string(),
                    }),
                    delete_old_steps,
                    keep_steps,
                })
            }
            (None, Some(_), Some(_), _, _) => {
                bail!("hub-repo and checkpoint-dir set, but no HF_TOKEN env variable.")
            }
            (_, Some(_), None, _, _) => {
                bail!("--hub-repo was set, but no --checkpoint-dir was passed!")
            }
            (_, None, Some(dir), delete_old_steps, keep_steps) => Some(CheckpointConfig {
                checkpoint_dir: dir,
                hub_upload: None,
                delete_old_steps,
                keep_steps,
            }),
            (_, None, _, _, _) => None,
        };

        Ok(checkpoint_upload_info)
    }

    pub fn eval_tasks(&self) -> Result<Vec<psyche_eval::Task>> {
        let eval_tasks = match &self.eval_tasks {
            Some(eval_tasks) => Self::eval_tasks_from_args(eval_tasks, self.eval_seed)?,
            None => Vec::new(),
        };
        Ok(eval_tasks)
    }

    pub fn eval_tasks_from_args(
        eval_tasks: &str,
        eval_seed: u64,
    ) -> Result<Vec<psyche_eval::Task>> {
        let result: Result<Vec<psyche_eval::Task>> = eval_tasks
            .split(",")
            .map(|eval_task| {
                let fewshot = match eval_task {
                    "mmlu_pro" => 5,
                    _ => 0,
                };
                tasktype_from_name(eval_task)
                    .map(|task_type| psyche_eval::Task::new(task_type, fewshot, eval_seed))
            })
            .collect();
        result
    }

    /// Build fault injection configuration from CLI args.
    ///
    /// Returns None if no fault injection is configured.
    pub fn fault_config(&self) -> Option<FaultConfig> {
        let config = FaultConfigBuilder::new()
            .latency(self.fault_latency_ms.clone())
            .packet_loss(self.fault_packet_loss)
            .bandwidth_limit(self.fault_bandwidth_limit)
            .seed(self.fault_seed)
            .build();

        if let Some(ref cfg) = config {
            cfg.log_summary();
        }

        config
    }
}

pub fn prepare_environment() {
    psyche_modeling::set_suggested_env_vars();

    #[cfg(target_os = "windows")]
    {
        // this is a gigantic hack to cover that called sdpa prints out
        // "Torch was not compiled with flash attention." via TORCH_WARN
        // on Windows, which screws with the TUI.
        // it's done once (really TORCH_WARN_ONCE), so elicit that behavior
        // before starting anything else
        use tch::Tensor;
        let device = tch::Device::Cuda(0);
        let _ = Tensor::scaled_dot_product_attention::<Tensor>(
            &Tensor::from_slice2(&[[0.]]).to(device),
            &Tensor::from_slice2(&[[0.]]).to(device),
            &Tensor::from_slice2(&[[0.]]).to(device),
            None,
            0.0,
            false,
            None,
        );
    }
}

fn parse_duration_from_seconds(s: &str) -> Result<Duration, String> {
    s.parse::<f64>()
        .map_err(|e| format!("Invalid number: {e}"))
        .and_then(|secs| {
            if secs < 0.0 {
                Err("Duration cannot be negative".to_string())
            } else {
                Ok(Duration::from_secs_f64(secs))
            }
        })
}
