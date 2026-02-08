use crate::{
    fetch_data::{BatchIdSet, DataFetcher, TrainingDataForStep},
    state::types::{DeserializeError, PayloadState},
    IntegrationTestLogMarker,
};

use futures::{future::try_join_all, stream::FuturesUnordered, StreamExt};
use half::f16;
use psyche_coordinator::{
    assign_data_for_state, get_batch_ids_for_node, get_batch_ids_for_round, model, Commitment,
    CommitteeSelection, Coordinator, CoordinatorError, HealthChecks, BLOOM_FALSE_RATE,
};
use psyche_core::{BatchId, Bloom, NodeIdentity, OptimizerDefinition};
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::{
    distillation_beta, ApplyDistroResultError, Batch, BatchData, BatchDataCPU, CausalLM,
    DistroResult, DistroValueMode, TeacherLogitTargets, TrainOutput, Trainer,
    TrainerThreadCommunicationError,
};
use psyche_network::{
    distro_results_to_bytes, AuthenticatableIdentity, CompressedTeacherLogits, Hash,
    SerializeDistroResultError, SerializedDistroResult, TransmittableDistroResult,
    TransmittableTeacherLogits,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tch::{Kind, Tensor};
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, trace_span, warn, Instrument};

use super::{
    evals::{MaybeRunningEvals, ModelTaskRunner},
    round_state::RoundState,
    types::DistroBroadcastAndPayload,
};
use sysinfo::{get_current_pid, Pid, ProcessesToUpdate, System};
use wandb::LogData;

const F16_MAX_ABS: f32 = 65504.0;

fn clamp_teacher_f16_bits(v: f32) -> u16 {
    if v.is_finite() {
        f16::from_f32(v.clamp(-F16_MAX_ABS, F16_MAX_ABS)).to_bits()
    } else {
        f16::from_f32(0.0).to_bits()
    }
}

#[derive(Debug)]
pub struct FinishedTrainers {
    pub evals_or_trainers: MaybeRunningEvals,
    pub round_losses: Vec<f32>,
    pub optim_stats: HashMap<String, f64>,
    pub round_duration: Duration,
}

#[derive(Error, Debug)]
pub enum TrainError {
    #[error("No trainers available when entering training step.")]
    NoTrainers,

    #[error("No training round in-progress")]
    NoActiveRound,

    #[error("No committee info for this round ")]
    NoCommitteeInfo,

    #[error("We're not in this round")]
    NotInThisRound,

    #[error("Apply thread crashed")]
    ApplyCrashed,

    #[error("Failed to apply distro results: {0}")]
    Apply(#[from] ApplyError),

    #[error("Training thread crashed")]
    TrainCrashed,

    #[error("Transmit thread crashed")]
    TransmitCrashed,

    #[error("Failed to train on batch: {0}")]
    TrainOnBatch(#[from] TrainerThreadCommunicationError),

    #[error("Model parameter count mismatch: got {got} DisTrO results, expected {expected}")]
    ParameterCountMismatch { expected: usize, got: usize },

    #[error("Failed to serialize distro result: {0}")]
    SerializeDistroResult(SerializeDistroResultError),

    #[error("Failed to send distro result, channel must be closed")]
    SendDistroResult,

    #[error("Failed to send health checks, channel must be closed")]
    SendHealthChecks,

    #[error("Healthcheck thread crashed")]
    HealthCheckCrashed,

    #[error("Coordinator error: {0}")]
    CoordinatorError(CoordinatorError),

    #[error("Failed to fetch data for batch {batch_id}: {source:#}")]
    FetchData {
        batch_id: BatchId,
        #[source]
        source: anyhow::Error,
    },

    #[error(
        "Timed out waiting for teacher logits for step {step} batch {batch_id} after {waited:?}"
    )]
    TeacherLogitsTimeout {
        step: u32,
        batch_id: BatchId,
        waited: Duration,
    },

    #[error("Teacher logits missing for step {step} batch {batch_id}")]
    TeacherLogitsMissing { step: u32, batch_id: BatchId },

    #[error("Invalid teacher logits for step {step} batch {batch_id}: {reason}")]
    InvalidTeacherLogits {
        step: u32,
        batch_id: BatchId,
        reason: String,
    },

    #[error("Failed to send teacher logits to broadcast loop (channel closed)")]
    SendTeacherLogits,
}

const TEACHER_LOGITS_WAIT_TIMEOUT: Duration = Duration::from_secs(120);
const TEACHER_LOGITS_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone)]
struct BatchShardCPU {
    start: usize,
    data: Vec<BatchDataCPU>,
}

fn split_batch_cpu_evenly(
    items: &[BatchDataCPU],
    parts: usize,
) -> Result<Vec<BatchShardCPU>, TrainError> {
    if parts == 0 {
        return Err(TrainError::TrainCrashed);
    }
    let total = items.len();
    if total < parts {
        return Err(TrainError::TrainCrashed);
    }
    let base = total / parts;
    let rem = total % parts;
    let mut start = 0usize;
    let mut out = Vec::with_capacity(parts);
    for i in 0..parts {
        let len = base + if i < rem { 1 } else { 0 };
        let end = start + len;
        out.push(BatchShardCPU {
            start,
            data: items[start..end].to_vec(),
        });
        start = end;
    }
    Ok(out)
}

async fn wait_get_teacher_logits(
    teacher_logits_store: Arc<Mutex<HashMap<BatchId, Arc<TransmittableTeacherLogits>>>>,
    step: u32,
    batch_id: BatchId,
) -> Result<Arc<TransmittableTeacherLogits>, TrainError> {
    let start = Instant::now();
    loop {
        if let Some(tl) = teacher_logits_store.lock().unwrap().get(&batch_id).cloned() {
            return Ok(tl);
        }
        let waited = start.elapsed();
        if waited >= TEACHER_LOGITS_WAIT_TIMEOUT {
            return Err(TrainError::TeacherLogitsTimeout {
                step,
                batch_id,
                waited,
            });
        }
        tokio::time::sleep(TEACHER_LOGITS_POLL_INTERVAL).await;
    }
}

fn teacher_targets_for_shard(
    teacher: &TransmittableTeacherLogits,
    step: u32,
    shard_start: usize,
    shard_len: usize,
    combine_mode: psyche_modeling::DistillationCombineMode,
    min_teacher_topk_mass: Option<f64>,
    kd_q_topk_mass_floor: f64,
) -> Result<TeacherLogitTargets, TrainError> {
    let logits = &teacher.logits;
    let batch_size = logits.batch_size as usize;
    let seq_len = logits.seq_len as usize;
    let top_k = logits.top_k as usize;

    if shard_start + shard_len > batch_size {
        return Err(TrainError::InvalidTeacherLogits {
            step,
            batch_id: teacher.batch_id,
            reason: format!(
                "shard range out of bounds: shard_start={shard_start} shard_len={shard_len} batch_size={batch_size}"
            ),
        });
    }

    let per_sample =
        seq_len
            .checked_mul(top_k)
            .ok_or_else(|| TrainError::InvalidTeacherLogits {
                step,
                batch_id: teacher.batch_id,
                reason: "seq_len*top_k overflow".to_string(),
            })?;
    let start_idx =
        shard_start
            .checked_mul(per_sample)
            .ok_or_else(|| TrainError::InvalidTeacherLogits {
                step,
                batch_id: teacher.batch_id,
                reason: "shard_start*per_sample overflow".to_string(),
            })?;
    let n = shard_len
        .checked_mul(per_sample)
        .ok_or_else(|| TrainError::InvalidTeacherLogits {
            step,
            batch_id: teacher.batch_id,
            reason: "shard_len*per_sample overflow".to_string(),
        })?;
    let end_idx = start_idx + n;

    if end_idx > logits.top_indices.len() || end_idx > logits.top_values_f16.len() {
        return Err(TrainError::InvalidTeacherLogits {
            step,
            batch_id: teacher.batch_id,
            reason: format!(
                "teacher logits slice out of bounds: end_idx={end_idx} indices_len={} values_len={}",
                logits.top_indices.len(),
                logits.top_values_f16.len()
            ),
        });
    }

    let indices: Vec<i64> = logits.top_indices[start_idx..end_idx]
        .iter()
        .map(|&x| x as i64)
        .collect();
    let values: Vec<f32> = logits.top_values_f16[start_idx..end_idx]
        .iter()
        .map(|&b| f16::from_bits(b).to_f32())
        .collect();
    let invalid_value_count = values.iter().filter(|v| !v.is_finite()).count();
    if invalid_value_count > 0 {
        return Err(TrainError::InvalidTeacherLogits {
            step,
            batch_id: teacher.batch_id,
            reason: format!(
                "teacher top values contain non-finite values: count={invalid_value_count}, expected={n}"
            ),
        });
    }

    let lse_start =
        shard_start
            .checked_mul(seq_len)
            .ok_or_else(|| TrainError::InvalidTeacherLogits {
                step,
                batch_id: teacher.batch_id,
                reason: "shard_start*seq_len overflow".to_string(),
            })?;
    let lse_n = shard_len
        .checked_mul(seq_len)
        .ok_or_else(|| TrainError::InvalidTeacherLogits {
            step,
            batch_id: teacher.batch_id,
            reason: "shard_len*seq_len overflow".to_string(),
        })?;
    let lse_end = lse_start + lse_n;
    if lse_end > logits.logsumexp_f16.len() {
        return Err(TrainError::InvalidTeacherLogits {
            step,
            batch_id: teacher.batch_id,
            reason: format!(
                "teacher logsumexp slice out of bounds: lse_end={lse_end} logsumexp_len={}",
                logits.logsumexp_f16.len()
            ),
        });
    }
    let logsumexp: Vec<f32> = logits.logsumexp_f16[lse_start..lse_end]
        .iter()
        .map(|&b| f16::from_bits(b).to_f32())
        .collect();
    let invalid_logsumexp_count = logsumexp.iter().filter(|v| !v.is_finite()).count();
    if invalid_logsumexp_count > 0 {
        return Err(TrainError::InvalidTeacherLogits {
            step,
            batch_id: teacher.batch_id,
            reason: format!(
                "teacher logsumexp contains non-finite values: count={invalid_logsumexp_count}, expected={lse_n}"
            ),
        });
    }

    let top_indices =
        Tensor::from_slice(&indices).view([shard_len as i64, seq_len as i64, top_k as i64]);
    let top_values =
        Tensor::from_slice(&values).view([shard_len as i64, seq_len as i64, top_k as i64]);
    let logsumexp = Tensor::from_slice(&logsumexp).view([shard_len as i64, seq_len as i64]);

    Ok(TeacherLogitTargets {
        top_indices,
        top_values,
        logsumexp,
        temperature: logits.temperature,
        top_k: logits.top_k as i64,
        combine_mode,
        min_teacher_topk_mass,
        kd_q_topk_mass_floor,
    })
}

pub struct TrainingStepMetadata<T: NodeIdentity, A: AuthenticatableIdentity> {
    pub identity: T,
    pub data_fetcher: DataFetcher<T, A>,
    pub tx_health_check: mpsc::UnboundedSender<HealthChecks<T>>,
    pub tx_distro_result: mpsc::UnboundedSender<DistroBroadcastAndPayload>,

    pub parameter_names: Arc<Vec<String>>,
    pub matformer_tier: u8,
    pub distro_apply_only_trainer_index: Option<u16>,

    pub write_gradients_dir: Option<PathBuf>,

    pub model_task_runner: ModelTaskRunner,
    pub log_memory_usage: bool,

    /// Enable step-level loss logging to WandB (per batch, not per round)
    pub step_logging_enabled: bool,
    /// WandB run for step-level logging (if enabled)
    pub wandb_run: Option<Arc<wandb::Run>>,

    /// Channel for tier-0 to send teacher logits to the client broadcast loop.
    pub tx_teacher_logits: mpsc::UnboundedSender<psyche_network::TransmittableTeacherLogits>,

    /// Distillation config (if enabled).
    pub distillation_config: Option<psyche_modeling::DistillationConfig>,

    /// DisTrO transmitted value mode.
    pub distro_value_mode: DistroValueMode,

    /// Experimental: number of local inner updates per outer coordinator step for
    /// smaller MatFormer tiers.
    pub matformer_local_inner_steps: u32,

    /// Periodic same-batch calibration schedule.
    pub same_batch_calibration_every_steps: u32,
    pub same_batch_calibration_start_step: u32,
    pub same_batch_calibration_no_apply: bool,
}

#[derive(Debug)]
pub struct TrainingStep {
    sending_health_checks: Option<JoinHandle<Result<(), TrainError>>>,
    cancel_training: CancellationToken,

    applying_and_training: JoinHandle<Result<FinishedTrainers, TrainError>>,
    finished: Arc<AtomicBool>,
}

impl TrainingStep {
    pub async fn finish(self) -> Result<FinishedTrainers, TrainError> {
        self.cancel_training.cancel();
        if let Some(hc) = self.sending_health_checks {
            hc.await.map_err(|_| TrainError::HealthCheckCrashed)??;
        }

        let finished = self.finished.clone();

        let trainers: FinishedTrainers = self
            .applying_and_training
            .await
            .map_err(|_| TrainError::TrainCrashed)??;

        if !finished.load(Ordering::SeqCst) {
            warn!("Training didn't finish when the Training round ended, we are likely to desync.");
        }

        Ok(trainers)
    }

    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
}

fn log_memory_snapshot_once(enable: bool) {
    static DID_LOG: AtomicBool = AtomicBool::new(false);
    if !enable || DID_LOG.swap(true, Ordering::Relaxed) {
        return;
    }

    let mut sys = System::new_all();
    let pid = get_current_pid().unwrap_or_else(|_| Pid::from_u32(0));
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);

    if let Some(proc_info) = sys.process(pid) {
        let rss_kb = proc_info.memory();
        let virt_kb = proc_info.virtual_memory();
        let cuda_mem_bytes: Option<u64> = None;

        info!(
            memory_rss_mb = (rss_kb as f64) / 1024.0,
            memory_virtual_mb = (virt_kb as f64) / 1024.0,
            cuda_mem_bytes,
            "memory_snapshot"
        );
    }
}

impl<T: NodeIdentity, A: AuthenticatableIdentity + 'static> TrainingStepMetadata<T, A> {
    pub fn start(
        &mut self,
        client_index: u64,
        state: &Coordinator<T>,
        trainers: Vec<Trainer>,
        previous_round: &mut RoundState<T>,
        current_round: &mut RoundState<T>,
    ) -> Result<TrainingStep, TrainError> {
        if trainers.is_empty() {
            return Err(TrainError::NoTrainers);
        }

        let parameter_names = self.parameter_names.clone();

        let applying = self.apply_results(trainers, state, previous_round, current_round)?;

        let sending_health_checks =
            start_sending_health_checks(current_round, state, self.tx_health_check.clone())?;

        debug!("Transitioning to train step {}", state.progress.step);
        log_memory_snapshot_once(self.log_memory_usage);

        let cancel_training = CancellationToken::new();
        let round_start = Instant::now();

        let round = state.current_round().ok_or(TrainError::NoActiveRound)?;

        *previous_round = std::mem::take(current_round);

        let committee_selection = CommitteeSelection::new(
            round.tie_breaker_tasks as usize,
            state.config.witness_nodes as usize,
            state.config.verification_percent,
            state.epoch_state.clients.len(),
            round.random_seed,
        )
        .map_err(TrainError::CoordinatorError)?;

        let have_training = !state.epoch_state.last_step_set();
        let (data_assignments, num_all_batch_ids, batch_ids_not_yet_trained_on) = if have_training {
            let data_assignments = assign_data_for_state(state, &committee_selection);
            let all_batch_ids = get_batch_ids_for_round(
                state.current_round().unwrap(),
                state,
                committee_selection.get_num_trainer_nodes(),
            );
            let num_all_batch_ids = all_batch_ids.len();
            let batch_ids_not_yet_trained_on: BatchIdSet = all_batch_ids.into_iter().collect();
            (
                data_assignments,
                num_all_batch_ids,
                Arc::new(Mutex::new(Some(batch_ids_not_yet_trained_on))),
            )
        } else {
            (BTreeMap::new(), 0, Arc::new(Mutex::new(None)))
        };

        let committee_proof = committee_selection.get_committee(client_index);
        let witness_proof = committee_selection.get_witness(client_index);

        let blooms = {
            let participant_bloom =
                Bloom::random(state.epoch_state.clients.len(), BLOOM_FALSE_RATE);
            let broadcast_bloom = Bloom::random(num_all_batch_ids, BLOOM_FALSE_RATE);
            trace!(
                "Participant bloom size: {} bits, {} keys",
                participant_bloom.bits.0.len(),
                participant_bloom.keys.len()
            );
            trace!(
                "Broadcast bloom size: {} bits, {} keys",
                broadcast_bloom.bits.0.len(),
                broadcast_bloom.keys.len()
            );
            Arc::new(Mutex::new(Some((participant_bloom, broadcast_bloom))))
        };

        *current_round = RoundState {
            height: round.height,
            step: state.progress.step,
            sent_witness: false,
            sent_finished: false,
            downloads: Default::default(),
            results: Default::default(),
            broadcasts: Default::default(),
            clients_finished: Default::default(),
            data_assignments: data_assignments.clone(),
            blooms,
            committee_info: Some((committee_proof, witness_proof, committee_selection)),
            batch_ids_not_yet_trained_on,
            self_distro_results: vec![],
            teacher_logits: Arc::new(Mutex::new(HashMap::new())),
            teacher_logits_downloads: Arc::new(Mutex::new(HashMap::new())),
        };

        let warmup_lr_between = state.get_cold_start_warmup_bounds();
        let zero_optim = warmup_lr_between.is_some_and(|_| round.height == 0);
        let epoch = state.progress.epoch;

        info!(
            integration_test_log_marker = %IntegrationTestLogMarker::WitnessElected,
            step = state.progress.step,
            round = round.height,
            epoch = epoch,
            index = client_index,
            comittee_position = committee_proof.position,
            committee = %committee_proof.committee,
            witness_position = witness_proof.position,
            witness = %witness_proof.witness,
            matformer_tier = self.matformer_tier,
            warmup_lr_between = ?warmup_lr_between,
            assigned_batches = ?get_batch_ids_for_node(&data_assignments, &self.identity),
            "Got training assignment for step {} (round {}/epoch {}): index={} committee position={} committee={} witness position={} witness={} warmup_lr_between={:?}",
            state.progress.step, round.height, epoch, client_index, committee_proof.position, committee_proof.committee, witness_proof.position, witness_proof.witness, warmup_lr_between
        );
        let model_task_runner = self.model_task_runner.clone();
        let finished = Arc::new(AtomicBool::new(false));

        let prev_self_distro_results = previous_round.self_distro_results.clone();
        let applying_and_training: JoinHandle<Result<FinishedTrainers, TrainError>> =
            if !have_training {
                let finished = finished.clone();

                // the last two rounds have no training (just applying the final results)
                tokio::task::spawn(async move {
                    let round_duration = Instant::now() - round_start;
                    debug!("Training for round finished, duration {:?}", round_duration);
                    finished.store(true, Ordering::SeqCst);
                    Ok(FinishedTrainers {
                        evals_or_trainers: MaybeRunningEvals::Running(
                            model_task_runner
                                .start(applying.await.map_err(|_| TrainError::ApplyCrashed)??),
                        ),
                        round_losses: vec![],
                        optim_stats: HashMap::new(),
                        round_duration,
                    })
                })
            } else {
                let identity = self.identity;
                let cancel_training = cancel_training.clone();
                let write_gradients_dir = self.write_gradients_dir.clone();
                let tx_distro_result = self.tx_distro_result.clone();
                let (quantize, optimizer_is_distro) = match &state.model {
                    model::Model::LLM(llm) => match llm.optimizer {
                        OptimizerDefinition::Distro { quantize_1bit, .. } => (quantize_1bit, true),
                        _ => (false, false),
                    },
                };
                let quantize = match self.distro_value_mode {
                    DistroValueMode::Auto => quantize,
                    DistroValueMode::Sign => {
                        if !quantize {
                            info!(
                                step = state.progress.step,
                                "DisTrO value mode=sign overrides quantize_1bit=false and forces sign payloads"
                            );
                        }
                        true
                    }
                    DistroValueMode::Raw => {
                        if quantize {
                            warn!(
                                step = state.progress.step,
                                "DisTrO value mode=raw overrides quantize_1bit=true and disables sign payloads"
                            );
                        }
                        false
                    }
                };
                let finished = finished.clone();
                let step_logging_enabled = self.step_logging_enabled;
                let wandb_run = self.wandb_run.clone();
                let tx_teacher_logits = self.tx_teacher_logits.clone();
                let distillation_config = self.distillation_config.clone();
                let matformer_tier = self.matformer_tier;
                let matformer_local_inner_steps = self.matformer_local_inner_steps.max(1);
                let teacher_logits_store = current_round.teacher_logits.clone();
                let data_provider = self.data_fetcher.data_provider_handle();

                let teacher_logits_top_k =
                    distillation_config.as_ref().map(|c| c.top_k).unwrap_or(64);
                let teacher_logits_temperature = distillation_config
                    .as_ref()
                    .map(|c| c.temperature)
                    .unwrap_or(1.0);
                let teacher_logits_to_keep: Option<i64> = distillation_config
                    .as_ref()
                    .and_then(|c| c.logits_to_keep)
                    .map(|n| n as i64);
                let distillation_combine_mode = distillation_config
                    .as_ref()
                    .map(|c| c.combine_mode)
                    .unwrap_or(psyche_modeling::DistillationCombineMode::Add);
                let distillation_min_teacher_topk_mass = distillation_config
                    .as_ref()
                    .and_then(|c| c.min_teacher_topk_mass);
                let distillation_kd_q_topk_mass_floor = distillation_config
                    .as_ref()
                    .map(|c| c.kd_q_topk_mass_floor)
                    .unwrap_or(0.0);
                let teacher_batch_ids: Vec<BatchId> = data_assignments
                    .iter()
                    .filter_map(|(batch_id, assigned)| {
                        if assigned != &identity {
                            Some(*batch_id)
                        } else {
                            None
                        }
                    })
                    .collect();

                let TrainingDataForStep {
                    step,
                    mut next_sample,
                } = self
                    .data_fetcher
                    .fetch_data(state, &data_assignments, &self.identity);

                tokio::task::spawn(async move {
                    let mut round_losses: Vec<f32> = Vec::new();
                    let mut batch_idx: usize = 0;
                    let mut optim_stats: HashMap<String, f64> = HashMap::new();

                    let mut available_trainers =
                        applying.await.map_err(|_| TrainError::ApplyCrashed)??;

                    let distill_beta = distillation_config
                        .as_ref()
                        .map(|cfg| distillation_beta(cfg, step))
                        .unwrap_or(0.0);
                    let distillation_active = distill_beta > 0.0;
                    if distillation_active && !optimizer_is_distro {
                        // Distillation still affects the student's local loss, but without DisTrO
                        // there is no cross-client gradient application, so tier-0 won't change.
                        warn!(
                            step = step,
                            "Distillation is enabled but optimizer is not DisTrO; student gradients will not be applied to other clients (tier-0 curves will look unchanged)."
                        );
                    }

                    let mut local_inner_steps = if matformer_tier > 0 {
                        matformer_local_inner_steps
                    } else {
                        1
                    };
                    if distillation_active && local_inner_steps > 1 {
                        warn!(
                            step = step,
                            tier = matformer_tier,
                            requested_inner_steps = local_inner_steps,
                            "Disabling local inner steps because distillation is active (teacher logits are only produced for assigned batches)"
                        );
                        local_inner_steps = 1;
                    }
                    if local_inner_steps > 1 {
                        info!(
                            step = step,
                            tier = matformer_tier,
                            local_inner_steps = local_inner_steps,
                            "Using local inner updates for this tier"
                        );
                    }

                    // Tier-0 teacher: produce WAN-friendly top-k logits for every batch not assigned to self.
                    // Students will block (no fallback) if distillation is active but teacher logits are missing.
                    if matformer_tier == 0 && distillation_active {
                        const MAX_RETRIES: u32 = 7;
                        const BASE_DELAY_MS: u64 = 2000;

                        info!(
                            step = step,
                            batches = teacher_batch_ids.len(),
                            top_k = teacher_logits_top_k,
                            temperature = teacher_logits_temperature,
                            logits_to_keep = teacher_logits_to_keep,
                            "Producing teacher logits for distillation",
                        );

                        for batch_id in teacher_batch_ids {
                            let mut retry_count = 0;
                            let samples = loop {
                                match data_provider.lock().await.get_samples(batch_id).await {
                                    Ok(batch) => break batch,
                                    Err(err) if retry_count < MAX_RETRIES => {
                                        retry_count += 1;
                                        let delay_ms = BASE_DELAY_MS * (retry_count as u64 - 1);
                                        warn!(
                                            step = step,
                                            batch_id = %batch_id,
                                            attempt = retry_count,
                                            max_attempts = MAX_RETRIES,
                                            "Teacher data fetch error: {err:#}. Retrying in {delay_ms}ms",
                                        );
                                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                                        continue;
                                    }
                                    Err(err) => {
                                        return Err(TrainError::FetchData {
                                            batch_id,
                                            source: err,
                                        });
                                    }
                                }
                            };

                            let cpu_items: Vec<BatchDataCPU> = samples
                                .into_iter()
                                .map(|batch| BatchDataCPU {
                                    input_ids: batch.input_ids,
                                    labels: batch.labels,
                                    position_ids: batch.position_ids,
                                    sequence_lengths: batch.sequence_lengths,
                                })
                                .collect();

                            let shards =
                                split_batch_cpu_evenly(&cpu_items, available_trainers.len())?;

                            let mut in_progress = FuturesUnordered::new();
                            for (trainer, shard) in
                                available_trainers.drain(..).zip(shards.into_iter())
                            {
                                let k = teacher_logits_top_k;
                                let temperature = teacher_logits_temperature;
                                let num_logits_to_keep = teacher_logits_to_keep;
                                in_progress.push(tokio::task::spawn_blocking(move || {
                                    let BatchShardCPU {
                                        start: shard_start,
                                        data: shard_data,
                                    } = shard;
                                    let device = trainer.device();
                                    let batch_gpu = Batch {
                                        id: batch_id,
                                        data: BatchData::CPU(shard_data),
                                    }
                                    .gpu(device);

                                    let BatchData::GPU(gpu) = batch_gpu.data else {
                                        unreachable!("Batch::gpu must produce GPU data");
                                    };

                                    let (logits, _) = trainer.forward(
                                        &gpu.input_ids,
                                        None,
                                        gpu.position_ids.as_ref(),
                                        gpu.sequence_lengths.as_ref(),
                                        num_logits_to_keep,
                                        None,
                                    );
                                    let logits =
                                        logits.ok_or_else(|| TrainError::InvalidTeacherLogits {
                                            step,
                                            batch_id,
                                            reason: "teacher forward returned no logits"
                                                .to_string(),
                                        })?;

                                    let k_i64 = k as i64;
                                    let (top_values, top_indices) =
                                        logits.topk(k_i64, -1, true, true);
                                    let size = top_indices.size();
                                    if size.len() != 3 {
                                        return Err(TrainError::InvalidTeacherLogits {
                                            step,
                                            batch_id,
                                            reason: format!(
                                                "unexpected teacher topk shape: {size:?}"
                                            ),
                                        });
                                    }
                                    let local_batch = size[0] as usize;
                                    let seq_len = size[1] as usize;

                                    // Compute per-token `logZ = logsumexp(logits / T)` (full vocab).
                                    // This enables student-side "top-k + tail bucket" KL without
                                    // renormalizing top-k to sum to 1.0.
                                    let logsumexp = (logits.to_kind(Kind::Float)
                                        / (temperature as f64))
                                        .logsumexp(-1, false);
                                    let lse_size = logsumexp.size();
                                    if lse_size.len() != 2
                                        || lse_size[0] != size[0]
                                        || lse_size[1] != size[1]
                                    {
                                        return Err(TrainError::InvalidTeacherLogits {
                                            step,
                                            batch_id,
                                            reason: format!(
                                                "unexpected teacher logsumexp shape: {lse_size:?} (expected [{}, {}])",
                                                size[0], size[1]
                                            ),
                                        });
                                    }
                                    let lse_flat = logsumexp
                                        .to(tch::Device::Cpu)
                                        .to_kind(Kind::Float)
                                        .contiguous()
                                        .view([-1]);
                                    let lse_flat: Vec<f32> =
                                        Vec::<f32>::try_from(lse_flat).map_err(|e| {
                                            TrainError::InvalidTeacherLogits {
                                                step,
                                                batch_id,
                                                reason: format!(
                                                    "failed to read logsumexp: {e:?}"
                                                ),
                                            }
                                        })?;
                                    let mut logsumexp_invalid = 0usize;
                                    let logsumexp_f16: Vec<u16> = lse_flat
                                        .iter()
                                        .map(|&v| {
                                            if !v.is_finite() {
                                                logsumexp_invalid += 1;
                                            }
                                            clamp_teacher_f16_bits(v)
                                        })
                                        .collect();
                                    if logsumexp_invalid > 0 {
                                        warn!(
                                            step = step,
                                            batch_id = %batch_id,
                                            invalid = logsumexp_invalid,
                                            "Sanitized non-finite teacher logsumexp to 0.0"
                                        );
                                    }

                                    let indices_flat = top_indices
                                        .to(tch::Device::Cpu)
                                        .to_kind(Kind::Int64)
                                        .contiguous()
                                        .view([-1]);
                                    let indices_flat: Vec<i64> = Vec::<i64>::try_from(indices_flat)
                                        .map_err(|e| TrainError::InvalidTeacherLogits {
                                            step,
                                            batch_id,
                                            reason: format!("failed to read top_indices: {e:?}"),
                                        })?;
                                    let mut top_indices_u16 =
                                        Vec::with_capacity(indices_flat.len());
                                    for idx in indices_flat {
                                        if idx < 0 || idx > (u16::MAX as i64) {
                                            return Err(TrainError::InvalidTeacherLogits {
                                                step,
                                                batch_id,
                                                reason: format!(
                                                    "top index out of u16 range: {idx}"
                                                ),
                                            });
                                        }
                                        top_indices_u16.push(idx as u16);
                                    }

                                    let values_flat = top_values
                                        .to(tch::Device::Cpu)
                                        .to_kind(Kind::Float)
                                        .contiguous()
                                        .view([-1]);
                                    let values_flat: Vec<f32> = Vec::<f32>::try_from(values_flat)
                                        .map_err(|e| {
                                        TrainError::InvalidTeacherLogits {
                                            step,
                                            batch_id,
                                            reason: format!("failed to read top_values: {e:?}"),
                                        }
                                    })?;
                                    let mut top_values_invalid = 0usize;
                                    let top_values_f16: Vec<u16> = values_flat
                                        .iter()
                                        .map(|&v| {
                                            if !v.is_finite() {
                                                top_values_invalid += 1;
                                            }
                                            clamp_teacher_f16_bits(v)
                                        })
                                        .collect();
                                    if top_values_invalid > 0 {
                                        warn!(
                                            step = step,
                                            batch_id = %batch_id,
                                            invalid = top_values_invalid,
                                            "Sanitized non-finite teacher top values to 0.0"
                                        );
                                    }

                                    let batch_size_u16: u16 =
                                        local_batch.try_into().map_err(|_| {
                                            TrainError::InvalidTeacherLogits {
                                                step,
                                                batch_id,
                                                reason: format!(
                                                    "local batch_size too large: {local_batch}"
                                                ),
                                            }
                                        })?;
                                    let seq_len_u16: u16 = seq_len.try_into().map_err(|_| {
                                        TrainError::InvalidTeacherLogits {
                                            step,
                                            batch_id,
                                            reason: format!("seq_len too large: {seq_len}"),
                                        }
                                    })?;

                                    let compressed = CompressedTeacherLogits {
                                        top_indices: top_indices_u16,
                                        top_values_f16,
                                        logsumexp_f16,
                                        batch_size: batch_size_u16,
                                        seq_len: seq_len_u16,
                                        top_k: k,
                                        temperature,
                                    };
                                    compressed.validate().map_err(|e| {
                                        TrainError::InvalidTeacherLogits {
                                            step,
                                            batch_id,
                                            reason: format!(
                                                "compressed teacher logits invalid: {e}"
                                            ),
                                        }
                                    })?;

                                    Ok::<_, TrainError>((trainer, shard_start, compressed))
                                }));
                            }

                            let mut shards_out: Vec<(usize, CompressedTeacherLogits)> = Vec::new();
                            while let Some(completed) = in_progress.next().await {
                                let (trainer, shard_start, logits) =
                                    completed.map_err(|_| TrainError::TrainCrashed)??;
                                available_trainers.push(trainer);
                                shards_out.push((shard_start, logits));
                            }

                            shards_out.sort_by_key(|(start, _)| *start);
                            let mut total_batch: usize = 0;
                            let mut seq_len: Option<u16> = None;
                            let mut top_k: Option<u16> = None;
                            let mut top_indices: Vec<u16> = Vec::new();
                            let mut top_values_f16: Vec<u16> = Vec::new();
                            let mut logsumexp_f16: Vec<u16> = Vec::new();
                            for (_, shard_logits) in shards_out {
                                total_batch += shard_logits.batch_size as usize;
                                seq_len.get_or_insert(shard_logits.seq_len);
                                top_k.get_or_insert(shard_logits.top_k);
                                if Some(shard_logits.seq_len) != seq_len
                                    || Some(shard_logits.top_k) != top_k
                                {
                                    return Err(TrainError::InvalidTeacherLogits {
                                        step,
                                        batch_id,
                                        reason: "inconsistent shard shapes in teacher logits"
                                            .to_string(),
                                    });
                                }
                                top_indices.extend_from_slice(&shard_logits.top_indices);
                                top_values_f16.extend_from_slice(&shard_logits.top_values_f16);
                                logsumexp_f16.extend_from_slice(&shard_logits.logsumexp_f16);
                            }

                            let batch_size_u16: u16 = total_batch.try_into().map_err(|_| {
                                TrainError::InvalidTeacherLogits {
                                    step,
                                    batch_id,
                                    reason: format!("combined batch_size too large: {total_batch}"),
                                }
                            })?;
                            let compressed = CompressedTeacherLogits {
                                top_indices,
                                top_values_f16,
                                logsumexp_f16,
                                batch_size: batch_size_u16,
                                seq_len: seq_len.unwrap_or(0),
                                top_k: top_k.unwrap_or(0),
                                temperature: teacher_logits_temperature,
                            };
                            compressed.validate().map_err(|e| {
                                TrainError::InvalidTeacherLogits {
                                    step,
                                    batch_id,
                                    reason: format!("combined teacher logits invalid: {e}"),
                                }
                            })?;

                            let transmittable = TransmittableTeacherLogits {
                                step,
                                batch_id,
                                logits: compressed,
                            };
                            if tx_teacher_logits.send(transmittable).is_err() {
                                return Err(TrainError::SendTeacherLogits);
                            }
                        }
                    }

                    while let Some(data) = next_sample.recv().await {
                        let tokens_processed_for_batch: u32 = match &data.data {
                            BatchData::CPU(items) => items
                                .iter()
                                .map(|item| item.input_ids.len() as u64)
                                .sum::<u64>()
                                .min(u32::MAX as u64)
                                as u32,
                            BatchData::GPU(_) => 0,
                        };
                        let mut in_progress = FuturesUnordered::new();

                        // reset the DP barriers
                        if let Some(trainer) = available_trainers.first() {
                            #[allow(irrefutable_let_patterns)]
                            if let Trainer::Local(trainer) = trainer {
                                if available_trainers.len() != trainer.data_parallel_world_size() {
                                    error!("Available trainers does not equal DP world size");
                                    return Err(TrainError::TrainCrashed);
                                }
                                trainer.data_parallel_barrier();
                            }
                        } else {
                            error!("No available trainers");
                            return Err(TrainError::TrainCrashed);
                        }

                        let shards = match &data.data {
                            BatchData::CPU(items) => {
                                split_batch_cpu_evenly(items, available_trainers.len())?
                            }
                            BatchData::GPU(_) => {
                                error!("Got data on GPU before distribution to trainers");
                                return Err(TrainError::TrainCrashed);
                            }
                        };

                        let teacher_for_batch = if matformer_tier > 0 && distillation_active {
                            Some(
                                wait_get_teacher_logits(
                                    teacher_logits_store.clone(),
                                    step,
                                    data.id,
                                )
                                .await?,
                            )
                        } else {
                            None
                        };
                        if let Some(tl) = teacher_for_batch.as_ref() {
                            let expected_batch = match &data.data {
                                BatchData::CPU(items) => items.len(),
                                BatchData::GPU(_) => 0,
                            };
                            if tl.step != step || tl.batch_id != data.id {
                                return Err(TrainError::InvalidTeacherLogits {
                                    step,
                                    batch_id: data.id,
                                    reason: format!(
                                        "teacher logits step/batch mismatch: got step={} batch={}, expected step={} batch={}",
                                        tl.step, tl.batch_id, step, data.id
                                    ),
                                });
                            }
                            if tl.logits.batch_size as usize != expected_batch {
                                return Err(TrainError::InvalidTeacherLogits {
                                    step,
                                    batch_id: data.id,
                                    reason: format!(
                                        "teacher batch_size mismatch: teacher={} student={expected_batch}",
                                        tl.logits.batch_size
                                    ),
                                });
                            }
                        }

                        for (trainer, shard) in available_trainers.drain(..).zip(shards) {
                            let batch_id = data.id;
                            let BatchShardCPU {
                                start: shard_start,
                                data: batch_data,
                            } = shard;
                            let shard_len = batch_data.len();
                            let cancel_training = cancel_training.clone();
                            let prev_self_distro_results = prev_self_distro_results.clone();

                            let teacher_targets_for_train: Option<(TeacherLogitTargets, f64)> =
                                match teacher_for_batch.as_ref() {
                                    Some(tl) => {
                                        let targets = teacher_targets_for_shard(
                                            tl,
                                            step,
                                            shard_start,
                                            shard_len,
                                            distillation_combine_mode,
                                            distillation_min_teacher_topk_mass,
                                            distillation_kd_q_topk_mass_floor,
                                        )?;
                                        Some((targets, distill_beta))
                                    }
                                    None => None,
                                };

                            in_progress.push(tokio::task::spawn_blocking(
                                move || -> Result<TrainOutput, TrainError> {
                                    let mut trainer = trainer;
                                    let mut prev_self_local = prev_self_distro_results;
                                    let inner_steps = local_inner_steps.max(1) as usize;

                                    for inner_idx in 0..inner_steps {
                                        let teacher_targets_this_step = if inner_idx == 0 {
                                            teacher_targets_for_train.clone()
                                        } else {
                                            None
                                        };
                                        let output = trainer.train(
                                            step,
                                            Batch {
                                                id: batch_id,
                                                data: BatchData::CPU(batch_data.clone()),
                                            },
                                            warmup_lr_between,
                                            zero_optim && inner_idx == 0,
                                            Vec::new(),
                                            Some(prev_self_local.clone()),
                                            cancel_training.clone(),
                                            false,
                                            teacher_logits_top_k,
                                            teacher_targets_this_step,
                                        )?;

                                        if inner_idx + 1 == inner_steps {
                                            return Ok(output);
                                        }

                                        let mut next_trainer = output.trainer;
                                        if let Some(local_results) = output.distro_results {
                                            if !local_results.is_empty() {
                                                prev_self_local = vec![local_results.clone()];
                                                next_trainer = next_trainer
                                                    .optimize(
                                                        step,
                                                        warmup_lr_between,
                                                        Some(vec![local_results]),
                                                    )
                                                    .map_err(|err| {
                                                        error!(
                                                            step = step,
                                                            batch_id = %batch_id,
                                                            inner_idx = inner_idx,
                                                            "Local inner optimize failed: {err:#}"
                                                        );
                                                        TrainError::TrainCrashed
                                                    })?;
                                            }
                                        }
                                        trainer = next_trainer;
                                    }

                                    Err(TrainError::TrainCrashed)
                                },
                            ));
                        }

                        // the distro results are identical across all ranks, so we just send the first one we get
                        let mut sent_results = false;

                        while let Some(completed_trainer) = in_progress.next().await {
                            let TrainOutput {
                                batch_id,
                                trainer,
                                loss,
                                step,
                                distro_results,
                                cancelled,
                                nonce,
                                grad_norm,
                                teacher_logits: _raw_teacher_logits,
                            } = completed_trainer.map_err(|_| TrainError::TrainCrashed)??;

                            debug!(step=step, loss=loss, batch_id=%batch_id, "Got training output, DisTrO results generated");

                            available_trainers.push(trainer);

                            if !sent_results {
                                let distro_results = distro_results.unwrap_or_default();

                                for result in &distro_results {
                                    if let Some(stats) = &result.stats {
                                        for (name, value) in stats {
                                            // a rolling average for this step :)
                                            optim_stats
                                                .entry(name.clone())
                                                .and_modify(|e| *e = (*e + value) / 2.0)
                                                .or_insert(*value);
                                        }
                                    }
                                }
                                let write_gradients_dir = write_gradients_dir.clone();
                                let tx_distro_result = tx_distro_result.clone();
                                let parameter_names = parameter_names.clone();
                                let aggregation_metadata =
                                    psyche_network::SerializedDistroAggregationMetadata {
                                        inner_steps_used: local_inner_steps.min(u16::MAX as u32)
                                            as u16,
                                        // Inner loop currently reuses outer step LR; use the
                                        // effective count as the LR-sum normalizer.
                                        sum_local_lr: local_inner_steps as f32,
                                        tokens_processed: tokens_processed_for_batch,
                                        delta_l2_preclip: grad_norm,
                                        // Post-clip norm is not currently surfaced by trainer
                                        // thread; keep equal to pre-clip until available.
                                        delta_l2_postclip: grad_norm,
                                        matformer_tier,
                                    };
                                let res: Result<(), TrainError> = tokio::task::spawn_blocking(move || {
                                    if cancelled {
                                        trace!("However, we were cancelled, so we're throwing away this result.");
                                        // we're throwing away this result.
                                        return Ok(());
                                    }

                                    let to_transmit = if quantize { Trainer::quantize_results(&distro_results) } else { distro_results.clone()};

                                    if !to_transmit.is_empty() && to_transmit.len() != parameter_names.len() {
                                        return Err(TrainError::ParameterCountMismatch {
                                            expected: parameter_names.len(),
                                            got: to_transmit.len(),
                                        });
                                    }

                                    let transmittable_distro_result = TransmittableDistroResult {
                                        step,
                                        batch_id,
                                        aggregation_metadata,
                                        distro_results: to_transmit
                                            .into_iter()
                                            .zip(parameter_names.iter())
                                            .map(|(x, name)| SerializedDistroResult::try_from((name.as_str(), &x)))
                                            .collect::<std::result::Result<Vec<_>, _>>()
                                            .map_err(TrainError::SerializeDistroResult)?,
                                        trainer_nonce: nonce,
                                    };

                                    if let Some(dir) = write_gradients_dir {
                                        let transmittable_distro_result = transmittable_distro_result.clone();
                                        let dir = dir.clone();
                                        tokio::spawn(async move {
                                            if let Err(err) =
                                                write_gradients_to_disk(dir, identity, transmittable_distro_result).await
                                            {
                                                error!("Failed to write gradients to disk: {err:#}");
                                            }
                                        });
                                    }

                                    let commitment_data_hash = transmittable_distro_result.comptue_hash();

                                    trace!("trying to queue tx distro result...");
                                    tx_distro_result
                                        .send(DistroBroadcastAndPayload {
                                            step,
                                            batch_id,
                                            commitment_data_hash,
                                            proof: committee_proof,
                                            distro_result: transmittable_distro_result,
                                            original_distro_result: distro_results,
                                        })
                                        .map_err(|_| TrainError::SendDistroResult)?;
                                    trace!("successfully queued tx distro result");
                                    Ok(())
                                }).await.map_err(|_| TrainError::TransmitCrashed)?;
                                res?;

                                round_losses.push(loss);

                                // Step-level logging to WandB if enabled
                                if step_logging_enabled {
                                    if let Some(ref run) = wandb_run {
                                        let mut step_log = LogData::new();
                                        step_log.insert("_step", step);
                                        step_log.insert("step/batch_idx", batch_idx);
                                        step_log.insert("step/loss", loss);
                                        step_log.insert("step/perplexity", loss.exp());
                                        let run = run.clone();
                                        tokio::spawn(async move {
                                            run.log(step_log).await;
                                        });
                                    }
                                }
                                batch_idx += 1;

                                sent_results = true;
                            }
                        }
                    }

                    let evals = if cancel_training.is_cancelled() {
                        // we got timed out, don't bother starting evals
                        MaybeRunningEvals::NotRunning(available_trainers)
                    } else {
                        // we finished before getting cancelled, have some time to start evals.
                        MaybeRunningEvals::Running(model_task_runner.start(available_trainers))
                    };
                    let round_duration = Instant::now() - round_start;
                    debug!("Training for round finished, duration {:?}", round_duration);
                    finished.store(true, Ordering::SeqCst);
                    Ok(FinishedTrainers {
                        evals_or_trainers: evals,
                        round_losses,
                        optim_stats,
                        round_duration,
                    })
                })
            };

        Ok(TrainingStep {
            applying_and_training,
            cancel_training,
            sending_health_checks,
            finished,
        })
    }

    fn apply_results(
        &mut self,
        trainers: Vec<Trainer>,
        state: &Coordinator<T>,
        previous_round: &mut RoundState<T>,
        current_round: &mut RoundState<T>,
    ) -> Result<JoinHandle<Result<Vec<Trainer>, ApplyError>>, ApplyError> {
        if current_round.height == 0 {
            // the first TWO training step of each epoch has no apply phase.
            // but, because we call this once with the default initalized RoundState (round 0)
            // and a second time (when transitioning from round 0 -> round 1), this check will skip
            // the two phases
            trace!("Skipping early apply");
            return Ok(tokio::task::spawn(async move { Ok(trainers) }));
        }

        let apply_start = Instant::now();
        let step = state.progress.step;
        let calibration_start = self.same_batch_calibration_start_step.max(1);
        let calibration_active = self.same_batch_calibration_every_steps > 0
            && step >= calibration_start
            && (step - calibration_start) % self.same_batch_calibration_every_steps == 0;
        let calibration_no_apply = self.same_batch_calibration_no_apply;
        let calibration_every_steps = self.same_batch_calibration_every_steps;
        let witness_quorum = state.witness_quorum(
            state
                .previous_round()
                .ok_or(ApplyError::NoActiveRound)?
                .witnesses
                .len() as u16,
        );
        let cold_start_warmup_steps = match &state.model {
            model::Model::LLM(llm) => llm.cold_start_warmup_steps,
        };
        let warmup_lr_between = state.get_cold_start_warmup_bounds();
        let epoch = state.progress.epoch;

        // coordinator has already advanced to the next round (unless we're in cooldown) but we haven't started ours yet.
        // so our current_round corresponds to the coordinator's previous_round
        // `previous_round` -> state.previous_previous_round()
        // `current_round` -> state.previous_round()
        let payloads = std::mem::take(&mut previous_round.downloads);
        let commitments = std::mem::take(&mut previous_round.results);

        // here, when dealing with the coordinator,
        let witnesses = state
            .previous_round()
            .ok_or(ApplyError::NoActiveRound)?
            .witnesses;
        let batch_ids = get_batch_ids_for_round(
            state
                .previous_previous_round()
                .ok_or(ApplyError::NoActiveRound)?,
            state,
            previous_round
                .committee_info
                .as_ref()
                .ok_or(ApplyError::NoActiveRound)?
                .2
                .get_num_trainer_nodes(),
        );

        let data_assignments = previous_round.data_assignments.clone();
        let apply_only_trainer_index = self.distro_apply_only_trainer_index;
        let apply_only_trainer_id: Option<T> = if let Some(index) = apply_only_trainer_index {
            let trainer_set: HashSet<T> = data_assignments.values().copied().collect();
            let trainers_in_order: Vec<T> = state
                .epoch_state
                .clients
                .iter()
                .filter_map(|c| trainer_set.contains(&c.id).then_some(c.id))
                .collect();

            let num_trainers = trainers_in_order.len();
            let Some(selected) = trainers_in_order.get(index as usize).copied() else {
                return Err(ApplyError::InvalidApplyOnlyTrainerIndex {
                    index,
                    num_trainers: num_trainers as u16,
                });
            };
            debug!(
                step,
                epoch,
                index,
                num_trainers,
                selected_trainer = %selected,
                "Applying updates from only one trainer slot (dropping other trainers' batches)"
            );
            Some(selected)
        } else {
            None
        };

        Ok(tokio::task::spawn(async move {
                let payloads = payloads.clone();
                let mut distro_results: Vec<Vec<DistroResult>> = Vec::new();
                let mut dropped_batches_for_apply: usize = 0;
                let mut kept_batches_for_apply: usize = 0;

                trace!("Have commitments for batches {:?}", commitments.keys().collect::<Vec<_>>());
                trace!("Have payloads for hashes {:?}", payloads.lock().unwrap().keys().collect::<Vec<_>>());

                for batch_id in batch_ids {
                    if let Some(selected) = apply_only_trainer_id {
                        match data_assignments.get(&batch_id) {
                            Some(assigned) if *assigned == selected => {}
                            Some(_) => {
                                dropped_batches_for_apply += 1;
                                continue;
                            }
                            None => {
                                dropped_batches_for_apply += 1;
                                continue;
                            }
                        }
                    }

                    let batch_commitments = match commitments.get(&batch_id) {
                        Some(x) => x,
                        None => {
                            let expected_trainer = data_assignments.get(&batch_id);
                            warn!(
                                integration_test_log_marker = %IntegrationTestLogMarker::UntrainedBatches,
                                batch_id = %batch_id,
                                expected_trainer = ?expected_trainer,
                                "No commitments for batch {batch_id}, assigned to node {expected_trainer:?}",
                            );
                            continue;
                        }
                    };
                    trace!("Commitments for batch {batch_id}: {batch_commitments:?}");
                    let consensus = match Coordinator::<T>::select_consensus_commitment_by_witnesses(
                        &batch_commitments
                            .iter()
                            .map(|x| x.1.0)
                            .collect::<Vec<_>>(),
                        &witnesses,
                        witness_quorum,
                    ) {
                        Some(x) => x,
                        None => {
                            warn!("No consensus commitment for batch {}", batch_id);
                            continue;
                        }
                    };
                    trace!("Consensus commitment for batch {batch_id}: {consensus:?}");

                    let (commitment, result) = &batch_commitments[consensus].1;
                    let payload_remove_result = payloads.lock().unwrap().remove(&result.ticket.hash());
                    let maybe_results: Result<(Vec<DistroResult>, u32), DeserializeError> = match payload_remove_result {
                        Some(PayloadState::Deserializing(x)) => match x.is_finished() {
                            true => x.await.unwrap(),
                            false => {
                                return Err(ApplyError::DidNotFinishDeserializingCommitment(
                                    Box::new(*commitment),
                                    batch_id,
                                ));
                            }
                        },
                        Some(PayloadState::Downloading((_, _, ticket))) => {
                            return Err(ApplyError::DidNotBeginDownloadingCommitment(
                                Box::new(*commitment),
                                batch_id,
                                ticket.hash()
                            ));
                        }
                        None => {
                            return Err(ApplyError::UnknownCommitment(
                                Box::new(*commitment),
                                batch_id,
                            ))
                        }
                    };

                    match maybe_results {
                        Ok((results, trainer_nonce)) => {
                            if trainer_nonce < cold_start_warmup_steps && epoch != 0 && warmup_lr_between.is_none()  {
                                // results are not actually applied for the first cold_start_warmup_steps of a trainer's lifetime
                                // note, we are relying on honest communication of this value here -- will need to harden with verification.
                                // the only exception is for the first steps of the first epoch
                                // or when doing a cold start (warmup_lr_between.is_some())
                                info!("Skipping apply of batch {batch_id}, trainer warming up ({trainer_nonce}/{cold_start_warmup_steps})");
                            } else {
                                kept_batches_for_apply += 1;
                                distro_results.push(results);
                            }
                        }
                        Err(err) => warn!("DESYNC: Got the following error when deserializing results for commitment 0x{}: {}", hex::encode(commitment.data_hash), err),
                    }
                }

                if let Some(selected) = apply_only_trainer_id {
                    debug!(
                        selected_trainer = %selected,
                        dropped_batches_for_apply,
                        kept_batches_for_apply,
                        "Apply-time trainer-slot filter summary"
                    );
                }

                if calibration_active && calibration_no_apply {
                    info!(
                        step = step,
                        every_steps = calibration_every_steps,
                        start_step = calibration_start,
                        dropped_batches_for_apply,
                        kept_batches_for_apply,
                        "Skipping distributed apply for same-batch calibration step"
                    );
                    return Ok(trainers);
                }

                let futures: Vec<JoinHandle<std::result::Result<Trainer, ApplyDistroResultError>>> =
                    trainers
                        .into_iter()
                        .map(|trainer| {
                            let distro_results = Some(distro_results.clone());

                            tokio::task::spawn_blocking(move || {
                                trainer.optimize(step, warmup_lr_between, distro_results)
                            })
                        })
                        .collect::<Vec<_>>();
                let trainers: Vec<_> = try_join_all(futures)
                    .await
                    .map_err(|_| ApplyDistroResultError::ThreadCrashed)?
                    .into_iter()
                    .collect::<Result<_, _>>()?;
                trace!(
                    "Apply time: {:.1}s, {} trainers ready",
                    (Instant::now() - apply_start).as_secs_f32(),
                    trainers.len()
                );
                Ok(trainers)
            }.instrument(trace_span!("Applying distro results"))))
    }
}

fn start_sending_health_checks<T: NodeIdentity>(
    round_state: &mut RoundState<T>,
    state: &Coordinator<T>,
    tx_health_check: mpsc::UnboundedSender<HealthChecks<T>>,
) -> Result<Option<JoinHandle<Result<(), TrainError>>>, TrainError> {
    // we won't have any information to health check with until at least one round of training has finished
    if round_state.height == 0 {
        return Ok(None);
    }
    let (_, witness_proof, committee_selection) = round_state
        .committee_info
        .as_ref()
        .ok_or(TrainError::NoCommitteeInfo)?;
    Ok(
        if state.epoch_state.first_round.is_false() && witness_proof.witness.is_true() {
            let clients = state.epoch_state.clients;
            let committee_selection = committee_selection.clone();
            let state = *state;
            Some(tokio::task::spawn(async move {
                let mut checks = HealthChecks::new();
                for (index, client) in clients.iter().enumerate() {
                    let proof = committee_selection.get_committee(index as u64);
                    if !state.healthy(&client.id, &proof).unwrap_or(false) {
                        warn!(
                            integration_test_log_marker = %IntegrationTestLogMarker::HealthCheck,
                            index = index,
                            client_id = %&client.id,
                            current_step = state.epoch_state.rounds_head,
                            "Found unhealthy trainer at index: {}", index,
                        );
                        checks.push((client.id, proof));
                    }
                }

                if !checks.is_empty() {
                    info!("Sending health check for following indicies: {:?}", checks);
                    tx_health_check
                        .send(checks)
                        .map_err(|_| TrainError::SendHealthChecks)
                } else {
                    Ok(())
                }
            }))
        } else {
            None
        },
    )
}

#[derive(Error, Debug)]
pub enum ApplyError {
    #[error("no active round")]
    NoActiveRound,

    #[error("invalid --distro-apply-only-trainer-index {index} (num_trainers={num_trainers})")]
    InvalidApplyOnlyTrainerIndex { index: u16, num_trainers: u16 },

    #[error("failed to apply distro result: {0}")]
    BadResult(#[from] ApplyDistroResultError),

    #[error("DESYNC: Did not finish deserializing payload for consensus commitment 0x{commitment} for batch {1}", commitment=hex::encode(.0.data_hash))]
    DidNotFinishDeserializingCommitment(Box<Commitment>, BatchId),

    #[error("DESYNC: Did not begin downloading payload for consensus commitment 0x{commitment} for batch {1} with blob hash {2}", commitment=hex::encode(.0.data_hash))]
    DidNotBeginDownloadingCommitment(Box<Commitment>, BatchId, Hash),

    #[error("DESYNC: Unknown consensus commitment 0x{commitment} for batch {1}", commitment=hex::encode(.0.data_hash))]
    UnknownCommitment(Box<Commitment>, BatchId),
}

#[derive(Debug, Error)]
enum WriteGradientsError {
    #[error("Failed to create write_gradients_dir: {0}")]
    CreateDir(tokio::io::Error),

    #[error("Failed to serialize distro result data {fname} to bytes: {err}")]
    Serialize { fname: String, err: postcard::Error },

    #[error("Failed to write distro result data {fname}: {err}")]
    Write {
        fname: String,
        err: tokio::io::Error,
    },
}

async fn write_gradients_to_disk<T: NodeIdentity>(
    write_gradients_dir: PathBuf,
    identity: T,
    distro_result: TransmittableDistroResult,
) -> Result<(), WriteGradientsError> {
    debug!("Trying to write distro result to disk...");
    tokio::fs::create_dir_all(&write_gradients_dir)
        .await
        .map_err(WriteGradientsError::CreateDir)?;

    let fname = format!(
        "result-{}-step{}-batch{}.vec-postcard",
        identity, distro_result.step, distro_result.batch_id
    );
    let fpath = write_gradients_dir.join(&fname);
    let serialized = distro_results_to_bytes(&distro_result.distro_results).map_err(|err| {
        WriteGradientsError::Serialize {
            fname: fname.clone(),
            err,
        }
    })?;
    tokio::fs::write(fpath, serialized)
        .await
        .map_err(|err| WriteGradientsError::Write {
            fname: fname.clone(),
            err,
        })?;
    debug!("Wrote distro result {fname}.");
    Ok(())
}
