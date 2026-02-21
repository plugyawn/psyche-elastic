use crate::{HubUploadInfo, IntegrationTestLogMarker};

use psyche_coordinator::{
    model::{self, HubRepo},
    Coordinator,
};
use psyche_core::{BatchId, ClosedInterval, FixedString, NodeIdentity};
use psyche_data_provider::{
    upload_model_repo_async, LengthKnownDataProvider, LocalDataProvider, PreprocessedDataProvider,
    TokenizedDataProvider, UploadModelError,
};
use psyche_modeling::{
    save_tensors_into_safetensors, Batch, BatchData, BatchDataCPU, CausalLM, SaveSafetensorsError,
    Trainer, TrainerThreadCommunicationError,
};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    path::PathBuf,
    sync::Arc,
};
use tch::Tensor;
use thiserror::Error;
use tokio::{
    sync::{mpsc, Mutex},
    task::JoinHandle,
};
use tracing::{error, info, info_span, warn, Instrument};
use wandb::LogData;

use super::{
    evals::{ModelTaskRunner, RunningEvals},
    CheckpointConfig,
};

fn want_log_suffix_gate_stats() -> bool {
    match std::env::var("PSYCHE_LOG_SUFFIX_GATE_STATS") {
        Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

fn log_suffix_gate_stats(extracted: &HashMap<String, Tensor>) {
    if !want_log_suffix_gate_stats() {
        return;
    }

    let mut logits: Vec<f64> = Vec::new();
    for (name, t) in extracted {
        if name.ends_with(".matformer_suffix_gate_logit") {
            // Extracted variables are unsharded and on CPU.
            logits.push(t.double_value(&[]));
        }
    }
    if logits.is_empty() {
        return;
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    let alphas: Vec<f64> = logits.iter().copied().map(sigmoid).collect();
    let n = logits.len() as f64;
    let logit_mean = logits.iter().sum::<f64>() / n;
    let alpha_mean = alphas.iter().sum::<f64>() / n;
    let logit_min = logits.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let logit_max = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let alpha_min = alphas.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let alpha_max = alphas
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    info!(
        suffix_gate_params = logits.len(),
        suffix_gate_logit_min = logit_min,
        suffix_gate_logit_mean = logit_mean,
        suffix_gate_logit_max = logit_max,
        suffix_gate_alpha_min = alpha_min,
        suffix_gate_alpha_mean = alpha_mean,
        suffix_gate_alpha_max = alpha_max,
        "MatFormer suffix-gate (learnable) stats (from extracted model)"
    );
}

#[derive(Clone, Copy, Debug)]
pub struct HeldoutEvalConfig {
    pub batches: usize,
    pub batch_size: usize,
}

pub enum HeldoutEvalDataProvider {
    Local(LocalDataProvider),
    Preprocessed(PreprocessedDataProvider),
}

pub struct HeldoutEvaluator {
    provider: HeldoutEvalDataProvider,
    config: HeldoutEvalConfig,
    next_index: u64,
}

#[derive(Debug, Clone, Copy)]
struct HeldoutEvalSummary {
    batches: usize,
    batch_size: usize,
    mean_loss: f32,
    min_loss: f32,
    max_loss: f32,
}

#[derive(Debug, Error)]
pub enum HeldoutEvalError {
    #[error("heldout eval is disabled (batches={batches}, batch_size={batch_size})")]
    Disabled { batches: usize, batch_size: usize },

    #[error("heldout data provider has no sequences")]
    EmptyData,

    #[error("failed to read heldout data for {batch_id}: {source:#}")]
    GetSamples {
        batch_id: BatchId,
        #[source]
        source: anyhow::Error,
    },

    #[error("model forward did not produce a loss value")]
    MissingLoss,

    #[error("non-finite heldout loss encountered")]
    NonFiniteLoss,
}

impl HeldoutEvaluator {
    pub fn new_local(provider: LocalDataProvider, config: HeldoutEvalConfig) -> Self {
        Self {
            provider: HeldoutEvalDataProvider::Local(provider),
            config,
            next_index: 0,
        }
    }

    pub fn new_preprocessed(provider: PreprocessedDataProvider, config: HeldoutEvalConfig) -> Self {
        Self {
            provider: HeldoutEvalDataProvider::Preprocessed(provider),
            config,
            next_index: 0,
        }
    }

    fn total_sequences(&self) -> usize {
        match &self.provider {
            HeldoutEvalDataProvider::Local(provider) => provider.num_sequences(),
            HeldoutEvalDataProvider::Preprocessed(provider) => provider.num_sequences(),
        }
    }

    async fn next_batch_cpu(&mut self) -> Result<(BatchId, Vec<BatchDataCPU>), HeldoutEvalError> {
        let total = self.total_sequences();
        if total == 0 {
            return Err(HeldoutEvalError::EmptyData);
        }

        let requested_batch_size = self.config.batch_size.max(1);
        let batch_size = requested_batch_size.min(total) as u64;
        let total = total as u64;

        let batch_id = match &self.provider {
            HeldoutEvalDataProvider::Local(_) => {
                if self.next_index + batch_size > total {
                    self.next_index = 0;
                }
                let start = self.next_index;
                let end = start + batch_size - 1;
                self.next_index = end + 1;
                BatchId(ClosedInterval::new(start, end))
            }
            HeldoutEvalDataProvider::Preprocessed(_) => {
                let start = self.next_index % total;
                let end = start + batch_size - 1;
                self.next_index = (start + batch_size) % total;
                BatchId(ClosedInterval::new(start, end))
            }
        };

        let samples = match &mut self.provider {
            HeldoutEvalDataProvider::Local(provider) => provider
                .get_samples(batch_id)
                .await
                .map_err(|source| HeldoutEvalError::GetSamples { batch_id, source })?,
            HeldoutEvalDataProvider::Preprocessed(provider) => provider
                .get_samples(batch_id)
                .await
                .map_err(|source| HeldoutEvalError::GetSamples { batch_id, source })?,
        };

        let cpu = samples
            .into_iter()
            .map(|sample| BatchDataCPU {
                input_ids: sample.input_ids,
                labels: sample.labels,
                position_ids: sample.position_ids,
                sequence_lengths: sample.sequence_lengths,
            })
            .collect();

        Ok((batch_id, cpu))
    }

    async fn evaluate(
        &mut self,
        trainer: &Trainer,
    ) -> Result<HeldoutEvalSummary, HeldoutEvalError> {
        if self.config.batches == 0 || self.config.batch_size == 0 {
            return Err(HeldoutEvalError::Disabled {
                batches: self.config.batches,
                batch_size: self.config.batch_size,
            });
        }

        let mut losses = Vec::with_capacity(self.config.batches);
        for _ in 0..self.config.batches {
            let (batch_id, cpu) = self.next_batch_cpu().await?;
            let batch = Batch {
                id: batch_id,
                data: BatchData::CPU(cpu),
            }
            .gpu(trainer.device());
            let BatchData::GPU(gpu) = batch.data else {
                unreachable!("Batch::gpu always returns GPU data");
            };

            let labels_fallback = if gpu.labels.is_none() {
                Some(gpu.input_ids.shallow_clone())
            } else {
                None
            };
            let labels = gpu.labels.as_ref().or(labels_fallback.as_ref());

            let (_, loss) = trainer.forward(
                &gpu.input_ids,
                labels,
                gpu.position_ids.as_ref(),
                gpu.sequence_lengths.as_ref(),
                None,
                None,
            );
            let Some(loss) = loss else {
                return Err(HeldoutEvalError::MissingLoss);
            };
            let loss_value = loss.double_value(&[]) as f32;
            if !loss_value.is_finite() {
                return Err(HeldoutEvalError::NonFiniteLoss);
            }
            losses.push(loss_value);
        }

        let mean_loss = losses.iter().copied().sum::<f32>() / losses.len() as f32;
        let min_loss = losses.iter().copied().fold(f32::INFINITY, f32::min);
        let max_loss = losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(HeldoutEvalSummary {
            batches: self.config.batches,
            batch_size: self.config.batch_size,
            mean_loss,
            min_loss,
            max_loss,
        })
    }
}

#[derive(Error, Debug)]
pub enum CooldownError {
    #[error("no trainers available for checkpointing!")]
    NoTrainers,

    #[error("checkpointing thread crashed")]
    CheckpointThreadCrashed,

    #[error("error while checkpointing: {0}")]
    Checkpoint(#[from] CheckpointError),
}

/// MatFormer tier information for checkpoint saving.
#[derive(Clone, Copy, Debug)]
pub struct MatformerCheckpointInfo {
    /// The effective tier used during training (0 = full width).
    pub effective_tier: u8,
    /// The base intermediate size before any tier slicing.
    pub base_intermediate_size: Option<u64>,
}

pub struct CooldownStepMetadata {
    tx_checkpoint: mpsc::UnboundedSender<model::HubRepo>,
    tx_model: mpsc::UnboundedSender<HashMap<String, Tensor>>,
    checkpoint_info: Option<CheckpointConfig>,
    checkpoint_extra_files: Vec<PathBuf>,
    matformer_info: MatformerCheckpointInfo,
    heldout_evaluator: Option<Arc<Mutex<HeldoutEvaluator>>>,
    wandb_run: Option<Arc<wandb::Run>>,

    model_task_runner: ModelTaskRunner,
    // use a heap here as a best-effort attempt to ensure we get rid of the lowest step number dir even if we spawn multiple tasks
    // which may not finish writing their dirs in order. We note that even if we were to take the more complicated
    // route of actually enumerating the checkpoint_dir there would still be a race condition, unless we took a lockfile
    // or the like on the entire checkpoint_dir which probably isn't worth it just to support disk cleanup
    // we don't really expect there to be contention on this lock or real race conditions in practice though
    // as by the time one task spawns after a training round the previous write/upload task(s) should (hopefully) be long done
    delete_queue: Arc<Mutex<BinaryHeap<Reverse<u32>>>>,
}

impl CooldownStepMetadata {
    pub fn new(
        tx_checkpoint: mpsc::UnboundedSender<model::HubRepo>,
        tx_model: mpsc::UnboundedSender<HashMap<String, Tensor>>,
        checkpoint_info: Option<CheckpointConfig>,
        checkpoint_extra_files: Vec<PathBuf>,
        matformer_info: MatformerCheckpointInfo,
        heldout_evaluator: Option<Arc<Mutex<HeldoutEvaluator>>>,
        wandb_run: Option<Arc<wandb::Run>>,
        model_task_runner: ModelTaskRunner,
    ) -> Self {
        Self {
            tx_checkpoint,
            tx_model,
            checkpoint_info,
            checkpoint_extra_files,
            matformer_info,
            heldout_evaluator,
            wandb_run,
            model_task_runner,
            delete_queue: Arc::new(Mutex::new(BinaryHeap::new())),
        }
    }
}

#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("Extract thread crashed")]
    ExtractThreadCrashed,

    #[error("Trainer extract error: {0}")]
    Extract(#[from] TrainerThreadCommunicationError),

    #[error("Write thread crashed")]
    WriteThreadCrashed,

    #[error("Writing safetensors to disk failed: {0}")]
    WriteSafetensors(#[from] SaveSafetensorsError),

    #[error("Writing extra file to disk failed: {0}")]
    WriteExtraFile(tokio::io::Error),

    #[error("Reading config.json failed: {0}")]
    ReadConfigJson(tokio::io::Error),

    #[error("Parsing config.json failed: {0}")]
    ParseConfigJson(serde_json::Error),

    #[error("Writing config.json failed: {0}")]
    WriteConfigJson(tokio::io::Error),

    #[error("Couldn't upload model to huggingface: {0}")]
    UploadError(#[from] UploadModelError),

    #[error("Couldn't send checkpoint - channel closed")]
    SendCheckpoint,
}

async fn cleanup_dirs(
    delete_queue: Arc<Mutex<BinaryHeap<Reverse<u32>>>>,
    keep_steps: u32,
    run_id: String,
    delete_old_steps: bool,
    step: u32,
    checkpoint_dir: PathBuf,
) {
    if delete_old_steps {
        let mut delete_queue_guard = delete_queue.lock().await;
        delete_queue_guard.push(Reverse(step));
        // in the happy case this could be an if but if previous iterations failed somewhere
        // then we may have more than 1 dir to clean up
        while delete_queue_guard.len() > keep_steps as usize {
            let delete_step = delete_queue_guard.pop().unwrap().0;
            let delete_path = checkpoint_dir.join(format!("{run_id}-step{delete_step}"));
            if let Err(err) = tokio::fs::remove_dir_all(delete_path.clone()).await {
                warn!("Error removing {} : {}", delete_path.display(), err);
            } else {
                info!("Successfully removed {}", delete_path.display());
            }
        }
    }
}

impl CooldownStepMetadata {
    pub fn start<T: NodeIdentity>(
        &self,
        mut trainers: Vec<Trainer>,
        state: &Coordinator<T>,
    ) -> Result<CooldownStep, CooldownError> {
        let Some(mut trainer) = trainers.pop() else {
            return Err(CooldownError::NoTrainers);
        };

        let step = state.progress.step - 1;
        let run_id = String::from(&state.run_id);
        let checkpoint_extra_files = self.checkpoint_extra_files.clone();
        let checkpoint_info = self.checkpoint_info.clone();
        let matformer_info = self.matformer_info;
        let heldout_evaluator = self.heldout_evaluator.clone();
        let wandb_run = self.wandb_run.clone();
        let tx_checkpoint = self.tx_checkpoint.clone();
        let tx_model = self.tx_model.clone();
        let model_task_runner = self.model_task_runner.clone();
        let delete_queue = self.delete_queue.clone();

        let checkpointing_and_evals: CheckpointAndEvalsHandle = tokio::task::spawn(
            async move {
                if let Some(heldout_evaluator) = heldout_evaluator {
                    let mut evaluator = heldout_evaluator.lock().await;
                    match evaluator.evaluate(&trainer).await {
                        Ok(summary) => {
                            info!(
                                integration_test_log_marker = %IntegrationTestLogMarker::HeldoutEval,
                                step = step,
                                matformer_tier = matformer_info.effective_tier,
                                heldout_eval_batches = summary.batches,
                                heldout_eval_batch_size = summary.batch_size,
                                heldout_loss = summary.mean_loss,
                                heldout_perplexity = summary.mean_loss.exp(),
                                heldout_loss_min = summary.min_loss,
                                heldout_loss_max = summary.max_loss,
                                "heldout_eval",
                            );

                            if let Some(run) = wandb_run.clone() {
                                let mut val_log = LogData::new();
                                // Keep validation points on human-friendly 1-based step indices
                                // so epoch_time=1000 yields points at 1000, 2000, ...
                                val_log.insert("_step", (step as u64) + 1);
                                val_log.insert("val/heldout_loss", summary.mean_loss);
                                val_log.insert("val/heldout_perplexity", summary.mean_loss.exp());
                                val_log.insert("val/heldout_loss_min", summary.min_loss);
                                val_log.insert("val/heldout_loss_max", summary.max_loss);
                                val_log.insert("val/heldout_eval_batches", summary.batches as u64);
                                val_log.insert(
                                    "val/heldout_eval_batch_size",
                                    summary.batch_size as u64,
                                );
                                val_log.insert(
                                    "val/matformer_tier_effective",
                                    matformer_info.effective_tier as u64,
                                );
                                run.log(val_log).await;
                            }
                        }
                        Err(err) => {
                            warn!(step = step, "Held-out eval failed: {err:#}");
                        }
                    }
                }

                info!("Extracting full model...");
                let (variables, trainer) =
                    tokio::task::spawn_blocking::<_, Result<_, CheckpointError>>(|| {
                        let variables = trainer.extract()?;
                        info!("Model extracted; {} parameters", variables.len());
                        Ok((variables, trainer))
                    })
                    .await
                    .map_err(|_| CheckpointError::ExtractThreadCrashed)??;

                log_suffix_gate_stats(&variables);

                let variables_clone: HashMap<String, Tensor> = variables
                    .iter()
                    .map(|(name, var)| (name.clone(), var.shallow_clone()))
                    .collect();

                trainers.push(trainer);
                let evals = model_task_runner.start(trainers);

                tx_model
                    .send(variables_clone)
                    .map_err(|_| CheckpointError::SendCheckpoint)?;

                let Some(CheckpointConfig {
                    hub_upload,
                    checkpoint_dir,
                    delete_old_steps,
                    keep_steps,
                }) = checkpoint_info
                else {
                    // If there was no HF checkpointing configuration, return immediately
                    return Ok((evals, None));
                };

                // Start the upload process of the updated model parameters in a separate task
                let upload_handle = tokio::task::spawn(async move {
                    let path = checkpoint_dir.join(format!("{run_id}-step{step}"));
                    info!("Saving to {}", path.display());
                    let mut local = tokio::task::spawn_blocking({
                        let path = path.clone();
                        move || save_tensors_into_safetensors(variables, path)
                    })
                    .await
                    .map_err(|_| CheckpointError::WriteThreadCrashed)??;

                    for extra in checkpoint_extra_files {
                        let filename = extra.file_name().unwrap();
                        let to = path.join(filename);

                        // Handle config.json specially: inject matformer_tier info
                        if filename == "config.json" {
                            let content = tokio::fs::read_to_string(&extra)
                                .await
                                .map_err(CheckpointError::ReadConfigJson)?;
                            let mut config: serde_json::Value = serde_json::from_str(&content)
                                .map_err(CheckpointError::ParseConfigJson)?;

                            if let Some(obj) = config.as_object_mut() {
                                let intermediate_size =
                                    obj.get("intermediate_size").and_then(|v| v.as_u64());
                                let base_size = matformer_info
                                    .base_intermediate_size
                                    .or_else(|| {
                                        obj.get("matformer_base_intermediate_size")
                                            .and_then(|v| v.as_u64())
                                    })
                                    .or(intermediate_size);

                                let actual_tier = match (base_size, intermediate_size) {
                                    (Some(base), Some(current))
                                        if current > 0
                                            && base >= current
                                            && base % current == 0 =>
                                    {
                                        let ratio = base / current;
                                        if ratio.is_power_of_two() {
                                            ratio.trailing_zeros() as u8
                                        } else {
                                            0
                                        }
                                    }
                                    _ => 0,
                                };

                                obj.insert(
                                    "matformer_tier".to_string(),
                                    serde_json::Value::from(actual_tier),
                                );

                                if let Some(base_size) = base_size {
                                    obj.insert(
                                        "matformer_base_intermediate_size".to_string(),
                                        serde_json::Value::from(base_size),
                                    );
                                }
                            }

                            let updated = serde_json::to_string_pretty(&config)
                                .map_err(CheckpointError::ParseConfigJson)?;
                            tokio::fs::write(&to, updated)
                                .await
                                .map_err(CheckpointError::WriteConfigJson)?;
                        } else {
                            tokio::fs::copy(extra.clone(), to.clone())
                                .await
                                .map_err(CheckpointError::WriteExtraFile)?;
                        }
                        local.push(to);
                    }

                    let Some(HubUploadInfo {
                        hub_repo,
                        hub_token,
                    }) = hub_upload
                    else {
                        cleanup_dirs(
                            delete_queue,
                            keep_steps,
                            run_id,
                            delete_old_steps,
                            step,
                            checkpoint_dir,
                        )
                        .await;
                        return Ok::<(), CheckpointError>(());
                    };

                    info!(repo = hub_repo, "Uploading checkpoint to HuggingFace");
                    let revision = match upload_model_repo_async(
                        hub_repo.clone(),
                        local,
                        hub_token.clone(),
                        Some(format!("step {step}")),
                        None,
                    )
                    .await
                    {
                        Ok(revision) => {
                            info!(
                                repo = hub_repo,
                                revision = revision,
                                "Upload to HuggingFace complete"
                            );
                            revision
                        }
                        Err(err) => {
                            error!(repo = hub_repo, "Error uploading to HuggingFace: {err:#}");
                            return Err(err.into());
                        }
                    };

                    tx_checkpoint
                        .send(HubRepo {
                            repo_id: FixedString::from_str_truncated(&hub_repo),
                            revision: Some(FixedString::from_str_truncated(&revision)),
                        })
                        .map_err(|_| CheckpointError::SendCheckpoint)?;

                    // we put the cleanup step at the end, so that if keep_steps == 0 the logic will still work
                    // we'll just delete the dir after we've uploaded it
                    // if we fail in any of the above steps we may wind up not queueing this dir for delete
                    // but that's probably better than risking having the dir deleted from under us
                    // for a relatively low priority disk cleanup task
                    // and this may actually be preferred anyway because if we failed to upload, we may want to keep
                    // the data around locally on disk
                    cleanup_dirs(
                        delete_queue,
                        keep_steps,
                        run_id,
                        delete_old_steps,
                        step,
                        checkpoint_dir,
                    )
                    .await;

                    Ok(())
                });

                Ok((evals, Some(upload_handle)))
            }
            .instrument(info_span!("checkpointing")),
        );
        Ok(CooldownStep {
            checkpointing_and_evals,
        })
    }
}

type CheckpointAndEvalsHandle = JoinHandle<
    Result<
        (
            RunningEvals,
            Option<JoinHandle<Result<(), CheckpointError>>>,
        ),
        CheckpointError,
    >,
>;

#[derive(Debug)]
pub struct CooldownStep {
    checkpointing_and_evals: CheckpointAndEvalsHandle,
}

impl CooldownStep {
    pub async fn finish(
        self,
    ) -> Result<
        (
            RunningEvals,
            Option<JoinHandle<Result<(), CheckpointError>>>,
        ),
        CooldownError,
    > {
        let (running_evals, upload_handle) = self
            .checkpointing_and_evals
            .await
            .map_err(|_| CooldownError::CheckpointThreadCrashed)??;

        Ok((running_evals, upload_handle))
    }
}
