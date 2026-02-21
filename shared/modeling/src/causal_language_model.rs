use crate::{
    AllReduce, AttentionImplementation, Communicator, CommunicatorId, DistillationCombineMode,
    ModelConfig, ModelLoadError, PretrainedSource, ReduceType, RoPEConfig, StableVarStoreIterator,
    StableVariableIterator,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::{fmt::Debug, sync::atomic::AtomicBool};
use tch::{
    nn::{self, Module},
    Device, Kind, Tensor,
};
use tracing::{error, info, trace};

#[cfg(feature = "parallelism")]
use tch::CNCCL;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum EosToks {
    Single(i64),
    Multiple(Vec<i64>),
}

/// This trait is for any Causal Language Model that can be inferred,
/// and thus can have backprop run on it.
/// Its internal implementation is completely hidden, so this can be impl'd
/// for a wrapper struct that does something like data parallelism.
pub trait CausalLM: Send {
    // returns (logits, loss)
    fn forward(
        &self,
        x: &Tensor,
        labels: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        num_logits_to_keep: Option<i64>,
        loss_scale: Option<f64>,
    ) -> (Option<Tensor>, Option<Tensor>);
    fn bos_token_id(&self) -> Option<i64>;
    fn eos_token_ids(&self) -> Option<EosToks>;
    fn device(&self) -> Device;
    fn max_context_length(&self) -> usize;
    fn variables(&self) -> StableVariableIterator;
    fn communicator(&self) -> Option<Arc<Communicator>>;
    fn prepare_for_training(&self);
    fn clip_grad_norm(&self, max_grad_norm: f64);
    fn shutdown(&self) {}
    /// Optional MatFormer debug hook: returns gradient stats for FFN tails (if applicable).
    ///
    /// Implementations that do not support MatFormer can return `None`.
    fn matformer_tail_grad_stats(&self) -> Option<MatformerTailGradSummary> {
        None
    }

    /// Forward pass with knowledge distillation loss.
    ///
    /// Computes either:
    /// - `(1 - β) * CE(student, labels) + β * KL(student/T, teacher/T) * T²` (`mix`)
    /// - `CE(student, labels) + β * KL(student/T, teacher/T) * T²` (`add`)
    ///
    /// Default impl falls back to regular forward (ignores teacher targets).
    fn forward_with_distillation(
        &self,
        x: &Tensor,
        labels: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        loss_scale: Option<f64>,
        _teacher_targets: &TeacherLogitTargets,
        _distillation_beta: f64,
    ) -> (Option<Tensor>, Option<Tensor>) {
        // Default: ignore distillation, just do regular forward
        self.forward(x, labels, position_ids, sequence_lengths, None, loss_scale)
    }
}

/// Reconstructed teacher logit targets for knowledge distillation.
///
/// Created from `CompressedTeacherLogits` on the student side.
/// The teacher distribution is stored as sparse top-k (WAN-friendly).
///
/// Note: we intentionally do *not* reconstruct a dense `[*, vocab]` teacher distribution,
/// since that is prohibitively expensive for real vocab sizes.
#[derive(Debug)]
pub struct TeacherLogitTargets {
    /// Top-k vocabulary indices: [batch, seq, top_k]
    pub top_indices: Tensor,
    /// Top-k logit values: [batch, seq, top_k]
    pub top_values: Tensor,
    /// Per-token log-normalizer: `logZ = logsumexp(logits / T)` over the full vocab.
    ///
    /// Shape: [batch, seq]
    pub logsumexp: Tensor,
    /// Temperature used for softening
    pub temperature: f32,
    /// Number of top-k entries per token
    pub top_k: i64,
    /// Distillation combine mode.
    pub combine_mode: DistillationCombineMode,
    /// Optional teacher-confidence gate threshold on mean q_topk_mass.
    pub min_teacher_topk_mass: Option<f64>,

    /// Optional KD scaling floor for the teacher's mean top-k mass.
    ///
    /// If > 0, scale KD by `1 / clamp(mean_q_topk_mass, floor, 1.0)`.
    pub kd_q_topk_mass_floor: f64,
}

impl Clone for TeacherLogitTargets {
    fn clone(&self) -> Self {
        Self {
            top_indices: self.top_indices.shallow_clone(),
            top_values: self.top_values.shallow_clone(),
            logsumexp: self.logsumexp.shallow_clone(),
            temperature: self.temperature,
            top_k: self.top_k,
            combine_mode: self.combine_mode,
            min_teacher_topk_mass: self.min_teacher_topk_mass,
            kd_q_topk_mass_floor: self.kd_q_topk_mass_floor,
        }
    }
}

/// Compute KD loss using sparse top-k teacher targets.
///
/// This computes KL(teacher || student) over the provided top-k support:
/// `sum_k q_k * (log q_k - log p_k)`, where `q` is teacher softmax over top-k logits
/// and `p_k` is student's probability at the same indices.
///
/// - `student_logits`: raw student logits `[N, vocab_size]` (already shifted/flattened)
/// - `teacher_top_indices`: `[N, top_k]` int64 indices into vocab
/// - `teacher_top_values`: `[N, top_k]` raw teacher logits corresponding to indices
/// - `temperature`: softening temperature
pub fn kd_loss_topk(
    student_logits: &Tensor,
    teacher_top_indices: &Tensor,
    teacher_top_values: &Tensor,
    temperature: f32,
) -> Tensor {
    kd_loss_topk_impl(
        student_logits,
        teacher_top_indices,
        teacher_top_values,
        temperature,
        None,
    )
}

/// Masked variant of `kd_loss_topk` that ignores positions where `valid_mask` is false.
///
/// `valid_mask` must be broadcastable to shape `[N]` where `N` is the flattened token count.
pub fn kd_loss_topk_masked(
    student_logits: &Tensor,
    teacher_top_indices: &Tensor,
    teacher_top_values: &Tensor,
    temperature: f32,
    valid_mask: &Tensor,
) -> Tensor {
    kd_loss_topk_impl(
        student_logits,
        teacher_top_indices,
        teacher_top_values,
        temperature,
        Some(valid_mask),
    )
}

fn kd_loss_topk_impl(
    student_logits: &Tensor,
    teacher_top_indices: &Tensor,
    teacher_top_values: &Tensor,
    temperature: f32,
    valid_mask: Option<&Tensor>,
) -> Tensor {
    let t = temperature as f64;
    let teacher_log_probs =
        (teacher_top_values.to_kind(Kind::Float) / t).log_softmax(-1, Kind::Float);
    let teacher_probs = teacher_log_probs.exp();

    // Student: avoid materializing full `[N, vocab]` log-softmax (VRAM-heavy).
    // Compute logZ = logsumexp(logits/T) and gather only the teacher top-k positions.
    let student_scaled = student_logits.to_kind(Kind::Float) / t;
    let student_logz = student_scaled.logsumexp(-1, false).reshape([-1, 1]); // [N,1]
    let gathered_logits =
        student_scaled.gather(1, &teacher_top_indices.to_kind(Kind::Int64), false);
    let gathered = gathered_logits - &student_logz;

    let per_token_kl =
        (&teacher_probs * (&teacher_log_probs - gathered)).sum_dim_intlist(-1, false, Kind::Float);
    let kl = match valid_mask {
        Some(mask) => {
            let mask_f = mask.to_kind(Kind::Float);
            let denom = mask_f.sum(Kind::Float).clamp_min(1.0);
            (&per_token_kl * mask_f).sum(Kind::Float) / denom
        }
        None => per_token_kl.mean(Kind::Float),
    };
    kl * (t * t)
}

/// Compute KD loss using sparse top-k teacher targets plus a single "tail" bucket.
///
/// Teacher transmits:
/// - top-k logits + indices, and
/// - `logZ = logsumexp(logits / T)` per token over the full vocab.
///
/// Student computes:
/// - `q_i` exactly for the top-k entries under the full normalization (`logZ`),
/// - `q_tail = 1 - sum_i q_i`,
/// - same for `p_i` and `p_tail` from student softmax,
/// and returns KL over `{topk} ∪ {tail}`.
///
/// This avoids the severe distortion of renormalizing teacher top-k to sum to 1.0.
pub fn kd_loss_topk_tail_bucket_masked(
    student_logits: &Tensor,
    teacher_top_indices: &Tensor,
    teacher_top_values: &Tensor,
    teacher_logsumexp: &Tensor,
    temperature: f32,
    valid_mask: &Tensor,
) -> Tensor {
    kd_loss_topk_tail_bucket_impl(
        student_logits,
        teacher_top_indices,
        teacher_top_values,
        teacher_logsumexp,
        temperature,
        Some(valid_mask),
    )
}

pub fn kd_loss_topk_tail_bucket(
    student_logits: &Tensor,
    teacher_top_indices: &Tensor,
    teacher_top_values: &Tensor,
    teacher_logsumexp: &Tensor,
    temperature: f32,
) -> Tensor {
    kd_loss_topk_tail_bucket_impl(
        student_logits,
        teacher_top_indices,
        teacher_top_values,
        teacher_logsumexp,
        temperature,
        None,
    )
}

fn kd_loss_topk_tail_bucket_impl(
    student_logits: &Tensor,
    teacher_top_indices: &Tensor,
    teacher_top_values: &Tensor,
    teacher_logsumexp: &Tensor,
    temperature: f32,
    valid_mask: Option<&Tensor>,
) -> Tensor {
    let t = temperature as f64;
    let eps = 1.0e-8;

    // Teacher: log q_i = (l_i / T) - logZ
    let teacher_logz = teacher_logsumexp.to_kind(Kind::Float).reshape([-1, 1]); // [N, 1]
    let teacher_scaled = teacher_top_values.to_kind(Kind::Float) / t;
    let teacher_log_probs_top = &teacher_scaled - &teacher_logz; // [N, top_k]
    let teacher_probs_top = teacher_log_probs_top.exp();
    let q_top_mass = teacher_probs_top.sum_dim_intlist(-1, false, Kind::Float); // [N]
    let q_tail = (Tensor::ones_like(&q_top_mass) - &q_top_mass).clamp(eps, 1.0);
    let log_q_tail = q_tail.log();

    // Student: avoid materializing full `[N, vocab]` log-softmax (VRAM-heavy).
    // Compute logZ = logsumexp(logits/T) and gather only the teacher top-k positions.
    let student_scaled = student_logits.to_kind(Kind::Float) / t;
    let student_logz = student_scaled.logsumexp(-1, false).reshape([-1, 1]); // [N,1]
    let gathered_logits =
        student_scaled.gather(1, &teacher_top_indices.to_kind(Kind::Int64), false);
    let gathered = gathered_logits - &student_logz; // log p_i at teacher indices
    let student_probs_top = gathered.exp();
    let p_top_mass = student_probs_top.sum_dim_intlist(-1, false, Kind::Float);
    let p_tail = (Tensor::ones_like(&p_top_mass) - &p_top_mass).clamp(eps, 1.0);
    let log_p_tail = p_tail.log();

    // KL over {topk} ∪ {tail}
    let kl_top = (&teacher_probs_top * (&teacher_log_probs_top - gathered)).sum_dim_intlist(
        -1,
        false,
        Kind::Float,
    );
    let kl_tail = &q_tail * (&log_q_tail - &log_p_tail);
    let per_token_kl = kl_top + kl_tail;

    let kl = match valid_mask {
        Some(mask) => {
            let mask_f = mask.to_kind(Kind::Float);
            let denom = mask_f.sum(Kind::Float).clamp_min(1.0);
            (&per_token_kl * mask_f).sum(Kind::Float) / denom
        }
        None => per_token_kl.mean(Kind::Float),
    };
    kl * (t * t)
}

fn kd_scale_from_q_topk_mass_mean(mean_q_topk_mass: f64, q_floor: f64) -> f64 {
    if q_floor <= 0.0 {
        return 1.0;
    }
    let denom = mean_q_topk_mass.clamp(q_floor, 1.0);
    1.0 / denom
}

#[derive(Debug, Clone, Default)]
pub struct MatformerTailGradSummary {
    pub gate_tail_max: f64,
    pub up_tail_max: f64,
    pub down_tail_max: f64,
    pub gate_prefix_max: f64,
    pub up_prefix_max: f64,
    pub down_prefix_max: f64,
}

pub trait LanguageModelForward: Send + Debug {
    fn forward(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        training: bool,
    ) -> Tensor;
}

pub trait LanguageModelConfig: ModelConfig + Send + Debug + serde::de::DeserializeOwned {
    fn tie_word_embeddings(&self) -> bool;
    fn set_max_position_embeddings(&mut self, set: usize);
    fn hidden_size(&self) -> usize;
    fn vocab_size(&self) -> usize;

    fn rope_config(&self) -> Option<RoPEConfig>;
    fn num_attention_heads(&self) -> usize;
    fn rope_theta(&self) -> f32;
    fn max_position_embeddings(&self) -> usize;
    fn bos_token_id(&self) -> Option<i64>;
    fn eos_token_ids(&self) -> Option<EosToks>;

    /// Returns softcap scale if logit softcapping is enabled.
    /// Formula: scale * sigmoid(logits / (scale / 4))
    /// Default: None (no softcapping)
    fn logit_softcap_scale(&self) -> Option<f64> {
        None
    }
}

#[derive(Debug)]
pub struct CausalLanguageModel<M: LanguageModelForward, C: LanguageModelConfig> {
    pub model: M,
    pub config: C,
    pub variables: StableVarStoreIterator,
    pub device: Device,
    pub lm_head: nn::Linear,
    pub comm: Option<Arc<Communicator>>,
    pub training: AtomicBool,
}

// this is absolutely unsafe, if you use it across threads with NCCL you will have a bad day
unsafe impl<M: LanguageModelForward, C: LanguageModelConfig> Send for CausalLanguageModel<M, C> {}

static LAST_DISTILL_INFO_STEP: AtomicU32 = AtomicU32::new(u32::MAX);

pub type LanguageModelBuilder<M, C> = fn(
    vs: nn::Path,
    config: &C,
    attn_implementation: Option<AttentionImplementation>,
    comm: Option<Arc<Communicator>>,
) -> Result<M, ModelLoadError>;

impl<M: LanguageModelForward, C: LanguageModelConfig> CausalLanguageModel<M, C> {
    pub fn from_builder_with_config_overrides<F: FnOnce(&mut C)>(
        builder: LanguageModelBuilder<M, C>,
        source: &PretrainedSource<C>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
        config_overrides: F,
    ) -> Result<Self, ModelLoadError> {
        let mut config = source.get_config()?;

        config_overrides(&mut config);

        if config.tie_word_embeddings() {
            return Err(ModelLoadError::ModelHasTiedEmbeddings);
        }

        if let Some(override_max_position_embeddings) = override_max_position_embeddings {
            config.set_max_position_embeddings(override_max_position_embeddings);
        }

        let device = device.unwrap_or(Device::cuda_if_available());

        #[cfg(feature = "parallelism")]
        let comm = match tensor_parallelism_world {
            #[allow(clippy::arc_with_non_send_sync)]
            // TODO: analyze how we're using Arc here, is this right?
            Some((id, rank, world_size)) => Some(Arc::new(
                CNCCL::new(
                    match id {
                        CommunicatorId::NCCL(cstore) => cstore,
                        _ => return Err(ModelLoadError::CommunicatorMismatch),
                    },
                    rank as i64,
                    world_size as i64,
                    device,
                )
                .map_err(ModelLoadError::TensorParallelismFailedInit)?
                .into(),
            )),
            None => None,
        };

        #[cfg(not(feature = "parallelism"))]
        let comm = match tensor_parallelism_world {
            Some(_) => return Err(ModelLoadError::TensorParallelismNotEnabled),
            None => None,
        };
        let mut variables: nn::VarStore = nn::VarStore::new(device);
        if let Some(kind) = kind {
            variables.set_kind(kind);
        }
        let (model, lm_head) = {
            let _no_grad = tch::no_grad_guard();
            let model = builder(variables.root(), &config, attn_implementation, comm.clone())?;
            let c = nn::LinearConfig {
                bias: false,
                ..Default::default()
            };
            let lm_head = nn::linear(
                &variables.root() / "lm_head",
                config.hidden_size() as i64,
                config.vocab_size() as i64,
                c,
            );

            source.load(&mut variables)?;

            (model, lm_head)
        };
        let variables = StableVarStoreIterator::new(&variables, comm.clone());
        Ok(Self {
            model,
            config,
            variables,
            device,
            lm_head,
            comm,
            training: AtomicBool::new(false),
        })
    }

    pub fn from_builder(
        builder: LanguageModelBuilder<M, C>,
        source: &PretrainedSource<C>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
    ) -> Result<Self, ModelLoadError> {
        Self::from_builder_with_config_overrides(
            builder,
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
            |_| {},
        )
    }
}

impl<M: LanguageModelForward, C: LanguageModelConfig> CausalLM for CausalLanguageModel<M, C> {
    fn forward(
        &self,
        x: &Tensor,
        labels: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        num_logits_to_keep: Option<i64>,
        loss_scale: Option<f64>,
    ) -> (Option<Tensor>, Option<Tensor>) {
        let (_, t) = x.size2().unwrap();
        let mut x = self.model.forward(
            x,
            position_ids,
            sequence_lengths,
            self.training.load(Ordering::Relaxed),
        );
        if let Some(num_logits_to_keep) = num_logits_to_keep {
            // Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            x = x.slice(1, t - num_logits_to_keep, t, 1);
        }
        let mut logits = self.lm_head.forward(&x);

        // Apply logit softcap if enabled (NanoGPT feature)
        // Formula: scale * sigmoid(logits / (scale / 4))
        if let Some(scale) = self.config.logit_softcap_scale() {
            logits = (logits / (scale / 4.0)).sigmoid() * scale;
        }

        let loss = match labels {
            Some(labels) => {
                // Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.to_kind(Kind::Float);
                // Shift so that tokens < n predict n
                let shift_logits = logits.slice(1, 0, -1, 1).contiguous();
                let shift_labels = labels.slice(1, 1, None, 1).contiguous();
                let shift_logits = shift_logits.view([-1i64, self.config.vocab_size() as i64]);
                let shift_targets = shift_labels.view(-1).to_kind(Kind::Int64);
                let mut loss = shift_logits.cross_entropy_loss::<Tensor>(
                    &shift_targets,
                    None,
                    tch::Reduction::Mean,
                    -100,
                    0.0,
                );
                if let Some(loss_scale) = loss_scale {
                    loss /= loss_scale;
                }
                Some(loss)
            }
            None => None,
        };
        (Some(logits), loss)
    }

    fn forward_with_distillation(
        &self,
        x: &Tensor,
        labels: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        loss_scale: Option<f64>,
        teacher_targets: &TeacherLogitTargets,
        distillation_beta: f64,
    ) -> (Option<Tensor>, Option<Tensor>) {
        let x = self.model.forward(
            x,
            position_ids,
            sequence_lengths,
            self.training.load(Ordering::Relaxed),
        );
        let mut logits = self.lm_head.forward(&x);

        if let Some(scale) = self.config.logit_softcap_scale() {
            logits = (logits / (scale / 4.0)).sigmoid() * scale;
        }

        let loss = match labels {
            Some(labels) => {
                logits = logits.to_kind(Kind::Float);
                let shift_logits = logits.slice(1, 0, -1, 1).contiguous();
                let shift_labels = labels.slice(1, 1, None, 1).contiguous();
                let vocab_size = self.config.vocab_size() as i64;
                let shift_logits_flat = shift_logits.view([-1i64, vocab_size]);
                let shift_targets = shift_labels.view(-1).to_kind(Kind::Int64);
                let _valid_mask_full = shift_targets.ne(-100);

                // CE loss
                let ce_loss = shift_logits_flat.cross_entropy_loss::<Tensor>(
                    &shift_targets,
                    None,
                    tch::Reduction::Mean,
                    -100,
                    0.0,
                );

                // KD loss from sparse top-k teacher targets (WAN-friendly)
                // Shift teacher targets the same way as CE: positions [0..seq-1] predict [1..seq]
                let teacher_seq_size = teacher_targets.top_indices.size();
                let teacher_seq_len = *teacher_seq_size.get(1).unwrap_or(&0);
                let teacher_values_size = teacher_targets.top_values.size();
                let teacher_logsumexp_size = teacher_targets.logsumexp.size();
                let teacher_top_k = teacher_targets.top_k;
                let step = crate::matformer_c2::current_step();
                let teacher_indices = teacher_targets
                    .top_indices
                    .to(self.device)
                    .to_kind(Kind::Int64);
                let teacher_values = teacher_targets
                    .top_values
                    .to(self.device)
                    .to_kind(Kind::Float);
                let teacher_logsumexp = teacher_targets
                    .logsumexp
                    .to(self.device)
                    .to_kind(Kind::Float);

                let (_, student_seq_len, _) = logits.size3().expect("logits must be 3D");
                let teacher_payload_ok = teacher_top_k > 0
                    && teacher_seq_size.len() == 3
                    && teacher_values_size.len() == 3
                    && teacher_logsumexp_size.len() == 2
                    && teacher_seq_size[0] == teacher_values_size[0]
                    && teacher_seq_size[1] == teacher_logsumexp_size[1]
                    && teacher_seq_size[0] == teacher_logsumexp_size[0]
                    && teacher_seq_size[2] == teacher_values_size[2]
                    && teacher_seq_size[2] == teacher_top_k
                    && teacher_indices
                        .lt(0)
                        .logical_or(&teacher_indices.ge(vocab_size))
                        .any()
                        .int64_value(&[])
                        == 0
                    && teacher_values.isfinite().all().int64_value(&[]) != 0
                    && teacher_logsumexp.isfinite().all().int64_value(&[]) != 0
                    && teacher_targets.temperature > 0.0;
                if !teacher_payload_ok {
                    error!(
                        step = step,
                        teacher_top_k = teacher_top_k,
                        temperature = teacher_targets.temperature,
                        teacher_indices_shape = ?teacher_seq_size,
                        teacher_values_shape = ?teacher_values_size,
                        teacher_logsumexp_shape = ?teacher_logsumexp_size,
                        "Invalid teacher payload for distillation"
                    );
                    return (Some(logits), None);
                }
                if teacher_seq_len < 2 {
                    error!(
                        teacher_seq_len = teacher_seq_len,
                        "Invalid teacher targets: seq_len must be >= 2 for distillation"
                    );
                    return (Some(logits), None);
                }
                if teacher_seq_len > student_seq_len {
                    error!(
                        teacher_seq_len = teacher_seq_len,
                        student_seq_len = student_seq_len,
                        "Invalid teacher targets: teacher seq_len exceeds student seq_len"
                    );
                    return (Some(logits), None);
                }

                // Teacher targets correspond to the *last* `teacher_seq_len` positions (when using
                // `num_logits_to_keep` on the teacher). Align student KD to the same suffix.
                let kd_start = student_seq_len - teacher_seq_len; // start in unshifted token positions
                let kd_len = teacher_seq_len - 1; // after shifting, we have one fewer prediction

                let shift_logits_kd = shift_logits
                    .narrow(1, kd_start, kd_len)
                    .contiguous()
                    .view([-1i64, vocab_size]);
                let kd_valid_mask = shift_labels
                    .narrow(1, kd_start, kd_len)
                    .contiguous()
                    .view(-1)
                    .to_kind(Kind::Int64)
                    .ne(-100);

                let teacher_indices_shift = teacher_indices
                    .slice(1, 0, -1, 1)
                    .contiguous()
                    .view([-1i64, teacher_targets.top_k]);
                let teacher_values_shift = teacher_values
                    .slice(1, 0, -1, 1)
                    .contiguous()
                    .view([-1i64, teacher_targets.top_k]);
                let teacher_logsumexp_shift = teacher_logsumexp
                    .slice(1, 0, -1, 1)
                    .contiguous()
                    .view([-1i64]);
                let kd_raw = kd_loss_topk_tail_bucket_masked(
                    &shift_logits_kd,
                    &teacher_indices_shift,
                    &teacher_values_shift,
                    &teacher_logsumexp_shift,
                    teacher_targets.temperature,
                    &kd_valid_mask,
                );

                let t = teacher_targets.temperature as f64;
                let eps = 1.0e-8;

                let teacher_logz = teacher_logsumexp_shift.to_kind(Kind::Float).view([-1, 1]); // [N, 1]
                let teacher_scaled = &teacher_values_shift / t;
                let teacher_probs_top = (&teacher_scaled - &teacher_logz).exp();
                let q_top_mass = teacher_probs_top
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .clamp(0.0, 1.0);
                let q_tail = (Tensor::ones_like(&q_top_mass) - &q_top_mass).clamp(eps, 1.0);

                let student_scaled = &shift_logits_kd / t;
                let student_logz = student_scaled.logsumexp(-1, false).reshape([-1, 1]);
                let student_gathered =
                    student_scaled.gather(1, &teacher_indices_shift.to_kind(Kind::Int64), false)
                        - &student_logz;
                let p_top_mass = student_gathered
                    .exp()
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .clamp(0.0, 1.0);
                let p_tail = (Tensor::ones_like(&p_top_mass) - &p_top_mass).clamp(eps, 1.0);

                let mask_f = kd_valid_mask.to_kind(Kind::Float);
                let denom = mask_f.sum(Kind::Float).clamp_min(1.0);
                let q_topk_mass_mean = (&q_top_mass * &mask_f).sum(Kind::Float) / &denom;
                let q_tail_mean = (&q_tail * &mask_f).sum(Kind::Float) / &denom;
                let p_topk_mass_mean = (&p_top_mass * &mask_f).sum(Kind::Float) / &denom;
                let p_tail_mean = (&p_tail * &mask_f).sum(Kind::Float) / &denom;

                let q_topk_mass_mean_f = q_topk_mass_mean.double_value(&[]);
                let beta_requested = distillation_beta;
                let mut beta = beta_requested;
                if let Some(min_teacher_topk_mass) = teacher_targets.min_teacher_topk_mass {
                    if q_topk_mass_mean_f < min_teacher_topk_mass {
                        beta = 0.0;
                    }
                }

                let kd_scale = kd_scale_from_q_topk_mass_mean(
                    q_topk_mass_mean_f,
                    teacher_targets.kd_q_topk_mass_floor,
                );
                let kd = &kd_raw * kd_scale;

                // Combined objective:
                // - Mix: (1 - β) * CE + β * KD
                // - Add: CE + β * KD
                let mut combined = match teacher_targets.combine_mode {
                    DistillationCombineMode::Mix => &ce_loss * (1.0 - beta) + &kd * beta,
                    DistillationCombineMode::Add => &ce_loss + &kd * beta,
                };
                trace!(
                    ce = ce_loss.double_value(&[]),
                    kd_raw = kd_raw.double_value(&[]),
                    kd = kd.double_value(&[]),
                    kd_scale = kd_scale,
                    beta_requested = beta_requested,
                    beta = beta,
                    combine_mode = ?teacher_targets.combine_mode,
                    teacher_seq_len = teacher_seq_len,
                    student_seq_len = student_seq_len,
                    "distillation loss"
                );

                // INFO log once per step: CE/KD/beta and teacher/student top-k mass diagnostics.
                if beta_requested > 0.0 {
                    let prev = LAST_DISTILL_INFO_STEP.load(Ordering::Relaxed);
                    if prev != step
                        && LAST_DISTILL_INFO_STEP
                            .compare_exchange(prev, step, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                    {
                        info!(
                            step = step,
                            ce = ce_loss.double_value(&[]),
                            kd_raw = kd_raw.double_value(&[]),
                            kd = kd.double_value(&[]),
                            kd_scale = kd_scale,
                            kd_q_topk_mass_floor = teacher_targets.kd_q_topk_mass_floor,
                            beta_requested = beta_requested,
                            beta = beta,
                            combine_mode = ?teacher_targets.combine_mode,
                            min_teacher_topk_mass = teacher_targets.min_teacher_topk_mass,
                            temperature = teacher_targets.temperature,
                            q_topk_mass = q_topk_mass_mean_f,
                            q_tail = q_tail_mean.double_value(&[]),
                            p_topk_mass = p_topk_mass_mean.double_value(&[]),
                            p_tail = p_tail_mean.double_value(&[]),
                            teacher_seq_len = teacher_seq_len,
                            "distillation stats"
                        );
                    }
                }

                if let Some(loss_scale) = loss_scale {
                    combined /= loss_scale;
                }
                Some(combined)
            }
            None => None,
        };
        (Some(logits), loss)
    }

    fn bos_token_id(&self) -> Option<i64> {
        self.config.bos_token_id()
    }

    fn eos_token_ids(&self) -> Option<EosToks> {
        self.config.eos_token_ids()
    }

    fn device(&self) -> Device {
        self.device
    }

    fn max_context_length(&self) -> usize {
        self.config.max_position_embeddings()
    }

    fn variables(&self) -> StableVariableIterator {
        Box::new(self.variables.clone())
    }

    fn communicator(&self) -> Option<Arc<Communicator>> {
        self.comm.clone()
    }

    fn prepare_for_training(&self) {
        self.training.store(true, Ordering::Relaxed);
    }

    /// Clips gradient norm, properly handling tensor-parallel parameters.
    ///
    /// For a model with both sharded and replicated parameters, the true L2 norm is:
    /// sqrt(||w_shared||^2 + ||w_replicated||^2) where:
    /// - w_shared are parameters sharded across ranks (like TP linear layers)
    /// - w_replicated are parameters replicated on all ranks (like layernorms)
    ///
    /// For sharded parameters, since each rank has an orthogonal slice of the full parameter:
    /// ||w_shared||^2 = ||w_shared_1||^2 + ||w_shared_2||^2 + ... + ||w_shared_n||^2
    /// where w_shared_i is the shard on rank i. We compute this via all_reduce_sum of local squared norms.
    ///
    /// For replicated parameters:
    /// ||w_replicated||^2 is identical on all ranks, so we compute it locally.
    ///
    /// The orthogonality of sharded parameters across ranks ensures that:
    /// total_norm = sqrt(all_reduce(||w_shared_local||^2) + ||w_replicated||^2)
    /// gives us the correct global L2 norm as if all parameters were on a single device.
    fn clip_grad_norm(&self, max_norm: f64) {
        let mut sharded_norm_sq = Tensor::zeros([], (Kind::Float, self.device));
        let mut replicated_norm_sq = Tensor::zeros([], (Kind::Float, self.device));

        for var in self.variables() {
            let grad = var.logical_tensor().grad();
            if grad.defined() {
                let local_norm = grad.norm();
                let local_norm_sq = &local_norm * &local_norm;

                if var.is_sharded() {
                    sharded_norm_sq += local_norm_sq
                } else {
                    replicated_norm_sq += local_norm_sq
                }
            }
        }

        sharded_norm_sq.all_reduce(&self.comm, ReduceType::Sum);

        let total_norm: f64 = (sharded_norm_sq + replicated_norm_sq)
            .sqrt()
            .try_into()
            .unwrap();

        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for var in self.variables() {
                let mut grad = var.logical_tensor().grad();
                if grad.defined() {
                    let _t = grad.g_mul_scalar_(scale);
                }
            }
        }
    }
}

impl EosToks {
    pub fn contains(&self, token: i64) -> bool {
        match self {
            EosToks::Single(x) => *x == token,
            EosToks::Multiple(items) => items.contains(&token),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn;

    #[test]
    fn test_kd_loss_topk_tail_bucket_identical_distributions_full_support() {
        // When student and teacher have the same distribution and teacher support covers full vocab,
        // KL should be ~0.
        let batch = 4;
        let vocab = 50;
        let student_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_logits = student_logits.shallow_clone();
        let teacher_top_indices = Tensor::arange(vocab, (Kind::Int64, Device::Cpu))
            .unsqueeze(0)
            .repeat(&[batch, 1]);
        let teacher_top_values = teacher_logits.shallow_clone();
        let teacher_logsumexp = (&teacher_logits / 1.0).logsumexp(-1, false);
        let loss = kd_loss_topk_tail_bucket(
            &student_logits,
            &teacher_top_indices,
            &teacher_top_values,
            &teacher_logsumexp,
            1.0,
        );
        let loss_val = loss.double_value(&[]);
        assert!(
            loss_val.abs() < 1e-4,
            "KD loss (tail-bucket) with identical distributions should be ~0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_kd_loss_topk_tail_bucket_different_distributions_full_support() {
        // When student and teacher differ, KL should be > 0.
        let batch = 4;
        let vocab = 50;
        let student_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_top_indices = Tensor::arange(vocab, (Kind::Int64, Device::Cpu))
            .unsqueeze(0)
            .repeat(&[batch, 1]);
        let teacher_top_values = teacher_logits.shallow_clone();
        let teacher_logsumexp = (&teacher_logits / 1.0).logsumexp(-1, false);
        let loss = kd_loss_topk_tail_bucket(
            &student_logits,
            &teacher_top_indices,
            &teacher_top_values,
            &teacher_logsumexp,
            1.0,
        );
        let loss_val = loss.double_value(&[]);
        assert!(
            loss_val > 0.0,
            "KD loss (tail-bucket) with different distributions should be > 0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_kd_loss_topk_identical_distributions_full_support() {
        // When student and teacher have the same distribution and teacher support covers full vocab,
        // KL should be ~0.
        let batch = 4;
        let vocab = 50;
        let student_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_logits = student_logits.shallow_clone();
        let teacher_top_indices = Tensor::arange(vocab, (Kind::Int64, Device::Cpu))
            .unsqueeze(0)
            .repeat(&[batch, 1]);
        let teacher_top_values = teacher_logits.shallow_clone();
        let loss = kd_loss_topk(
            &student_logits,
            &teacher_top_indices,
            &teacher_top_values,
            1.0,
        );
        let loss_val = loss.double_value(&[]);
        assert!(
            loss_val.abs() < 1e-4,
            "KD loss with identical distributions should be ~0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_kd_loss_topk_different_distributions_full_support() {
        // When student and teacher differ, KL should be > 0.
        let batch = 4;
        let vocab = 50;
        let student_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_top_indices = Tensor::arange(vocab, (Kind::Int64, Device::Cpu))
            .unsqueeze(0)
            .repeat(&[batch, 1]);
        let teacher_top_values = teacher_logits.shallow_clone();
        let loss = kd_loss_topk(
            &student_logits,
            &teacher_top_indices,
            &teacher_top_values,
            1.0,
        );
        let loss_val = loss.double_value(&[]);
        assert!(
            loss_val > 0.0,
            "KD loss with different distributions should be > 0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_kd_loss_topk_temperature_is_finite() {
        let batch = 4;
        let vocab = 50;
        let student_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_logits = Tensor::randn([batch, vocab], (Kind::Float, Device::Cpu));
        let teacher_top_indices = Tensor::arange(vocab, (Kind::Int64, Device::Cpu))
            .unsqueeze(0)
            .repeat(&[batch, 1]);
        let teacher_top_values = teacher_logits.shallow_clone();

        let loss_t1 = kd_loss_topk(
            &student_logits,
            &teacher_top_indices,
            &teacher_top_values,
            1.0,
        )
        .double_value(&[]);
        let loss_t2 = kd_loss_topk(
            &student_logits,
            &teacher_top_indices,
            &teacher_top_values,
            2.0,
        )
        .double_value(&[]);

        assert!(loss_t1.is_finite());
        assert!(loss_t2.is_finite());
    }

    #[test]
    fn test_teacher_logit_targets_shift_and_flatten_shapes() {
        let batch = 2;
        let seq = 4;
        let top_k = 3;
        let vocab_size = 10; // only used for student logits in this test

        // Create known top-k targets
        let top_indices = Tensor::from_slice(&[
            0i64, 1, 2, // token 0
            3, 4, 5, // token 1
            6, 7, 8, // token 2
            0, 3, 6, // token 3
            1, 4, 7, // token 0 (batch 2)
            2, 5, 8, // token 1
            0, 1, 2, // token 2
            3, 4, 5, // token 3
        ])
        .reshape([batch, seq, top_k]);

        let top_values = Tensor::from_slice(&[
            5.0f32, 3.0, 1.0, // token 0
            4.0, 2.0, 0.5, // token 1
            6.0, 4.0, 2.0, // token 2
            3.0, 2.0, 1.0, // token 3
            5.0, 3.0, 1.0, // token 0 (batch 2)
            4.0, 2.0, 0.5, // token 1
            6.0, 4.0, 2.0, // token 2
            3.0, 2.0, 1.0, // token 3
        ])
        .reshape([batch, seq, top_k]);

        let logsumexp = Tensor::full([batch, seq], 10.0, (Kind::Float, Device::Cpu));
        let targets = TeacherLogitTargets {
            top_indices,
            top_values,
            logsumexp,
            temperature: 2.0,
            top_k,
            combine_mode: DistillationCombineMode::Mix,
            min_teacher_topk_mass: None,
            kd_q_topk_mass_floor: 0.05,
        };

        // Shift by 1 to match causal LM convention and flatten.
        let idx_shift = targets
            .top_indices
            .slice(1, 0, -1, 1)
            .contiguous()
            .view([-1, top_k]);
        let val_shift = targets
            .top_values
            .slice(1, 0, -1, 1)
            .contiguous()
            .view([-1, top_k]);

        assert_eq!(idx_shift.size(), vec![batch * (seq - 1), top_k]);
        assert_eq!(val_shift.size(), vec![batch * (seq - 1), top_k]);

        let student_logits =
            Tensor::randn([batch * (seq - 1), vocab_size], (Kind::Float, Device::Cpu));
        let loss = kd_loss_topk(&student_logits, &idx_shift, &val_shift, targets.temperature);
        assert!(loss.double_value(&[]).is_finite());
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct ToyConfig {
        hidden_size: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
    }

    impl ModelConfig for ToyConfig {
        fn get_parameter_names(&self) -> Vec<String> {
            vec![]
        }
    }

    impl LanguageModelConfig for ToyConfig {
        fn tie_word_embeddings(&self) -> bool {
            false
        }

        fn set_max_position_embeddings(&mut self, set: usize) {
            self.max_position_embeddings = set;
        }

        fn hidden_size(&self) -> usize {
            self.hidden_size
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn rope_config(&self) -> Option<RoPEConfig> {
            None
        }

        fn num_attention_heads(&self) -> usize {
            1
        }

        fn rope_theta(&self) -> f32 {
            10000.0
        }

        fn max_position_embeddings(&self) -> usize {
            self.max_position_embeddings
        }

        fn bos_token_id(&self) -> Option<i64> {
            None
        }

        fn eos_token_ids(&self) -> Option<EosToks> {
            None
        }
    }

    #[derive(Debug)]
    struct ToyModel {
        wte: nn::Embedding,
    }

    impl LanguageModelForward for ToyModel {
        fn forward(
            &self,
            x: &Tensor,
            _position_ids: Option<&Tensor>,
            _sequence_lengths: Option<&Vec<Vec<i32>>>,
            _training: bool,
        ) -> Tensor {
            self.wte.forward(x)
        }
    }

    #[test]
    fn test_forward_with_distillation_truncated_teacher_seq_len() {
        let batch = 2;
        let student_seq = 16;
        let teacher_seq = 8; // last-N distillation window
        let hidden = 8;
        let vocab = 32;
        let top_k = 4;

        let mut cfg = ToyConfig {
            hidden_size: hidden,
            vocab_size: vocab,
            max_position_embeddings: student_seq,
        };
        // ensure config trait method is exercised
        cfg.set_max_position_embeddings(student_seq);

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let wte = nn::embedding(
            &root / "wte",
            vocab as i64,
            hidden as i64,
            Default::default(),
        );
        let model = ToyModel { wte };
        let lm_head = nn::linear(
            &root / "lm_head",
            hidden as i64,
            vocab as i64,
            Default::default(),
        );

        let clm = CausalLanguageModel {
            model,
            config: cfg,
            variables: StableVarStoreIterator::new(&vs, None),
            device: Device::Cpu,
            lm_head,
            comm: None,
            training: AtomicBool::new(true),
        };

        let input_ids = Tensor::randint(
            vocab as i64,
            [batch as i64, student_seq as i64],
            (Kind::Int64, Device::Cpu),
        );
        let labels = input_ids.shallow_clone();

        let teacher_top_indices = Tensor::randint(
            vocab as i64,
            [batch as i64, teacher_seq as i64, top_k as i64],
            (Kind::Int64, Device::Cpu),
        );
        let teacher_top_values = Tensor::randn(
            [batch as i64, teacher_seq as i64, top_k as i64],
            (Kind::Float, Device::Cpu),
        );
        let teacher_logsumexp = Tensor::full(
            [batch as i64, teacher_seq as i64],
            10.0,
            (Kind::Float, Device::Cpu),
        );
        let teacher_targets = TeacherLogitTargets {
            top_indices: teacher_top_indices,
            top_values: teacher_top_values,
            logsumexp: teacher_logsumexp,
            temperature: 2.0,
            top_k: top_k as i64,
            combine_mode: DistillationCombineMode::Mix,
            min_teacher_topk_mass: None,
            kd_q_topk_mass_floor: 0.05,
        };

        let (_logits, loss) = clm.forward_with_distillation(
            &input_ids,
            Some(&labels),
            None,
            None,
            None,
            &teacher_targets,
            0.5,
        );
        let loss = loss.expect("loss should be present for valid truncated teacher targets");
        assert!(loss.double_value(&[]).is_finite());
    }

    #[test]
    fn test_forward_with_distillation_invalid_teacher_payload_returns_none() {
        let batch = 2;
        let student_seq = 12;
        let teacher_seq = 6;
        let hidden = 8;
        let vocab = 24;
        let top_k = 4;

        let mut cfg = ToyConfig {
            hidden_size: hidden,
            vocab_size: vocab,
            max_position_embeddings: student_seq,
        };
        cfg.set_max_position_embeddings(student_seq);

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let wte = nn::embedding(
            &root / "wte",
            vocab as i64,
            hidden as i64,
            Default::default(),
        );
        let model = ToyModel { wte };
        let lm_head = nn::linear(
            &root / "lm_head",
            hidden as i64,
            vocab as i64,
            Default::default(),
        );
        let clm = CausalLanguageModel {
            model,
            config: cfg,
            variables: StableVarStoreIterator::new(&vs, None),
            device: Device::Cpu,
            lm_head,
            comm: None,
            training: AtomicBool::new(true),
        };

        let input_ids = Tensor::randint(
            vocab as i64,
            [batch as i64, student_seq as i64],
            (Kind::Int64, Device::Cpu),
        );
        let labels = input_ids.shallow_clone();

        let invalid_teacher_top_indices = Tensor::full(
            [batch as i64, teacher_seq as i64, top_k as i64],
            vocab as i64,
            (Kind::Int64, Device::Cpu),
        );
        let invalid_teacher_top_values = Tensor::randn(
            [batch as i64, teacher_seq as i64, top_k as i64],
            (Kind::Float, Device::Cpu),
        );
        let teacher_logsumexp = Tensor::full(
            [batch as i64, teacher_seq as i64],
            0.0,
            (Kind::Float, Device::Cpu),
        );
        let invalid_teacher_targets = TeacherLogitTargets {
            top_indices: invalid_teacher_top_indices,
            top_values: invalid_teacher_top_values,
            logsumexp: teacher_logsumexp,
            temperature: 2.0,
            top_k: top_k as i64,
            combine_mode: DistillationCombineMode::Add,
            min_teacher_topk_mass: None,
            kd_q_topk_mass_floor: 0.05,
        };

        let (_, invalid_loss) = clm.forward_with_distillation(
            &input_ids,
            Some(&labels),
            None,
            None,
            None,
            &invalid_teacher_targets,
            1.0,
        );
        assert!(
            invalid_loss.is_none(),
            "loss should be None when teacher payload is invalid"
        );

        let nan_teacher_top_values = Tensor::full(
            [batch as i64, teacher_seq as i64, top_k as i64],
            f64::NAN,
            (Kind::Float, Device::Cpu),
        );
        let non_finite_teacher_targets = TeacherLogitTargets {
            top_indices: Tensor::randint(
                vocab as i64,
                [batch as i64, teacher_seq as i64, top_k as i64],
                (Kind::Int64, Device::Cpu),
            ),
            top_values: nan_teacher_top_values,
            logsumexp: Tensor::full(
                [batch as i64, teacher_seq as i64],
                f64::NAN,
                (Kind::Float, Device::Cpu),
            ),
            temperature: 2.0,
            top_k: top_k as i64,
            combine_mode: DistillationCombineMode::Add,
            min_teacher_topk_mass: None,
            kd_q_topk_mass_floor: 0.05,
        };
        let (_, nan_loss) = clm.forward_with_distillation(
            &input_ids,
            Some(&labels),
            None,
            None,
            None,
            &non_finite_teacher_targets,
            1.0,
        );
        assert!(
            nan_loss.is_none(),
            "loss should be None when teacher payload is non-finite"
        );
    }

    #[test]
    fn test_kd_scale_from_q_topk_mass_mean() {
        assert_eq!(kd_scale_from_q_topk_mass_mean(0.5, 0.0), 1.0);
        assert!((kd_scale_from_q_topk_mass_mean(0.5, 0.05) - 2.0).abs() < 1e-12);
        assert!((kd_scale_from_q_topk_mass_mean(0.01, 0.05) - 20.0).abs() < 1e-12);
        assert!((kd_scale_from_q_topk_mass_mean(1.2, 0.05) - 1.0).abs() < 1e-12);
    }
}
