use crate::{
    AllReduce, AttentionImplementation, Communicator, CommunicatorId, ModelConfig, ModelLoadError,
    PretrainedSource, ReduceType, RoPEConfig, StableVarStoreIterator, StableVariableIterator,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::{fmt::Debug, sync::atomic::AtomicBool};
use tch::{
    nn::{self, Module},
    Device, Kind, Tensor,
};
use tracing::trace;

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
    /// Computes: `(1 - β) * CE(student, labels) + β * KL(student/T, teacher/T) * T²`
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
/// The teacher distribution is stored as sparse top-k + uniform remainder.
#[derive(Debug)]
pub struct TeacherLogitTargets {
    /// Top-k vocabulary indices: [batch, seq, top_k]
    pub top_indices: Tensor,
    /// Top-k logit values: [batch, seq, top_k]
    pub top_values: Tensor,
    /// Temperature used for softening
    pub temperature: f32,
    /// Number of top-k entries per token
    pub top_k: i64,
    /// Vocab size for reconstructing full distribution
    pub vocab_size: i64,
}

impl TeacherLogitTargets {
    /// Reconstruct a full teacher log-probability distribution from sparse top-k.
    ///
    /// For tokens where we have top-k logits, the remaining probability mass
    /// is distributed uniformly across the (vocab_size - top_k) remaining tokens.
    ///
    /// Returns: [batch * (seq-1), vocab_size] log-probabilities (shifted to match CE convention).
    pub fn to_shifted_log_probs(&self, device: Device) -> Tensor {
        let top_indices = self.top_indices.to(device);
        let top_values = self.top_values.to(device).to_kind(Kind::Float);

        // Apply temperature scaling
        let scaled = &top_values / (self.temperature as f64);

        // Compute softmax over top-k to get the "known" probability mass
        let top_probs = scaled.softmax(-1, Kind::Float);

        // Allocate full distribution with a small floor (uniform over remaining vocab)
        let (batch, seq, _k) = top_indices.size3().unwrap();
        let floor_val = 1e-8_f64 / (self.vocab_size as f64);
        let mut full_log_probs =
            Tensor::full([batch, seq, self.vocab_size], floor_val, (Kind::Float, device)).log();

        // Scatter top-k probabilities into the full distribution
        let top_probs_adjusted = &top_probs * (1.0 - 1e-8); // leave floor for remaining
        let log_top_probs = top_probs_adjusted.log();
        let _ = full_log_probs.scatter_(-1, &top_indices.to_kind(Kind::Int64), &log_top_probs);

        // Shift by 1 to match causal LM convention: token[i] predicts token[i+1]
        // Same shift as CE loss: take positions [0..seq-1]
        full_log_probs.slice(1, 0, -1, 1).contiguous().view([-1, self.vocab_size])
    }
}

/// Compute KD loss: `β * KL(student/T || teacher/T) * T²`.
///
/// - `student_logits`: raw student logits [batch*(seq-1), vocab_size]
/// - `teacher_log_probs`: teacher log-probs [batch*(seq-1), vocab_size]
/// - `temperature`: softening temperature
///
/// Returns a scalar loss tensor.
pub fn kd_loss(student_logits: &Tensor, teacher_log_probs: &Tensor, temperature: f32) -> Tensor {
    let t = temperature as f64;
    let student_log_probs = (student_logits / t).log_softmax(-1, Kind::Float);
    // KL(student || teacher) = sum(teacher * (log(teacher) - log(student)))
    // = sum(exp(teacher_log_probs) * (teacher_log_probs - student_log_probs))
    let teacher_probs = teacher_log_probs.exp();
    let kl = (&teacher_probs * (teacher_log_probs - &student_log_probs))
        .sum_dim_intlist(-1, false, Kind::Float)
        .mean(Kind::Float);
    kl * (t * t)
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

                // CE loss
                let ce_loss = shift_logits_flat.cross_entropy_loss::<Tensor>(
                    &shift_targets,
                    None,
                    tch::Reduction::Mean,
                    -100,
                    0.0,
                );

                // KD loss from teacher targets
                let teacher_log_probs = teacher_targets.to_shifted_log_probs(self.device);
                let kd = kd_loss(&shift_logits_flat, &teacher_log_probs, teacher_targets.temperature);

                // Combined: (1 - β) * CE + β * KD
                let beta = distillation_beta;
                let mut combined = &ce_loss * (1.0 - beta) + &kd * beta;
                trace!(
                    ce = ce_loss.double_value(&[]),
                    kd = kd.double_value(&[]),
                    beta = beta,
                    "distillation loss"
                );

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

    #[test]
    fn test_kd_loss_identical_distributions() {
        // When student and teacher have the same distribution, KL should be ~0
        let logits = Tensor::randn([4, 100], (Kind::Float, Device::Cpu));
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let loss = kd_loss(&logits, &log_probs, 1.0);
        let loss_val = loss.double_value(&[]);
        assert!(
            loss_val.abs() < 1e-4,
            "KD loss with identical distributions should be ~0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_kd_loss_different_distributions() {
        // When student and teacher differ, KL should be > 0
        let student_logits = Tensor::randn([4, 100], (Kind::Float, Device::Cpu));
        let teacher_logits = Tensor::randn([4, 100], (Kind::Float, Device::Cpu));
        let teacher_log_probs = teacher_logits.log_softmax(-1, Kind::Float);
        let loss = kd_loss(&student_logits, &teacher_log_probs, 1.0);
        let loss_val = loss.double_value(&[]);
        assert!(
            loss_val > 0.0,
            "KD loss with different distributions should be > 0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_kd_loss_temperature_scaling() {
        // Higher temperature should soften the distribution, reducing KL
        let student_logits = Tensor::randn([4, 100], (Kind::Float, Device::Cpu));
        let teacher_logits = Tensor::randn([4, 100], (Kind::Float, Device::Cpu));

        let teacher_log_probs_t1 = teacher_logits.log_softmax(-1, Kind::Float);
        let loss_t1 = kd_loss(&student_logits, &teacher_log_probs_t1, 1.0).double_value(&[]);

        // For T=2, we need to compute teacher log probs at that temperature
        let teacher_log_probs_t2 = (&teacher_logits / 2.0).log_softmax(-1, Kind::Float);
        let loss_t2 = kd_loss(&student_logits, &teacher_log_probs_t2, 2.0).double_value(&[]);

        // Both should be positive
        assert!(loss_t1 > 0.0);
        assert!(loss_t2 > 0.0);
    }

    #[test]
    fn test_teacher_logit_targets_reconstruction() {
        let batch = 2;
        let seq = 4;
        let top_k = 3;
        let vocab_size = 10;

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

        let targets = TeacherLogitTargets {
            top_indices,
            top_values,
            temperature: 2.0,
            top_k,
            vocab_size,
        };

        let log_probs = targets.to_shifted_log_probs(Device::Cpu);
        // Should be [batch * (seq-1), vocab_size]
        assert_eq!(log_probs.size(), vec![batch * (seq - 1), vocab_size]);

        // All log probs should be <= 0
        let max_lp = log_probs.max().double_value(&[]);
        assert!(
            max_lp <= 0.0 + 1e-6,
            "log probs should be <= 0, got max {}",
            max_lp
        );

        // Exponentiated probs should sum to ~1
        let probs = log_probs.exp();
        let sums = probs.sum_dim_intlist(-1, false, Kind::Float);
        for i in 0..sums.size()[0] {
            let s = sums.double_value(&[i]);
            assert!(
                (s - 1.0).abs() < 0.01,
                "probs should sum to ~1, got {} at position {}",
                s,
                i
            );
        }
    }
}
