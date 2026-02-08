use crate::{
    default_rope,
    matformer_c2::{MatformerStabilizationConfig, SuffixGateConfig},
    matformer_helper::HelperConfig,
    parallelism::Communicator,
    AttentionImplementation, AutoConfig, CausalLanguageModel, CausalSelfAttention,
    ColumnParallelLinear, CommunicatorId, EosToks, LanguageModelConfig, LanguageModelForward,
    ModelConfig, ModelLoadError, PretrainedSource, RMSNorm, RoPECache, RoPEConfig,
    RowParallelLinear,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tch::IndexOp;
use tch::{
    nn::{self, Module},
    Device, Kind, Tensor,
};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Optional base (tier-0) intermediate size for tier-sliced checkpoints.
    ///
    /// For sliced checkpoints, `intermediate_size` is already the active width (e.g. 128),
    /// while `matformer_base_intermediate_size` stores the original tier-0 width (e.g. 256).
    /// This allows tier-conditioned stabilization knobs (norm gain / residual scaling) to be
    /// computed consistently across sliced and universal checkpoints.
    #[serde(default)]
    pub matformer_base_intermediate_size: Option<usize>,
    #[serde(default)]
    pub matformer_tier: u8,
    /// Fraction of suffix neurons to help train (0.0-1.0) for stochastic suffix sampling.
    #[serde(default)]
    pub matformer_helper_fraction: f32,
    /// Rounds to keep helper indices fixed before rotating.
    #[serde(default = "default_helper_rotation_interval")]
    pub matformer_helper_rotation_interval: u64,
    /// MatFormer stabilization knobs (C2): width-rescale and optional tier-0 suffix gate.
    #[serde(default)]
    pub matformer_stabilization: MatformerStabilizationConfig,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<EosToks>,
    pub rope_scaling: Option<RoPEConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub attention_bias: Option<bool>,
    /// Sliding window attention size. If set, each token only attends to the
    /// last `sliding_window` tokens instead of the full sequence.
    /// Compatible with Flash Attention 2 only.
    #[serde(default)]
    pub sliding_window: Option<i64>,
}

fn default_helper_rotation_interval() -> u64 {
    16
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn matformer_intermediate_sizes(&self) -> (i64, i64) {
        let base = self
            .matformer_base_intermediate_size
            .unwrap_or(self.intermediate_size) as i64;
        let active = match self.matformer_tier {
            // For tier-sliced checkpoints we force `matformer_tier=0` (effective tier) to avoid
            // double-slicing, but `intermediate_size` is already the active width. Use it.
            0 => self.intermediate_size as i64,
            tier => {
                let divisor = 1_i64
                    .checked_shl(tier as u32)
                    .expect("matformer_tier too large");
                base / divisor
            }
        };
        (base, active)
    }

    pub fn dummy() -> Self {
        Self {
            hidden_size: 1,
            intermediate_size: 1,
            matformer_base_intermediate_size: None,
            matformer_tier: 0,
            matformer_helper_fraction: 0.0,
            matformer_helper_rotation_interval: 16,
            matformer_stabilization: Default::default(),
            vocab_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: Some(1),
            rms_norm_eps: 0.00001,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(EosToks::Single(1)),
            rope_scaling: None,
            max_position_embeddings: 2048,
            tie_word_embeddings: false,
            attention_bias: None,
            sliding_window: None,
        }
    }

    /// Get helper config if helper mode is enabled.
    pub fn helper_config(&self) -> Option<HelperConfig> {
        if self.matformer_helper_fraction > 0.0 && self.matformer_tier > 0 {
            Some(HelperConfig::new(
                self.matformer_helper_fraction,
                self.matformer_helper_rotation_interval,
            ))
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Mlp {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
    /// Prefix size for this tier (None = tier 0, full width)
    matformer_hidden_size: Option<i64>,
    /// Full intermediate size for helper index generation
    full_intermediate_size: i64,
    /// Base (tier-0) intermediate size, even when this model is loaded from a sliced checkpoint.
    ///
    /// Used to avoid applying tier-0 suffix gating logic to already-sliced models.
    matformer_base_intermediate_size: i64,
    /// Layer index for deterministic helper sampling
    layer_idx: usize,
    /// Helper config (None = no helper mode)
    helper_config: Option<HelperConfig>,
    /// Shared reference to current round number
    current_round: Arc<AtomicU64>,
    is_tensor_parallel: bool,
    // MatFormer stabilization knobs
    width_rescale_mlp_output: bool,
    width_rescale_power: f64,
    suffix_gate: Option<SuffixGateConfig>,
    /// Optional learnable scalar for suffix gate: `sigmoid(logit)` in (0,1).
    /// Present on all tiers when enabled to keep model schema consistent.
    suffix_gate_logit: Option<Tensor>,
}

impl Mlp {
    fn new(
        vs: nn::Path,
        n_embd: i64,
        n_hidden: i64,
        matformer_base_intermediate_size: i64,
        matformer_hidden_size: Option<i64>,
        layer_idx: usize,
        helper_config: Option<HelperConfig>,
        current_round: Arc<AtomicU64>,
        matformer_stabilization: MatformerStabilizationConfig,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        let is_tensor_parallel = comm.as_ref().map(|x| x.size()).unwrap_or(1) > 1;
        let tp_size = comm.as_ref().map(|x| x.size()).unwrap_or(1);
        assert_eq!(
            n_hidden % tp_size,
            0,
            "n_hidden must be divisible by tp_size"
        );
        if let Some(ref config) = helper_config {
            assert!(
                config.helper_fraction <= 0.0,
                "MatFormer helper mode is temporarily disabled (suffix sampling not wired)"
            );
        }
        if let Some(matformer_hidden_size) = matformer_hidden_size {
            assert!(
                matformer_hidden_size > 0,
                "matformer_hidden_size must be > 0"
            );
            assert!(
                matformer_hidden_size <= n_hidden,
                "matformer_hidden_size must be <= n_hidden"
            );
            assert_eq!(
                matformer_hidden_size % tp_size,
                0,
                "matformer_hidden_size must be divisible by tp_size"
            );
        }

        let gate_proj = ColumnParallelLinear::new(
            &vs / "gate_proj",
            n_embd,
            n_hidden,
            false,
            false,
            comm.clone(),
        );
        let up_proj = ColumnParallelLinear::new(
            &vs / "up_proj",
            n_embd,
            n_hidden,
            false,
            false,
            comm.clone(),
        );
        let down_proj =
            RowParallelLinear::new(&vs / "down_proj", n_hidden, n_embd, false, true, comm);

        if let Some(ref config) = helper_config {
            eprintln!(
                "[MLP layer {}] Helper mode enabled: fraction={}, rotation_interval={}, matformer_hidden={}",
                layer_idx,
                config.helper_fraction,
                config.rotation_interval,
                matformer_hidden_size.unwrap_or(n_hidden)
            );
        }

        let suffix_gate_logit = match matformer_stabilization.suffix_gate.as_ref() {
            Some(cfg) if cfg.learnable => {
                // Keep the gate logit in fp32 even when the rest of the model is fp16/bf16.
                // This prevents LR steps smaller than fp16 ULPs from being rounded away.
                let mut logit = vs.var(
                    "matformer_suffix_gate_logit",
                    &[1],
                    nn::Init::Const(cfg.learnable_init_logit),
                );
                if logit.kind() != Kind::Float {
                    logit.set_data(&logit.to_kind(Kind::Float));
                }
                Some(logit)
            }
            _ => None,
        };

        Self {
            gate_proj,
            up_proj,
            down_proj,
            matformer_hidden_size,
            full_intermediate_size: n_hidden,
            matformer_base_intermediate_size,
            layer_idx,
            helper_config,
            current_round,
            is_tensor_parallel,
            width_rescale_mlp_output: matformer_stabilization.width_rescale_mlp_output,
            width_rescale_power: matformer_stabilization.width_rescale_power,
            suffix_gate: matformer_stabilization.suffix_gate,
            suffix_gate_logit,
        }
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // Tier 0 (full width): use standard forward
        let Some(matformer_hidden_size) = self.matformer_hidden_size else {
            let gate = self.gate_proj.forward(xs);
            let up = self.up_proj.forward(xs);
            let hidden = gate.silu() * up;

            // Suffix gate only supported for non-TP tier-0.
            if !self.is_tensor_parallel {
                if let Some(ref cfg) = self.suffix_gate {
                    // Do not apply the suffix gate to already-sliced checkpoints (no suffix).
                    // Still "touch" the learnable scalar later so gradients are defined.
                    let can_gate =
                        self.matformer_base_intermediate_size == self.full_intermediate_size;
                    if !can_gate {
                        let mut out = self.down_proj.forward(&hidden);
                        if let Some(ref logit) = self.suffix_gate_logit {
                            let out_kind = out.kind();
                            let noop = logit.sum(Kind::Float).to_kind(out_kind) * 0.0;
                            out = out + noop;
                        }
                        return out;
                    }

                    let step = crate::matformer_c2::current_step();
                    let beta = crate::matformer_c2::suffix_beta(cfg, step);
                    let suffix_alpha = if cfg.learnable {
                        let alpha = self
                            .suffix_gate_logit
                            .as_ref()
                            .expect("suffix_gate_logit must be present when learnable=true")
                            .sigmoid();
                        // Cast to activation dtype to avoid promoting large tensors to fp32.
                        Some(alpha.to_kind(hidden.kind()))
                    } else {
                        None
                    };
                    // If the gate is learnable, always take the split path so gradients can flow
                    // through `suffix_alpha` even after the schedule saturates (beta=1).
                    if beta < 1.0 || suffix_alpha.is_some() {
                        let prefix_len = crate::matformer_c2::gate_prefix_len(
                            self.full_intermediate_size,
                            cfg.gate_tier,
                        );
                        let suffix_len = self.full_intermediate_size - prefix_len;
                        if suffix_len > 0 {
                            let hidden_prefix = hidden.narrow(2, 0, prefix_len);
                            let hidden_suffix = hidden.narrow(2, prefix_len, suffix_len);

                            let down_w_prefix = self.down_proj.linear.ws.narrow(1, 0, prefix_len);
                            let down_w_suffix =
                                self.down_proj.linear.ws.narrow(1, prefix_len, suffix_len);

                            let prefix_out = hidden_prefix.matmul(&down_w_prefix.transpose(0, 1));
                            let suffix_out = hidden_suffix.matmul(&down_w_suffix.transpose(0, 1));
                            return if let Some(alpha) = suffix_alpha {
                                // beta is a schedule scalar; alpha is a learned scalar tensor.
                                prefix_out + suffix_out * alpha.multiply_scalar(beta)
                            } else {
                                prefix_out + suffix_out.multiply_scalar(beta)
                            };
                        }
                    }
                }
            }

            let mut out = self.down_proj.forward(&hidden);
            if let Some(ref logit) = self.suffix_gate_logit {
                let out_kind = out.kind();
                let noop = logit.sum(Kind::Float).to_kind(out_kind) * 0.0;
                out = out + noop;
            }
            return out;
        };
        assert!(
            !self.is_tensor_parallel,
            "matformer_tier is not yet supported with tensor parallelism"
        );

        // Check if we have helper mode enabled
        let use_helper =
            self.helper_config.is_some() && matformer_hidden_size < self.full_intermediate_size;

        if use_helper {
            // Helper mode: use index_select with prefix + stochastic suffix indices
            let helper_config = self.helper_config.as_ref().unwrap();
            let round = self.current_round.load(Ordering::Relaxed);

            let indices = crate::matformer_helper::get_matformer_indices(
                matformer_hidden_size as usize,
                self.full_intermediate_size as usize,
                helper_config,
                round,
                self.layer_idx,
            );

            let indices_tensor = Tensor::from_slice(&indices).to_device(xs.device());

            // Weight shapes:
            // - gate_proj/up_proj: [n_hidden, n_embd] - select rows
            // - down_proj: [n_embd, n_hidden] - select columns
            let gate_w = self.gate_proj.linear.ws.index_select(0, &indices_tensor);
            let up_w = self.up_proj.linear.ws.index_select(0, &indices_tensor);
            let down_w = self.down_proj.linear.ws.index_select(1, &indices_tensor);

            let gate = xs.matmul(&gate_w.transpose(0, 1));
            let up = xs.matmul(&up_w.transpose(0, 1));
            let hidden = gate.silu() * up;
            let mut out = hidden.matmul(&down_w.transpose(0, 1));
            if self.width_rescale_mlp_output {
                let active = indices.len() as i64;
                let scale = crate::matformer_c2::width_rescale_factor(
                    self.full_intermediate_size,
                    active,
                    self.width_rescale_power,
                );
                if (scale - 1.0).abs() > f64::EPSILON {
                    out = out.multiply_scalar(scale);
                }
            }
            if let Some(ref logit) = self.suffix_gate_logit {
                let out_kind = out.kind();
                let noop = logit.sum(Kind::Float).to_kind(out_kind) * 0.0;
                out = out + noop;
            }
            out
        } else {
            // Standard MatFormer: use narrow for contiguous prefix (more efficient)
            // Weight shapes:
            // - gate_proj/up_proj: [n_hidden, n_embd]
            // - down_proj: [n_embd, n_hidden]
            let gate_w = self.gate_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
            let up_w = self.up_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
            let down_w = self.down_proj.linear.ws.narrow(1, 0, matformer_hidden_size);

            let gate = xs.matmul(&gate_w.transpose(0, 1));
            let up = xs.matmul(&up_w.transpose(0, 1));
            let hidden = gate.silu() * up;
            let mut out = hidden.matmul(&down_w.transpose(0, 1));
            if self.width_rescale_mlp_output {
                let scale = crate::matformer_c2::width_rescale_factor(
                    self.full_intermediate_size,
                    matformer_hidden_size,
                    self.width_rescale_power,
                );
                if (scale - 1.0).abs() > f64::EPSILON {
                    out = out.multiply_scalar(scale);
                }
            }
            if let Some(ref logit) = self.suffix_gate_logit {
                let out_kind = out.kind();
                let noop = logit.sum(Kind::Float).to_kind(out_kind) * 0.0;
                out = out + noop;
            }
            out
        }
    }
}

#[derive(Debug)]
struct Block {
    rms_1: RMSNorm,
    attn: CausalSelfAttention,
    rms_2: RMSNorm,
    mlp: Mlp,
    /// Per-tier residual scaling for MLP output (plain f64, NOT in VarStore).
    residual_scale_mlp: Option<f64>,
    /// Per-tier residual scaling for attention output (plain f64, NOT in VarStore).
    residual_scale_attn: Option<f64>,
    /// Learnable per-tier residual gate for MLP output: `alpha = exp(log_alpha[tier])`.
    residual_gate_mlp_log: Option<Tensor>,
    /// Learnable per-tier residual gate for attention output: `alpha = exp(log_alpha[tier])`.
    residual_gate_attn_log: Option<Tensor>,
    /// Tier index used to select a gate element (derived from base/active widths; works for slices).
    residual_gate_tier_idx: i64,
}

impl Block {
    fn new(
        vs: nn::Path,
        config: &LlamaConfig,
        layer_idx: usize,
        current_round: Arc<AtomicU64>,
        attn_implementation: AttentionImplementation,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        // Compute MatFormer width ratio for stabilization knobs.
        // Note: for tier-sliced checkpoints, `matformer_tier` is forced to 0 (effective tier),
        // but `matformer_base_intermediate_size` lets us still compute (active/base) correctly.
        let (full_intermediate, active_intermediate) = config.matformer_intermediate_sizes();
        let residual_gate_tier_idx =
            crate::matformer_c2::infer_matformer_tier_from_intermediate_sizes(
                full_intermediate,
                active_intermediate,
            ) as i64;

        // Compute tier gain for RMSNorm (if configured)
        let norm_tier_gain = config
            .matformer_stabilization
            .norm_tier_gain
            .as_ref()
            .and_then(|cfg| {
                let gain = crate::matformer_c2::residual_scale_factor(
                    full_intermediate,
                    active_intermediate,
                    cfg.power,
                );
                if (gain - 1.0).abs() > f64::EPSILON {
                    Some(gain)
                } else {
                    None
                }
            });

        let rms_1 = RMSNorm::new_with_tier_gain(
            &vs / "input_layernorm",
            config.hidden_size as i64,
            config.rms_norm_eps,
            norm_tier_gain,
        );
        let attn = CausalSelfAttention::new(
            &vs / "self_attn",
            config.num_attention_heads as i64,
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads) as i64,
            config.hidden_size as i64,
            (config.max_position_embeddings + 1) as i64,
            attn_implementation,
            comm.clone(),
            config.sliding_window,
        );
        let rms_2 = RMSNorm::new_with_tier_gain(
            &vs / "post_attention_layernorm",
            config.hidden_size as i64,
            config.rms_norm_eps,
            norm_tier_gain,
        );
        let mlp = Mlp::new(
            &vs / "mlp",
            config.hidden_size as i64,
            config.intermediate_size as i64,
            full_intermediate,
            match config.matformer_tier {
                0 => None,
                tier => {
                    let divisor = 1_i64
                        .checked_shl(tier as u32)
                        .expect("matformer_tier too large");
                    Some((config.intermediate_size as i64) / divisor)
                }
            },
            layer_idx,
            config.helper_config(),
            current_round,
            config.matformer_stabilization.clone(),
            comm,
        );

        // Compute per-tier residual scaling factors
        let (residual_scale_mlp, residual_scale_attn) =
            match &config.matformer_stabilization.residual_scale {
                Some(rs_cfg) => {
                    let mlp_alpha = if rs_cfg.apply_to_mlp {
                        let alpha = crate::matformer_c2::residual_scale_factor(
                            full_intermediate,
                            active_intermediate,
                            rs_cfg.power,
                        );
                        if (alpha - 1.0).abs() > f64::EPSILON {
                            Some(alpha)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    // Attention residual scaling: uses the same width ratio.
                    // Typically disabled (attention is full-width), but available for experiments.
                    let attn_alpha = if rs_cfg.apply_to_attn {
                        let alpha = crate::matformer_c2::residual_scale_factor(
                            full_intermediate,
                            active_intermediate,
                            rs_cfg.power,
                        );
                        if (alpha - 1.0).abs() > f64::EPSILON {
                            Some(alpha)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    (mlp_alpha, attn_alpha)
                }
                None => (None, None),
            };

        // Learnable per-tier residual gates (log-space, exp in forward).
        // These are real parameters (included in DisTrO updates).
        let (residual_gate_mlp_log, residual_gate_attn_log) =
            match &config.matformer_stabilization.residual_gates {
                Some(cfg) => {
                    let init_logs =
                        crate::matformer_c2::residual_gates_init_log_alphas(cfg.init_power);
                    let mk_gate = |name: &str| {
                        let mut gate = vs.var(
                            name,
                            &[crate::matformer_c2::RESIDUAL_GATES_MAX_TIERS],
                            nn::Init::Const(0.0),
                        );
                        if cfg.init_power != 0.0 {
                            let init = Tensor::f_from_slice(&init_logs)
                                .expect("failed to build residual gate init tensor")
                                .to_device(gate.device())
                                .to_kind(gate.kind());
                            gate.f_copy_(&init).expect("failed to init residual gate");
                        }
                        gate
                    };
                    let mlp_gate = cfg
                        .apply_to_mlp
                        .then(|| mk_gate("matformer_resid_gate_mlp_log"));
                    let attn_gate = cfg
                        .apply_to_attn
                        .then(|| mk_gate("matformer_resid_gate_attn_log"));
                    (mlp_gate, attn_gate)
                }
                None => (None, None),
            };

        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            residual_scale_mlp,
            residual_scale_attn,
            residual_gate_mlp_log,
            residual_gate_attn_log,
            residual_gate_tier_idx,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&(Tensor, i32)>,
        cache: &RoPECache,
    ) -> Tensor {
        let attn_out = self.attn.forward(
            &self.rms_1.forward(x),
            position_ids,
            sequence_lengths,
            cache,
        );
        let mut attn_scaled = match self.residual_scale_attn {
            Some(alpha) => attn_out * alpha,
            None => attn_out,
        };
        if let Some(ref log_gates) = self.residual_gate_attn_log {
            let alpha = log_gates.i(self.residual_gate_tier_idx).exp();
            attn_scaled = attn_scaled * alpha;
        }
        let x = attn_scaled + x;
        let mlp_out = self.mlp.forward(&self.rms_2.forward(&x));
        let mut mlp_scaled = match self.residual_scale_mlp {
            Some(alpha) => mlp_out * alpha,
            None => mlp_out,
        };
        if let Some(ref log_gates) = self.residual_gate_mlp_log {
            let alpha = log_gates.i(self.residual_gate_tier_idx).exp();
            mlp_scaled = mlp_scaled * alpha;
        }
        mlp_scaled + x
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Llama {
    wte: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: RMSNorm,
    attn_implementation: AttentionImplementation,
    rope_cache: RoPECache,
    /// Shared round counter for helper index generation
    current_round: Arc<AtomicU64>,
}

impl Llama {
    pub fn new(
        vs: nn::Path,
        config: &LlamaConfig,
        attn_implementation: AttentionImplementation,
        comm: Option<Arc<Communicator>>,
    ) -> Self {
        let current_round = Arc::new(AtomicU64::new(0));
        let wte = nn::embedding(
            &vs / "model" / "embed_tokens",
            config.vocab_size as i64,
            config.hidden_size as i64,
            Default::default(),
        );
        // Apply tier-conditioned RMSNorm gain consistently (including final RMSNorm).
        let (full_intermediate, active_intermediate) = config.matformer_intermediate_sizes();
        let ln_f_tier_gain = config
            .matformer_stabilization
            .norm_tier_gain
            .as_ref()
            .and_then(|cfg| {
                let gain = crate::matformer_c2::residual_scale_factor(
                    full_intermediate,
                    active_intermediate,
                    cfg.power,
                );
                if (gain - 1.0).abs() > f64::EPSILON {
                    Some(gain)
                } else {
                    None
                }
            });
        let ln_f = RMSNorm::new_with_tier_gain(
            &vs / "model" / "norm",
            config.hidden_size as i64,
            config.rms_norm_eps,
            ln_f_tier_gain,
        );
        let blocks = (0..config.num_hidden_layers)
            .map(|i| {
                Block::new(
                    &vs / "model" / "layers" / i,
                    config,
                    i,
                    current_round.clone(),
                    attn_implementation,
                    comm.clone(),
                )
            })
            .collect::<Vec<_>>();
        let rope_cache = RoPECache::new(
            &config.rope_config(),
            config.hidden_size() / config.num_attention_heads(),
            config.rope_theta(),
            &vs.device(),
        );
        Self {
            wte,
            blocks,
            ln_f,
            attn_implementation,
            rope_cache,
            current_round,
        }
    }

    /// Set the current round number for helper index generation.
    /// This should be called before each forward pass during training.
    pub fn set_round(&self, round: u64) {
        self.current_round.store(round, Ordering::Relaxed);
    }

    /// Get the current round number.
    pub fn get_round(&self) -> u64 {
        self.current_round.load(Ordering::Relaxed)
    }
}

impl LanguageModelForward for Llama {
    #[allow(unused_variables)]
    fn forward(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        _training: bool,
    ) -> Tensor {
        let sequence_lengths = sequence_lengths.map(|sequence_lengths| {
            #[cfg(feature = "parallelism")]
            {
                if self.attn_implementation == AttentionImplementation::FlashAttention2 {
                    crate::attention::create_cu_seqlens(sequence_lengths, x.device())
                } else {
                    panic!("`sequence_lengths` only supported for FlashAttention2");
                }
            }

            #[cfg(not(feature = "parallelism"))]
            {
                panic!("`sequence_lengths` only supported for FlashAttention2");
            }
        });

        let mut x = self.wte.forward(x);
        for block in &self.blocks {
            x = block.forward(
                &x,
                position_ids,
                sequence_lengths.as_ref(),
                &self.rope_cache,
            );
        }
        self.ln_f.forward(&x)
    }
}

pub type LlamaForCausalLM = CausalLanguageModel<Llama, LlamaConfig>;

impl LlamaForCausalLM {
    fn builder(
        vs: nn::Path,
        config: &LlamaConfig,
        attn_implementation: Option<AttentionImplementation>,
        comm: Option<Arc<Communicator>>,
    ) -> Result<Llama, ModelLoadError> {
        Ok(Llama::new(
            vs,
            config,
            attn_implementation.unwrap_or_default(),
            comm,
        ))
    }

    pub fn from_pretrained(
        source: &PretrainedSource<LlamaConfig>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
    ) -> Result<Self, ModelLoadError> {
        Self::from_builder(
            Self::builder,
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
        )
    }

    /// Load a pretrained model while applying caller-provided config overrides.
    ///
    /// This is useful for runtime MatFormer knobs (suffix gate / distillation scheduling)
    /// that should not require editing a checkpoint's `config.json`.
    pub fn from_pretrained_with_config_overrides<F: FnOnce(&mut LlamaConfig)>(
        source: &PretrainedSource<LlamaConfig>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
        config_overrides: F,
    ) -> Result<Self, ModelLoadError> {
        Self::from_builder_with_config_overrides(
            Self::builder,
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
            config_overrides,
        )
    }

    pub fn from_pretrained_with_matformer_tier(
        source: &PretrainedSource<LlamaConfig>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
        matformer_tier: u8,
    ) -> Result<Self, ModelLoadError> {
        Self::from_pretrained_with_matformer_config(
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
            matformer_tier,
            0.0,
            16,
        )
    }

    pub fn from_pretrained_with_matformer_config(
        source: &PretrainedSource<LlamaConfig>,
        kind: Option<Kind>,
        attn_implementation: Option<AttentionImplementation>,
        device: Option<Device>,
        tensor_parallelism_world: Option<(CommunicatorId, usize, usize)>,
        override_max_position_embeddings: Option<usize>,
        matformer_tier: u8,
        matformer_helper_fraction: f32,
        matformer_helper_rotation_interval: u64,
    ) -> Result<Self, ModelLoadError> {
        Self::from_builder_with_config_overrides(
            Self::builder,
            source,
            kind,
            attn_implementation,
            device,
            tensor_parallelism_world,
            override_max_position_embeddings,
            |config| {
                config.matformer_tier = matformer_tier;
                config.matformer_helper_fraction = matformer_helper_fraction;
                config.matformer_helper_rotation_interval = matformer_helper_rotation_interval;
            },
        )
    }
}

impl ModelConfig for LlamaConfig {
    fn get_parameter_names(&self) -> Vec<String> {
        let mut variables = Vec::new();
        let gates_cfg = self.matformer_stabilization.residual_gates.as_ref();
        for layer_idx in 0..self.num_hidden_layers {
            let layer_prefix = format!("model.layers.{}", layer_idx);

            variables.push(format!("{}.self_attn.q_proj.weight", layer_prefix));
            variables.push(format!("{}.self_attn.k_proj.weight", layer_prefix));
            variables.push(format!("{}.self_attn.v_proj.weight", layer_prefix));
            variables.push(format!("{}.self_attn.o_proj.weight", layer_prefix));

            variables.push(format!("{}.mlp.gate_proj.weight", layer_prefix));
            variables.push(format!("{}.mlp.up_proj.weight", layer_prefix));
            variables.push(format!("{}.mlp.down_proj.weight", layer_prefix));

            variables.push(format!("{}.input_layernorm.weight", layer_prefix));
            variables.push(format!("{}.post_attention_layernorm.weight", layer_prefix));

            if let Some(cfg) = gates_cfg {
                if cfg.apply_to_mlp {
                    variables.push(format!("{}.matformer_resid_gate_mlp_log", layer_prefix));
                }
                if cfg.apply_to_attn {
                    variables.push(format!("{}.matformer_resid_gate_attn_log", layer_prefix));
                }
            }

            if self
                .matformer_stabilization
                .suffix_gate
                .as_ref()
                .is_some_and(|cfg| cfg.learnable)
            {
                variables.push(format!("{}.mlp.matformer_suffix_gate_logit", layer_prefix));
            }

            if self.attention_bias.unwrap_or(false) {
                variables.push(format!("{}.self_attn.q_proj.bias", layer_prefix));
                variables.push(format!("{}.self_attn.k_proj.bias", layer_prefix));
                variables.push(format!("{}.self_attn.v_proj.bias", layer_prefix));
            }
        }

        variables.push("lm_head.weight".to_string());
        variables.push("model.norm.weight".to_string());
        variables.push("model.embed_tokens.weight".to_string());

        variables
    }
}

impl TryFrom<AutoConfig> for LlamaConfig {
    type Error = ModelLoadError;

    fn try_from(value: AutoConfig) -> Result<Self, Self::Error> {
        match value {
            AutoConfig::Llama(llama_config) => Ok(llama_config),
            _ => Err(ModelLoadError::WrongConfigType),
        }
    }
}

impl TryFrom<PretrainedSource<AutoConfig>> for PretrainedSource<LlamaConfig> {
    type Error = ModelLoadError;

    fn try_from(value: PretrainedSource<AutoConfig>) -> Result<Self, Self::Error> {
        match value {
            PretrainedSource::RepoFiles(path_bufs) => Ok(PretrainedSource::RepoFiles(path_bufs)),
            PretrainedSource::ConfigAndTensors(AutoConfig::Llama(config), hash_map) => {
                Ok(PretrainedSource::ConfigAndTensors(config, hash_map))
            }
            _ => Err(ModelLoadError::WrongConfigType),
        }
    }
}

impl LanguageModelConfig for LlamaConfig {
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
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
        self.rope_scaling.clone()
    }

    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn bos_token_id(&self) -> Option<i64> {
        self.bos_token_id
    }

    fn eos_token_ids(&self) -> Option<EosToks> {
        self.eos_token_id.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::Mlp;
    use std::sync::{atomic::AtomicU64, Arc};
    use tch::nn::Module;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn matformer_mlp_has_zero_tail_grads() {
        let vs = nn::VarStore::new(Device::Cpu);
        let n_embd = 4;
        let n_hidden = 8;
        let matformer_hidden = 4;
        let current_round = Arc::new(AtomicU64::new(0));

        let mlp = Mlp::new(
            vs.root(),
            n_embd,
            n_hidden,
            n_hidden,
            Some(matformer_hidden),
            0,    // layer_idx
            None, // helper_config (no helper mode for this test)
            current_round,
            Default::default(),
            None,
        );

        let xs = Tensor::randn([2, 3, n_embd], (Kind::Float, Device::Cpu));
        let out = mlp.forward(&xs);
        let loss = out.sum(Kind::Float);
        loss.backward();

        let gate_grad = mlp.gate_proj.linear.ws.grad();
        let up_grad = mlp.up_proj.linear.ws.grad();
        let down_grad = mlp.down_proj.linear.ws.grad();

        // gate/up: [n_hidden, n_embd] => tail rows must have zero grad
        let gate_tail = gate_grad.narrow(0, matformer_hidden, n_hidden - matformer_hidden);
        let up_tail = up_grad.narrow(0, matformer_hidden, n_hidden - matformer_hidden);

        // down: [n_embd, n_hidden] => tail cols must have zero grad
        let down_tail = down_grad.narrow(1, matformer_hidden, n_hidden - matformer_hidden);

        let gate_tail_max = gate_tail.abs().max().double_value(&[]);
        let up_tail_max = up_tail.abs().max().double_value(&[]);
        let down_tail_max = down_tail.abs().max().double_value(&[]);

        assert_eq!(gate_tail_max, 0.0);
        assert_eq!(up_tail_max, 0.0);
        assert_eq!(down_tail_max, 0.0);
    }

    #[test]
    #[should_panic(expected = "MatFormer helper mode is temporarily disabled")]
    fn matformer_mlp_helper_mode_has_helper_grads() {
        use crate::matformer_helper::HelperConfig;

        let vs = nn::VarStore::new(Device::Cpu);
        let n_embd = 4;
        let n_hidden = 16;
        let matformer_hidden = 4;
        let current_round = Arc::new(AtomicU64::new(0));
        let helper_config = HelperConfig::new(0.5, 16); // 50% helper

        let _ = Mlp::new(
            vs.root(),
            n_embd,
            n_hidden,
            n_hidden,
            Some(matformer_hidden),
            0,
            Some(helper_config),
            current_round,
            Default::default(),
            None,
        );
    }
}
