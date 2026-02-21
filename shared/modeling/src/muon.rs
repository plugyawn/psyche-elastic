//! Muon Optimizer Implementation
//!
//! Muon is an optimizer that applies Polar Express orthogonalization to momentum.
//! It uses Nesterov-style momentum with orthogonalization for 2D (matrix) parameters,
//! and falls back to AdamW for non-2D parameters (embeddings, norms, biases).
//!
//! # Algorithm
//!
//! For 2D parameters (where Polar Express can be applied):
//! ```text
//! m_t = β * m_{t-1} + (1 - β) * g_t
//! m_ortho = PolarExpress(m_t, 5)
//! θ_t = θ_{t-1} - lr * m_ortho - wd * θ_{t-1}
//! ```
//!
//! For 1D/0D parameters (embeddings, norms, biases):
//! ```text
//! Standard AdamW update
//! ```
//!
//! # References
//!
//! - NorMuon optimizer in modded-nanogpt
//! - "Shampoo: Preconditioned Stochastic Tensor Optimization" (related work)

use crate::{
    kernels::{CpuOrthogonalize, Orthogonalize},
    CausalLM,
};
use regex::Regex;
use std::collections::HashMap;
use tch::{Kind, Tensor};

/// Configuration for a parameter group in Muon
#[derive(Debug, Clone)]
pub struct MuonParamGroupConfig {
    /// Regex pattern for matching parameter names
    pub pattern: String,
    /// Whether to use Muon (true) or AdamW fallback (false)
    pub use_muon: bool,
    /// Whether to apply Polar Express orthogonalization (only for 2D)
    pub orthogonalize: bool,
}

impl MuonParamGroupConfig {
    /// Create a Muon param group with orthogonalization
    pub fn muon(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            use_muon: true,
            orthogonalize: true,
        }
    }

    /// Create an AdamW fallback param group
    pub fn adamw(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            use_muon: false,
            orthogonalize: false,
        }
    }
}

/// Configuration for the Muon optimizer
#[derive(Debug, Clone)]
pub struct MuonConfig {
    /// Learning rate
    pub lr: f64,
    /// Momentum coefficient for Muon (typically 0.95)
    pub momentum: f64,
    /// Weight decay (decoupled, applied to weights directly)
    pub weight_decay: f64,
    /// Beta1 for AdamW fallback parameters
    pub adamw_beta1: f64,
    /// Beta2 for AdamW fallback parameters
    pub adamw_beta2: f64,
    /// Epsilon for AdamW numerical stability
    pub adamw_eps: f64,
    /// Gradient clipping norm (None = no clipping)
    pub clip_grad_norm: Option<f64>,
    /// Parameter groups (ordered, first match wins)
    pub param_groups: Vec<MuonParamGroupConfig>,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.95,
            weight_decay: 0.0,
            adamw_beta1: 0.9,
            adamw_beta2: 0.999,
            adamw_eps: 1e-8,
            clip_grad_norm: Some(1.0),
            // Default param groups matching modded-nanogpt:
            // - Embeddings use AdamW
            // - Norms use AdamW
            // - lm_head uses AdamW
            // - Everything else uses Muon with orthogonalization
            param_groups: vec![
                MuonParamGroupConfig::adamw(".*embed.*"),
                MuonParamGroupConfig::adamw(".*norm.*"),
                MuonParamGroupConfig::adamw(".*lm_head.*"),
                MuonParamGroupConfig::adamw(".*bias.*"),
                MuonParamGroupConfig::muon(".*"), // Catch-all for matrices
            ],
        }
    }
}

impl MuonConfig {
    /// Create a config with custom learning rate
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Create a config with custom momentum
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Create a config with custom weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Create a config with custom gradient clipping
    pub fn with_clip_grad_norm(mut self, clip: Option<f64>) -> Self {
        self.clip_grad_norm = clip;
        self
    }
}

/// Per-parameter optimizer state
#[derive(Debug)]
struct ParamState {
    /// Which param group this parameter belongs to
    group_idx: usize,
    /// Momentum buffer for Muon
    momentum_buffer: Option<Tensor>,
    /// Exponential moving average of gradients (for AdamW)
    exp_avg: Option<Tensor>,
    /// Exponential moving average of squared gradients (for AdamW)
    exp_avg_sq: Option<Tensor>,
    /// Whether this parameter is 2D (can be orthogonalized)
    is_2d: bool,
}

/// The Muon optimizer
pub struct Muon {
    config: MuonConfig,
    states: HashMap<String, ParamState>,
    compiled_patterns: Vec<Regex>,
    orthogonalizer: Box<dyn Orthogonalize>,
    step: u64,
}

impl Muon {
    /// Create a new Muon optimizer with the given configuration
    pub fn new(config: MuonConfig) -> Self {
        // Pre-compile regex patterns
        let compiled_patterns: Vec<Regex> = config
            .param_groups
            .iter()
            .map(|pg| Regex::new(&pg.pattern).expect("Invalid param group regex pattern"))
            .collect();

        Self {
            config,
            states: HashMap::new(),
            compiled_patterns,
            orthogonalizer: Box::new(CpuOrthogonalize::new()),
            step: 0,
        }
    }

    /// Get the parameter group index for a given parameter name
    fn get_param_group(&self, name: &str) -> usize {
        for (idx, pattern) in self.compiled_patterns.iter().enumerate() {
            if pattern.is_match(name) {
                return idx;
            }
        }
        // Should not happen if there's a catch-all pattern
        self.config.param_groups.len() - 1
    }

    /// Initialize state for a parameter
    fn init_state(&mut self, name: &str, tensor: &Tensor) -> &mut ParamState {
        let group_idx = self.get_param_group(name);
        let is_2d = tensor.dim() == 2;
        let device = tensor.device();
        let kind = Kind::Float; // State always in FP32

        let group = &self.config.param_groups[group_idx];

        let state = if group.use_muon {
            // Muon: only need momentum buffer
            ParamState {
                group_idx,
                momentum_buffer: Some(Tensor::zeros(tensor.size(), (kind, device))),
                exp_avg: None,
                exp_avg_sq: None,
                is_2d,
            }
        } else {
            // AdamW: need exp_avg and exp_avg_sq
            ParamState {
                group_idx,
                momentum_buffer: None,
                exp_avg: Some(Tensor::zeros(tensor.size(), (kind, device))),
                exp_avg_sq: Some(Tensor::zeros(tensor.size(), (kind, device))),
                is_2d,
            }
        };

        self.states.insert(name.to_string(), state);
        self.states.get_mut(name).unwrap()
    }

    /// Perform a single optimization step
    ///
    /// # Arguments
    ///
    /// * `model` - The model to optimize
    /// * `lr` - Learning rate for this step (allows LR scheduling)
    pub fn step(&mut self, model: &dyn CausalLM, lr: f64) {
        self.step += 1;
        let current_step = self.step;

        let _guard = tch::no_grad_guard();

        // Copy config values to avoid borrow issues
        let momentum = self.config.momentum;
        let weight_decay = self.config.weight_decay;
        let adamw_beta1 = self.config.adamw_beta1;
        let adamw_beta2 = self.config.adamw_beta2;
        let adamw_eps = self.config.adamw_eps;

        // Optional gradient clipping
        if let Some(max_norm) = self.config.clip_grad_norm {
            model.clip_grad_norm(max_norm);
        }

        // First pass: initialize any missing states
        for var in model.variables() {
            let name = var.name();
            let tensor = var.local_tensor();
            if !self.states.contains_key(name) {
                self.init_state(name, &tensor);
            }
        }

        // Second pass: perform updates
        for var in model.variables() {
            let name = var.name();
            let mut tensor = var.local_tensor();
            let grad = tensor.grad();

            if !grad.defined() {
                continue;
            }

            let state = self.states.get_mut(name).unwrap();
            let group_idx = state.group_idx;
            let is_2d = state.is_2d;
            let use_muon = self.config.param_groups[group_idx].use_muon;
            let orthogonalize = self.config.param_groups[group_idx].orthogonalize;

            // Convert gradient to FP32 for state updates
            let grad_fp32 = grad.to_kind(Kind::Float);

            if use_muon {
                // Muon update
                let momentum_buffer = state.momentum_buffer.as_mut().unwrap();

                // m_t = β * m_{t-1} + (1 - β) * g_t
                let _ = momentum_buffer.g_mul_scalar_(momentum);
                let grad_scaled = &grad_fp32 * (1.0 - momentum);
                let _ = momentum_buffer.g_add_(&grad_scaled);

                // Orthogonalize if 2D and configured
                let update = if is_2d && orthogonalize {
                    self.orthogonalizer.polar_express(momentum_buffer, 5)
                } else {
                    momentum_buffer.shallow_clone()
                };

                // Apply update: θ = θ - lr * update
                let update = update.to_kind(tensor.kind());
                let update_scaled = &update * (-lr);
                let _ = tensor.g_add_(&update_scaled);

                // Apply decoupled weight decay: θ = θ * (1 - wd * lr)
                if weight_decay > 0.0 {
                    let decay_factor = 1.0 - weight_decay * lr;
                    let _ = tensor.g_mul_scalar_(decay_factor);
                }
            } else {
                // AdamW update
                let exp_avg = state.exp_avg.as_mut().unwrap();
                let exp_avg_sq = state.exp_avg_sq.as_mut().unwrap();

                // m_t = β1 * m_{t-1} + (1 - β1) * g_t
                let _ = exp_avg.g_mul_scalar_(adamw_beta1);
                let grad_scaled = &grad_fp32 * (1.0 - adamw_beta1);
                let _ = exp_avg.g_add_(&grad_scaled);

                // v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                let _ = exp_avg_sq.g_mul_scalar_(adamw_beta2);
                let grad_sq = &grad_fp32 * &grad_fp32;
                let grad_sq_scaled = &grad_sq * (1.0 - adamw_beta2);
                let _ = exp_avg_sq.g_add_(&grad_sq_scaled);

                // Bias correction
                let bias_correction1 = 1.0 - adamw_beta1.powi(current_step as i32);
                let bias_correction2 = 1.0 - adamw_beta2.powi(current_step as i32);

                // Compute update: m_hat / (sqrt(v_hat) + eps)
                let exp_avg_corrected = exp_avg.shallow_clone() / bias_correction1;
                let exp_avg_sq_corrected = exp_avg_sq.shallow_clone() / bias_correction2;
                let denom = exp_avg_sq_corrected.sqrt() + adamw_eps;
                let update = exp_avg_corrected / denom;

                // Apply update
                let update = update.to_kind(tensor.kind());
                let update_scaled = &update * (-lr);
                let _ = tensor.g_add_(&update_scaled);

                // Apply decoupled weight decay: θ = θ * (1 - wd * lr)
                if weight_decay > 0.0 {
                    let decay_factor = 1.0 - weight_decay * lr;
                    let _ = tensor.g_mul_scalar_(decay_factor);
                }
            }
        }
    }

    /// Zero all gradients in the model
    pub fn zero_grad(&self, model: &dyn CausalLM) {
        for var in model.variables() {
            let tensor = var.local_tensor();
            if tensor.grad().defined() {
                let _ = tensor.grad().zero_();
            }
        }
    }

    /// Get the current step count
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Get the configuration
    pub fn config(&self) -> &MuonConfig {
        &self.config
    }
}

impl std::fmt::Debug for Muon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Muon")
            .field("config", &self.config)
            .field("step", &self.step)
            .field("num_params", &self.states.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CausalLM, Communicator, EosToks, StableVarStoreIterator, StableVariableIterator};
    use std::sync::Arc;
    use tch::nn::{self, Module, VarStore};

    #[test]
    fn test_muon_config_default() {
        let config = MuonConfig::default();
        assert_eq!(config.momentum, 0.95);
        assert_eq!(config.adamw_beta1, 0.9);
        assert!(!config.param_groups.is_empty());
    }

    #[test]
    fn test_param_group_matching() {
        let config = MuonConfig::default();
        let muon = Muon::new(config);

        // Test pattern matching
        assert_eq!(muon.get_param_group("model.embed_tokens.weight"), 0); // embed
        assert_eq!(
            muon.get_param_group("model.layers.0.input_layernorm.weight"),
            1
        ); // norm
        assert_eq!(muon.get_param_group("lm_head.weight"), 2); // lm_head
        assert_eq!(
            muon.get_param_group("model.layers.0.mlp.gate_proj.weight"),
            4
        ); // catch-all
    }

    // ==================== Training Convergence Test ====================

    /// A tiny transformer for testing training convergence.
    /// Architecture: embed -> 2 transformer blocks -> output projection
    /// Each block: layernorm -> attention (simplified) -> layernorm -> MLP
    struct TinyTransformer {
        var_store: VarStore,
        embed: nn::Embedding,
        blocks: Vec<TinyBlock>,
        ln_f: nn::LayerNorm,
        lm_head: nn::Linear,
        config: TinyConfig,
    }

    struct TinyConfig {
        vocab_size: i64,
        hidden_size: i64,
        num_layers: usize,
        num_heads: i64,
        intermediate_size: i64,
    }

    impl TinyConfig {
        fn tiny() -> Self {
            Self {
                vocab_size: 256,
                hidden_size: 64,
                num_layers: 2,
                num_heads: 4,
                intermediate_size: 128,
            }
        }
    }

    struct TinyBlock {
        ln1: nn::LayerNorm,
        attn_qkv: nn::Linear,
        attn_out: nn::Linear,
        ln2: nn::LayerNorm,
        mlp_up: nn::Linear,
        mlp_down: nn::Linear,
        num_heads: i64,
        head_dim: i64,
    }

    impl TinyBlock {
        fn new(vs: &nn::Path, hidden_size: i64, num_heads: i64, intermediate_size: i64) -> Self {
            let head_dim = hidden_size / num_heads;
            let ln_config = nn::LayerNormConfig::default();

            Self {
                ln1: nn::layer_norm(vs / "ln1", vec![hidden_size], ln_config),
                attn_qkv: nn::linear(
                    vs / "attn_qkv",
                    hidden_size,
                    3 * hidden_size,
                    Default::default(),
                ),
                attn_out: nn::linear(
                    vs / "attn_out",
                    hidden_size,
                    hidden_size,
                    Default::default(),
                ),
                ln2: nn::layer_norm(vs / "ln2", vec![hidden_size], ln_config),
                mlp_up: nn::linear(
                    vs / "mlp_up",
                    hidden_size,
                    intermediate_size,
                    Default::default(),
                ),
                mlp_down: nn::linear(
                    vs / "mlp_down",
                    intermediate_size,
                    hidden_size,
                    Default::default(),
                ),
                num_heads,
                head_dim,
            }
        }

        fn forward(&self, x: &Tensor) -> Tensor {
            let (batch, seq_len, hidden) = (x.size()[0], x.size()[1], x.size()[2]);

            // Self-attention with residual
            let h = self.ln1.forward(x);
            let qkv = self.attn_qkv.forward(&h);
            let qkv = qkv.reshape([batch, seq_len, 3, self.num_heads, self.head_dim]);
            let qkv = qkv.permute([2, 0, 3, 1, 4]); // [3, B, H, S, D]
            let q = qkv.select(0, 0);
            let k = qkv.select(0, 1);
            let v = qkv.select(0, 2);

            // Scaled dot-product attention (causal)
            let scale = (self.head_dim as f64).sqrt();
            let attn = q.matmul(&k.transpose(-2, -1)) / scale;

            // Causal mask
            let mask = Tensor::ones([seq_len, seq_len], (Kind::Float, x.device()))
                .tril(0)
                .reshape([1, 1, seq_len, seq_len]);
            let attn = attn.masked_fill(&mask.eq(0.0), f64::NEG_INFINITY);
            let attn = attn.softmax(-1, Kind::Float);

            let out = attn.matmul(&v); // [B, H, S, D]
            let out = out.permute([0, 2, 1, 3]).reshape([batch, seq_len, hidden]);
            let out = self.attn_out.forward(&out);
            let x = x + out;

            // MLP with residual
            let h = self.ln2.forward(&x);
            let h = self.mlp_up.forward(&h).gelu("none");
            let h = self.mlp_down.forward(&h);
            x + h
        }
    }

    impl TinyTransformer {
        fn new(config: TinyConfig, device: tch::Device) -> Self {
            let var_store = VarStore::new(device);
            let vs = var_store.root();

            // Use path separators that tch-rs allows (no dots in names)
            let model_path = &vs / "model";
            let embed = nn::embedding(
                &model_path / "embed_tokens",
                config.vocab_size,
                config.hidden_size,
                Default::default(),
            );

            let mut blocks = Vec::new();
            let layers_path = &model_path / "layers";
            for i in 0..config.num_layers {
                let layer_path = &layers_path / i;
                blocks.push(TinyBlock::new(
                    &layer_path,
                    config.hidden_size,
                    config.num_heads,
                    config.intermediate_size,
                ));
            }

            let ln_config = nn::LayerNormConfig::default();
            let ln_f = nn::layer_norm(&model_path / "norm", vec![config.hidden_size], ln_config);
            let lm_head = nn::linear(
                &vs / "lm_head",
                config.hidden_size,
                config.vocab_size,
                Default::default(),
            );

            Self {
                var_store,
                embed,
                blocks,
                ln_f,
                lm_head,
                config,
            }
        }
    }

    impl CausalLM for TinyTransformer {
        fn forward(
            &self,
            x: &Tensor,
            labels: Option<&Tensor>,
            _position_ids: Option<&Tensor>,
            _sequence_lengths: Option<&Vec<Vec<i32>>>,
            _num_logits_to_keep: Option<i64>,
            loss_scale: Option<f64>,
        ) -> (Option<Tensor>, Option<Tensor>) {
            // x: [batch, seq_len] of token ids
            let mut h = self.embed.forward(x); // [batch, seq_len, hidden]

            for block in &self.blocks {
                h = block.forward(&h);
            }

            h = self.ln_f.forward(&h);
            let logits = self.lm_head.forward(&h); // [batch, seq_len, vocab]

            let loss = labels.map(|labels| {
                // Shift logits and labels for next-token prediction
                let shift_logits = logits.narrow(1, 0, logits.size()[1] - 1);
                let shift_labels = labels.narrow(1, 1, labels.size()[1] - 1);

                let batch_seq = shift_logits.size()[0] * shift_logits.size()[1];
                let flat_logits = shift_logits.reshape([batch_seq, self.config.vocab_size]);
                let flat_labels = shift_labels.reshape([batch_seq]);

                let loss = flat_logits.cross_entropy_for_logits(&flat_labels);
                match loss_scale {
                    Some(scale) => loss / scale,
                    None => loss,
                }
            });

            (Some(logits), loss)
        }

        fn bos_token_id(&self) -> Option<i64> {
            Some(1)
        }

        fn eos_token_ids(&self) -> Option<EosToks> {
            Some(EosToks::Single(2))
        }

        fn device(&self) -> tch::Device {
            self.var_store.device()
        }

        fn max_context_length(&self) -> usize {
            512
        }

        fn variables(&self) -> StableVariableIterator {
            Box::new(StableVarStoreIterator::new(&self.var_store, None))
        }

        fn communicator(&self) -> Option<Arc<Communicator>> {
            None
        }

        fn prepare_for_training(&self) {
            for var in self.variables() {
                let _ = var.local_tensor().set_requires_grad(true);
            }
        }

        fn clip_grad_norm(&self, max_grad_norm: f64) {
            let _guard = tch::no_grad_guard();
            let mut total_norm_sq = 0.0f64;

            for var in self.variables() {
                let grad = var.local_tensor().grad();
                if grad.defined() {
                    let norm_sq: f64 = grad.square().sum(Kind::Float).double_value(&[]);
                    total_norm_sq += norm_sq;
                }
            }

            let total_norm = total_norm_sq.sqrt();
            if total_norm > max_grad_norm {
                let clip_coef = max_grad_norm / (total_norm + 1e-6);
                for var in self.variables() {
                    let mut grad = var.local_tensor().grad();
                    if grad.defined() {
                        let _ = grad.g_mul_scalar_(clip_coef);
                    }
                }
            }
        }
    }

    #[test]
    fn test_muon_training_convergence() {
        // Test that Muon optimizer reduces loss over training steps
        let device = tch::Device::Cpu;
        let model = TinyTransformer::new(TinyConfig::tiny(), device);
        model.prepare_for_training();

        let mut optimizer = Muon::new(MuonConfig::default().with_lr(1e-3));

        // Generate synthetic training data (random tokens)
        let batch_size = 4;
        let seq_len = 32;
        let data = Tensor::randint(256, [batch_size, seq_len], (Kind::Int64, device));

        // Training loop
        let num_steps = 50;
        let mut losses = Vec::new();

        for step in 0..num_steps {
            optimizer.zero_grad(&model);

            // Forward pass with labels = input (next token prediction)
            let (_, loss) = model.forward(&data, Some(&data), None, None, None, None);
            let loss = loss.unwrap();

            // Record loss
            let loss_val = loss.double_value(&[]);
            losses.push(loss_val);

            // Backward pass
            loss.backward();

            // Optimizer step
            optimizer.step(&model, 1e-3);

            if step % 10 == 0 {
                eprintln!("Step {}: loss = {:.4}", step, loss_val);
            }
        }

        // Verify convergence: loss should decrease
        let initial_loss = losses[0];
        let final_loss = losses[num_steps - 1];

        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={:.4}, final={:.4}",
            initial_loss,
            final_loss
        );

        // Loss should decrease by at least 20%
        let improvement = (initial_loss - final_loss) / initial_loss;
        assert!(
            improvement > 0.2,
            "Loss should improve by >20%: initial={:.4}, final={:.4}, improvement={:.1}%",
            initial_loss,
            final_loss,
            improvement * 100.0
        );

        // No NaN losses
        for (i, &loss) in losses.iter().enumerate() {
            assert!(
                loss.is_finite(),
                "Loss became non-finite at step {}: {}",
                i,
                loss
            );
        }

        eprintln!(
            "Training convergence test passed: {:.4} -> {:.4} ({:.1}% improvement)",
            initial_loss,
            final_loss,
            improvement * 100.0
        );
    }

    #[test]
    fn test_muon_vs_adamw_comparison() {
        // Compare Muon with orthogonalization vs pure AdamW
        // Both should converge, but we verify Muon doesn't break anything
        let device = tch::Device::Cpu;

        // Same seed for fair comparison
        tch::manual_seed(42);
        let data = Tensor::randint(256, [4, 32], (Kind::Int64, device));

        // Train with Muon
        tch::manual_seed(42);
        let model_muon = TinyTransformer::new(TinyConfig::tiny(), device);
        model_muon.prepare_for_training();
        let mut optimizer_muon = Muon::new(MuonConfig::default().with_lr(1e-3));

        let mut losses_muon = Vec::new();
        for _ in 0..30 {
            optimizer_muon.zero_grad(&model_muon);
            let (_, loss) = model_muon.forward(&data, Some(&data), None, None, None, None);
            let loss = loss.unwrap();
            losses_muon.push(loss.double_value(&[]));
            loss.backward();
            optimizer_muon.step(&model_muon, 1e-3);
        }

        // Train with AdamW-only config (no orthogonalization)
        tch::manual_seed(42);
        let model_adamw = TinyTransformer::new(TinyConfig::tiny(), device);
        model_adamw.prepare_for_training();
        let adamw_config = MuonConfig {
            param_groups: vec![MuonParamGroupConfig::adamw(".*")], // All AdamW
            ..MuonConfig::default()
        };
        let mut optimizer_adamw = Muon::new(adamw_config.with_lr(1e-3));

        let mut losses_adamw = Vec::new();
        for _ in 0..30 {
            optimizer_adamw.zero_grad(&model_adamw);
            let (_, loss) = model_adamw.forward(&data, Some(&data), None, None, None, None);
            let loss = loss.unwrap();
            losses_adamw.push(loss.double_value(&[]));
            loss.backward();
            optimizer_adamw.step(&model_adamw, 1e-3);
        }

        // Both should converge
        let muon_improvement = (losses_muon[0] - losses_muon[29]) / losses_muon[0];
        let adamw_improvement = (losses_adamw[0] - losses_adamw[29]) / losses_adamw[0];

        assert!(
            muon_improvement > 0.1,
            "Muon should improve: {:.1}%",
            muon_improvement * 100.0
        );
        assert!(
            adamw_improvement > 0.1,
            "AdamW should improve: {:.1}%",
            adamw_improvement * 100.0
        );

        eprintln!(
            "Muon improvement: {:.1}%, AdamW improvement: {:.1}%",
            muon_improvement * 100.0,
            adamw_improvement * 100.0
        );
    }
}
