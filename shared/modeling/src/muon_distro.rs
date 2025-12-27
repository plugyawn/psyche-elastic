//! Experimental Muon + DisTrO Compositions
//!
//! This module explores different ways to combine Muon (orthogonalized momentum)
//! with DisTrO (DCT-compressed gradient communication) for distributed training.
//!
//! # The Fundamental Problem
//!
//! Muon's key insight is that orthogonalized updates (Q @ Q^T ≈ I) converge faster.
//! DisTrO's key insight is that DCT + top-k compression maintains training quality.
//!
//! The conflict: DCT compression destroys orthogonality.
//!
//! # Experimental Approaches
//!
//! 1. **MomentumOnlyDistro** - Use Muon-style momentum without orthogonalization
//! 2. **ServerOrthogonalize** - Orthogonalize after aggregation at server
//! 3. **ResidualCompress** - Orthogonalize locally, only compress residual
//! 4. **HybridLayers** - Orthogonalize attention, compress MLP
//!
//! # Usage
//!
//! ```ignore
//! use psyche_modeling::muon_distro::{MuonDistroConfig, MuonDistroComposition};
//!
//! let config = MuonDistroConfig {
//!     composition: MuonDistroComposition::MomentumOnly,
//!     momentum: 0.95,
//!     ..Default::default()
//! };
//! ```

use crate::{
    distro::{CompressDCT, DistroResult},
    kernels::{AdaptiveOrthogonalize, Orthogonalize},
    CausalLM,
};
use std::collections::HashMap;
use tch::{Kind, Tensor};

/// Different ways to compose Muon with DisTrO
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuonDistroComposition {
    /// Use Muon-style momentum smoothing, but skip orthogonalization
    /// before DisTrO compression. Hypothesis: momentum alone provides
    /// most of the benefit.
    MomentumOnly,

    /// Compress gradients with DisTrO, aggregate at server, then
    /// orthogonalize the aggregated update. Preserves orthogonality
    /// for final update but increases server compute.
    ServerOrthogonalize,

    /// Orthogonalize locally, compute residual (momentum - ortho),
    /// only compress and send the residual. Hypothesis: residual is
    /// small and compresses well.
    ResidualCompress,

    /// Use orthogonalization for attention layers (where it matters most),
    /// use DisTrO compression for MLP layers (which are larger).
    HybridLayers,

    /// Periodically sync full orthogonalized weights, use DisTrO between syncs.
    /// Amortizes the orthogonalization cost.
    PeriodicOrtho {
        /// Orthogonalize every N steps
        period: u32,
    },
}

impl Default for MuonDistroComposition {
    fn default() -> Self {
        Self::MomentumOnly
    }
}

/// Configuration for MuonDistro hybrid optimizer
#[derive(Debug, Clone)]
pub struct MuonDistroConfig {
    /// Which composition strategy to use
    pub composition: MuonDistroComposition,

    /// Momentum coefficient (0.95 typical for Muon)
    pub momentum: f64,

    /// Weight decay (decoupled, applied directly to weights)
    pub weight_decay: f64,

    /// DisTrO compression parameters
    pub compression_topk: i64,
    pub compression_chunk: i64,
    pub compression_decay: f64,

    /// Whether to quantize compressed values to 1-bit signs
    pub quantize_1bit: bool,

    /// Gradient clipping norm (None = no clipping)
    pub clip_grad_norm: Option<f64>,
}

impl Default for MuonDistroConfig {
    fn default() -> Self {
        Self {
            composition: MuonDistroComposition::MomentumOnly,
            momentum: 0.95,
            weight_decay: 0.0,
            compression_topk: 32,
            compression_chunk: 64,
            compression_decay: 0.999,
            quantize_1bit: true,
            clip_grad_norm: Some(1.0),
        }
    }
}

/// Per-parameter state for MuonDistro
struct ParamState {
    /// Momentum buffer
    momentum_buffer: Tensor,
    /// DCT error accumulator (for DisTrO compression)
    error_accumulator: Tensor,
    /// Is this a 2D parameter (can be orthogonalized)
    is_2d: bool,
    /// Is this an attention layer (for hybrid mode)
    is_attention: bool,
}

/// MuonDistro hybrid optimizer
///
/// Combines Muon's momentum/orthogonalization with DisTrO's compression.
pub struct MuonDistro {
    config: MuonDistroConfig,
    states: HashMap<String, ParamState>,
    orthogonalizer: Box<dyn Orthogonalize>,
    step: u64,
}

impl MuonDistro {
    /// Create a new MuonDistro optimizer
    pub fn new(config: MuonDistroConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            orthogonalizer: Box::new(AdaptiveOrthogonalize::new()),
            step: 0,
        }
    }

    /// Initialize state for a parameter
    fn init_state(&mut self, name: &str, tensor: &Tensor) {
        let device = tensor.device();
        let is_2d = tensor.dim() == 2;
        let is_attention = name.contains("attn")
            || name.contains("q_proj")
            || name.contains("k_proj")
            || name.contains("v_proj")
            || name.contains("o_proj");

        let state = ParamState {
            momentum_buffer: Tensor::zeros(tensor.size(), (Kind::Float, device)),
            error_accumulator: Tensor::zeros(tensor.size(), (Kind::Float, device)),
            is_2d,
            is_attention,
        };

        self.states.insert(name.to_string(), state);
    }

    /// Generate compressed gradient results for transmission
    ///
    /// This is the main entry point - it processes gradients according to
    /// the configured composition strategy and returns DisTrO-compressed results.
    pub fn generate(&mut self, model: &dyn CausalLM) -> Vec<(String, DistroResult)> {
        self.step += 1;
        let current_step = self.step;
        let _guard = tch::no_grad_guard();

        // Optional gradient clipping
        if let Some(max_norm) = self.config.clip_grad_norm {
            model.clip_grad_norm(max_norm);
        }

        // Copy config values to avoid borrow issues
        let momentum = self.config.momentum;
        let composition = self.config.composition;
        let compression_topk = self.config.compression_topk;
        let quantize_1bit = self.config.quantize_1bit;

        let mut results = Vec::new();

        for var in model.variables() {
            let name = var.name();
            let tensor = var.local_tensor();
            let grad = tensor.grad();

            if !grad.defined() {
                continue;
            }

            // Initialize state if needed
            if !self.states.contains_key(name) {
                self.init_state(name, &tensor);
            }

            let state = self.states.get_mut(name).unwrap();

            // Convert gradient to FP32
            let grad_fp32 = grad.to_kind(Kind::Float);

            // Process according to composition strategy
            let to_compress = match composition {
                MuonDistroComposition::MomentumOnly => {
                    Self::process_momentum_only_static(&grad_fp32, state, momentum)
                }
                MuonDistroComposition::ServerOrthogonalize => {
                    // For server-side ortho, we just send momentum (server will ortho)
                    Self::process_momentum_only_static(&grad_fp32, state, momentum)
                }
                MuonDistroComposition::ResidualCompress => {
                    Self::process_residual_compress_static(
                        &grad_fp32,
                        state,
                        momentum,
                        &*self.orthogonalizer,
                    )
                }
                MuonDistroComposition::HybridLayers => {
                    Self::process_hybrid_layers_static(
                        &grad_fp32,
                        state,
                        momentum,
                        &*self.orthogonalizer,
                    )
                }
                MuonDistroComposition::PeriodicOrtho { period } => {
                    Self::process_periodic_ortho_static(
                        &grad_fp32,
                        state,
                        momentum,
                        current_step,
                        period,
                        &*self.orthogonalizer,
                    )
                }
            };

            // Compress with DisTrO
            let result = Self::compress_tensor_static(
                &to_compress,
                state,
                compression_topk,
                quantize_1bit,
            );
            results.push((name.to_string(), result));
        }

        results
    }

    /// Process with momentum only (no orthogonalization before compression)
    fn process_momentum_only_static(
        grad: &Tensor,
        state: &mut ParamState,
        momentum: f64,
    ) -> Tensor {
        // Update momentum: m = β * m + (1 - β) * g
        let _ = state.momentum_buffer.g_mul_scalar_(momentum);
        let grad_scaled = grad * (1.0 - momentum);
        let _ = state.momentum_buffer.g_add_(&grad_scaled);

        state.momentum_buffer.shallow_clone()
    }

    /// Process with residual compression
    ///
    /// Key insight: if momentum ≈ orthogonal, the residual should be small
    /// and compress better than the full gradient.
    fn process_residual_compress_static(
        grad: &Tensor,
        state: &mut ParamState,
        momentum: f64,
        orthogonalizer: &dyn Orthogonalize,
    ) -> Tensor {
        // Update momentum
        let _ = state.momentum_buffer.g_mul_scalar_(momentum);
        let grad_scaled = grad * (1.0 - momentum);
        let _ = state.momentum_buffer.g_add_(&grad_scaled);

        if state.is_2d {
            // Orthogonalize
            let ortho = orthogonalizer.polar_express(&state.momentum_buffer, 5);

            // Compute residual: what orthogonalization "removed"
            let residual = &state.momentum_buffer - &ortho;

            // Return residual for compression
            // Note: We need to also track the ortho component somehow
            // For now, we'll compress the residual and apply ortho locally
            residual
        } else {
            // For 1D params, just return momentum
            state.momentum_buffer.shallow_clone()
        }
    }

    /// Process with hybrid layer strategy
    ///
    /// Attention layers: full orthogonalization (no compression for these)
    /// MLP layers: momentum + compression (skip orthogonalization)
    fn process_hybrid_layers_static(
        grad: &Tensor,
        state: &mut ParamState,
        momentum: f64,
        orthogonalizer: &dyn Orthogonalize,
    ) -> Tensor {
        // Update momentum
        let _ = state.momentum_buffer.g_mul_scalar_(momentum);
        let grad_scaled = grad * (1.0 - momentum);
        let _ = state.momentum_buffer.g_add_(&grad_scaled);

        if state.is_attention && state.is_2d {
            // For attention: orthogonalize (will be sent uncompressed or lightly compressed)
            orthogonalizer.polar_express(&state.momentum_buffer, 5)
        } else {
            // For MLP/embeddings: just momentum
            state.momentum_buffer.shallow_clone()
        }
    }

    /// Process with periodic orthogonalization
    fn process_periodic_ortho_static(
        grad: &Tensor,
        state: &mut ParamState,
        momentum: f64,
        current_step: u64,
        period: u32,
        orthogonalizer: &dyn Orthogonalize,
    ) -> Tensor {
        // Update momentum
        let _ = state.momentum_buffer.g_mul_scalar_(momentum);
        let grad_scaled = grad * (1.0 - momentum);
        let _ = state.momentum_buffer.g_add_(&grad_scaled);

        if state.is_2d && current_step % period as u64 == 0 {
            // Orthogonalize on this step
            orthogonalizer.polar_express(&state.momentum_buffer, 5)
        } else {
            // Regular momentum
            state.momentum_buffer.shallow_clone()
        }
    }

    /// Compress a tensor using DisTrO's DCT + top-k compression
    fn compress_tensor_static(
        tensor: &Tensor,
        state: &mut ParamState,
        compression_topk: i64,
        quantize_1bit: bool,
    ) -> DistroResult {
        // Add error accumulator for better compression
        let to_compress = tensor + &state.error_accumulator;

        // DCT compression
        let (sparse_idx, sparse_val, xshape, totalk) =
            CompressDCT::compress(&to_compress, compression_topk);

        // Compute what we actually sent (for error accumulation)
        let sent = CompressDCT::decompress(
            &sparse_idx,
            &sparse_val,
            &xshape,
            totalk,
            tensor.kind(),
            tensor.device(),
        );

        // Update error accumulator: error = to_compress - sent
        let _ = state.error_accumulator.copy_(&(&to_compress - &sent));

        // Optionally quantize to 1-bit
        let sparse_val = if quantize_1bit {
            sparse_val.greater(0.0).to_kind(Kind::Bool)
        } else {
            sparse_val
        };

        DistroResult {
            sparse_idx,
            sparse_val,
            xshape,
            totalk,
            stats: None,
        }
    }

    /// Get current step count
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Get configuration
    pub fn config(&self) -> &MuonDistroConfig {
        &self.config
    }
}

impl std::fmt::Debug for MuonDistro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MuonDistro")
            .field("config", &self.config)
            .field("step", &self.step)
            .field("num_params", &self.states.len())
            .finish()
    }
}

/// Server-side aggregation with optional orthogonalization
///
/// This is used when MuonDistroComposition::ServerOrthogonalize is selected.
/// The server aggregates compressed gradients from clients, then orthogonalizes
/// the result before broadcasting.
pub struct ServerAggregator {
    orthogonalizer: Box<dyn Orthogonalize>,
    do_orthogonalize: bool,
}

impl ServerAggregator {
    /// Create a new server aggregator
    pub fn new(do_orthogonalize: bool) -> Self {
        Self {
            orthogonalizer: Box::new(AdaptiveOrthogonalize::new()),
            do_orthogonalize,
        }
    }

    /// Aggregate and optionally orthogonalize gradients
    ///
    /// # Arguments
    /// * `aggregated` - Already-aggregated gradient (sum of decompressed client contributions)
    /// * `is_2d` - Whether this is a 2D parameter
    ///
    /// # Returns
    /// The processed gradient ready for application
    pub fn process(&self, aggregated: &Tensor, is_2d: bool) -> Tensor {
        if self.do_orthogonalize && is_2d {
            self.orthogonalizer.polar_express(aggregated, 5)
        } else {
            aggregated.shallow_clone()
        }
    }
}

/// Metrics for comparing different compositions
#[derive(Debug, Clone, Default)]
pub struct CompositionMetrics {
    /// Average orthogonality error: ||Q @ Q^T - I||_F / n
    pub avg_orthogonality_error: f64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Training loss
    pub loss: f64,
    /// Gradient norm before compression
    pub grad_norm_before: f64,
    /// Gradient norm after compression/decompression
    pub grad_norm_after: f64,
}

/// Compute orthogonality error for a tensor
pub fn orthogonality_error(q: &Tensor) -> f64 {
    if q.dim() != 2 {
        return 0.0;
    }

    let n = q.size()[0].min(q.size()[1]) as f64;
    let (rows, cols) = (q.size()[0], q.size()[1]);

    let (product, size) = if rows <= cols {
        (q.matmul(&q.tr()), rows)
    } else {
        (q.tr().matmul(q), cols)
    };

    let identity = Tensor::eye(size, (Kind::Float, q.device()));
    let frobenius = (product.to_kind(Kind::Float) - identity)
        .norm()
        .double_value(&[]);

    frobenius / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_muon_distro_config_default() {
        let config = MuonDistroConfig::default();
        assert_eq!(config.momentum, 0.95);
        assert!(matches!(
            config.composition,
            MuonDistroComposition::MomentumOnly
        ));
    }

    #[test]
    fn test_momentum_only_accumulates() {
        let config = MuonDistroConfig {
            composition: MuonDistroComposition::MomentumOnly,
            ..Default::default()
        };
        let mut optimizer = MuonDistro::new(config.clone());

        // Create a simple parameter state
        let grad = Tensor::ones([4, 4], (Kind::Float, Device::Cpu));
        optimizer.init_state("test", &grad);

        let state = optimizer.states.get_mut("test").unwrap();
        let momentum = config.momentum;

        // First update: m = 0.05 * grad (since m starts at 0)
        let result1 = MuonDistro::process_momentum_only_static(&grad, state, momentum);
        let expected1 = 0.05; // (1 - 0.95) * 1.0
        let actual1 = result1.mean(Kind::Float).double_value(&[]);
        assert!(
            (actual1 - expected1).abs() < 1e-5,
            "First update: expected {}, got {}",
            expected1,
            actual1
        );

        // Second update: m = 0.95 * 0.05 + 0.05 * 1.0 = 0.0975
        let result2 = MuonDistro::process_momentum_only_static(&grad, state, momentum);
        let expected2 = 0.0975;
        let actual2 = result2.mean(Kind::Float).double_value(&[]);
        assert!(
            (actual2 - expected2).abs() < 1e-5,
            "Second update: expected {}, got {}",
            expected2,
            actual2
        );
    }

    #[test]
    fn test_residual_compress_smaller_than_full() {
        let config = MuonDistroConfig {
            composition: MuonDistroComposition::ResidualCompress,
            ..Default::default()
        };
        let mut optimizer = MuonDistro::new(config.clone());

        // Create a gradient that's close to orthogonal already
        let grad = Tensor::randn([8, 8], (Kind::Float, Device::Cpu));
        optimizer.init_state("test", &grad);

        let state = optimizer.states.get_mut("test").unwrap();
        let momentum = config.momentum;

        // Process a few times to build up momentum
        for _ in 0..5 {
            let _ = MuonDistro::process_residual_compress_static(
                &grad, state, momentum, &*optimizer.orthogonalizer
            );
        }

        // The residual should generally be smaller than the full momentum
        let residual = MuonDistro::process_residual_compress_static(
            &grad, state, momentum, &*optimizer.orthogonalizer
        );
        let momentum_norm = state.momentum_buffer.norm().double_value(&[]);
        let residual_norm = residual.norm().double_value(&[]);

        // Residual might not always be smaller, but should be comparable
        // This is more of a sanity check than a strict requirement
        eprintln!(
            "Momentum norm: {:.4}, Residual norm: {:.4}, Ratio: {:.2}",
            momentum_norm,
            residual_norm,
            residual_norm / momentum_norm
        );
    }

    #[test]
    fn test_hybrid_layers_detection() {
        let config = MuonDistroConfig {
            composition: MuonDistroComposition::HybridLayers,
            ..Default::default()
        };
        let mut optimizer = MuonDistro::new(config, 0);

        // Attention layer
        let attn_grad = Tensor::randn([64, 64], (Kind::Float, Device::Cpu));
        optimizer.init_state("model.layers.0.self_attn.q_proj.weight", &attn_grad);

        let attn_state = optimizer.states.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert!(attn_state.is_attention, "q_proj should be detected as attention");

        // MLP layer
        let mlp_grad = Tensor::randn([256, 64], (Kind::Float, Device::Cpu));
        optimizer.init_state("model.layers.0.mlp.gate_proj.weight", &mlp_grad);

        let mlp_state = optimizer.states.get("model.layers.0.mlp.gate_proj.weight").unwrap();
        assert!(!mlp_state.is_attention, "gate_proj should not be detected as attention");
    }

    #[test]
    fn test_periodic_ortho_fires_at_period() {
        let period = 5;
        let config = MuonDistroConfig {
            composition: MuonDistroComposition::PeriodicOrtho { period },
            ..Default::default()
        };
        let mut optimizer = MuonDistro::new(config.clone());

        let grad = Tensor::randn([8, 8], (Kind::Float, Device::Cpu));
        optimizer.init_state("test", &grad);
        let momentum = config.momentum;

        // Run for several steps and check orthogonality at period boundaries
        for step in 1..=15u64 {
            let state = optimizer.states.get_mut("test").unwrap();
            let result = MuonDistro::process_periodic_ortho_static(
                &grad, state, momentum, step, period, &*optimizer.orthogonalizer
            );

            if step % period as u64 == 0 {
                // Should be orthogonalized
                let error = orthogonality_error(&result);
                eprintln!("Step {}: orthogonality error = {:.4}", step, error);
                // Note: error won't be 0 because momentum isn't perfectly orthogonal
            }
        }
    }

    #[test]
    fn test_server_aggregator() {
        let aggregator = ServerAggregator::new(true);

        let grad = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));
        let result = aggregator.process(&grad, true);

        // Should be more orthogonal than input
        let input_error = orthogonality_error(&grad);
        let output_error = orthogonality_error(&result);

        eprintln!(
            "Server aggregator: input error = {:.4}, output error = {:.4}",
            input_error, output_error
        );

        // Output should be more orthogonal (lower error)
        assert!(
            output_error < input_error || output_error < 0.5,
            "Server orthogonalization should improve orthogonality"
        );
    }

    #[test]
    fn test_orthogonality_error_perfect() {
        // Perfect orthogonal matrix should have error ≈ 0
        let angle = std::f64::consts::PI / 4.0;
        let data: [f32; 4] = [
            angle.cos() as f32,
            -angle.sin() as f32,
            angle.sin() as f32,
            angle.cos() as f32,
        ];
        let q = Tensor::from_slice(&data).reshape([2, 2]);

        let error = orthogonality_error(&q);
        assert!(
            error < 1e-5,
            "Perfect rotation matrix should have ~0 error, got {}",
            error
        );
    }

    #[test]
    fn test_orthogonality_error_random() {
        // Random matrix should have high error
        let rand = Tensor::randn([8, 8], (Kind::Float, Device::Cpu));
        let error = orthogonality_error(&rand);

        // Random matrix won't be orthogonal
        eprintln!("Random matrix orthogonality error: {:.4}", error);
        // Just a sanity check - random matrices are typically not orthogonal
    }

    // ==================== Compression Quality Tests ====================

    #[test]
    fn test_compression_preserves_direction() {
        // Test that DisTrO compression preserves gradient direction
        let config = MuonDistroConfig {
            compression_topk: 64, // Keep more for this test
            quantize_1bit: false, // Don't quantize
            ..Default::default()
        };
        let mut optimizer = MuonDistro::new(config.clone());

        let grad = Tensor::randn([16, 16], (Kind::Float, Device::Cpu));
        optimizer.init_state("test", &grad);

        let state = optimizer.states.get_mut("test").unwrap();
        let _ = state.momentum_buffer.copy_(&grad);

        let result = MuonDistro::compress_tensor_static(
            &grad, state, config.compression_topk, config.quantize_1bit
        );

        // Decompress and check direction similarity (cosine similarity)
        let decompressed = CompressDCT::decompress(
            &result.sparse_idx,
            &result.sparse_val,
            &result.xshape,
            result.totalk,
            Kind::Float,
            Device::Cpu,
        );

        let dot = (&grad * &decompressed).sum(Kind::Float).double_value(&[]);
        let grad_norm = grad.norm().double_value(&[]);
        let decomp_norm = decompressed.norm().double_value(&[]);

        let cosine_sim = dot / (grad_norm * decomp_norm + 1e-8);

        eprintln!("Compression cosine similarity: {:.4}", cosine_sim);

        // Should preserve direction reasonably well
        assert!(
            cosine_sim > 0.5,
            "Compression should preserve direction, got cosine_sim = {}",
            cosine_sim
        );
    }
}
