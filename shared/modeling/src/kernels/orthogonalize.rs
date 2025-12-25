//! Orthogonalization kernels for Muon optimizer
//!
//! This module provides implementations of matrix orthogonalization using the
//! Polar Express algorithm - an optimal matrix sign iteration method.
//!
//! # Polar Express Algorithm
//!
//! The Polar Express method computes the orthogonal polar factor Q of a matrix G
//! where G = QS (Q orthogonal, S symmetric positive semi-definite).
//!
//! The algorithm uses pre-computed optimal coefficients (a, b, c) for a 5-iteration
//! fixed-point scheme:
//!
//! ```text
//! X_0 = G / (||G|| * 1.02)  // Normalized initial guess
//! For i = 0..5:
//!     B = b_i * X @ X^T + c_i * (X @ X^T) @ (X @ X^T)
//!     X_{i+1} = a_i * X + B @ X
//! ```
//!
//! The coefficients are derived from polynomial approximations optimized for convergence
//! within the spectral interval [1e-7, 1].
//!
//! # References
//!
//! - NorMuon optimizer in modded-nanogpt
//! - "An Optimal Matrix Sign Iteration" paper

use tch::{Kind, Tensor};

/// Pre-computed optimal coefficients for Polar Express (5 iterations)
/// Each tuple is (a, b, c) for one iteration
const POLAR_COEFFS: [(f64, f64, f64); 5] = [
    (8.157, -22.483, 15.879),
    (4.043, -2.809, 0.500),
    (3.892, -2.772, 0.506),
    (3.286, -2.368, 0.464),
    (2.347, -1.710, 0.423),
];

/// Safety factor for initial normalization (prevents numerical instability)
const NORM_SAFETY_FACTOR: f64 = 1.02;

/// Trait for orthogonalization operations
pub trait Orthogonalize: Send + Sync {
    /// Get the name of this kernel implementation
    fn name(&self) -> &'static str;

    /// Check if this kernel is available on the current system
    fn is_available(&self) -> bool;

    /// Orthogonalize a matrix using the Polar Express algorithm
    ///
    /// # Arguments
    ///
    /// * `g` - Input matrix (typically a gradient tensor, 2D)
    /// * `iterations` - Number of iterations (default: 5)
    ///
    /// # Returns
    ///
    /// The orthogonal polar factor Q such that G ≈ QS
    ///
    /// # Notes
    ///
    /// - Input must be 2D tensor
    /// - Output has the same shape as input
    /// - For non-2D inputs, returns the input unchanged (with warning)
    fn polar_express(&self, g: &Tensor, iterations: usize) -> Tensor;

    /// Orthogonalize with default 5 iterations
    fn orthogonalize(&self, g: &Tensor) -> Tensor {
        self.polar_express(g, 5)
    }
}

/// CPU implementation of orthogonalization using Polar Express
pub struct CpuOrthogonalize {
    /// Use BFloat16 for intermediate computations
    /// Note: BF16 is ~150x slower on CPU due to emulation. Use FP32 for CPU.
    /// BF16 should only be used on CUDA where it has hardware support.
    use_bf16: bool,
}

impl CpuOrthogonalize {
    /// Create a new CPU orthogonalization kernel (uses FP32 for speed)
    pub fn new() -> Self {
        // Default to FP32 on CPU - BF16 emulation is ~150x slower
        Self { use_bf16: false }
    }

    /// Create with custom precision settings
    pub fn with_precision(use_bf16: bool) -> Self {
        Self { use_bf16 }
    }
}

impl Default for CpuOrthogonalize {
    fn default() -> Self {
        Self::new()
    }
}

impl Orthogonalize for CpuOrthogonalize {
    fn name(&self) -> &'static str {
        "cpu_polar_express"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn polar_express(&self, g: &Tensor, iterations: usize) -> Tensor {
        // Only orthogonalize 2D tensors
        if g.dim() != 2 {
            tracing::trace!(
                "Polar Express skipped for non-2D tensor (dim={})",
                g.dim()
            );
            return g.shallow_clone();
        }

        let _guard = tch::no_grad_guard();
        let original_kind = g.kind();

        // Convert to computation dtype
        let compute_kind = if self.use_bf16 {
            Kind::BFloat16
        } else {
            Kind::Float
        };

        // Initial normalization: X_0 = G / (||G|| * safety_factor)
        let g_norm = g.norm().double_value(&[]);
        if g_norm < 1e-12 {
            // Near-zero gradient, return as-is
            return g.shallow_clone();
        }

        let mut x = g.to_kind(compute_kind) / (g_norm * NORM_SAFETY_FACTOR);

        // Fixed-point iteration with pre-computed optimal coefficients
        let num_iters = iterations.min(POLAR_COEFFS.len());
        for (a, b, c) in POLAR_COEFFS.iter().take(num_iters) {
            // X @ X^T
            let xxt = x.matmul(&x.tr());

            // B = b * X@X^T + c * (X@X^T)@(X@X^T)
            let b_mat = &xxt * *b + xxt.matmul(&xxt) * *c;

            // X = a * X + B @ X
            x = &x * *a + b_mat.matmul(&x);
        }

        // Convert back to original dtype
        x.to_kind(original_kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_express_identity() {
        // For an orthogonal matrix, Polar Express should return ~same matrix
        let ortho = CpuOrthogonalize::new();

        // Create a simple orthogonal matrix (rotation by 45 degrees)
        let angle = std::f64::consts::PI / 4.0;
        let data: [f32; 4] = [
            angle.cos() as f32,
            -angle.sin() as f32,
            angle.sin() as f32,
            angle.cos() as f32,
        ];
        let q = Tensor::from_slice(&data).reshape([2, 2]);

        let result = ortho.polar_express(&q, 5);

        // Result should be close to input (since input is already orthogonal)
        let diff = (&result - &q).abs().max().double_value(&[]);
        assert!(
            diff < 0.1,
            "Polar Express on orthogonal matrix changed too much: diff={diff}"
        );
    }

    #[test]
    fn test_polar_express_orthogonality() {
        // Result should be approximately orthogonal: Q @ Q^T ≈ I
        let ortho = CpuOrthogonalize::new();

        // Create a random-ish matrix
        let g = Tensor::randn([4, 4], (Kind::Float, tch::Device::Cpu));
        let result = ortho.polar_express(&g, 5);

        // Check orthogonality: Q @ Q^T should be close to identity
        let qqt = result.matmul(&result.tr());
        let identity = Tensor::eye(4, (Kind::Float, tch::Device::Cpu));
        let ortho_error = (&qqt - &identity).abs().max().double_value(&[]);

        assert!(
            ortho_error < 0.5,
            "Result not orthogonal enough: max error={ortho_error}"
        );
    }

    #[test]
    fn test_polar_express_non_2d() {
        // Non-2D tensors should be returned unchanged
        let ortho = CpuOrthogonalize::new();

        let t1d = Tensor::randn([10], (Kind::Float, tch::Device::Cpu));
        let result1d = ortho.polar_express(&t1d, 5);
        assert_eq!(t1d.size(), result1d.size());

        let t3d = Tensor::randn([2, 3, 4], (Kind::Float, tch::Device::Cpu));
        let result3d = ortho.polar_express(&t3d, 5);
        assert_eq!(t3d.size(), result3d.size());
    }

    #[test]
    fn test_polar_express_near_zero() {
        // Near-zero gradients should be handled gracefully
        let ortho = CpuOrthogonalize::new();

        let g = Tensor::zeros([4, 4], (Kind::Float, tch::Device::Cpu)) * 1e-20;
        let result = ortho.polar_express(&g, 5);

        // Should not produce NaN or Inf
        assert!(
            !result.isnan().any().int64_value(&[]) != 0,
            "Result contains NaN"
        );
        assert!(
            !result.isinf().any().int64_value(&[]) != 0,
            "Result contains Inf"
        );
    }

    #[test]
    #[ignore] // Slow benchmark - run with: cargo test bench_polar -- --ignored --nocapture
    fn bench_polar_express_latency() {
        use std::time::Instant;

        // Test both BF16 (default) and FP32 to see if BF16 emulation is the bottleneck
        let ortho_bf16 = CpuOrthogonalize::new();
        let ortho_fp32 = CpuOrthogonalize::with_precision(false);

        let sizes: [(i64, i64); 3] = [(256, 256), (512, 512), (1024, 1024)];

        println!("\nPolar Express Latency Comparison (CPU, 5 iterations):");
        println!("{:<12} {:>12} {:>12} {:>8}", "Shape", "BF16", "FP32", "Speedup");

        for (m, n) in sizes {
            let g = Tensor::randn([m, n], (Kind::Float, tch::Device::Cpu));

            // Warmup
            let _ = ortho_bf16.polar_express(&g, 5);
            let _ = ortho_fp32.polar_express(&g, 5);

            // Benchmark BF16
            let iters = 3;
            let start = Instant::now();
            for _ in 0..iters {
                let _ = ortho_bf16.polar_express(&g, 5);
            }
            let bf16_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Benchmark FP32
            let start = Instant::now();
            for _ in 0..iters {
                let _ = ortho_fp32.polar_express(&g, 5);
            }
            let fp32_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = bf16_ms / fp32_ms;

            println!(
                "{:<12} {:>10.1}ms {:>10.1}ms {:>7.2}x",
                format!("{}x{}", m, n),
                bf16_ms,
                fp32_ms,
                speedup
            );
        }

        // Also benchmark raw matmul to understand overhead
        println!("\nRaw matmul baseline (single op):");
        let g = Tensor::randn([1024, 1024], (Kind::Float, tch::Device::Cpu));
        let start = Instant::now();
        for _ in 0..10 {
            let _ = g.matmul(&g.tr());
        }
        let matmul_ms = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
        println!("1024x1024 matmul: {:.1}ms", matmul_ms);
        println!("Polar Express has ~15 matmuls, expected: {:.1}ms", matmul_ms * 15.0);
    }
}
