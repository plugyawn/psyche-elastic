//! Polar Express: Fast Orthogonalization for Muon Optimizer
//!
//! Implementation of the Polar Express Sign Method from:
//! "Polar Express: High Performance Sign Methods for Low Precision Training"
//! by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower
//! https://arxiv.org/pdf/2505.16932
//!
//! This replaces Newton-Schulz iteration with a faster, more stable method.

use tch::{Kind, Tensor};

/// Pre-computed coefficients for 5 iterations of Polar Express
/// Computed for: num_iters=5, safety_factor=2e-2, cushion=2
const POLAR_COEFFS: [(f64, f64, f64); 5] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];

/// Safety factor for spectral norm estimation (1 + safety_factor)
const SAFETY_FACTOR: f64 = 0.02;

/// Small epsilon for numerical stability
const EPS: f64 = 1e-6;

/// Compute the Polar Express orthogonalization of a matrix.
///
/// Given a matrix G, finds the nearest orthogonal matrix U such that
/// U @ U.T ≈ I (for tall matrices) or U.T @ U ≈ I (for wide matrices).
///
/// # Arguments
/// * `g` - Input gradient/momentum tensor (2D)
/// * `iterations` - Number of Polar Express iterations (default: 5)
///
/// # Returns
/// Orthogonalized tensor with same shape as input
pub fn polar_express(g: &Tensor, iterations: usize) -> Tensor {
    assert!(g.dim() == 2, "polar_express requires 2D tensor, got {}D", g.dim());

    let (rows, cols) = (g.size()[0], g.size()[1]);
    let device = g.device();

    // Work in BFloat16 for speed (matches modded-nanogpt)
    let mut x = g.to_kind(Kind::BFloat16);

    // For tall matrices, transpose to work with wide matrices
    // (more efficient for the X @ X.T computation)
    let transposed = rows > cols;
    if transposed {
        x = x.tr();
    }

    // Normalize to ensure spectral norm is at most 1
    let norm = x.norm().double_value(&[]);
    x = &x / (norm * (1.0 + SAFETY_FACTOR) + EPS);

    // Apply Polar Express iterations
    let num_iters = iterations.min(POLAR_COEFFS.len());
    for i in 0..num_iters {
        let (a, b, c) = POLAR_COEFFS[i];

        // A = X @ X.T (symmetric matrix multiply)
        let xxt = x.matmul(&x.tr());

        // B = b * A + c * A @ A
        let b_mat = &xxt * b + xxt.matmul(&xxt) * c;

        // X = a * X + B @ X
        x = &x * a + b_mat.matmul(&x);
    }

    // Transpose back if needed
    if transposed {
        x = x.tr();
    }

    // Convert back to original dtype
    x.to_kind(g.kind()).to_device(device)
}

/// Compute polar_express for a batch of matrices.
///
/// # Arguments
/// * `g` - Input tensor of shape (batch, rows, cols)
/// * `iterations` - Number of iterations (default: 5)
///
/// # Returns
/// Orthogonalized tensor with same shape
pub fn polar_express_batched(g: &Tensor, iterations: usize) -> Tensor {
    assert!(g.dim() == 3, "polar_express_batched requires 3D tensor");

    let batch_size = g.size()[0];
    let results: Vec<Tensor> = (0..batch_size)
        .map(|i| polar_express(&g.get(i), iterations))
        .collect();

    Tensor::stack(&results, 0)
}

/// Check how close a matrix is to being orthogonal.
///
/// Returns the Frobenius norm of (X @ X.T - I) for wide matrices
/// or (X.T @ X - I) for tall matrices.
pub fn orthogonality_error(x: &Tensor) -> f64 {
    assert!(x.dim() == 2);
    let (rows, cols) = (x.size()[0], x.size()[1]);

    let x_float = x.to_kind(Kind::Float);

    let (product, size) = if rows <= cols {
        // Wide or square: check X @ X.T
        (x_float.matmul(&x_float.tr()), rows)
    } else {
        // Tall: check X.T @ X
        (x_float.tr().matmul(&x_float), cols)
    };

    let identity = Tensor::eye(size, (Kind::Float, x.device()));
    let diff = product - identity;

    diff.norm().double_value(&[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_polar_express_square() {
        let device = Device::Cpu;

        // Random square matrix
        let g = Tensor::randn([64, 64], (Kind::Float, device));

        let u = polar_express(&g, 5);

        // Check shape preserved
        assert_eq!(u.size(), g.size());

        // Check approximate orthogonality
        let error = orthogonality_error(&u);
        println!("Square matrix orthogonality error: {}", error);
        assert!(error < 0.1, "Expected orthogonality error < 0.1, got {}", error);
    }

    #[test]
    fn test_polar_express_tall() {
        let device = Device::Cpu;

        // Tall matrix (more rows than cols)
        let g = Tensor::randn([128, 64], (Kind::Float, device));

        let u = polar_express(&g, 5);

        assert_eq!(u.size(), g.size());

        let error = orthogonality_error(&u);
        println!("Tall matrix orthogonality error: {}", error);
        assert!(error < 0.1, "Expected orthogonality error < 0.1, got {}", error);
    }

    #[test]
    fn test_polar_express_wide() {
        let device = Device::Cpu;

        // Wide matrix (more cols than rows)
        let g = Tensor::randn([64, 128], (Kind::Float, device));

        let u = polar_express(&g, 5);

        assert_eq!(u.size(), g.size());

        let error = orthogonality_error(&u);
        println!("Wide matrix orthogonality error: {}", error);
        assert!(error < 0.1, "Expected orthogonality error < 0.1, got {}", error);
    }

    #[test]
    fn test_polar_express_iterations() {
        let device = Device::Cpu;
        let g = Tensor::randn([64, 64], (Kind::Float, device));

        // More iterations should give better orthogonality
        let u1 = polar_express(&g, 1);
        let u3 = polar_express(&g, 3);
        let u5 = polar_express(&g, 5);

        let e1 = orthogonality_error(&u1);
        let e3 = orthogonality_error(&u3);
        let e5 = orthogonality_error(&u5);

        println!("Orthogonality errors: 1 iter={}, 3 iter={}, 5 iter={}", e1, e3, e5);
        assert!(e3 <= e1 + 0.01, "3 iterations should be better than 1");
        assert!(e5 <= e3 + 0.01, "5 iterations should be better than 3");
    }

    #[test]
    fn test_polar_express_preserves_device() {
        let device = Device::Cpu;
        let g = Tensor::randn([32, 32], (Kind::Float, device));

        let u = polar_express(&g, 5);

        assert_eq!(u.device(), device);
    }

    #[test]
    fn test_polar_express_deterministic() {
        let device = Device::Cpu;
        let g = Tensor::randn([32, 32], (Kind::Float, device));

        let u1 = polar_express(&g, 5);
        let u2 = polar_express(&g, 5);

        let diff = (&u1 - &u2).abs().max().double_value(&[]);
        assert!(diff < 1e-5, "polar_express should be deterministic");
    }
}
