use crate::{
    CausalLM, StableVariableIterator, Variable,
    optimizer::{
        DistroAggregateMode, DistroApplyMode, DistroDilocoLiteConfig, DistroRawConfig,
        DistroRawMissingSidecarPolicy, DistroRawNormMode, DistroValueMode,
    },
};

use std::{
    cmp::Ordering,
    collections::HashMap,
    f64::consts::PI,
    sync::{OnceLock, atomic::AtomicU32, atomic::Ordering as AtomicOrdering},
};
use tch::{COptimizer, Device, Kind, Tensor};
use tracing::{info, warn};

fn cosine_mixer_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let v = std::env::var("PSYCHE_DISTRO_COSINE_MIXER")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "y" | "on")
    })
}

fn cosine_mixer_shadow_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let v = std::env::var("PSYCHE_DISTRO_COSINE_MIXER_SHADOW")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "y" | "on")
    })
}

fn sign_flip_sample_elems() -> i64 {
    static SAMPLE: OnceLock<i64> = OnceLock::new();
    *SAMPLE.get_or_init(|| {
        std::env::var("PSYCHE_DISTRO_SIGN_FLIP_SAMPLE_ELEMS")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .filter(|v| *v > 0)
            // Cheap, stable default for per-step diagnostics.
            .unwrap_or(4096)
    })
}

fn raw_align_sign_scale_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var("PSYCHE_DISTRO_RAW_ALIGN_SIGN_SCALE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if raw.is_empty() {
            // Raw mode otherwise under-updates badly vs sign at the same LR.
            true
        } else {
            matches!(raw.as_str(), "1" | "true" | "yes" | "y" | "on")
        }
    })
}

fn raw_align_sign_scale_multiplier() -> f64 {
    static MULTIPLIER: OnceLock<f64> = OnceLock::new();
    *MULTIPLIER.get_or_init(|| {
        let raw = std::env::var("PSYCHE_DISTRO_RAW_ALIGN_SIGN_SCALE_MULTIPLIER")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0);
        raw.unwrap_or(1.0)
    })
}

fn raw_align_sign_scale_max() -> f64 {
    static MAX_SCALE: OnceLock<f64> = OnceLock::new();
    *MAX_SCALE.get_or_init(|| {
        std::env::var("PSYCHE_DISTRO_RAW_ALIGN_SIGN_SCALE_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            // Guardrail against exploding tiny-norm tensors in raw mode.
            // Tuned to avoid hard under-updates while still capping pathological scales.
            .unwrap_or(1.0e7)
    })
}

fn raw_align_abs_clip() -> f64 {
    static ABS_CLIP: OnceLock<f64> = OnceLock::new();
    *ABS_CLIP.get_or_init(|| {
        std::env::var("PSYCHE_DISTRO_RAW_ALIGN_ABS_CLIP")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            // Keep magnitude info, but suppress outlier coordinates that sign() would saturate.
            // Slightly looser than sign to preserve useful magnitude detail.
            .unwrap_or(4.0)
    })
}

fn modular_geometry_align_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let v = std::env::var("PSYCHE_DISTRO_MATFORMER_GEOMETRY_ALIGN")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "y" | "on")
    })
}

fn modular_geometry_scale_power() -> f64 {
    static POWER: OnceLock<f64> = OnceLock::new();
    *POWER.get_or_init(|| {
        std::env::var("PSYCHE_DISTRO_MATFORMER_GEOMETRY_SCALE_POWER")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.5)
    })
}

fn modular_geometry_transport_scale(
    name: &str,
    full_shape: &[i64],
    active_shape: &[i64],
) -> Option<f64> {
    if !modular_geometry_align_enabled() {
        return None;
    }

    let prefix_dim = matformer_prefix_dim(name)?;
    if full_shape.len() != 2 || active_shape.len() != 2 {
        return None;
    }
    let full_len = full_shape[prefix_dim] as f64;
    let active_len = active_shape.get(prefix_dim).copied().unwrap_or(0) as f64;
    if !(full_len > 0.0) || !(active_len > 0.0) || !full_len.is_finite() || !active_len.is_finite()
    {
        return None;
    }

    let ratio = (active_len / full_len).min(1.0).max(0.0);
    if ratio <= 0.0 {
        return None;
    }

    let power = modular_geometry_scale_power();
    let base_scale = ratio.powf(power);
    if !base_scale.is_finite() {
        return None;
    }

    // For W1-type matrices (gate/up, rows) we transport from width m_t to m_f
    // by √(m_t / m_f); for W2 (down) we need the inverse mapping.
    if prefix_dim == 0 {
        Some(base_scale)
    } else {
        Some(1.0 / base_scale)
    }
}

static LAST_APPLY_INFO_STEP: AtomicU32 = AtomicU32::new(u32::MAX);
static RAW_HEAVY_CLIP_STREAK: AtomicU32 = AtomicU32::new(0);

fn sign_flip_stats_sampled(
    combined: &Tensor,
    reference: &Tensor,
    prefix_dim: i64,
    shared_len: i64,
    sample_elems: i64,
) -> Option<(f64, f64)> {
    if shared_len <= 0 {
        return None;
    }
    let combined = combined
        .narrow(prefix_dim, 0, shared_len)
        .to_kind(Kind::Float)
        .contiguous()
        .view([-1]);
    let reference = reference
        .narrow(prefix_dim, 0, shared_len)
        .to_kind(Kind::Float)
        .contiguous()
        .view([-1]);
    let n = combined.numel().max(0) as i64;
    if n <= 0 {
        return None;
    }
    let n_sample = if sample_elems > 0 {
        n.min(sample_elems)
    } else {
        n
    };
    let combined = if n_sample < n {
        combined.narrow(0, 0, n_sample)
    } else {
        combined
    };
    let reference = if n_sample < n {
        reference.narrow(0, 0, n_sample)
    } else {
        reference
    };

    // In sign-mode, only sign changes matter. Count sign mismatches (including 0 vs +/- 1).
    let diff = (&combined.sign() - &reference.sign()).ne(0);
    let flips = diff.to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
    Some((flips, n_sample as f64))
}

fn diloco_peer_metadata(
    results: &[Vec<DistroResult>],
    parameter_index: usize,
) -> Vec<DistroPeerMetadata> {
    results
        .iter()
        .map(|peer| {
            peer.get(parameter_index)
                .and_then(|r| r.peer_metadata)
                .unwrap_or_else(|| DistroPeerMetadata {
                    inner_steps_used: 1,
                    sum_local_lr: 1.0,
                    tokens_processed: 1,
                    delta_l2_preclip: 0.0,
                    delta_l2_postclip: 0.0,
                })
        })
        .collect()
}

fn diloco_peer_weights(
    metas: &[DistroPeerMetadata],
    tier_weight_cap: f64,
) -> (Vec<f64>, f64, f64, f64) {
    if metas.is_empty() {
        return (Vec::new(), 0.0, 0.0, 0.0);
    }

    let raw_weights = metas
        .iter()
        .map(|meta| {
            let tokens = meta.tokens_processed.max(1) as f64;
            let inner_steps = meta.inner_steps_used.max(1) as f64;
            let lr_sum = if meta.sum_local_lr.is_finite() && meta.sum_local_lr > 1.0e-8 {
                meta.sum_local_lr as f64
            } else {
                inner_steps
            };
            tokens / lr_sum.max(1.0e-8)
        })
        .collect::<Vec<_>>();

    let sum = raw_weights.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        let uniform = vec![1.0; raw_weights.len()];
        return (uniform, 1.0, 1.0, 1.0);
    }

    // Normalize to mean-1 with an explicit upper bound projection so the cap is truly enforced.
    let n = raw_weights.len();
    let cap = tier_weight_cap.max(1.0);
    let mut normalized = vec![0.0f64; n];
    let mut capped = vec![false; n];
    let mut active: Vec<usize> = (0..n).collect();
    let mut remaining_sum = n as f64;

    loop {
        if active.is_empty() {
            break;
        }

        let active_raw_sum = active.iter().map(|&i| raw_weights[i]).sum::<f64>();
        if !active_raw_sum.is_finite() || active_raw_sum <= 0.0 {
            let fill = (remaining_sum / active.len() as f64).max(0.0);
            for &i in &active {
                normalized[i] = fill;
            }
            break;
        }

        let scale = remaining_sum / active_raw_sum;
        let mut newly_capped = Vec::new();
        for &i in &active {
            let candidate = raw_weights[i] * scale;
            if candidate > cap {
                newly_capped.push(i);
            }
        }

        if newly_capped.is_empty() {
            for &i in &active {
                normalized[i] = (raw_weights[i] * scale).max(0.0);
            }
            break;
        }

        for i in newly_capped {
            if !capped[i] {
                capped[i] = true;
                normalized[i] = cap;
                remaining_sum = (remaining_sum - cap).max(0.0);
            }
        }
        active.retain(|&i| !capped[i]);
    }

    let normalized_sum = normalized.iter().sum::<f64>();
    if !normalized_sum.is_finite() || normalized_sum <= 0.0 {
        let uniform = vec![1.0; n];
        return (uniform, 1.0, 1.0, 1.0);
    }

    let mean = normalized_sum / n as f64;
    let min = normalized
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let max = normalized
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    (normalized, mean, min, max)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PrefixAlignError {
    UnsupportedParameter(String),
    RankMismatch { expected: usize, got: usize },
    ShapeMismatch { full: Vec<i64>, grad: Vec<i64> },
}

fn matformer_prefix_dim(name: &str) -> Option<usize> {
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        Some(0)
    } else if name.ends_with("down_proj.weight") {
        Some(1)
    } else {
        None
    }
}

/// Extract layer index from parameter name like "model.layers.5.mlp.gate_proj.weight"
// TODO: Use this when helper-mode sparse indices are wired (map params -> layer for index lookup).
// TODO: Remove if helper-mode sparse indices are dropped or moved elsewhere.
#[allow(dead_code)]
fn extract_layer_index(name: &str) -> Option<usize> {
    // Look for "layers.N" pattern
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

// TODO: Wire this into DisTrO apply once helper-mode indices are transmitted or computed.
// TODO: Plumb helper indices through DistroResult or compute via helper config + layer index.
/// Align gradient with non-contiguous MatFormer helper indices.
///
/// This function scatters a gradient from a subset of indices back to the full shape.
/// Used when helper mode is enabled and gradients cover prefix + stochastic suffix indices.
///
/// # Arguments
/// * `name` - Parameter name (e.g., "model.layers.0.mlp.gate_proj.weight")
/// * `full_shape` - Full shape of the parameter [intermediate_size, hidden_size] or [hidden_size, intermediate_size]
/// * `grad` - Gradient tensor covering the selected indices
/// * `indices` - Which indices the gradient covers (e.g., [0,1,2,...,767] + [891,1024,...])
///
/// # Returns
/// Full-sized gradient tensor with values scattered to the correct positions
#[allow(dead_code)]
pub fn align_matformer_sparse_grad(
    name: &str,
    full_shape: &[i64],
    grad: Tensor,
    indices: &[i64],
) -> Result<Tensor, PrefixAlignError> {
    let grad_shape = grad.size();

    // If already full size, return as-is
    if grad_shape == full_shape {
        return Ok(grad);
    }

    let Some(prefix_dim) = matformer_prefix_dim(name) else {
        return Err(PrefixAlignError::UnsupportedParameter(name.to_string()));
    };

    if full_shape.len() != 2 || grad_shape.len() != 2 {
        return Err(PrefixAlignError::RankMismatch {
            expected: 2,
            got: grad_shape.len(),
        });
    }

    // Verify the other dimension matches
    let other_dim = if prefix_dim == 0 { 1 } else { 0 };
    if grad_shape[other_dim] != full_shape[other_dim] {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    // Verify grad and indices have compatible sizes
    if grad_shape[prefix_dim] != indices.len() as i64 {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    // Create full-size zero tensor
    let mut full_grad = Tensor::zeros(full_shape, (grad.kind(), grad.device()));

    // Create index tensor
    let indices_tensor = Tensor::from_slice(indices).to_device(grad.device());

    // Scatter gradient values to their correct positions
    if prefix_dim == 0 {
        // gate_proj, up_proj: indices select rows
        let _ = full_grad.index_copy_(0, &indices_tensor, &grad);
    } else {
        // down_proj: indices select columns
        let _ = full_grad.index_copy_(1, &indices_tensor, &grad);
    }

    Ok(full_grad)
}

fn align_matformer_prefix_grad(
    name: &str,
    full_shape: &[i64],
    grad: Tensor,
) -> Result<Tensor, PrefixAlignError> {
    let grad_shape = grad.size();
    if grad_shape == full_shape {
        return Ok(grad);
    }

    let Some(prefix_dim) = matformer_prefix_dim(name) else {
        return Err(PrefixAlignError::UnsupportedParameter(name.to_string()));
    };

    if full_shape.len() != grad_shape.len() {
        return Err(PrefixAlignError::RankMismatch {
            expected: full_shape.len(),
            got: grad_shape.len(),
        });
    }
    if full_shape.len() != 2 {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    let other_dim = if prefix_dim == 0 { 1 } else { 0 };
    if grad_shape[other_dim] != full_shape[other_dim] {
        return Err(PrefixAlignError::ShapeMismatch {
            full: full_shape.to_vec(),
            grad: grad_shape,
        });
    }

    let prefix_len = full_shape[prefix_dim];
    let grad_len = grad_shape[prefix_dim];
    if grad_len == prefix_len {
        return Ok(grad);
    }

    if grad_len > prefix_len {
        Ok(grad.narrow(prefix_dim as i64, 0, prefix_len))
    } else {
        let expanded = Tensor::zeros(full_shape, (grad.kind(), grad.device()));
        let mut prefix_view = expanded.narrow(prefix_dim as i64, 0, grad_len);
        prefix_view.copy_(&grad);
        Ok(expanded)
    }
}

fn scale_matformer_prefix_grad_for_geometry(
    grad: &mut Tensor,
    name: &str,
    full_shape: &[i64],
    active_shape: &[i64],
    telemetry_sum: &mut f64,
    telemetry_min: &mut f64,
    telemetry_max: &mut f64,
    telemetry_count: &mut u64,
) {
    let Some(scale) = modular_geometry_transport_scale(name, full_shape, active_shape) else {
        return;
    };
    if !scale.is_finite() || scale <= 0.0 {
        return;
    }
    if (scale - 1.0).abs() > f64::EPSILON {
        let _ = grad.g_mul_scalar_(scale);
    }
    *telemetry_sum += scale;
    *telemetry_min = telemetry_min.min(scale);
    *telemetry_max = telemetry_max.max(scale);
    *telemetry_count += 1;
}

pub struct TransformDCT {
    shape_dict: HashMap<i64, i64>,
    f_dict: HashMap<i64, Tensor>,
    b_dict: HashMap<i64, Tensor>,
}

impl TransformDCT {
    pub fn new(variables: StableVariableIterator, target_chunk: i64) -> Self {
        let _no_grad = tch::no_grad_guard();
        let mut shape_dict = HashMap::new();
        let mut f_dict = HashMap::new();
        let mut b_dict = HashMap::new();

        // Get all variants of model tensor sizes
        // Generate all possible valid DCT sizes for model tensors
        for variable in variables {
            let size = variable.full_tensor_shape();
            let variable = variable.local_tensor();
            for s in size {
                // Get the closest smallest divisor to the targeted DCT size
                let sc = match shape_dict.get(&s) {
                    Some(sc) => *sc,
                    None => {
                        let sc = Self::get_smaller_split(s, target_chunk);
                        shape_dict.insert(s, sc);
                        sc
                    }
                };

                // Pregenerate DCT basis matrices
                if let std::collections::hash_map::Entry::Vacant(e) = f_dict.entry(sc) {
                    let i = Tensor::eye(sc, (Kind::Float, variable.device()));
                    e.insert(
                        Self::dct(&i, true)
                            .to_kind(variable.kind())
                            .to(variable.device()),
                    );
                    b_dict.insert(
                        sc,
                        Self::idct(&i, true)
                            .to_kind(variable.kind())
                            .to(variable.device()),
                    );
                }
            }
        }
        Self {
            shape_dict,
            f_dict,
            b_dict,
        }
    }

    fn get_prime_divisors(mut n: i64) -> Vec<i64> {
        if n == 0 {
            return Vec::new();
        }
        let mut divisors = Vec::new();
        while n % 2 == 0 {
            divisors.push(2);
            n /= 2;
        }
        while n % 3 == 0 {
            divisors.push(3);
            n /= 3;
        }
        let mut i = 5;
        while i * i <= n {
            for k in [i, i + 2].iter() {
                while n % k == 0 {
                    divisors.push(*k);
                    n /= k;
                }
            }
            i += 6;
        }
        if n > 1 {
            divisors.push(n);
        }
        divisors
    }

    fn get_divisors(n: i64) -> Vec<i64> {
        let mut divisors = Vec::new();
        match n.cmp(&1) {
            Ordering::Equal => {
                divisors.push(1);
            }
            Ordering::Greater => {
                let prime_factors = Self::get_prime_divisors(n);
                divisors = vec![1];
                let mut last_prime = 0;
                let mut factor = 0;
                let mut slice_len = 0;
                // Find all the products that are divisors of n
                for prime in prime_factors {
                    if last_prime != prime {
                        slice_len = divisors.len();
                        factor = prime;
                    } else {
                        factor *= prime;
                    }
                    for i in 0..slice_len {
                        divisors.push(divisors[i] * factor);
                    }
                    last_prime = prime;
                }
                divisors.sort_unstable();
            }
            Ordering::Less => {}
        }
        divisors
    }

    fn get_smaller_split(n: i64, close_to: i64) -> i64 {
        let all_divisors = Self::get_divisors(n);
        for (ix, &val) in all_divisors.iter().enumerate() {
            if val == close_to {
                return val;
            }
            if val > close_to {
                if ix == 0 {
                    return val;
                }
                return all_divisors[ix - 1];
            }
        }
        n
    }

    fn dct_fft_impl(v: &Tensor) -> Tensor {
        v.fft_fft(None, 1, "backward").view_as_real()
    }

    #[allow(unused)]
    fn dct(x: &Tensor, ortho: bool) -> Tensor {
        let x_shape = x.size();
        let n = { *x_shape.last().unwrap() };
        let x = x.contiguous().view([-1, n]);

        let v = Tensor::cat(
            &[x.slice(1, 0, None, 2), x.slice(1, 1, None, 2).flip([1])],
            1,
        );

        let vc = Self::dct_fft_impl(&v);

        let k = -Tensor::arange(n, (Kind::Float, x.device()))
            .unsqueeze(0)
            .g_mul_scalar(PI / (2.0 * n as f64));
        let w_r = k.cos();
        let w_i = k.sin();

        let mut v = vc.select(2, 0) * &w_r - vc.select(2, 1) * &w_i;

        if ortho {
            v.select(1, 0).g_div_scalar_((n as f64).sqrt() * 2.0);
            v.slice(1, 1, None, 1)
                .g_div_scalar_((n as f64 / 2.0).sqrt() * 2.0);
        }

        v.g_mul_scalar_(2.0).view(x_shape.as_slice())
    }

    fn idct_irfft_impl(v: &Tensor) -> Tensor {
        let complex_v = v.view_as_complex();
        let n = v.size()[1];
        complex_v.fft_irfft(Some(n), 1, "backward")
    }

    #[allow(unused)]
    fn idct(x: &Tensor, ortho: bool) -> Tensor {
        let x_shape = x.size();
        let n = { *x_shape.last().unwrap() };

        let mut x_v = x.contiguous().view([-1, n]).f_div_scalar(2.0).unwrap();

        if ortho {
            x_v.slice(1, 0, 1, 1)
                .f_mul_scalar_((n as f64).sqrt() * 2.0)
                .unwrap();
            x_v.slice(1, 1, n, 1)
                .f_mul_scalar_((n as f64 / 2.0).sqrt() * 2.0)
                .unwrap();
        }

        let k = Tensor::arange(n, (Kind::Float, x.device()))
            .f_mul_scalar(PI / (2.0 * n as f64))
            .unwrap()
            .unsqueeze(0);

        let w_r = k.cos();
        let w_i = k.sin();

        let v_t_r = &x_v;
        let v_t_i = Tensor::cat(
            &[
                x_v.slice(1, 0, 1, 1).f_mul_scalar(0.0).unwrap(),
                x_v.flip([1]).slice(1, 0, n - 1, 1).f_neg().unwrap(),
            ],
            1,
        );

        let v_r = v_t_r.f_mul(&w_r).unwrap() - v_t_i.f_mul(&w_i).unwrap();
        let v_i = v_t_r.f_mul(&w_i).unwrap() + v_t_i.f_mul(&w_r).unwrap();

        let v = Tensor::cat(&[v_r.unsqueeze(2), v_i.unsqueeze(2)], 2);

        let v = Self::idct_irfft_impl(&v);

        let mut x = Tensor::zeros(v.size(), (Kind::Float, v.device()));

        x.slice(1, 0, n, 2)
            .f_add_(&v.slice(1, 0, n - (n / 2), 1))
            .unwrap();
        x.slice(1, 1, n, 2)
            .f_add_(&v.flip([1]).slice(1, 0, n / 2, 1))
            .unwrap();

        x.view(x_shape.as_slice())
    }

    fn einsum_2d(x: &Tensor, b: &Tensor, d: Option<&Tensor>) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        match d {
            None => Tensor::einsum("...ij, jb -> ...ib", &[x, b], None::<i64>),
            Some(d_tensor) => {
                // Note: b-c axis output is transposed to chunk DCT in 2D
                Tensor::einsum("...ijkl, jb, ld -> ...ikbd", &[x, b, d_tensor], None::<i64>)
            }
        }
    }

    fn einsum_2d_t(x: &Tensor, b: &Tensor, d: Option<&Tensor>) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        match d {
            None => Tensor::einsum("...ij, jb -> ...ib", &[x, b], None::<i64>),
            Some(d_tensor) => {
                // Note: b-c axis output is transposed to chunk DCT in 2D
                Tensor::einsum("...ijkl, kb, ld -> ...ibjd", &[x, b, d_tensor], None::<i64>)
            }
        }
    }

    pub fn encode(&mut self, x: &Tensor) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        if x.size().len() > 1 {
            // 2D weights
            let n1 = *self.shape_dict.get(&x.size()[0]).unwrap();
            let n2 = *self.shape_dict.get(&x.size()[1]).unwrap();
            let n1w = self.f_dict.get(&n1).unwrap().to_device(x.device());
            let n2w = self.f_dict.get(&n2).unwrap().to_device(x.device());
            self.f_dict.insert(n1, n1w.copy());
            self.f_dict.insert(n2, n2w.copy());

            // Equivalent to rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            let x = x.view([x.size()[0] / n1, n1, x.size()[1] / n2, n2]);
            Self::einsum_2d(&x, &n1w, Some(&n2w))
        } else {
            // 1D weights
            let n1 = *self.shape_dict.get(&x.size()[0]).unwrap();
            let n1w = self.f_dict.get(&n1).unwrap().to_device(x.device());
            self.f_dict.insert(n1, n1w.copy());

            // Equivalent to rearrange(x, "(x w) -> x w", w=n1)
            let x = x.view([-1, n1]);
            Self::einsum_2d(&x, &n1w, None)
        }
    }

    pub fn decode(&mut self, x: &Tensor) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x_shape = x.size();

        if x_shape.len() > 2 {
            // 2D weights
            let n1 = x_shape[2];
            let n2 = x_shape[3];
            let device = x.device();

            if !self.b_dict.contains_key(&n1) {
                let i = Tensor::eye(n1, (Kind::Float, device));
                self.b_dict
                    .insert(n1, Self::idct(&i, true).to_kind(x.kind()));
            }
            if !self.b_dict.contains_key(&n2) {
                let i = Tensor::eye(n2, (Kind::Float, device));
                self.b_dict
                    .insert(n2, Self::idct(&i, true).to_kind(x.kind()));
            }
            let n1w = self.b_dict.get(&n1).unwrap().to_device(device);
            let n2w = self.b_dict.get(&n2).unwrap().to_device(device);

            self.b_dict.insert(n1, n1w.copy());
            self.b_dict.insert(n2, n2w.copy());

            let x = Self::einsum_2d_t(x, &n1w, Some(&n2w));
            let x_shape = x.size();

            // Equivalent to rearrange(x, "y h x w -> (y h) (x w)")
            let (y, h, x_, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
            x.reshape([y * h, x_ * w])
        } else {
            // 1D weights
            let n1 = x_shape[1];
            let device = x.device();

            if !self.b_dict.contains_key(&n1) {
                let i = Tensor::eye(n1, (Kind::Float, device));
                self.b_dict
                    .insert(n1, Self::idct(&i, true).to_kind(x.kind()));
            }
            let n1w = self.b_dict.get(&n1).unwrap().to_device(device);
            self.b_dict.insert(n1, n1w.copy());

            let x = Self::einsum_2d_t(x, &n1w, None);
            let x_shape = x.size();

            // Equivalent to rearrange(x, "x w -> (x w)")
            let (x_, w) = (x_shape[0], x_shape[1]);
            x.reshape([x_ * w])
        }
    }
}

pub struct CompressDCT;

impl CompressDCT {
    fn clamp_topk(x: &Tensor, topk: i64) -> i64 {
        let last_dim = x.size()[x.dim() - 1];

        if topk > last_dim {
            last_dim
        } else if topk < 1 {
            1
        } else {
            topk
        }
    }

    pub fn compress(x: &Tensor, topk: i64) -> (Tensor, Tensor, Vec<i64>, i64) {
        let _no_grad = tch::no_grad_guard();
        let xshape = x.size();
        let x = if xshape.len() > 2 {
            // Equivalent to rearrange(x, "y x h w -> y x (h w)")
            let y = xshape[0];
            let x_dim = xshape[1];
            let h = xshape[2];
            let w = xshape[3];
            x.view([y, x_dim, h * w])
        } else {
            x.shallow_clone()
        };

        let totalk = *x.size().last().unwrap();
        let topk = Self::clamp_topk(&x, topk);

        let idx = x.abs().topk(topk, -1, true, false).1;
        let val = x.gather(-1, &idx, false);

        let idx = compress_idx(totalk, &idx);

        (idx, val, xshape, totalk)
    }

    #[allow(unused)]
    pub fn decompress(
        idx: &Tensor,
        val: &Tensor,
        xshape: &[i64],
        totalk: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let totalk = totalk.abs();

        let idx = decompress_idx(totalk, idx);

        let val = val.to_kind(kind);

        let mut x: Tensor = Tensor::zeros(xshape, (kind, device));

        if xshape.len() > 2 {
            // 2D weights
            // Equivalent to rearrange(x, "y x h w -> y x (h w)")
            let y = xshape[0];
            let x_dim = xshape[1];
            let h = xshape[2];
            let w = xshape[3];
            x = x.view([y, x_dim, h * w]);
        }

        x.internal_scatter_reduce_(-1, &idx, &val, "mean", false);

        x = x.reshape(xshape);

        if x.size().len() > 2 {
            // 2D weights
            // Equivalent to rearrange(x, "y x (h w) -> y x h w", h=xshape[2])
            let y = xshape[0];
            let x_dim = xshape[1];
            let h = xshape[2];
            let w = xshape[3];
            x = x.view([y, x_dim, h, w]);
        }

        x
    }

    pub fn batch_decompress(
        idx: &[Tensor],
        val: &[Tensor],
        xshape: &[i64],
        totalk: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let idx_concat = Tensor::cat(idx, -1).to_device(device);
        let val_concat = Tensor::cat(val, -1).to_device(device);
        // Call the decompress method
        Self::decompress(&idx_concat, &val_concat, xshape, totalk, kind, device)
    }
}

fn compress_idx(max_value: i64, idx: &Tensor) -> Tensor {
    if max_value <= 256 {
        idx.to_kind(Kind::Uint8)
    } else if max_value <= 65536 {
        idx.to_kind(Kind::UInt16).view_dtype(Kind::Uint8)
    } else if max_value <= 4294967296 {
        idx.to_kind(Kind::UInt32).view_dtype(Kind::Uint8)
    } else {
        idx.shallow_clone()
    }
}

fn decompress_idx(max_value: i64, idx: &Tensor) -> Tensor {
    if max_value <= 256 {
        idx.view_dtype(Kind::Uint8)
    } else if max_value <= 65536 {
        idx.view_dtype(Kind::UInt16)
    } else if max_value <= 4294967296 {
        idx.view_dtype(Kind::UInt32)
    } else {
        idx.shallow_clone()
    }
    .to_kind(Kind::Int64)
}

struct State {
    delta: Box<dyn Variable>,
    outer_momentum: Box<dyn Variable>,
}

#[derive(Debug)]
pub struct DistroResult {
    pub sparse_idx: Tensor,
    pub sparse_val: Tensor,
    pub xshape: Vec<i64>,
    pub totalk: i64,
    pub norm_sidecar: Option<DistroNormSidecar>,
    pub peer_metadata: Option<DistroPeerMetadata>,
    pub stats: Option<HashMap<String, f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistroNormSidecar {
    pub full_pre_l2: f32,
    pub full_pre_abs_mean: f32,
    pub numel: u32,
    pub nnz: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DistroPeerMetadata {
    pub inner_steps_used: u16,
    pub sum_local_lr: f32,
    pub tokens_processed: u32,
    pub delta_l2_preclip: f32,
    pub delta_l2_postclip: f32,
}

impl Clone for DistroResult {
    fn clone(&self) -> Self {
        Self {
            sparse_idx: self.sparse_idx.shallow_clone(),
            sparse_val: self.sparse_val.shallow_clone(),
            xshape: self.xshape.clone(),
            totalk: self.totalk,
            norm_sidecar: self.norm_sidecar,
            peer_metadata: self.peer_metadata,
            stats: self.stats.clone(),
        }
    }
}

pub struct Distro {
    sgd: COptimizer,
    compression_decay: f64,
    compression_topk: i64,
    weight_decay: f64,
    apply_mode: DistroApplyMode,
    aggregate_mode: DistroAggregateMode,
    value_mode: DistroValueMode,
    raw_config: DistroRawConfig,
    diloco_lite_config: DistroDilocoLiteConfig,
    state: Vec<State>,
    transform: TransformDCT,
}

impl Distro {
    pub fn new(
        vs: &dyn CausalLM,
        compression_decay: f64,
        compression_chunk: i64,
        compression_topk: i64,
        weight_decay: f64,
        apply_mode: DistroApplyMode,
        aggregate_mode: DistroAggregateMode,
        value_mode: DistroValueMode,
        raw_config: DistroRawConfig,
        diloco_lite_config: DistroDilocoLiteConfig,
    ) -> Self {
        let _no_grad = tch::no_grad_guard();
        let mut sgd = COptimizer::sgd(0.1, 0.0, 0.0, 0.0, false).unwrap();

        let mut state = Vec::new();
        for variable in vs.variables() {
            state.push(State {
                delta: variable.zeros_like(format!("{}.delta", variable.name())),
                outer_momentum: variable.zeros_like(format!("{}.outer_momentum", variable.name())),
            });

            let logical_tensor = variable.logical_tensor();
            sgd.add_parameters(&logical_tensor, 0).unwrap();
            variable.zero_grad();
        }

        let transform = TransformDCT::new(vs.variables(), compression_chunk);

        Self {
            sgd,
            compression_decay,
            compression_topk,
            weight_decay,
            apply_mode,
            aggregate_mode,
            value_mode,
            raw_config,
            diloco_lite_config,
            state,
            transform,
        }
    }

    fn lookahead_delta(delta: &Tensor, mode: DistroApplyMode) -> Tensor {
        match mode {
            DistroApplyMode::Sign => delta.sign(),
            DistroApplyMode::Raw => delta.shallow_clone(),
        }
    }

    pub fn generate(
        &mut self,
        variables: &dyn CausalLM,
        prev_self_results: &[Vec<DistroResult>],
        prev_lr: f64,
        lr: f64,
        stats: bool,
    ) -> Vec<DistroResult> {
        let _no_grad = tch::no_grad_guard();

        let mut ret = Vec::new();
        for (index, var) in variables.variables().enumerate() {
            let mut variable = var.logical_tensor();

            let grad_energy: Option<f64> = match stats {
                true => Some(
                    variable
                        .grad()
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                _ => None,
            };

            let delta_var = &mut self.state.get_mut(index).unwrap().delta;
            let mut delta = delta_var.logical_tensor();

            let lookahead = Self::lookahead_delta(&delta, self.apply_mode).multiply_scalar(prev_lr);
            let _t = variable.g_add_(&lookahead);

            if !prev_self_results.is_empty() {
                let device = variable.device();
                let indicies = prev_self_results
                    .iter()
                    .map(|x| x[index].sparse_idx.to_device(device))
                    .collect::<Vec<_>>();

                let val_kind: Kind = variable.kind();
                let values = prev_self_results
                    .iter()
                    .map(|x| {
                        let sparse_val = x[index].sparse_val.to_device(device);
                        if sparse_val.kind() == Kind::Bool {
                            Self::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                        } else {
                            sparse_val
                        }
                    })
                    .collect::<Vec<_>>();

                // Decode grad from all nodes
                let decompressed = CompressDCT::batch_decompress(
                    &indicies,
                    &values,
                    &prev_self_results[0][index].xshape,
                    prev_self_results[0][index].totalk,
                    val_kind,
                    device,
                );
                let transmit_grad = self.transform.decode(&decompressed);

                // Remove transmitted from delta
                let _t = delta.g_sub_(&var.shard_other_tensor_like_me(transmit_grad));
            }

            // weight decay
            if self.weight_decay != 0.0 {
                let _t = variable.g_mul_scalar_(1.0 - lr * self.weight_decay);
            }

            // decay delta
            if self.compression_decay != 1.0 {
                let _t = delta.g_mul_scalar_(self.compression_decay);
            }

            // add delta to new gradient
            let _t = delta.g_add_(&variable.grad().multiply_scalar(lr));

            // Compress delta
            let full_delta = delta_var.gather_full_tensor();
            let (sparse_idx, sparse_val, xshape, totalk) =
                CompressDCT::compress(&self.transform.encode(&full_delta), self.compression_topk);

            let delta_energy: Option<f64> = match stats {
                true => Some(
                    full_delta
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                false => None,
            };

            let norm_sidecar = if self.raw_config.enabled || self.value_mode == DistroValueMode::Raw
            {
                let numel = full_delta.numel();
                let nnz = sparse_val.numel();
                Some(DistroNormSidecar {
                    full_pre_l2: full_delta
                        .to_kind(Kind::Float)
                        .pow_tensor_scalar(2)
                        .sum(Kind::Float)
                        .sqrt()
                        .double_value(&[]) as f32,
                    full_pre_abs_mean: full_delta
                        .to_kind(Kind::Float)
                        .abs()
                        .mean(Kind::Float)
                        .double_value(&[]) as f32,
                    numel: (numel.min(u32::MAX as usize)) as u32,
                    nnz: (nnz.min(u32::MAX as usize)) as u32,
                })
            } else {
                None
            };

            ret.push(DistroResult {
                sparse_idx,
                sparse_val,
                xshape,
                totalk,
                norm_sidecar,
                peer_metadata: None,
                stats: match stats {
                    true => {
                        let name = var.name();
                        Some(HashMap::from([
                            (format!("{name}.delta_energy"), delta_energy.unwrap()),
                            (format!("{name}.grad_energy"), grad_energy.unwrap()),
                        ]))
                    }
                    false => None,
                },
            });
        }
        ret
    }

    pub fn apply(&mut self, vars: &dyn CausalLM, results: &[Vec<DistroResult>], lr: f64) {
        let _no_grad = tch::no_grad_guard();
        if results.is_empty() {
            return;
        }

        let use_cosine_mixer = cosine_mixer_enabled();
        let shadow_cosine_mixer = cosine_mixer_shadow_enabled();
        let want_cosine_stats = use_cosine_mixer || shadow_cosine_mixer;
        let want_sign_flip_stats = want_cosine_stats && self.apply_mode == DistroApplyMode::Sign;
        let sign_flip_sample = if want_sign_flip_stats {
            sign_flip_sample_elems()
        } else {
            0
        };
        let step = crate::matformer_c2::current_step();
        let mut mixer_cos_sum: f64 = 0.0;
        let mut mixer_cos_count: u64 = 0;
        let mut mixer_cos_min: f64 = 1.0;
        let mut mixer_cos_max: f64 = -1.0;
        let mut mixer_cos_dropped: u64 = 0;
        let mut telemetry_sign_flip_flips: f64 = 0.0;
        let mut telemetry_sign_flip_total: f64 = 0.0;
        let mut telemetry_sign_flip_params: u64 = 0;
        let mut telemetry_sign_flip_min: f64 = f64::INFINITY;
        let mut telemetry_sign_flip_max: f64 = 0.0;
        let mut telemetry_pre_grad_l2_sq: f64 = 0.0;
        let mut telemetry_pre_grad_linf: f64 = 0.0;
        let mut telemetry_post_grad_l2_sq: f64 = 0.0;
        let mut telemetry_post_grad_linf: f64 = 0.0;
        let mut telemetry_param_l2_sq: f64 = 0.0;
        let mut telemetry_nonfinite_skipped: u64 = 0;
        let mut telemetry_grad_tensors: u64 = 0;
        let raw_legacy_scale = if lr.abs() > 1.0e-12 {
            Some(1.0 / lr)
        } else {
            None
        };
        let raw_align_sign_scale = raw_align_sign_scale_enabled();
        let raw_align_scale_multiplier = raw_align_sign_scale_multiplier();
        let raw_align_scale_cap = raw_align_sign_scale_max();
        let raw_align_abs_clip = raw_align_abs_clip();
        let raw_v2_enabled = self.raw_config.enabled && self.apply_mode == DistroApplyMode::Raw;
        let raw_v2_match_pre_l2 =
            raw_v2_enabled && self.raw_config.norm_mode == DistroRawNormMode::MatchPreL2;
        let raw_v2_match_sign_equiv =
            raw_v2_enabled && self.raw_config.norm_mode == DistroRawNormMode::MatchSignEquivalent;
        let raw_v2_match_sign_equiv_nnz = raw_v2_enabled
            && self.raw_config.norm_mode == DistroRawNormMode::MatchSignEquivalentNnz;
        let raw_v2_match_target_l2 =
            raw_v2_match_pre_l2 || raw_v2_match_sign_equiv || raw_v2_match_sign_equiv_nnz;
        let mut telemetry_raw_align_scale_sum: f64 = 0.0;
        let mut telemetry_raw_align_scale_min: f64 = f64::INFINITY;
        let mut telemetry_raw_align_scale_max: f64 = 0.0;
        let mut telemetry_raw_align_scale_count: u64 = 0;
        let mut telemetry_raw_align_scale_clamped: u64 = 0;
        let mut telemetry_raw_clip_tensors: u64 = 0;
        let mut telemetry_raw_v2_missing_sidecar: u64 = 0;
        let mut telemetry_raw_v2_invalid_sidecar: u64 = 0;
        let mut telemetry_raw_v2_target_l2_sum: f64 = 0.0;
        let mut telemetry_raw_v2_target_support_sum: f64 = 0.0;
        let mut telemetry_raw_v2_obs_l2_sum: f64 = 0.0;
        let mut telemetry_raw_v2_target_l2_count: u64 = 0;
        let mut telemetry_raw_v2_target_to_obs_ratio_sum: f64 = 0.0;
        let mut telemetry_raw_preclip_l2_sq: f64 = 0.0;
        let mut telemetry_raw_preclip_linf: f64 = 0.0;
        let mut telemetry_raw_preclip_tensors: u64 = 0;
        let mut telemetry_modular_scale_sum: f64 = 0.0;
        let mut telemetry_modular_scale_min: f64 = f64::INFINITY;
        let mut telemetry_modular_scale_max: f64 = 0.0;
        let mut telemetry_modular_scale_count: u64 = 0;
        let diloco_lite_enabled = self.aggregate_mode == DistroAggregateMode::DilocoLite;
        let mut telemetry_diloco_weight_mean_sum: f64 = 0.0;
        let mut telemetry_diloco_weight_min: f64 = f64::INFINITY;
        let mut telemetry_diloco_weight_max: f64 = 0.0;
        let mut telemetry_diloco_weight_params: u64 = 0;
        let mut telemetry_diloco_trust_scale_sum: f64 = 0.0;
        let mut telemetry_diloco_trust_scale_min: f64 = f64::INFINITY;
        let mut telemetry_diloco_trust_scale_max: f64 = 0.0;
        let mut telemetry_diloco_trust_scale_count: u64 = 0;

        for (index, var) in vars.variables().enumerate() {
            let variable = var.logical_tensor();
            let device = variable.device();
            let full_shape = var.full_tensor_shape();
            let indicies = results
                .iter()
                .map(|x| x[index].sparse_idx.to_device(device))
                .collect::<Vec<_>>();

            let diloco_peer_weights = if diloco_lite_enabled {
                let metas = diloco_peer_metadata(results, index);
                let (weights, mean_w, min_w, max_w) =
                    diloco_peer_weights(&metas, self.diloco_lite_config.tier_weight_cap);
                if !weights.is_empty() {
                    telemetry_diloco_weight_mean_sum += mean_w;
                    telemetry_diloco_weight_min = telemetry_diloco_weight_min.min(min_w);
                    telemetry_diloco_weight_max = telemetry_diloco_weight_max.max(max_w);
                    telemetry_diloco_weight_params += 1;
                }
                Some(weights)
            } else {
                None
            };

            let val_kind: Kind = variable.kind();
            let values = results
                .iter()
                .enumerate()
                .map(|(peer_idx, x)| {
                    let sparse_val = x[index].sparse_val.to_device(device);
                    let mut value = if sparse_val.kind() == Kind::Bool {
                        Self::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                    } else {
                        sparse_val
                    };
                    if let Some(weights) = &diloco_peer_weights {
                        let w = weights.get(peer_idx).copied().unwrap_or(1.0);
                        if w.is_finite() && (w - 1.0).abs() > f64::EPSILON {
                            value = value.multiply_scalar(w);
                        }
                    }
                    value
                })
                .collect::<Vec<_>>();

            let mut raw_v2_target_l2: Option<f64> = None;
            let mut raw_v2_target_numel: Option<f64> = None;
            let mut raw_v2_target_support: Option<f64> = None;
            let mut raw_v2_target_clip_basis: Option<f64> = None;
            if raw_v2_enabled {
                let mut valid_sidecars = 0u64;
                let mut target_l2_sum = 0.0f64;
                let mut numel_sum = 0.0f64;
                let mut nnz_sum = 0.0f64;
                for peer_results in results {
                    match peer_results[index].norm_sidecar {
                        Some(sc)
                            if sc.full_pre_l2.is_finite()
                                && sc.full_pre_l2 >= 0.0
                                && sc.full_pre_abs_mean.is_finite()
                                && sc.numel > 0 =>
                        {
                            valid_sidecars += 1;
                            target_l2_sum += sc.full_pre_l2 as f64;
                            numel_sum += sc.numel as f64;
                            nnz_sum += sc.nnz.max(1) as f64;
                        }
                        Some(_) => {
                            telemetry_raw_v2_invalid_sidecar += 1;
                        }
                        None => {
                            telemetry_raw_v2_missing_sidecar += 1;
                        }
                    }
                }

                let numel_effective = if valid_sidecars > 0 {
                    numel_sum / valid_sidecars as f64
                } else {
                    variable.numel() as f64
                }
                .max(1.0);
                let nnz_effective = if valid_sidecars > 0 {
                    nnz_sum / valid_sidecars as f64
                } else {
                    numel_effective
                }
                .max(1.0);
                raw_v2_target_numel = Some(numel_effective);
                raw_v2_target_clip_basis = Some(numel_effective);

                if raw_v2_match_pre_l2 {
                    if valid_sidecars > 0 {
                        raw_v2_target_l2 = Some(target_l2_sum / valid_sidecars as f64);
                    } else if self.raw_config.missing_sidecar_policy
                        == DistroRawMissingSidecarPolicy::Fail
                    {
                        panic!(
                            "DisTrO raw-v2(match-pre-l2) requires valid norm sidecars, but none were available for parameter {} at step {}",
                            var.name(),
                            step
                        );
                    }
                } else if raw_v2_match_sign_equiv {
                    raw_v2_target_support = Some(numel_effective);
                    raw_v2_target_clip_basis = Some(numel_effective);
                    raw_v2_target_l2 =
                        Some(self.raw_config.sign_equiv_mult.max(0.0) * numel_effective.sqrt());
                } else if raw_v2_match_sign_equiv_nnz {
                    raw_v2_target_support = Some(nnz_effective);
                    raw_v2_target_clip_basis = Some(nnz_effective);
                    raw_v2_target_l2 =
                        Some(self.raw_config.sign_equiv_mult.max(0.0) * nnz_effective.sqrt());
                }
            }

            let same_shape = results.iter().all(|x| {
                x[index].xshape == results[0][index].xshape
                    && x[index].totalk == results[0][index].totalk
            });

            if same_shape {
                let prefix_dim = matformer_prefix_dim(var.name());
                let want_stats_for_param =
                    want_cosine_stats && prefix_dim.is_some() && results.len() > 1;
                let apply_mixer_for_param =
                    use_cosine_mixer && prefix_dim.is_some() && results.len() > 1;

                if apply_mixer_for_param {
                    // Same-shape MatFormer: decode per-peer so we can do a cosine-weighted mix.
                    // This is a soft guardrail to downweight negatively-aligned updates.
                    let prefix_dim = prefix_dim.expect("checked above") as i64;
                    let mut grads: Vec<Tensor> = Vec::new();
                    let mut grad_norms_sq: Vec<f64> = Vec::new();

                    for (peer_results, sparse_val) in results.iter().zip(values.iter()) {
                        let res = &peer_results[index];
                        let sparse_idx = res.sparse_idx.to_device(device);
                        let decompressed = CompressDCT::decompress(
                            &sparse_idx,
                            sparse_val,
                            &res.xshape,
                            res.totalk,
                            val_kind,
                            device,
                        );
                        let decoded = self.transform.decode(&decompressed);
                        let decoded_shape = decoded.size();
                        let mut aligned = match align_matformer_prefix_grad(
                            var.name(),
                            &full_shape,
                            decoded,
                        ) {
                            Ok(tensor) => tensor,
                            Err(err) => {
                                warn!(
                                    parameter = var.name(),
                                    full_shape = ?full_shape,
                                    "Skipping incompatible grad shape in DisTrO apply (cosine mixer): {err:?}"
                                );
                                continue;
                            }
                        };
                        scale_matformer_prefix_grad_for_geometry(
                            &mut aligned,
                            var.name(),
                            &full_shape,
                            &decoded_shape,
                            &mut telemetry_modular_scale_sum,
                            &mut telemetry_modular_scale_min,
                            &mut telemetry_modular_scale_max,
                            &mut telemetry_modular_scale_count,
                        );
                        let n_sq = aligned
                            .to_kind(Kind::Float)
                            .pow_tensor_scalar(2)
                            .sum(Kind::Float)
                            .double_value(&[]);
                        grads.push(aligned);
                        grad_norms_sq.push(n_sq);
                    }

                    if grads.is_empty() {
                        warn!(
                            parameter = var.name(),
                            "Skipping DisTrO apply: no compatible grads found (cosine mixer enabled)"
                        );
                        continue;
                    }
                    if grads.len() == 1 {
                        var.set_grad(grads.pop().expect("len==1"));
                    } else {
                        // Reference: largest-norm gradient. In MatFormer, this is typically the
                        // full-width tier, but we don't assume peer ordering.
                        let ref_idx = grad_norms_sq
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let ref_grad = grads[ref_idx].shallow_clone();

                        let mut combined = Tensor::zeros(&full_shape, (Kind::Float, device));
                        let mut weight_sum = Tensor::zeros([], (Kind::Float, device));

                        let len = full_shape[prefix_dim as usize].clamp(1, i64::MAX);
                        for (i, g) in grads.iter().enumerate() {
                            let a = g
                                .narrow(prefix_dim, 0, len)
                                .to_kind(Kind::Float)
                                .contiguous()
                                .view([-1]);
                            let b = ref_grad
                                .narrow(prefix_dim, 0, len)
                                .to_kind(Kind::Float)
                                .contiguous()
                                .view([-1]);

                            let dot = (&a * &b).sum(Kind::Float);
                            let norm_a = a.pow_tensor_scalar(2).sum(Kind::Float).sqrt();
                            let norm_b = b.pow_tensor_scalar(2).sum(Kind::Float).sqrt();
                            let cos = dot / (norm_a * norm_b + 1.0e-12);

                            let w = cos.clamp_min(0.0);
                            combined += g.to_kind(Kind::Float) * &w;
                            weight_sum += &w;

                            if i != ref_idx {
                                let cos_v = cos.double_value(&[]);
                                mixer_cos_sum += cos_v;
                                mixer_cos_count += 1;
                                mixer_cos_min = mixer_cos_min.min(cos_v);
                                mixer_cos_max = mixer_cos_max.max(cos_v);
                                if cos_v <= 0.0 {
                                    mixer_cos_dropped += 1;
                                }
                            }
                        }

                        let weight_sum = weight_sum.clamp_min(1.0e-6);
                        let normalized = (combined / weight_sum).to_kind(val_kind);
                        if want_sign_flip_stats {
                            if let Some((flips, total)) = sign_flip_stats_sampled(
                                &normalized,
                                &ref_grad,
                                prefix_dim,
                                len,
                                sign_flip_sample,
                            ) {
                                let frac = flips / total.max(1.0);
                                telemetry_sign_flip_flips += flips;
                                telemetry_sign_flip_total += total;
                                telemetry_sign_flip_params += 1;
                                telemetry_sign_flip_min = telemetry_sign_flip_min.min(frac);
                                telemetry_sign_flip_max = telemetry_sign_flip_max.max(frac);
                            }
                        }
                        var.set_grad(normalized);
                    }
                } else {
                    let mut sign_flip_ref_grad: Option<Tensor> = None;
                    let mut sign_flip_prefix_dim: i64 = 0;
                    let mut sign_flip_shared_len: i64 = 0;

                    if want_stats_for_param {
                        // Shadow-mode: compute cosine stats exactly as the mixer would, but do
                        // not change the aggregation behavior.
                        let prefix_dim = prefix_dim.expect("checked above") as i64;
                        let mut grads: Vec<Tensor> = Vec::new();
                        let mut grad_norms_sq: Vec<f64> = Vec::new();

                        for (peer_results, sparse_val) in results.iter().zip(values.iter()) {
                            let res = &peer_results[index];
                            let sparse_idx = res.sparse_idx.to_device(device);
                            let decompressed = CompressDCT::decompress(
                                &sparse_idx,
                                sparse_val,
                                &res.xshape,
                                res.totalk,
                                val_kind,
                                device,
                            );
                            let decoded = self.transform.decode(&decompressed);
                            let decoded_shape = decoded.size();
                            let mut aligned =
                                match align_matformer_prefix_grad(var.name(), &full_shape, decoded)
                                {
                                    Ok(tensor) => tensor,
                                    Err(_err) => {
                                        // Keep behavior unchanged; just skip stats for this param.
                                        continue;
                                    }
                                };
                            scale_matformer_prefix_grad_for_geometry(
                                &mut aligned,
                                var.name(),
                                &full_shape,
                                &decoded_shape,
                                &mut telemetry_modular_scale_sum,
                                &mut telemetry_modular_scale_min,
                                &mut telemetry_modular_scale_max,
                                &mut telemetry_modular_scale_count,
                            );
                            let n_sq = aligned
                                .to_kind(Kind::Float)
                                .pow_tensor_scalar(2)
                                .sum(Kind::Float)
                                .double_value(&[]);
                            grads.push(aligned);
                            grad_norms_sq.push(n_sq);
                        }

                        if grads.len() > 1 {
                            let ref_idx = grad_norms_sq
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                                .map(|(i, _)| i)
                                .unwrap_or(0);
                            let ref_grad = grads[ref_idx].shallow_clone();
                            let len = full_shape[prefix_dim as usize].clamp(1, i64::MAX);
                            if want_sign_flip_stats {
                                sign_flip_ref_grad = Some(ref_grad.shallow_clone());
                                sign_flip_prefix_dim = prefix_dim;
                                sign_flip_shared_len = len;
                            }
                            for (i, g) in grads.iter().enumerate() {
                                if i == ref_idx {
                                    continue;
                                }
                                let a = g
                                    .narrow(prefix_dim, 0, len)
                                    .to_kind(Kind::Float)
                                    .contiguous()
                                    .view([-1]);
                                let b = ref_grad
                                    .narrow(prefix_dim, 0, len)
                                    .to_kind(Kind::Float)
                                    .contiguous()
                                    .view([-1]);

                                let dot = (&a * &b).sum(Kind::Float);
                                let norm_a = a.pow_tensor_scalar(2).sum(Kind::Float).sqrt();
                                let norm_b = b.pow_tensor_scalar(2).sum(Kind::Float).sqrt();
                                let cos = dot / (norm_a * norm_b + 1.0e-12);

                                let cos_v = cos.double_value(&[]);
                                mixer_cos_sum += cos_v;
                                mixer_cos_count += 1;
                                mixer_cos_min = mixer_cos_min.min(cos_v);
                                mixer_cos_max = mixer_cos_max.max(cos_v);
                                if cos_v <= 0.0 {
                                    mixer_cos_dropped += 1;
                                }
                            }
                        }
                    }

                    // Decode grad from all nodes (fast path)
                    let decompressed = CompressDCT::batch_decompress(
                        &indicies,
                        &values,
                        &results[0][index].xshape,
                        results[0][index].totalk,
                        val_kind,
                        device,
                    );

                    let decoded = self.transform.decode(&decompressed);
                    let decoded_shape = decoded.size();
                    let mut aligned =
                        match align_matformer_prefix_grad(var.name(), &full_shape, decoded) {
                            Ok(tensor) => tensor,
                            Err(err) => {
                                warn!(
                                    parameter = var.name(),
                                    full_shape = ?full_shape,
                                    "Skipping incompatible grad shape in DisTrO apply: {err:?}"
                                );
                                continue;
                            }
                        };
                    scale_matformer_prefix_grad_for_geometry(
                        &mut aligned,
                        var.name(),
                        &full_shape,
                        &decoded_shape,
                        &mut telemetry_modular_scale_sum,
                        &mut telemetry_modular_scale_min,
                        &mut telemetry_modular_scale_max,
                        &mut telemetry_modular_scale_count,
                    );
                    if want_sign_flip_stats {
                        if let Some(ref_grad) = sign_flip_ref_grad.take() {
                            if let Some((flips, total)) = sign_flip_stats_sampled(
                                &aligned,
                                &ref_grad,
                                sign_flip_prefix_dim,
                                sign_flip_shared_len,
                                sign_flip_sample,
                            ) {
                                let frac = flips / total.max(1.0);
                                telemetry_sign_flip_flips += flips;
                                telemetry_sign_flip_total += total;
                                telemetry_sign_flip_params += 1;
                                telemetry_sign_flip_min = telemetry_sign_flip_min.min(frac);
                                telemetry_sign_flip_max = telemetry_sign_flip_max.max(frac);
                            }
                        }
                    }

                    // Set the gradients!!!
                    var.set_grad(aligned);
                }
            } else {
                // Heterogeneous shapes: decode individually, then align to local shape.
                let prefix_dim = matformer_prefix_dim(var.name());
                let want_stats_for_param =
                    want_cosine_stats && prefix_dim.is_some() && results.len() > 1;
                let apply_mixer_for_param =
                    use_cosine_mixer && prefix_dim.is_some() && results.len() > 1;

                if want_stats_for_param {
                    let prefix_dim = prefix_dim.expect("checked above") as i64;
                    let mut grads: Vec<(Tensor, i64)> = Vec::new();
                    for (peer_results, sparse_val) in results.iter().zip(values.iter()) {
                        let res = &peer_results[index];
                        let sparse_idx = res.sparse_idx.to_device(device);
                        let decompressed = CompressDCT::decompress(
                            &sparse_idx,
                            sparse_val,
                            &res.xshape,
                            res.totalk,
                            val_kind,
                            device,
                        );
                        let decoded = self.transform.decode(&decompressed);
                        let decoded_shape = decoded.size();
                        let active_len =
                            decoded_shape.get(prefix_dim as usize).copied().unwrap_or(0);
                        let mut aligned = match align_matformer_prefix_grad(
                            var.name(),
                            &full_shape,
                            decoded,
                        ) {
                            Ok(tensor) => tensor,
                            Err(err) => {
                                warn!(
                                    parameter = var.name(),
                                    full_shape = ?full_shape,
                                    "Skipping incompatible grad shape in DisTrO apply (cosine stats): {err:?}"
                                );
                                continue;
                            }
                        };
                        scale_matformer_prefix_grad_for_geometry(
                            &mut aligned,
                            var.name(),
                            &full_shape,
                            &decoded_shape,
                            &mut telemetry_modular_scale_sum,
                            &mut telemetry_modular_scale_min,
                            &mut telemetry_modular_scale_max,
                            &mut telemetry_modular_scale_count,
                        );
                        grads.push((aligned, active_len));
                    }

                    if grads.is_empty() {
                        warn!(
                            parameter = var.name(),
                            "Skipping DisTrO apply: no compatible grads found (cosine stats enabled)"
                        );
                        continue;
                    }
                    if grads.len() == 1 {
                        var.set_grad(grads.pop().expect("len==1").0);
                    } else {
                        // Reference: the widest variant for this parameter (largest active_len).
                        let ref_idx = grads
                            .iter()
                            .enumerate()
                            .max_by_key(|(_i, (_g, active_len))| *active_len)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let ref_grad = grads[ref_idx].0.shallow_clone();
                        let shared_len = grads
                            .iter()
                            .map(|(_g, active_len)| *active_len)
                            .min()
                            .unwrap_or(0)
                            .clamp(1, full_shape[prefix_dim as usize]);

                        let mut combined = Tensor::zeros(&full_shape, (Kind::Float, device));
                        let mut weight_sum = Tensor::zeros([], (Kind::Float, device));

                        for (i, (g, active_len)) in grads.iter().enumerate() {
                            let len = (*active_len).clamp(1, full_shape[prefix_dim as usize]);
                            let a = g
                                .narrow(prefix_dim, 0, len)
                                .to_kind(Kind::Float)
                                .contiguous()
                                .view([-1]);
                            let b = ref_grad
                                .narrow(prefix_dim, 0, len)
                                .to_kind(Kind::Float)
                                .contiguous()
                                .view([-1]);

                            let dot = (&a * &b).sum(Kind::Float);
                            let norm_a = a.pow_tensor_scalar(2).sum(Kind::Float).sqrt();
                            let norm_b = b.pow_tensor_scalar(2).sum(Kind::Float).sqrt();
                            let cos = dot / (norm_a * norm_b + 1.0e-12);

                            if apply_mixer_for_param {
                                // Soft guardrail: drop negatively-aligned updates.
                                let w = cos.clamp_min(0.0);
                                combined += g.to_kind(Kind::Float) * &w;
                                weight_sum += &w;
                            } else {
                                combined += g.to_kind(Kind::Float);
                            }

                            if i != ref_idx {
                                let cos_v = cos.double_value(&[]);
                                mixer_cos_sum += cos_v;
                                mixer_cos_count += 1;
                                mixer_cos_min = mixer_cos_min.min(cos_v);
                                mixer_cos_max = mixer_cos_max.max(cos_v);
                                if cos_v <= 0.0 {
                                    mixer_cos_dropped += 1;
                                }
                            }
                        }

                        let denom = if apply_mixer_for_param {
                            weight_sum.clamp_min(1.0e-6)
                        } else {
                            Tensor::from(grads.len() as f64).to_device(device)
                        };
                        let normalized = (combined / denom).to_kind(val_kind);
                        if want_sign_flip_stats {
                            if let Some((flips, total)) = sign_flip_stats_sampled(
                                &normalized,
                                &ref_grad,
                                prefix_dim,
                                shared_len,
                                sign_flip_sample,
                            ) {
                                let frac = flips / total.max(1.0);
                                telemetry_sign_flip_flips += flips;
                                telemetry_sign_flip_total += total;
                                telemetry_sign_flip_params += 1;
                                telemetry_sign_flip_min = telemetry_sign_flip_min.min(frac);
                                telemetry_sign_flip_max = telemetry_sign_flip_max.max(frac);
                            }
                        }
                        var.set_grad(normalized);
                    }
                } else {
                    let mut combined: Option<Tensor> = None;
                    let mut contributing_peers: usize = 0;
                    for (peer_results, sparse_val) in results.iter().zip(values.iter()) {
                        let res = &peer_results[index];
                        let sparse_idx = res.sparse_idx.to_device(device);
                        let decompressed = CompressDCT::decompress(
                            &sparse_idx,
                            sparse_val,
                            &res.xshape,
                            res.totalk,
                            val_kind,
                            device,
                        );
                        let decoded = self.transform.decode(&decompressed);
                        let decoded_shape = decoded.size();
                        let mut aligned =
                            match align_matformer_prefix_grad(var.name(), &full_shape, decoded) {
                                Ok(tensor) => tensor,
                                Err(err) => {
                                    warn!(
                                        parameter = var.name(),
                                        full_shape = ?full_shape,
                                        "Skipping incompatible grad shape in DisTrO apply: {err:?}"
                                    );
                                    continue;
                                }
                            };
                        scale_matformer_prefix_grad_for_geometry(
                            &mut aligned,
                            var.name(),
                            &full_shape,
                            &decoded_shape,
                            &mut telemetry_modular_scale_sum,
                            &mut telemetry_modular_scale_min,
                            &mut telemetry_modular_scale_max,
                            &mut telemetry_modular_scale_count,
                        );
                        combined = Some(match combined {
                            Some(acc) => acc + aligned,
                            None => aligned,
                        });
                        contributing_peers += 1;
                    }

                    if let Some(combined) = combined {
                        // Normalize by contributing peer count (consistent with "mean" in batch_decompress)
                        let normalized = if contributing_peers > 1 {
                            combined / (contributing_peers as f64)
                        } else {
                            combined
                        };
                        var.set_grad(normalized);
                    } else {
                        warn!(
                            parameter = var.name(),
                            "Skipping DisTrO apply: no compatible grads found"
                        );
                        continue;
                    }
                }
            }

            let mut grad = variable.grad();
            if !grad.defined() {
                continue;
            }

            if diloco_lite_enabled {
                let state = self.state.get_mut(index).expect("state len matches vars");
                let mut outer_momentum = state.outer_momentum.logical_tensor();
                let beta = self.diloco_lite_config.outer_momentum.clamp(0.0, 0.999_999);
                let one_minus_beta = 1.0 - beta;
                let grad_f32 = grad.to_kind(Kind::Float);
                let _ = outer_momentum.g_mul_scalar_(beta);
                let _ = outer_momentum.g_add_(&grad_f32.multiply_scalar(one_minus_beta));

                let mut aggregate = outer_momentum.shallow_clone();
                let param_l2 = variable
                    .to_kind(Kind::Float)
                    .pow_tensor_scalar(2)
                    .sum(Kind::Float)
                    .sqrt()
                    .double_value(&[]);
                let agg_l2 = aggregate
                    .pow_tensor_scalar(2)
                    .sum(Kind::Float)
                    .sqrt()
                    .double_value(&[]);
                let mut trust_scale = 1.0f64;
                let target_ratio = self.diloco_lite_config.trust_region_target.max(0.0);
                if target_ratio > 0.0 && param_l2 > 0.0 && agg_l2 > 0.0 {
                    let target_l2 = param_l2 * target_ratio;
                    trust_scale = target_l2 / agg_l2;
                    let max_scale = self.diloco_lite_config.trust_region_max_scale.max(1.0);
                    trust_scale = trust_scale.clamp(1.0 / max_scale, max_scale);
                }

                let outer_lr_multiplier = self.diloco_lite_config.outer_lr_multiplier;
                let final_scale = trust_scale * outer_lr_multiplier;
                if final_scale.is_finite() && (final_scale - 1.0).abs() > f64::EPSILON {
                    aggregate = aggregate.multiply_scalar(final_scale);
                }

                if final_scale.is_finite() {
                    telemetry_diloco_trust_scale_sum += final_scale;
                    telemetry_diloco_trust_scale_min =
                        telemetry_diloco_trust_scale_min.min(final_scale);
                    telemetry_diloco_trust_scale_max =
                        telemetry_diloco_trust_scale_max.max(final_scale);
                    telemetry_diloco_trust_scale_count += 1;
                }

                grad.copy_(&aggregate.to_kind(grad.kind()));
            }

            let pre_grad = grad.to_kind(Kind::Float);
            let pre_finite = pre_grad.isfinite().all().int64_value(&[]) != 0;
            if !pre_finite {
                telemetry_nonfinite_skipped += 1;
                warn!(
                    step = step,
                    parameter = var.name(),
                    apply_mode = ?self.apply_mode,
                    "Skipping non-finite DisTrO grad before apply-mode transform"
                );
                let _ = grad.zero_();
                continue;
            }
            let pre_grad_l2_sq_this = pre_grad
                .pow_tensor_scalar(2)
                .sum(Kind::Float)
                .double_value(&[]);
            let pre_grad_l2_this = pre_grad_l2_sq_this.sqrt();
            telemetry_pre_grad_l2_sq += pre_grad_l2_sq_this;
            telemetry_pre_grad_linf =
                telemetry_pre_grad_linf.max(pre_grad.abs().max().double_value(&[]));

            match self.apply_mode {
                DistroApplyMode::Sign => {
                    let _ = grad.sign_();
                }
                DistroApplyMode::Raw => {
                    if raw_v2_match_target_l2 {
                        let target_l2 = raw_v2_target_l2.unwrap_or(0.0);
                        let target_numel = raw_v2_target_clip_basis
                            .or(raw_v2_target_numel)
                            .unwrap_or_else(|| pre_grad.numel() as f64)
                            .max(1.0);
                        if target_l2 <= 0.0 || !target_l2.is_finite() {
                            telemetry_nonfinite_skipped += 1;
                            if self.raw_config.missing_sidecar_policy
                                == DistroRawMissingSidecarPolicy::Fail
                            {
                                panic!(
                                    "DisTrO raw-v2 got invalid target_l2 for parameter {} at step {}: {}",
                                    var.name(),
                                    step,
                                    target_l2
                                );
                            }
                            warn!(
                                step = step,
                                parameter = var.name(),
                                target_l2 = target_l2,
                                "Skipping raw-v2 norm matching for invalid sidecar target"
                            );
                        } else if pre_grad_l2_this > 1.0e-20 {
                            let mut align_scale = (target_l2 / pre_grad_l2_this)
                                * self.raw_config.scale_multiplier.max(0.0);
                            if align_scale > self.raw_config.scale_max {
                                align_scale = self.raw_config.scale_max;
                                telemetry_raw_align_scale_clamped += 1;
                            }
                            if align_scale.is_finite() && align_scale > 0.0 {
                                let _ = grad.g_mul_scalar_(align_scale);
                                let raw_preclip = grad.to_kind(Kind::Float);
                                let raw_preclip_l2_sq_this = raw_preclip
                                    .pow_tensor_scalar(2)
                                    .sum(Kind::Float)
                                    .double_value(&[]);
                                telemetry_raw_preclip_l2_sq += raw_preclip_l2_sq_this;
                                telemetry_raw_preclip_linf = telemetry_raw_preclip_linf
                                    .max(raw_preclip.abs().max().double_value(&[]));
                                telemetry_raw_preclip_tensors += 1;
                                let clip_abs = self.raw_config.abs_clip_mult.max(0.0) * target_l2
                                    / target_numel.sqrt();
                                if clip_abs.is_finite() && clip_abs > 0.0 {
                                    let max_abs_before_clip =
                                        grad.to_kind(Kind::Float).abs().max().double_value(&[]);
                                    if max_abs_before_clip > clip_abs {
                                        telemetry_raw_clip_tensors += 1;
                                        let clipped = grad.clamp(-clip_abs, clip_abs);
                                        grad.copy_(&clipped);
                                    }
                                }
                                telemetry_raw_align_scale_sum += align_scale;
                                telemetry_raw_align_scale_min =
                                    telemetry_raw_align_scale_min.min(align_scale);
                                telemetry_raw_align_scale_max =
                                    telemetry_raw_align_scale_max.max(align_scale);
                                telemetry_raw_align_scale_count += 1;
                                telemetry_raw_v2_target_l2_sum += target_l2;
                                telemetry_raw_v2_target_support_sum += raw_v2_target_support
                                    .or(raw_v2_target_numel)
                                    .unwrap_or(variable.numel() as f64);
                                telemetry_raw_v2_obs_l2_sum += pre_grad_l2_this;
                                telemetry_raw_v2_target_to_obs_ratio_sum +=
                                    target_l2 / pre_grad_l2_this;
                                telemetry_raw_v2_target_l2_count += 1;
                            } else {
                                telemetry_nonfinite_skipped += 1;
                                if self.raw_config.missing_sidecar_policy
                                    == DistroRawMissingSidecarPolicy::Fail
                                {
                                    panic!(
                                        "DisTrO raw-v2 got non-finite align scale for parameter {} at step {}: {}",
                                        var.name(),
                                        step,
                                        align_scale
                                    );
                                }
                                warn!(
                                    step = step,
                                    parameter = var.name(),
                                    align_scale = align_scale,
                                    "Skipping raw-v2 DisTrO apply for non-finite align scale"
                                );
                                let _ = grad.zero_();
                                continue;
                            }
                        } else {
                            telemetry_nonfinite_skipped += 1;
                            if self.raw_config.missing_sidecar_policy
                                == DistroRawMissingSidecarPolicy::Fail
                            {
                                panic!(
                                    "DisTrO raw-v2 got tiny gradient norm for parameter {} at step {}: {}",
                                    var.name(),
                                    step,
                                    pre_grad_l2_this
                                );
                            }
                            warn!(
                                step = step,
                                parameter = var.name(),
                                pre_grad_l2 = pre_grad_l2_this,
                                "Skipping raw-v2 DisTrO apply for tiny gradient norm"
                            );
                            let _ = grad.zero_();
                            continue;
                        }
                    } else if raw_align_sign_scale {
                        let numel = pre_grad.numel() as f64;
                        if pre_grad_l2_this > 1.0e-20 && numel > 0.0 {
                            // Match the sign-mode norm scale per tensor:
                            // sign(g) has ||.||2 ~= sqrt(numel), so normalize raw grads to that.
                            // This keeps LR schedules comparable across sign/raw modes.
                            let target_l2 = numel.sqrt();
                            let mut align_scale =
                                (target_l2 / pre_grad_l2_this) * raw_align_scale_multiplier;
                            if align_scale > raw_align_scale_cap {
                                align_scale = raw_align_scale_cap;
                                telemetry_raw_align_scale_clamped += 1;
                            }
                            if align_scale.is_finite() && align_scale > 0.0 {
                                let _ = grad.g_mul_scalar_(align_scale);
                                let raw_preclip = grad.to_kind(Kind::Float);
                                let raw_preclip_l2_sq_this = raw_preclip
                                    .pow_tensor_scalar(2)
                                    .sum(Kind::Float)
                                    .double_value(&[]);
                                telemetry_raw_preclip_l2_sq += raw_preclip_l2_sq_this;
                                telemetry_raw_preclip_linf = telemetry_raw_preclip_linf
                                    .max(raw_preclip.abs().max().double_value(&[]));
                                telemetry_raw_preclip_tensors += 1;
                                if raw_align_abs_clip > 0.0 {
                                    let max_abs_before_clip =
                                        grad.to_kind(Kind::Float).abs().max().double_value(&[]);
                                    if max_abs_before_clip > raw_align_abs_clip {
                                        telemetry_raw_clip_tensors += 1;
                                        let clipped =
                                            grad.clamp(-raw_align_abs_clip, raw_align_abs_clip);
                                        grad.copy_(&clipped);
                                    }
                                }
                                telemetry_raw_align_scale_sum += align_scale;
                                telemetry_raw_align_scale_min =
                                    telemetry_raw_align_scale_min.min(align_scale);
                                telemetry_raw_align_scale_max =
                                    telemetry_raw_align_scale_max.max(align_scale);
                                telemetry_raw_align_scale_count += 1;
                            } else {
                                telemetry_nonfinite_skipped += 1;
                                warn!(
                                    step = step,
                                    parameter = var.name(),
                                    align_scale = align_scale,
                                    "Skipping raw DisTrO apply for non-finite align scale"
                                );
                                let _ = grad.zero_();
                                continue;
                            }
                        } else {
                            telemetry_nonfinite_skipped += 1;
                            warn!(
                                step = step,
                                parameter = var.name(),
                                pre_grad_l2 = pre_grad_l2_this,
                                numel = numel,
                                "Skipping raw DisTrO apply for tiny gradient norm"
                            );
                            let _ = grad.zero_();
                            continue;
                        }
                    } else {
                        if let Some(scale) = raw_legacy_scale {
                            // Legacy raw behavior: decode lr-scaled residuals back to grad-space.
                            let _ = grad.g_mul_scalar_(scale);
                            let raw_preclip = grad.to_kind(Kind::Float);
                            let raw_preclip_l2_sq_this = raw_preclip
                                .pow_tensor_scalar(2)
                                .sum(Kind::Float)
                                .double_value(&[]);
                            telemetry_raw_preclip_l2_sq += raw_preclip_l2_sq_this;
                            telemetry_raw_preclip_linf = telemetry_raw_preclip_linf
                                .max(raw_preclip.abs().max().double_value(&[]));
                            telemetry_raw_preclip_tensors += 1;
                        } else {
                            telemetry_nonfinite_skipped += 1;
                            warn!(
                                step = step,
                                lr = lr,
                                parameter = var.name(),
                                "Skipping raw DisTrO apply for near-zero lr"
                            );
                            let _ = grad.zero_();
                            continue;
                        }
                    }
                }
            }

            let post_grad = grad.to_kind(Kind::Float);
            let post_finite = post_grad.isfinite().all().int64_value(&[]) != 0;
            if !post_finite {
                telemetry_nonfinite_skipped += 1;
                warn!(
                    step = step,
                    parameter = var.name(),
                    apply_mode = ?self.apply_mode,
                    "Skipping non-finite DisTrO grad after apply-mode transform"
                );
                let _ = grad.zero_();
                continue;
            }
            telemetry_post_grad_l2_sq += post_grad
                .pow_tensor_scalar(2)
                .sum(Kind::Float)
                .double_value(&[]);
            telemetry_post_grad_linf =
                telemetry_post_grad_linf.max(post_grad.abs().max().double_value(&[]));
            telemetry_param_l2_sq += variable
                .to_kind(Kind::Float)
                .pow_tensor_scalar(2)
                .sum(Kind::Float)
                .double_value(&[]);
            telemetry_grad_tensors += 1;
        }

        if want_cosine_stats && mixer_cos_count > 0 {
            let mean_cos = mixer_cos_sum / (mixer_cos_count as f64);
            let dropped_frac = mixer_cos_dropped as f64 / (mixer_cos_count as f64);
            info!(
                step = step,
                enabled = use_cosine_mixer,
                shadow = shadow_cosine_mixer,
                mean_cos = mean_cos,
                min_cos = mixer_cos_min,
                max_cos = mixer_cos_max,
                would_drop = mixer_cos_dropped,
                would_drop_frac = dropped_frac,
                count = mixer_cos_count,
                "DisTrO cosine mixer stats (MatFormer prefix params)"
            );
        }
        if telemetry_grad_tensors > 0 {
            let should_log = {
                let prev = LAST_APPLY_INFO_STEP.load(AtomicOrdering::Relaxed);
                prev != step
                    && LAST_APPLY_INFO_STEP
                        .compare_exchange(
                            prev,
                            step,
                            AtomicOrdering::Relaxed,
                            AtomicOrdering::Relaxed,
                        )
                        .is_ok()
            };
            if should_log {
                let pre_grad_l2 = telemetry_pre_grad_l2_sq.sqrt();
                let post_grad_l2 = telemetry_post_grad_l2_sq.sqrt();
                let param_l2 = telemetry_param_l2_sq.sqrt();
                let effective_update_l2 = post_grad_l2 * lr.abs();
                let update_to_param_ratio = if param_l2 > 0.0 {
                    effective_update_l2 / param_l2
                } else {
                    0.0
                };
                let raw_align_scale_mean = if telemetry_raw_align_scale_count > 0 {
                    telemetry_raw_align_scale_sum / telemetry_raw_align_scale_count as f64
                } else {
                    0.0
                };
                let raw_align_scale_min = if telemetry_raw_align_scale_count > 0 {
                    telemetry_raw_align_scale_min
                } else {
                    0.0
                };
                let raw_align_scale_observed_max = if telemetry_raw_align_scale_count > 0 {
                    telemetry_raw_align_scale_max
                } else {
                    0.0
                };
                let raw_v2_target_l2_mean = if telemetry_raw_v2_target_l2_count > 0 {
                    telemetry_raw_v2_target_l2_sum / telemetry_raw_v2_target_l2_count as f64
                } else {
                    0.0
                };
                let raw_v2_obs_l2_mean = if telemetry_raw_v2_target_l2_count > 0 {
                    telemetry_raw_v2_obs_l2_sum / telemetry_raw_v2_target_l2_count as f64
                } else {
                    0.0
                };
                let raw_v2_target_to_obs_ratio_mean = if telemetry_raw_v2_target_l2_count > 0 {
                    telemetry_raw_v2_target_to_obs_ratio_sum
                        / telemetry_raw_v2_target_l2_count as f64
                } else {
                    0.0
                };
                let raw_v2_target_support_mean = if telemetry_raw_v2_target_l2_count > 0 {
                    telemetry_raw_v2_target_support_sum / telemetry_raw_v2_target_l2_count as f64
                } else {
                    0.0
                };
                let modular_scale_mean = if telemetry_modular_scale_count > 0 {
                    telemetry_modular_scale_sum / telemetry_modular_scale_count as f64
                } else {
                    0.0
                };
                let modular_scale_min = if telemetry_modular_scale_count > 0 {
                    telemetry_modular_scale_min
                } else {
                    0.0
                };
                let modular_scale_max = if telemetry_modular_scale_count > 0 {
                    telemetry_modular_scale_max
                } else {
                    0.0
                };
                let diloco_weight_mean = if telemetry_diloco_weight_params > 0 {
                    telemetry_diloco_weight_mean_sum / telemetry_diloco_weight_params as f64
                } else {
                    0.0
                };
                let diloco_weight_min = if telemetry_diloco_weight_params > 0 {
                    telemetry_diloco_weight_min
                } else {
                    0.0
                };
                let diloco_weight_max = if telemetry_diloco_weight_params > 0 {
                    telemetry_diloco_weight_max
                } else {
                    0.0
                };
                let diloco_trust_scale_mean = if telemetry_diloco_trust_scale_count > 0 {
                    telemetry_diloco_trust_scale_sum / telemetry_diloco_trust_scale_count as f64
                } else {
                    0.0
                };
                let diloco_trust_scale_min = if telemetry_diloco_trust_scale_count > 0 {
                    telemetry_diloco_trust_scale_min
                } else {
                    0.0
                };
                let diloco_trust_scale_max = if telemetry_diloco_trust_scale_count > 0 {
                    telemetry_diloco_trust_scale_max
                } else {
                    0.0
                };
                let raw_v2_target_mode = if raw_v2_match_pre_l2 {
                    "match-pre-l2"
                } else if raw_v2_match_sign_equiv {
                    "match-sign-equivalent"
                } else if raw_v2_match_sign_equiv_nnz {
                    "match-sign-equivalent-nnz"
                } else {
                    "off"
                };
                let raw_clip_frac = if telemetry_raw_align_scale_count > 0 {
                    telemetry_raw_clip_tensors as f64 / telemetry_raw_align_scale_count as f64
                } else {
                    0.0
                };
                let raw_preclip_l2 = telemetry_raw_preclip_l2_sq.sqrt();
                let raw_preclip_linf = if telemetry_raw_preclip_tensors > 0 {
                    telemetry_raw_preclip_linf
                } else {
                    0.0
                };
                let sign_flip_frac = if telemetry_sign_flip_total > 0.0 {
                    telemetry_sign_flip_flips / telemetry_sign_flip_total
                } else {
                    0.0
                };
                let sign_flip_min = if telemetry_sign_flip_params > 0 {
                    telemetry_sign_flip_min
                } else {
                    0.0
                };
                let sign_flip_max = if telemetry_sign_flip_params > 0 {
                    telemetry_sign_flip_max
                } else {
                    0.0
                };
                info!(
                    step = step,
                    apply_mode = ?self.apply_mode,
                    peer_count = results.len(),
                    raw_v2_enabled = raw_v2_enabled,
                    raw_v2_target_mode = raw_v2_target_mode,
                    raw_v2_norm_mode = ?self.raw_config.norm_mode,
                    raw_v2_missing_sidecar_policy = ?self.raw_config.missing_sidecar_policy,
                    raw_v2_scale_multiplier = self.raw_config.scale_multiplier,
                    raw_v2_scale_max = self.raw_config.scale_max,
                    raw_v2_abs_clip_mult = self.raw_config.abs_clip_mult,
                    raw_v2_sign_equiv_mult = self.raw_config.sign_equiv_mult,
                    raw_v2_missing_sidecar = telemetry_raw_v2_missing_sidecar,
                    raw_v2_invalid_sidecar = telemetry_raw_v2_invalid_sidecar,
                    raw_v2_target_l2_mean = raw_v2_target_l2_mean,
                    raw_v2_target_support_mean = raw_v2_target_support_mean,
                    raw_v2_obs_l2_mean = raw_v2_obs_l2_mean,
                    raw_v2_target_to_obs_ratio_mean = raw_v2_target_to_obs_ratio_mean,
                    raw_align_sign_scale = raw_align_sign_scale,
                    raw_align_scale_multiplier = raw_align_scale_multiplier,
                    raw_align_scale_cap = raw_align_scale_cap,
                    raw_align_abs_clip = raw_align_abs_clip,
                    raw_align_scale_count = telemetry_raw_align_scale_count,
                    raw_align_scale_clamped = telemetry_raw_align_scale_clamped,
                    raw_clip_tensors = telemetry_raw_clip_tensors,
                    raw_clip_frac = raw_clip_frac,
                    raw_preclip_tensors = telemetry_raw_preclip_tensors,
                    raw_preclip_l2 = raw_preclip_l2,
                    raw_preclip_linf = raw_preclip_linf,
                    raw_align_scale_mean = raw_align_scale_mean,
                    raw_align_scale_min = raw_align_scale_min,
                    raw_align_scale_max = raw_align_scale_observed_max,
                    sign_flip_sample_elems = sign_flip_sample,
                    sign_flip_params = telemetry_sign_flip_params,
                    sign_flip_frac = sign_flip_frac,
                    sign_flip_min = sign_flip_min,
                    sign_flip_max = sign_flip_max,
                    aggregate_mode = ?self.aggregate_mode,
                    diloco_lite_enabled = diloco_lite_enabled,
                    diloco_outer_momentum = self.diloco_lite_config.outer_momentum,
                    diloco_outer_lr_multiplier = self.diloco_lite_config.outer_lr_multiplier,
                    diloco_trust_region_target = self.diloco_lite_config.trust_region_target,
                    diloco_trust_region_max_scale = self.diloco_lite_config.trust_region_max_scale,
                    diloco_tier_weight_cap = self.diloco_lite_config.tier_weight_cap,
                    diloco_weight_params = telemetry_diloco_weight_params,
                    diloco_weight_mean = diloco_weight_mean,
                    diloco_weight_min = diloco_weight_min,
                    diloco_weight_max = diloco_weight_max,
                    diloco_trust_scale_count = telemetry_diloco_trust_scale_count,
                    diloco_trust_scale_mean = diloco_trust_scale_mean,
                    diloco_trust_scale_min = diloco_trust_scale_min,
                    diloco_trust_scale_max = diloco_trust_scale_max,
                    modular_geometry_align = modular_geometry_align_enabled(),
                    modular_geometry_scale_power = modular_geometry_scale_power(),
                    modular_geometry_scale_count = telemetry_modular_scale_count,
                    modular_geometry_scale_mean = modular_scale_mean,
                    modular_geometry_scale_min = modular_scale_min,
                    modular_geometry_scale_max = modular_scale_max,
                    lr = lr,
                    grad_tensors = telemetry_grad_tensors,
                    skipped_nonfinite_tensors = telemetry_nonfinite_skipped,
                    pre_apply_grad_l2 = pre_grad_l2,
                    pre_apply_grad_linf = telemetry_pre_grad_linf,
                    post_mode_grad_l2 = post_grad_l2,
                    post_mode_grad_linf = telemetry_post_grad_linf,
                    effective_update_l2 = effective_update_l2,
                    param_l2 = param_l2,
                    update_to_param_ratio = update_to_param_ratio,
                    "DisTrO apply norm stats"
                );
                if raw_v2_enabled && telemetry_raw_align_scale_count > 0 {
                    let clamp_frac = telemetry_raw_align_scale_clamped as f64
                        / telemetry_raw_align_scale_count as f64;
                    if clamp_frac >= 0.5 {
                        warn!(
                            step = step,
                            clamp_frac = clamp_frac,
                            raw_v2_scale_max = self.raw_config.scale_max,
                            raw_v2_target_mode = raw_v2_target_mode,
                            "DisTrO raw-v2 is heavily scale-clamped; increase distro_raw_scale_max or adjust target mode/multiplier"
                        );
                    }
                    if raw_clip_frac > 0.5 {
                        let streak =
                            RAW_HEAVY_CLIP_STREAK.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                        if streak == 5 || streak % 10 == 0 {
                            warn!(
                                step = step,
                                clip_frac = raw_clip_frac,
                                clip_streak = streak,
                                raw_v2_abs_clip_mult = self.raw_config.abs_clip_mult,
                                "DisTrO raw-v2 is heavily clipping for sustained steps; increase distro_raw_abs_clip_mult or disable per-tensor clipping"
                            );
                        }
                    } else {
                        RAW_HEAVY_CLIP_STREAK.store(0, AtomicOrdering::Relaxed);
                    }
                } else {
                    RAW_HEAVY_CLIP_STREAK.store(0, AtomicOrdering::Relaxed);
                }
            }
        }
        // SGD step
        self.sgd.set_learning_rate(lr).unwrap();
        let _ = self.sgd.step();
        for var in vars.variables() {
            var.zero_grad();
        }
    }

    pub fn error_correction(&mut self, vars: &dyn CausalLM, prev_lr: f64) {
        let _no_grad = tch::no_grad_guard();
        for (index, var) in vars.variables().enumerate() {
            let mut variable = var.logical_tensor();

            let state = self.state.get_mut(index).unwrap();

            // Remove the lookahead delta that was applied in `generate`.
            let lookahead = Self::lookahead_delta(&state.delta.logical_tensor(), self.apply_mode)
                .multiply_scalar(prev_lr);
            let _t = variable.g_sub_(&lookahead);
        }
    }

    pub fn zero_optim(&mut self) {
        for state in &mut self.state {
            let _ = state.delta.logical_tensor().zero_();
            let _ = state.outer_momentum.logical_tensor().zero_();
        }
    }

    pub fn quantize_nozeros_tensor_to_boolean_sign(tensor: &Tensor) -> Tensor {
        let original_size = tensor.size();
        let tensor = tensor.signbit();
        debug_assert_eq!(tensor.kind(), Kind::Bool);
        debug_assert_eq!(tensor.size(), original_size);
        tensor
    }

    fn unpack_tensor_sign_from_boolean(tensor: Tensor, unpack_kind: Kind) -> Tensor {
        tensor.to_kind(unpack_kind) * -2 + 1
    }
}

unsafe impl Send for Distro {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Variable, set_torch_rng_seed};
    use itertools::iproduct;

    impl Variable for Tensor {
        fn name(&self) -> &str {
            unimplemented!()
        }

        fn local_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn logical_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn gather_full_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn shard_other_tensor_like_me(&self, tensor: Tensor) -> Tensor {
            tensor
        }

        fn full_tensor_shape(&self) -> Vec<i64> {
            self.size()
        }

        fn is_sharded(&self) -> bool {
            false
        }

        fn zeros_like(&self, _name: String) -> Box<dyn Variable> {
            Box::new(self.zeros_like())
        }

        fn set_grad(&self, tensor: Tensor) {
            self.grad().copy_(&tensor);
        }

        fn zero_grad(&self) {
            let grad = self.grad();
            if grad.defined() {
                let _ = self.grad().zero_();
            }
        }
    }

    fn vars(vars: Vec<Tensor>) -> StableVariableIterator {
        Box::new(vars.into_iter().map(|x| Box::new(x) as Box<dyn Variable>))
    }

    #[test]
    fn test_diloco_peer_weights_normalized() {
        let metas = vec![
            DistroPeerMetadata {
                inner_steps_used: 1,
                sum_local_lr: 1.0,
                tokens_processed: 100,
                delta_l2_preclip: 0.0,
                delta_l2_postclip: 0.0,
            },
            DistroPeerMetadata {
                inner_steps_used: 2,
                sum_local_lr: 2.0,
                tokens_processed: 100,
                delta_l2_preclip: 0.0,
                delta_l2_postclip: 0.0,
            },
        ];
        let (weights, mean, min_w, max_w) = diloco_peer_weights(&metas, 2.0);
        assert_eq!(weights.len(), 2);
        assert!((mean - 1.0).abs() < 1.0e-6);
        assert!(min_w > 0.0);
        assert!(max_w < 2.0);
        assert!((weights[0] - (4.0 / 3.0)).abs() < 1.0e-5);
        assert!((weights[1] - (2.0 / 3.0)).abs() < 1.0e-5);
    }

    #[test]
    fn test_diloco_peer_weights_cap_enforced() {
        let metas = vec![
            DistroPeerMetadata {
                inner_steps_used: 1,
                sum_local_lr: 1.0,
                tokens_processed: 1000,
                delta_l2_preclip: 0.0,
                delta_l2_postclip: 0.0,
            },
            DistroPeerMetadata {
                inner_steps_used: 1,
                sum_local_lr: 1.0,
                tokens_processed: 10,
                delta_l2_preclip: 0.0,
                delta_l2_postclip: 0.0,
            },
        ];
        let cap = 1.2;
        let (weights, mean, _min_w, max_w) = diloco_peer_weights(&metas, cap);
        assert_eq!(weights.len(), 2);
        assert!((mean - 1.0).abs() < 1.0e-6);
        assert!(max_w <= cap + 1.0e-6);
    }

    #[test]
    fn test_get_prime_divisors() {
        assert_eq!(TransformDCT::get_prime_divisors(1), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_prime_divisors(2), vec![2]);
        assert_eq!(TransformDCT::get_prime_divisors(12), vec![2, 2, 3]);
        assert_eq!(TransformDCT::get_prime_divisors(15), vec![3, 5]);
        assert_eq!(TransformDCT::get_prime_divisors(100), vec![2, 2, 5, 5]);
        assert_eq!(TransformDCT::get_prime_divisors(2310), vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_get_divisors() {
        assert_eq!(TransformDCT::get_divisors(1), vec![1]);
        assert_eq!(TransformDCT::get_divisors(2), vec![1, 2]);
        assert_eq!(TransformDCT::get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(TransformDCT::get_divisors(15), vec![1, 3, 5, 15]);
        assert_eq!(
            TransformDCT::get_divisors(100),
            vec![1, 2, 4, 5, 10, 20, 25, 50, 100]
        );
    }

    #[test]
    fn test_get_smaller_split() {
        assert_eq!(TransformDCT::get_smaller_split(12, 3), 3);
        assert_eq!(TransformDCT::get_smaller_split(12, 4), 4);
        assert_eq!(TransformDCT::get_smaller_split(12, 5), 4);
        assert_eq!(TransformDCT::get_smaller_split(100, 7), 5);
        assert_eq!(TransformDCT::get_smaller_split(100, 26), 25);
        assert_eq!(TransformDCT::get_smaller_split(100, 101), 100);
        assert_eq!(TransformDCT::get_smaller_split(1, 1), 1);
    }

    #[test]
    fn test_edge_cases() {
        assert_eq!(TransformDCT::get_prime_divisors(0), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_divisors(0), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_smaller_split(0, 1), 0);
    }

    #[test]
    fn test_large_numbers() {
        assert_eq!(
            TransformDCT::get_prime_divisors(1000000007),
            vec![1000000007]
        ); // Large prime
        assert_eq!(TransformDCT::get_divisors(1000000007), vec![1, 1000000007]);
        assert_eq!(TransformDCT::get_smaller_split(1000000007, 500000000), 1);
    }

    #[test]
    fn test_dct() {
        let eye = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [0.5000, 0.6533, 0.5000, 0.2706],
            [0.5000, 0.2706, -0.5000, -0.6533],
            [0.5000, -0.2706, -0.5000, 0.6533],
            [0.5000, -0.6533, 0.5000, -0.2706],
        ]);
        let result = TransformDCT::dct(&eye, true);
        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    fn _2d_float<T: AsRef<[f64]>>(x: &[T]) -> Tensor {
        Tensor::from_slice2(x).to_kind(Kind::Float).to(Device::Cpu)
    }

    fn _2d_int<T: AsRef<[i64]>>(x: &[T]) -> Tensor {
        Tensor::from_slice2(x).to_kind(Kind::Int64).to(Device::Cpu)
    }

    fn _1d_float(x: &[f64]) -> Tensor {
        Tensor::from_slice(x).to_kind(Kind::Float).to(Device::Cpu)
    }

    fn _1d_int(x: &[i64]) -> Tensor {
        Tensor::from_slice(x).to_kind(Kind::Int64).to(Device::Cpu)
    }

    #[test]
    fn test_idct() {
        let eye = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ]);
        let result = TransformDCT::idct(&eye, true);
        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_compress_2d() {
        let r = _2d_float(&[
            [0.1911, 0.4076, 0.1649, 0.8059],
            [0.2803, 0.9381, 0.9071, 0.2573],
            [0.4070, 0.5765, 0.7226, 0.9486],
            [0.0737, 0.7378, 0.1898, 0.2990],
        ]);
        let truth = (
            _2d_int(&[[3, 1], [1, 2], [3, 2], [1, 3]]),
            _2d_float(&[
                [0.8059, 0.4076],
                [0.9381, 0.9071],
                [0.9486, 0.7226],
                [0.7378, 0.2990],
            ]),
            vec![4i64, 4i64],
            4i64,
        );
        let ret = CompressDCT::compress(&r, 2);
        assert_eq!(truth.0, ret.0);
        assert!(truth.1.allclose(&ret.1, 1e-4, 1e-8, false));
        assert_eq!(truth.2, ret.2);
        assert_eq!(4, ret.3);
    }

    #[test]
    fn test_compress_1d() {
        let r = _1d_float(&[
            0.5223, 0.9625, 0.5487, 0.2152, 0.2161, 0.0363, 0.4944, 0.0974,
        ]);
        let truth = (
            _1d_int(&[1, 2]),
            _1d_float(&[0.9625, 0.5487]),
            vec![8i64],
            8i64,
        );
        let ret = CompressDCT::compress(&r, 2);
        assert_eq!(truth.0, ret.0);
        assert!(truth.1.allclose(&ret.1, 1e-4, 1e-8, false));
        assert_eq!(truth.2, ret.2);
        assert_eq!(8, ret.3);
    }

    #[test]
    fn test_decompress_1d() {
        let p = _1d_float(&[0.0]);
        let idx = _1d_int(&[1, 2]);
        let val = _1d_float(&[0.9625, 0.5487]);
        let xshape = vec![8i64];
        let truth = _1d_float(&[
            0.0000, 0.9625, 0.5487, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        ]);
        let ret = CompressDCT::decompress(&idx, &val, &xshape, i64::MAX, p.kind(), p.device());
        assert!(truth.allclose(&ret, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_decompress_2d() {
        let p = _1d_float(&[0.0]);
        let idx = _2d_int(&[[0, 2], [1, 2], [2, 3], [3, 1]]);
        let val = _2d_float(&[
            [0.8988, 0.5175],
            [0.9882, 0.8945],
            [0.8285, 0.8163],
            [0.9093, 0.7600],
        ]);
        let xshape = vec![4i64, 4i64];
        let truth = _2d_float(&[
            [0.8988, 0.0000, 0.5175, 0.0000],
            [0.0000, 0.9882, 0.8945, 0.0000],
            [0.0000, 0.0000, 0.8285, 0.8163],
            [0.0000, 0.7600, 0.0000, 0.9093],
        ]);
        let ret = CompressDCT::decompress(&idx, &val, &xshape, i64::MAX, p.kind(), p.device());
        assert!(truth.allclose(&ret, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_encode_1d() {
        let a = Tensor::arange(8, (Kind::Float, Device::Cpu));
        let truth = _1d_float(&[
            9.8995e+00,
            -6.4423e+00,
            -4.7684e-07,
            -6.7345e-01,
            2.3842e-07,
            -2.0090e-01,
            -1.1921e-07,
            -5.0702e-02,
        ]);
        let ret = TransformDCT::new(vars(vec![a.copy()]), 64)
            .encode(&a)
            .squeeze();
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_encode_2d() {
        let b = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00, 0.0000e+00, -5.9605e-08],
            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
            [0.0000e+00, -5.9605e-08, 0.0000e+00, 1.0000e+00],
        ]);
        let ret = TransformDCT::new(vars(vec![b.copy()]), 64)
            .encode(&b)
            .squeeze();
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_decode_1d() {
        let a = Tensor::arange(8, (Kind::Float, Device::Cpu));
        let a_ = _2d_float(&[[
            9.8995e+00,
            -6.4423e+00,
            -4.7684e-07,
            -6.7345e-01,
            2.3842e-07,
            -2.0090e-01,
            -1.1921e-07,
            -5.0702e-02,
        ]]);
        let truth = _1d_float(&[
            -2.2352e-07,
            1.0000e+00,
            2.0000e+00,
            3.0000e+00,
            4.0000e+00,
            5.0000e+00,
            6.0000e+00,
            7.0000e+00,
        ]);
        let ret = TransformDCT::new(vars(vec![a]), 64).decode(&a_);
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_decode_2d() {
        let b = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let b_ = _2d_float(&[
            [1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00, 0.0000e+00, -5.9605e-08],
            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
            [0.0000e+00, -5.9605e-08, 0.0000e+00, 1.0000e+00],
        ])
        .unsqueeze(0)
        .unsqueeze(0);
        let truth = _2d_float(&[
            [1.0000e+00, 1.4901e-08, 4.4703e-08, 4.4703e-08],
            [2.9802e-08, 1.0000e+00, -2.9802e-08, 4.4703e-08],
            [4.4703e-08, -2.9802e-08, 1.0000e+00, 2.9802e-08],
            [4.4703e-08, 4.4703e-08, 1.4901e-08, 1.0000e+00],
        ]);
        let ret = TransformDCT::new(vars(vec![b]), 64).decode(&b_);
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_align_matformer_prefix_grad_expand_gate() {
        let grad = _2d_float(&[[1.0, 2.0], [3.0, 4.0]]);
        let full_shape = vec![4i64, 2i64];
        let aligned =
            align_matformer_prefix_grad("model.layers.0.mlp.gate_proj.weight", &full_shape, grad)
                .unwrap();
        assert_eq!(aligned.size(), full_shape);
        let prefix = aligned.narrow(0, 0, 2);
        let tail = aligned.narrow(0, 2, 2);
        assert!(prefix.allclose(&_2d_float(&[[1.0, 2.0], [3.0, 4.0]]), 1e-6, 1e-6, false));
        assert!(tail.allclose(&_2d_float(&[[0.0, 0.0], [0.0, 0.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_prefix_grad_slice_down() {
        let grad = _2d_float(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let full_shape = vec![2i64, 2i64];
        let aligned =
            align_matformer_prefix_grad("model.layers.0.mlp.down_proj.weight", &full_shape, grad)
                .unwrap();
        assert_eq!(aligned.size(), full_shape);
        assert!(aligned.allclose(&_2d_float(&[[1.0, 2.0], [5.0, 6.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_prefix_grad_rejects_non_mlp() {
        let grad = _2d_float(&[[1.0, 2.0]]);
        let full_shape = vec![2i64, 2i64];
        let err = align_matformer_prefix_grad(
            "model.layers.0.self_attn.q_proj.weight",
            &full_shape,
            grad,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PrefixAlignError::UnsupportedParameter(
                "model.layers.0.self_attn.q_proj.weight".to_string()
            )
        );
    }

    #[test]
    fn test_signed_vals_reconstructs_original_sign() {
        let truth = Tensor::from_slice2(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ])
        .to_kind(Kind::Float)
        .to(Device::Cpu);

        let signed_truth = truth.sign();

        let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(&truth, i64::MAX);
        let signed_sparse_val = sparse_val.sign();

        let decompressed_signed = CompressDCT::decompress(
            &sparse_idx,
            &signed_sparse_val,
            &xshape,
            totalk,
            truth.kind(),
            Device::Cpu,
        );
        assert!(decompressed_signed.equal(&signed_truth));
    }

    #[test]
    fn test_artifical_distro_results_roundtrip() {
        use tch::{Kind, Tensor};

        /// Generates a dummy estimate_val tensor of shape (r0, r1, k), where r is the remainder shape after DCT chunking
        /// r1 can be set to 0 to simulate a 1D DCT
        fn generate_random_estimate_val(r0: i64, r1: i64, k: i64, dtype: Kind) -> Tensor {
            // Warning: only works if dtype bits size is divisible by 8, should always be true for current torch tensors
            // but who knows what would happen one day... fp4?

            let randbytes = match dtype {
                Kind::BFloat16 => 2,
                Kind::Float => 4,
                Kind::Double => 8,
                _ => panic!("Unsupported dtype"),
            };

            // 1D DCT estimates
            let randsize = if r1 == 0 {
                vec![r0, k * randbytes]
            }
            // 2D DCT estimates
            else {
                vec![r0, r1, k * randbytes]
            };

            Tensor::randint(256, &randsize, (Kind::Uint8, tch::Device::Cpu)).view_dtype(dtype)
        }

        /// Generates a dummy indices tensor when given estimate_val. indices are between 0 and s0*s1 (exclusive),
        /// where s0 and s1 is the DCT chunk shape
        /// s1 can be set to 0 to simulate a 1D DCT
        fn generate_random_estimate_idx(val: &Tensor, s0: i64, s1: i64) -> (Tensor, i64) {
            // Note: Some indices will collide, just like real estimates
            // Warning: At the current moment of writing this test, we assume indices must always be int64
            // for correct torch indexing

            // 1D DCT estimates
            let s1 = if s1 == 0 { 1 } else { s1 };

            let max_value = s0 * s1;
            (
                Tensor::randint(max_value, val.size(), (Kind::Int64, tch::Device::Cpu)),
                max_value,
            )
        }

        set_torch_rng_seed();

        let range_r0 = 1..10;
        let range_r1 = 0..10;
        let range_s0 = [1, 7, 512];
        let range_s1 = [1, 4, 64];
        let range_k = [1, 2, 3, 4, 5, 7, 9, 16, 32, 64, 96, 128];
        let range_dtype = [Kind::BFloat16, Kind::Float];

        for (r0, r1, s0, s1, k, d) in
            iproduct!(range_r0, range_r1, range_s0, range_s1, range_k, range_dtype)
        {
            let val = generate_random_estimate_val(r0, r1, k, d);
            let (idx, max_idx_val) = generate_random_estimate_idx(&val, s0, s1);

            let roundtripped_val = Distro::unpack_tensor_sign_from_boolean(
                Distro::quantize_nozeros_tensor_to_boolean_sign(&val),
                val.kind(),
            );

            // we need to make a reference to compare the compression to.
            // this compression should hold Infinity and +0 and some NaNs as 1
            // and -Infinity and -0 and some NaNs as -1
            let val_signed: Tensor = (-2.0 * val.signbit().to_kind(Kind::Float)) + 1.0;
            assert!(val_signed.equal(&roundtripped_val));

            let roundtripped_idx = decompress_idx(max_idx_val, &compress_idx(max_idx_val, &idx));
            assert!(idx.equal(&roundtripped_idx));
        }
    }
    #[test]
    fn test_1bit_matches_non_quant() {
        set_torch_rng_seed();
        let input = Tensor::rand(
            [51, 35, 5, 13, 6],
            (Kind::BFloat16, Device::cuda_if_available()),
        ) - 0.5;
        // ensure no zeros in our ground truth!
        let input = (&input) + (input.sign() + 0.1);

        let quant = Distro::quantize_nozeros_tensor_to_boolean_sign(&input);
        let unquant = Distro::unpack_tensor_sign_from_boolean(quant, input.kind());

        assert!(input.sign().equal(&unquant));
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(
            extract_layer_index("model.layers.0.mlp.gate_proj.weight"),
            Some(0)
        );
        assert_eq!(
            extract_layer_index("model.layers.5.mlp.down_proj.weight"),
            Some(5)
        );
        assert_eq!(
            extract_layer_index("model.layers.12.self_attn.q_proj.weight"),
            Some(12)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn test_align_matformer_sparse_grad_gate_proj() {
        // Test scattering sparse gradients for gate_proj
        // Simulates prefix [0,1] + helper indices [4,6]
        let grad = _2d_float(&[
            [1.0, 2.0], // idx 0
            [3.0, 4.0], // idx 1
            [5.0, 6.0], // idx 4
            [7.0, 8.0], // idx 6
        ]);
        let full_shape = vec![8i64, 2i64];
        let indices: Vec<i64> = vec![0, 1, 4, 6];

        let aligned = align_matformer_sparse_grad(
            "model.layers.0.mlp.gate_proj.weight",
            &full_shape,
            grad,
            &indices,
        )
        .unwrap();

        assert_eq!(aligned.size(), full_shape);

        // Check row 0
        let row0 = aligned.narrow(0, 0, 1);
        assert!(row0.allclose(&_2d_float(&[[1.0, 2.0]]), 1e-6, 1e-6, false));

        // Check row 1
        let row1 = aligned.narrow(0, 1, 1);
        assert!(row1.allclose(&_2d_float(&[[3.0, 4.0]]), 1e-6, 1e-6, false));

        // Check rows 2,3 are zeros
        let row2 = aligned.narrow(0, 2, 1);
        let row3 = aligned.narrow(0, 3, 1);
        assert!(row2.allclose(&_2d_float(&[[0.0, 0.0]]), 1e-6, 1e-6, false));
        assert!(row3.allclose(&_2d_float(&[[0.0, 0.0]]), 1e-6, 1e-6, false));

        // Check row 4
        let row4 = aligned.narrow(0, 4, 1);
        assert!(row4.allclose(&_2d_float(&[[5.0, 6.0]]), 1e-6, 1e-6, false));

        // Check row 5 is zeros
        let row5 = aligned.narrow(0, 5, 1);
        assert!(row5.allclose(&_2d_float(&[[0.0, 0.0]]), 1e-6, 1e-6, false));

        // Check row 6
        let row6 = aligned.narrow(0, 6, 1);
        assert!(row6.allclose(&_2d_float(&[[7.0, 8.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_sparse_grad_down_proj() {
        // Test scattering sparse gradients for down_proj (columns)
        // Simulates prefix [0,1] + helper indices [4,6]
        let grad = _2d_float(&[
            [1.0, 2.0, 5.0, 7.0], // row 0, cols [0,1,4,6]
            [3.0, 4.0, 6.0, 8.0], // row 1, cols [0,1,4,6]
        ]);
        let full_shape = vec![2i64, 8i64];
        let indices: Vec<i64> = vec![0, 1, 4, 6];

        let aligned = align_matformer_sparse_grad(
            "model.layers.0.mlp.down_proj.weight",
            &full_shape,
            grad,
            &indices,
        )
        .unwrap();

        assert_eq!(aligned.size(), full_shape);

        // Check columns 0,1,4,6 have values, others are zeros
        let col0 = aligned.narrow(1, 0, 1);
        assert!(col0.allclose(&_2d_float(&[[1.0], [3.0]]), 1e-6, 1e-6, false));

        let col4 = aligned.narrow(1, 4, 1);
        assert!(col4.allclose(&_2d_float(&[[5.0], [6.0]]), 1e-6, 1e-6, false));

        let col2 = aligned.narrow(1, 2, 1);
        assert!(col2.allclose(&_2d_float(&[[0.0], [0.0]]), 1e-6, 1e-6, false));
    }

    #[test]
    fn test_align_matformer_sparse_grad_rejects_non_mlp() {
        let grad = _2d_float(&[[1.0, 2.0]]);
        let full_shape = vec![2i64, 2i64];
        let indices: Vec<i64> = vec![0];
        let err = align_matformer_sparse_grad(
            "model.layers.0.self_attn.q_proj.weight",
            &full_shape,
            grad,
            &indices,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PrefixAlignError::UnsupportedParameter(
                "model.layers.0.self_attn.q_proj.weight".to_string()
            )
        );
    }
}

// #[cfg(test)]
// #[cfg(feature = "parallelism")]
// mod tp_tests {
//     use super::*;
//     use crate::tensor_parallelism::CommunicatorId;
//     use crate::{
//         set_suggested_env_vars, set_torch_rng_seed, unsharded_cpu_variables, ColumnParallelLinear,
//     };
//     use std::sync::{Arc, Barrier, Mutex};
//     use tch::{nn, Device, Kind, Tensor, CNCCL};

//     const TEST_LR: f64 = 0.01;
//     const COMPRESSION_DECAY: f64 = 0.99;
//     const COMPRESSION_CHUNK: i64 = 64;
//     const COMPRESSION_TOPK: i64 = 16;
//     const WEIGHT_DECAY: f64 = 0.0;
//     const NUM_STEPS: u32 = 10;

//     fn run_parallel_test<F>(world_size: usize, test_fn: F)
//     where
//         F: Fn(Arc<CommunicatorId>, usize, Arc<Barrier>, Device) -> anyhow::Result<()>
//             + Send
//             + Sync
//             + 'static,
//     {
//         if !tch::utils::has_cuda() || tch::Cuda::device_count() < world_size as i64 {
//             println!(
//                 "Skipping parallel test: requires CUDA and {} GPUs.",
//                 world_size
//             );
//             return;
//         }

//         let barrier = Arc::new(Barrier::new(world_size));
//         let comm_id = Arc::new(CommunicatorId::new());
//         let test_fn = Arc::new(test_fn);

//         let threads: Vec<_> = (0..world_size)
//             .map(|rank| {
//                 let barrier = barrier.clone();
//                 let comm_id = comm_id.clone();
//                 let test_fn = test_fn.clone();
//                 let device = Device::Cuda(rank);

//                 std::thread::spawn(move || {
//                     test_fn(comm_id, rank, barrier, device).unwrap();
//                 })
//             })
//             .collect();

//         for thread in threads {
//             thread.join().expect("Thread panicked");
//         }
//     }

//     // Helper to run a simple training loop step with Distro
//     fn run_distro_step(
//         step_num: u32,
//         model: &dyn nn::Module,
//         input: &Tensor,
//         target: &Tensor,
//         optimizer: &mut Distro,
//         lr: f64,
//         all_rank_results: Arc<Mutex<HashMap<u32, Vec<Vec<DistroResult>>>>>,
//         _rank: usize,
//         _world_size: usize,
//         _comm: &Option<Arc<Communicator>>,
//         barrier: &Arc<Barrier>,
//     ) -> anyhow::Result<Vec<DistroResult>> {
//         optimizer.zero_grad();
//         barrier.wait();

//         let output = model.forward(input);
//         let loss = output.mse_loss(target, tch::Reduction::Mean);
//         barrier.wait();

//         loss.backward();
//         barrier.wait();

//         let current_step_results = optimizer.generate(&vec![], 0.0, lr, false);
//         barrier.wait();

//         {
//             let mut results_map = all_rank_results.lock().unwrap();
//             let step_results = results_map.entry(step_num).or_default();
//             step_results.push(current_step_results.clone());
//         }
//         barrier.wait();

//         let results_to_apply = {
//             let results_map = all_rank_results.lock().unwrap();
//             results_map
//                 .get(&step_num)
//                 .expect(&format!("missing results for current step {step_num}"))
//                 .clone()
//         };
//         barrier.wait();

//         optimizer.apply(&results_to_apply, lr);
//         barrier.wait();

//         Ok(current_step_results)
//     }

//     #[test]
//     fn test_distro_tp_consistency() -> anyhow::Result<()> {
//         const WORLD_SIZE: usize = 8;
//         const BATCH_SIZE: i64 = 4;
//         const SEQ_LEN: i64 = 32;
//         const IN_FEATURES: i64 = 128;
//         const OUT_FEATURES: i64 = 256;

//         set_suggested_env_vars();
//         set_torch_rng_seed();

//         let device = Device::cuda_if_available();
//         if !device.is_cuda() {
//             println!("Skipping TP test as CUDA is not available.");
//             return Ok(());
//         }

//         let input = Arc::new(Mutex::new(Tensor::randn(
//             &[BATCH_SIZE, SEQ_LEN, IN_FEATURES],
//             (Kind::Float, device),
//         )));
//         let target = Arc::new(Mutex::new(Tensor::randn(
//             &[BATCH_SIZE, SEQ_LEN, OUT_FEATURES],
//             (Kind::Float, device),
//         )));

//         // single gpu
//         let (final_weights_non_tp, linear_layer_weights) = {
//             let vs_non_tp = nn::VarStore::new(device);
//             let model_non_tp = nn::linear(
//                 vs_non_tp.root() / "layer",
//                 IN_FEATURES,
//                 OUT_FEATURES,
//                 nn::LinearConfig {
//                     bias: false,
//                     ..Default::default()
//                 },
//             );
//             let original_weights = model_non_tp.ws.copy();

//             let mut optimizer_non_tp = Distro::new(
//                 &vs_non_tp,
//                 COMPRESSION_DECAY,
//                 COMPRESSION_CHUNK,
//                 COMPRESSION_TOPK,
//                 WEIGHT_DECAY,
//                 None,
//             );

//             let dummy_barrier = Arc::new(Barrier::new(1));
//             let dummy_all_results = Arc::new(Mutex::new(HashMap::new()));

//             for step in 0..NUM_STEPS {
//                 let _ = run_distro_step(
//                     step,
//                     &model_non_tp,
//                     &input.lock().unwrap(),
//                     &target.lock().unwrap(),
//                     &mut optimizer_non_tp,
//                     TEST_LR,
//                     dummy_all_results.clone(),
//                     0,
//                     1,
//                     &None,
//                     &dummy_barrier,
//                 )?;
//             }

//             let mut final_weights = HashMap::new();
//             for (name, tensor) in vs_non_tp.variables() {
//                 final_weights.insert(name, tensor.detach().to_device(Device::Cpu));
//             }
//             (final_weights, original_weights)
//         };

//         let final_weights_tp_rank0 = Arc::new(Mutex::new(HashMap::new()));
//         let all_rank_results_tp: Arc<Mutex<HashMap<u32, Vec<Vec<DistroResult>>>>> =
//             Arc::new(Mutex::new(HashMap::new()));

//         {
//             let final_weights_tp_rank0 = final_weights_tp_rank0.clone();
//             let all_rank_results_tp = all_rank_results_tp.clone();
//             let ref_linear_weights = Arc::new(Mutex::new(linear_layer_weights));

//             run_parallel_test(
//                 WORLD_SIZE,
//                 move |comm_id, rank, barrier, device| -> anyhow::Result<()> {
//                     let vs_tp = nn::VarStore::new(device);
//                     let comm = Arc::new(CNCCL::new(
//                         comm_id.clone(),
//                         rank as i64,
//                         WORLD_SIZE as i64,
//                         device,
//                     )?);

//                     let mut model_tp = ColumnParallelLinear::new(
//                         vs_tp.root() / "layer",
//                         IN_FEATURES,
//                         OUT_FEATURES,
//                         false,
//                         true,
//                         Some(comm.clone()),
//                     );

//                     let (input, target) = {
//                         let _no_grad = tch::no_grad_guard();
//                         model_tp.linear.ws.copy_(&tensor_shard(
//                             &ref_linear_weights.lock().unwrap(),
//                             &Shard {
//                                 dim: 0,
//                                 rank,
//                                 world_size: WORLD_SIZE,
//                             },
//                         ));

//                         barrier.wait();

//                         comm.group_start().unwrap();
//                         if rank == 0 {
//                             let input = input.lock().unwrap();
//                             for i in 0..WORLD_SIZE {
//                                 comm.send(&[input.as_ref()], i as i64).unwrap();
//                             }
//                         }
//                         let input = Tensor::zeros(
//                             &[BATCH_SIZE, SEQ_LEN, IN_FEATURES],
//                             (Kind::Float, device),
//                         );
//                         comm.recv(&[input.shallow_clone()], 0).unwrap();
//                         comm.group_end().unwrap();

//                         barrier.wait();

//                         comm.group_start().unwrap();
//                         if rank == 0 {
//                             let target = target.lock().unwrap();
//                             for i in 0..WORLD_SIZE {
//                                 comm.send(&[target.as_ref()], i as i64).unwrap();
//                             }
//                         }
//                         let target = Tensor::zeros(
//                             &[BATCH_SIZE, SEQ_LEN, OUT_FEATURES],
//                             (Kind::Float, device),
//                         );
//                         comm.recv(&[target.shallow_clone()], 0).unwrap();
//                         comm.group_end().unwrap();

//                         barrier.wait();

//                         (input, target)
//                     };

//                     let mut optimizer_tp = Distro::new(
//                         &vs_tp,
//                         COMPRESSION_DECAY,
//                         COMPRESSION_CHUNK,
//                         COMPRESSION_TOPK,
//                         WEIGHT_DECAY,
//                         Some(comm.clone()),
//                     );

//                     for step in 0..NUM_STEPS {
//                         let current_rank_results = run_distro_step(
//                             step,
//                             &model_tp,
//                             &input,
//                             &target,
//                             &mut optimizer_tp,
//                             TEST_LR,
//                             all_rank_results_tp.clone(),
//                             rank,
//                             WORLD_SIZE,
//                             &Some(comm.clone()),
//                             &barrier,
//                         )?;
//                         let _ = current_rank_results;
//                         barrier.wait();
//                     }

//                     let unsharded_vars = unsharded_cpu_variables(&vs_tp, Some(comm.clone()))?;
//                     if rank == 0 {
//                         *final_weights_tp_rank0.lock().unwrap() = unsharded_vars;
//                     }

//                     Ok(())
//                 },
//             );
//         }

//         let final_weights_tp = final_weights_tp_rank0.lock().unwrap();

//         assert_eq!(
//             final_weights_non_tp.len(),
//             final_weights_tp.len(),
//             "Number of parameters differs between TP and non-TP runs."
//         );

//         for (name, non_tp_tensor) in &final_weights_non_tp {
//             let tp_tensor = final_weights_tp
//                 .get(name)
//                 .ok_or_else(|| anyhow::anyhow!("Parameter '{}' missing in TP results", name))?;

//             assert_eq!(
//                 non_tp_tensor.size(),
//                 tp_tensor.size(),
//                 "Shape mismatch for parameter '{}': Non-TP {:?}, TP {:?}",
//                 name,
//                 non_tp_tensor.size(),
//                 tp_tensor.size()
//             );

//             assert!(
//                 non_tp_tensor.allclose(tp_tensor, 1e-5, 1e-4, false),
//                 "Parameter '{}' differs significantly between TP and non-TP runs.\nNon-TP:\n{}\nTP:\n{}", name, non_tp_tensor, tp_tensor
//             );
//         }

//         Ok(())
//     }
// }
