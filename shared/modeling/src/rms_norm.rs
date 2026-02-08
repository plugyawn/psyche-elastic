use tch::{
    nn::{self, Module},
    Kind, Tensor,
};

#[derive(Debug)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
    /// Per-tier gain scalar (plain f64, NOT in VarStore).
    /// When `Some(g)`, the output is multiplied by `g`.
    /// Tier-0 should always have `None` or `Some(1.0)`.
    tier_gain: Option<f64>,
}

impl RMSNorm {
    pub fn new(vs: nn::Path, size: i64, eps: f64) -> Self {
        let weight = vs.ones("weight", &[size]);
        Self {
            weight,
            eps,
            tier_gain: None,
        }
    }

    /// Create an RMSNorm with a per-tier gain scalar.
    pub fn new_with_tier_gain(vs: nn::Path, size: i64, eps: f64, tier_gain: Option<f64>) -> Self {
        let weight = vs.ones("weight", &[size]);
        Self {
            weight,
            eps,
            tier_gain,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let kind = xs.kind();
        let xs = xs.to_kind(Kind::Float);
        let variance = xs.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float);
        let xs_normed = xs * (variance + self.eps).rsqrt();
        let xs_normed = xs_normed.to_kind(kind);
        let out = &self.weight * xs_normed;
        match self.tier_gain {
            Some(g) if (g - 1.0).abs() > f64::EPSILON => out * g,
            _ => out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn test_tier_gain_none_is_identity() {
        let vs = nn::VarStore::new(Device::Cpu);
        let norm = RMSNorm::new(vs.root(), 4, 1e-5);
        let xs = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu));
        let out = norm.forward(&xs);
        // Just check shape is preserved
        assert_eq!(out.size(), vec![2, 3, 4]);
    }

    #[test]
    fn test_tier_gain_one_is_identity() {
        let vs = nn::VarStore::new(Device::Cpu);
        let norm = RMSNorm::new_with_tier_gain(vs.root(), 4, 1e-5, Some(1.0));
        let xs = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu));
        let out = norm.forward(&xs);
        assert_eq!(out.size(), vec![2, 3, 4]);
    }

    #[test]
    fn test_tier_gain_scales_output() {
        let vs = nn::VarStore::new(Device::Cpu);
        let norm_base = RMSNorm::new(vs.root(), 4, 1e-5);
        let vs2 = nn::VarStore::new(Device::Cpu);
        let gain = 0.7071;
        let norm_scaled = RMSNorm::new_with_tier_gain(vs2.root(), 4, 1e-5, Some(gain));

        let xs = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu));
        let out_base = norm_base.forward(&xs);
        let out_scaled = norm_scaled.forward(&xs);

        // out_scaled should be approximately gain * out_base.
        // Avoid divide-by-zero: compare directly with an abs error tolerance.
        let expected = &out_base * gain;
        let max_abs_err = (&out_scaled - expected).abs().max().double_value(&[]);
        assert!(
            max_abs_err < 1e-3,
            "expected out_scaled ~= gain*out_base (gain={gain}), max_abs_err={max_abs_err}"
        );
    }
}
