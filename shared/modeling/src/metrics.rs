//! Metrics recording infrastructure for regression testing and golden baseline comparison.
//!
//! This module provides tools to record per-step training metrics in JSONL format,
//! enabling comparison between different implementations and detecting convergence regressions.

use crate::CausalLM;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

/// Metrics recorded for a single training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Training step number
    pub step: u32,
    /// Loss value for this step
    pub loss: f64,
    /// Learning rate at this step
    pub lr: f64,
    /// Global gradient norm (L2 norm across all parameters)
    pub global_grad_norm: f64,
    /// Global weight norm (L2 norm across all parameters)
    pub global_weight_norm: f64,
    /// Per-layer gradient norms (optional, for detailed analysis)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_grad_norms: Option<HashMap<String, f64>>,
    /// Per-layer weight norms (optional, for detailed analysis)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_weight_norms: Option<HashMap<String, f64>>,
    /// Timestamp in milliseconds since UNIX epoch
    pub timestamp_ms: u64,
    /// Optional metadata (e.g., optimizer type, batch size)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl StepMetrics {
    /// Create a new StepMetrics with current timestamp
    pub fn new(step: u32, loss: f64, lr: f64) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            step,
            loss,
            lr,
            global_grad_norm: 0.0,
            global_weight_norm: 0.0,
            layer_grad_norms: None,
            layer_weight_norms: None,
            timestamp_ms,
            metadata: None,
        }
    }

    /// Compute gradient and weight norms from model variables
    pub fn with_norms_from_model(mut self, model: &dyn CausalLM, detailed: bool) -> Self {
        let (global_grad_norm, global_weight_norm, layer_grad_norms, layer_weight_norms) =
            compute_norms(model, detailed);

        self.global_grad_norm = global_grad_norm;
        self.global_weight_norm = global_weight_norm;
        if detailed {
            self.layer_grad_norms = Some(layer_grad_norms);
            self.layer_weight_norms = Some(layer_weight_norms);
        }
        self
    }

    /// Add metadata key-value pair
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }
}

/// Compute global and per-layer gradient and weight norms from model
fn compute_norms(
    model: &dyn CausalLM,
    detailed: bool,
) -> (f64, f64, HashMap<String, f64>, HashMap<String, f64>) {
    let _guard = tch::no_grad_guard();

    let mut grad_squared_sum = 0.0f64;
    let mut weight_squared_sum = 0.0f64;
    let mut layer_grad_norms = HashMap::new();
    let mut layer_weight_norms = HashMap::new();

    for var in model.variables() {
        let tensor = var.local_tensor();
        let grad = tensor.grad();

        // Weight norm
        let weight_norm_sq: f64 = tensor
            .to_kind(tch::Kind::Float)
            .square()
            .sum(tch::Kind::Float)
            .double_value(&[]);
        weight_squared_sum += weight_norm_sq;

        // Gradient norm (only if gradient exists)
        let grad_norm_sq = if grad.defined() {
            grad.to_kind(tch::Kind::Float)
                .square()
                .sum(tch::Kind::Float)
                .double_value(&[])
        } else {
            0.0
        };
        grad_squared_sum += grad_norm_sq;

        if detailed {
            let name = var.name().to_string();
            layer_weight_norms.insert(name.clone(), weight_norm_sq.sqrt());
            layer_grad_norms.insert(name, grad_norm_sq.sqrt());
        }
    }

    (
        grad_squared_sum.sqrt(),
        weight_squared_sum.sqrt(),
        layer_grad_norms,
        layer_weight_norms,
    )
}

/// Configuration for metrics recording
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Whether metrics recording is enabled
    pub enabled: bool,
    /// Path to the JSONL output file
    pub output_path: PathBuf,
    /// Record detailed per-layer norms (increases file size)
    pub detailed_norms: bool,
    /// Record metrics every N steps (1 = every step)
    pub record_every_n_steps: u32,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_path: PathBuf::from("training_metrics.jsonl"),
            detailed_norms: false,
            record_every_n_steps: 1,
        }
    }
}

impl MetricsConfig {
    /// Create a new config with metrics enabled
    pub fn enabled(output_path: impl Into<PathBuf>) -> Self {
        Self {
            enabled: true,
            output_path: output_path.into(),
            ..Default::default()
        }
    }

    /// Enable detailed per-layer norm recording
    pub fn with_detailed_norms(mut self) -> Self {
        self.detailed_norms = true;
        self
    }

    /// Set recording frequency
    pub fn record_every(mut self, n: u32) -> Self {
        self.record_every_n_steps = n.max(1);
        self
    }
}

/// Recorder that writes metrics to a JSONL file
pub struct MetricsRecorder {
    config: MetricsConfig,
    writer: Option<BufWriter<File>>,
    steps_recorded: u32,
}

impl MetricsRecorder {
    /// Create a new metrics recorder with the given config
    pub fn new(config: MetricsConfig) -> std::io::Result<Self> {
        let writer = if config.enabled {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&config.output_path)?;
            Some(BufWriter::new(file))
        } else {
            None
        };

        Ok(Self {
            config,
            writer,
            steps_recorded: 0,
        })
    }

    /// Create a disabled recorder (no-op)
    pub fn disabled() -> Self {
        Self {
            config: MetricsConfig::default(),
            writer: None,
            steps_recorded: 0,
        }
    }

    /// Check if recording is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && self.writer.is_some()
    }

    /// Check if this step should be recorded based on config
    pub fn should_record(&self, step: u32) -> bool {
        self.is_enabled() && step.is_multiple_of(self.config.record_every_n_steps)
    }

    /// Record metrics for a training step
    pub fn record(&mut self, metrics: &StepMetrics) -> std::io::Result<()> {
        if let Some(writer) = &mut self.writer {
            let json = serde_json::to_string(metrics)?;
            writeln!(writer, "{}", json)?;
            writer.flush()?;
            self.steps_recorded += 1;
        }
        Ok(())
    }

    /// Record metrics from model state (convenience method)
    pub fn record_step(
        &mut self,
        step: u32,
        loss: f64,
        lr: f64,
        model: &dyn CausalLM,
    ) -> std::io::Result<()> {
        if !self.should_record(step) {
            return Ok(());
        }

        let metrics = StepMetrics::new(step, loss, lr)
            .with_norms_from_model(model, self.config.detailed_norms);

        self.record(&metrics)
    }

    /// Get the number of steps recorded
    pub fn steps_recorded(&self) -> u32 {
        self.steps_recorded
    }

    /// Get the output path
    pub fn output_path(&self) -> &Path {
        &self.config.output_path
    }

    /// Flush the writer
    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }
}

impl Drop for MetricsRecorder {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Load metrics from a JSONL file
pub fn load_metrics(path: impl AsRef<Path>) -> std::io::Result<Vec<StepMetrics>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut metrics = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let step_metrics: StepMetrics = serde_json::from_str(&line)?;
            metrics.push(step_metrics);
        }
    }

    Ok(metrics)
}

/// Comparison result between two metrics runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsComparison {
    /// Steps compared
    pub steps_compared: u32,
    /// Maximum absolute loss difference
    pub max_loss_diff: f64,
    /// Mean absolute loss difference
    pub mean_loss_diff: f64,
    /// Maximum relative loss difference (as percentage)
    pub max_loss_diff_pct: f64,
    /// Step with maximum loss difference
    pub max_diff_step: u32,
    /// Final loss difference
    pub final_loss_diff: f64,
    /// Whether the comparison passed (within tolerance)
    pub passed: bool,
    /// Tolerance used for comparison
    pub tolerance: f64,
}

/// Compare two metrics runs
pub fn compare_metrics(
    baseline: &[StepMetrics],
    current: &[StepMetrics],
    tolerance: f64,
) -> MetricsComparison {
    let steps_compared = baseline.len().min(current.len()) as u32;
    let mut max_loss_diff = 0.0f64;
    let mut max_loss_diff_pct = 0.0f64;
    let mut max_diff_step = 0u32;
    let mut total_diff = 0.0f64;

    for (b, c) in baseline.iter().zip(current.iter()) {
        let diff = (b.loss - c.loss).abs();
        let diff_pct = if b.loss.abs() > 1e-10 {
            diff / b.loss.abs() * 100.0
        } else {
            0.0
        };

        total_diff += diff;

        if diff > max_loss_diff {
            max_loss_diff = diff;
            max_loss_diff_pct = diff_pct;
            max_diff_step = b.step;
        }
    }

    let mean_loss_diff = if steps_compared > 0 {
        total_diff / steps_compared as f64
    } else {
        0.0
    };

    let final_loss_diff = if !baseline.is_empty() && !current.is_empty() {
        (baseline.last().unwrap().loss - current.last().unwrap().loss).abs()
    } else {
        0.0
    };

    let passed = max_loss_diff <= tolerance;

    MetricsComparison {
        steps_compared,
        max_loss_diff,
        mean_loss_diff,
        max_loss_diff_pct,
        max_diff_step,
        final_loss_diff,
        passed,
        tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_metrics_serialization() {
        let metrics = StepMetrics::new(100, 2.5, 0.001)
            .with_metadata("optimizer", "distro")
            .with_metadata("batch_size", "32");

        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: StepMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.step, 100);
        assert!((parsed.loss - 2.5).abs() < 1e-10);
        assert!((parsed.lr - 0.001).abs() < 1e-10);
        assert!(parsed.metadata.is_some());
    }

    #[test]
    fn test_metrics_comparison() {
        let baseline = vec![
            StepMetrics::new(0, 5.0, 0.001),
            StepMetrics::new(1, 4.0, 0.001),
            StepMetrics::new(2, 3.0, 0.001),
        ];

        let current = vec![
            StepMetrics::new(0, 5.01, 0.001),
            StepMetrics::new(1, 3.99, 0.001),
            StepMetrics::new(2, 3.02, 0.001),
        ];

        let comparison = compare_metrics(&baseline, &current, 0.05);
        assert!(comparison.passed);
        assert_eq!(comparison.steps_compared, 3);
        assert!(comparison.max_loss_diff < 0.05);
    }

    #[test]
    fn test_metrics_comparison_fail() {
        let baseline = vec![StepMetrics::new(0, 5.0, 0.001)];
        let current = vec![StepMetrics::new(0, 5.5, 0.001)];

        let comparison = compare_metrics(&baseline, &current, 0.1);
        assert!(!comparison.passed);
        assert!((comparison.max_loss_diff - 0.5).abs() < 1e-10);
    }
}
