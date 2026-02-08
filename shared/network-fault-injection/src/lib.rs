//! Network fault injection for stress testing Psyche distributed training.
//!
//! This crate provides deterministic, reproducible fault injection capabilities
//! for testing network resilience in heterogeneous MatFormer training.
//!
//! # Example
//!
//! ```rust
//! use psyche_network_fault_injection::{FaultConfig, LatencyConfig, Distribution};
//!
//! let config = FaultConfig::new(42)  // Seed for reproducibility
//!     .with_latency(LatencyConfig {
//!         base_ms: 50,
//!         jitter_ms: 20,
//!         distribution: Distribution::Normal { std_dev: 10.0 },
//!     })
//!     .with_packet_loss(0.1);  // 10% packet loss
//!
//! // In your network code:
//! // config.inject_latency().await;
//! // if config.should_drop_packet() { return Err(...); }
//! ```

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution as RandDistribution, Exp, Normal};
use serde::{Deserialize, Serialize};
use std::sync::{atomic::AtomicU64, Arc, Mutex};
use thiserror::Error;
use tokio::time::{sleep, Duration};
use tracing::{debug, trace, warn};

/// Errors that can occur during fault injection.
#[derive(Error, Debug)]
pub enum FaultInjectionError {
    #[error("Simulated packet drop")]
    SimulatedDrop,

    #[error("Simulated timeout after {0}ms")]
    SimulatedTimeout(u64),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Statistical distribution for latency injection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Distribution {
    /// Uniform distribution: latency = base_ms +/- jitter_ms
    Uniform,

    /// Normal/Gaussian distribution with specified standard deviation
    Normal { std_dev: f64 },

    /// Exponential distribution with specified rate parameter
    Exponential { rate: f64 },
}

impl Default for Distribution {
    fn default() -> Self {
        Distribution::Uniform
    }
}

/// Configuration for latency injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    /// Base latency in milliseconds
    pub base_ms: u64,

    /// Jitter range in milliseconds (+/- for uniform, scaling for others)
    pub jitter_ms: u64,

    /// Statistical distribution to use
    pub distribution: Distribution,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            base_ms: 0,
            jitter_ms: 0,
            distribution: Distribution::Uniform,
        }
    }
}

impl LatencyConfig {
    /// Create a new latency config from a string like "50-100" (base-jitter).
    pub fn from_str(s: &str) -> Result<Self, FaultInjectionError> {
        let parts: Vec<&str> = s.split('-').collect();
        match parts.len() {
            1 => {
                let base_ms = parts[0].parse::<u64>().map_err(|_| {
                    FaultInjectionError::InvalidConfig(format!("Invalid latency value: {}", s))
                })?;
                Ok(Self {
                    base_ms,
                    jitter_ms: 0,
                    distribution: Distribution::Uniform,
                })
            }
            2 => {
                let base_ms = parts[0].parse::<u64>().map_err(|_| {
                    FaultInjectionError::InvalidConfig(format!(
                        "Invalid base latency: {}",
                        parts[0]
                    ))
                })?;
                let jitter_ms = parts[1].parse::<u64>().map_err(|_| {
                    FaultInjectionError::InvalidConfig(format!("Invalid jitter: {}", parts[1]))
                })?;
                Ok(Self {
                    base_ms,
                    jitter_ms,
                    distribution: Distribution::Uniform,
                })
            }
            _ => Err(FaultInjectionError::InvalidConfig(format!(
                "Invalid latency format '{}', expected 'base' or 'base-jitter'",
                s
            ))),
        }
    }

    /// Sample a latency value using the configured distribution.
    fn sample(&self, rng: &mut StdRng) -> u64 {
        if self.jitter_ms == 0 {
            return self.base_ms;
        }

        match self.distribution {
            Distribution::Uniform => {
                let jitter = rng.gen_range(0..=self.jitter_ms * 2) as i64 - self.jitter_ms as i64;
                (self.base_ms as i64 + jitter).max(0) as u64
            }
            Distribution::Normal { std_dev } => {
                let normal = Normal::new(self.base_ms as f64, std_dev).unwrap();
                normal.sample(rng).max(0.0) as u64
            }
            Distribution::Exponential { rate } => {
                let exp = Exp::new(rate).unwrap();
                let extra: f64 = exp.sample(rng);
                self.base_ms + (extra * self.jitter_ms as f64) as u64
            }
        }
    }
}

/// Metrics collected during fault injection.
#[derive(Debug, Default)]
pub struct FaultMetrics {
    /// Total latency injected in milliseconds
    pub total_latency_injected_ms: AtomicU64,

    /// Number of packets dropped
    pub packets_dropped: AtomicU64,

    /// Number of operations that experienced latency
    pub latency_events: AtomicU64,
}

impl FaultMetrics {
    pub fn total_latency_injected_ms(&self) -> u64 {
        self.total_latency_injected_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn packets_dropped(&self) -> u64 {
        self.packets_dropped
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn latency_events(&self) -> u64 {
        self.latency_events
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Main fault injection configuration.
///
/// This struct is thread-safe and can be shared across async tasks.
#[derive(Debug, Clone)]
pub struct FaultConfig {
    /// Latency injection configuration (None = disabled)
    latency: Option<LatencyConfig>,

    /// Packet loss probability (0.0 to 1.0)
    packet_loss: Option<f64>,

    /// Bandwidth limit in bytes per second (None = unlimited)
    bandwidth_limit: Option<u64>,

    /// Random number generator (seeded for reproducibility)
    rng: Arc<Mutex<StdRng>>,

    /// Collected metrics
    metrics: Arc<FaultMetrics>,

    /// Whether fault injection is enabled
    enabled: bool,
}

impl FaultConfig {
    /// Create a new fault configuration with a specific seed for reproducibility.
    pub fn new(seed: u64) -> Self {
        Self {
            latency: None,
            packet_loss: None,
            bandwidth_limit: None,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
            metrics: Arc::new(FaultMetrics::default()),
            enabled: true,
        }
    }

    /// Create a disabled (no-op) fault configuration.
    pub fn disabled() -> Self {
        Self {
            latency: None,
            packet_loss: None,
            bandwidth_limit: None,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0))),
            metrics: Arc::new(FaultMetrics::default()),
            enabled: false,
        }
    }

    /// Check if any fault injection is configured.
    pub fn is_active(&self) -> bool {
        self.enabled && (self.latency.is_some() || self.packet_loss.is_some())
    }

    /// Configure latency injection.
    pub fn with_latency(mut self, config: LatencyConfig) -> Self {
        self.latency = Some(config);
        self
    }

    /// Configure latency from a string like "50-100".
    pub fn with_latency_str(mut self, s: &str) -> Result<Self, FaultInjectionError> {
        self.latency = Some(LatencyConfig::from_str(s)?);
        Ok(self)
    }

    /// Configure packet loss probability (0.0 to 1.0).
    pub fn with_packet_loss(mut self, probability: f64) -> Self {
        if probability > 0.0 && probability <= 1.0 {
            self.packet_loss = Some(probability);
        }
        self
    }

    /// Configure bandwidth limit in bytes per second.
    pub fn with_bandwidth_limit(mut self, bytes_per_sec: u64) -> Self {
        if bytes_per_sec > 0 {
            self.bandwidth_limit = Some(bytes_per_sec);
        }
        self
    }

    /// Get a reference to the collected metrics.
    pub fn metrics(&self) -> &FaultMetrics {
        &self.metrics
    }

    /// Inject latency according to configuration.
    ///
    /// This is an async function that will sleep for the configured duration.
    /// Returns the actual latency injected in milliseconds.
    pub async fn inject_latency(&self) -> u64 {
        if !self.enabled {
            return 0;
        }

        if let Some(ref latency_config) = self.latency {
            let latency_ms = {
                let mut rng = self.rng.lock().unwrap();
                latency_config.sample(&mut rng)
            };

            if latency_ms > 0 {
                trace!(latency_ms, "Injecting latency");
                sleep(Duration::from_millis(latency_ms)).await;

                self.metrics
                    .total_latency_injected_ms
                    .fetch_add(latency_ms, std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .latency_events
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            return latency_ms;
        }

        0
    }

    /// Check if a packet should be dropped.
    ///
    /// Returns true if the packet should be dropped (simulated loss).
    pub fn should_drop_packet(&self) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(probability) = self.packet_loss {
            let drop = {
                let mut rng = self.rng.lock().unwrap();
                rng.gen::<f64>() < probability
            };

            if drop {
                debug!(probability, "Simulating packet drop");
                self.metrics
                    .packets_dropped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            return drop;
        }

        false
    }

    /// Calculate delay for bandwidth limiting.
    ///
    /// Returns the delay in milliseconds needed to simulate the bandwidth limit
    /// for a transfer of the given size.
    pub fn bandwidth_delay(&self, bytes: u64) -> u64 {
        if !self.enabled {
            return 0;
        }

        if let Some(limit) = self.bandwidth_limit {
            // Time to transfer at limit = bytes / (bytes_per_sec)
            // Convert to milliseconds
            let delay_ms = (bytes * 1000) / limit;
            if delay_ms > 0 {
                trace!(bytes, limit, delay_ms, "Bandwidth limiting");
            }
            return delay_ms;
        }

        0
    }

    /// Apply bandwidth limiting for a transfer.
    ///
    /// Sleeps for the appropriate duration based on transfer size.
    pub async fn apply_bandwidth_limit(&self, bytes: u64) {
        let delay_ms = self.bandwidth_delay(bytes);
        if delay_ms > 0 {
            sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    /// Log a summary of fault injection configuration.
    pub fn log_summary(&self) {
        if !self.enabled {
            debug!("Fault injection: DISABLED");
            return;
        }

        let mut parts = Vec::new();

        if let Some(ref latency) = self.latency {
            parts.push(format!(
                "latency={}ms+/-{}ms ({:?})",
                latency.base_ms, latency.jitter_ms, latency.distribution
            ));
        }

        if let Some(loss) = self.packet_loss {
            parts.push(format!("packet_loss={:.1}%", loss * 100.0));
        }

        if let Some(limit) = self.bandwidth_limit {
            parts.push(format!("bandwidth_limit={} B/s", limit));
        }

        if parts.is_empty() {
            debug!("Fault injection: ENABLED (no faults configured)");
        } else {
            tracing::info!("Fault injection: ENABLED - {}", parts.join(", "));
        }
    }
}

/// Builder for creating FaultConfig from CLI arguments.
#[derive(Debug, Default)]
pub struct FaultConfigBuilder {
    latency_str: Option<String>,
    packet_loss: Option<f64>,
    bandwidth_limit: Option<u64>,
    seed: Option<u64>,
}

impl FaultConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn latency(mut self, latency_str: Option<String>) -> Self {
        self.latency_str = latency_str;
        self
    }

    pub fn packet_loss(mut self, probability: Option<f64>) -> Self {
        self.packet_loss = probability;
        self
    }

    pub fn bandwidth_limit(mut self, bytes_per_sec: Option<u64>) -> Self {
        self.bandwidth_limit = bytes_per_sec;
        self
    }

    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// Build the FaultConfig.
    ///
    /// Returns None if no fault injection is configured.
    pub fn build(self) -> Option<FaultConfig> {
        let has_faults = self.latency_str.is_some()
            || self.packet_loss.is_some()
            || self.bandwidth_limit.is_some();

        if !has_faults {
            return None;
        }

        let seed = self.seed.unwrap_or_else(|| {
            // Use current time as seed if not specified
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(42)
        });

        let mut config = FaultConfig::new(seed);

        if let Some(latency_str) = self.latency_str {
            match LatencyConfig::from_str(&latency_str) {
                Ok(latency_config) => {
                    config = config.with_latency(latency_config);
                }
                Err(e) => {
                    warn!("Invalid latency config '{}': {}", latency_str, e);
                }
            }
        }

        if let Some(probability) = self.packet_loss {
            config = config.with_packet_loss(probability);
        }

        if let Some(limit) = self.bandwidth_limit {
            config = config.with_bandwidth_limit(limit);
        }

        Some(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_config_from_str() {
        // Simple base latency
        let config = LatencyConfig::from_str("100").unwrap();
        assert_eq!(config.base_ms, 100);
        assert_eq!(config.jitter_ms, 0);

        // Base with jitter
        let config = LatencyConfig::from_str("50-20").unwrap();
        assert_eq!(config.base_ms, 50);
        assert_eq!(config.jitter_ms, 20);

        // Invalid format
        assert!(LatencyConfig::from_str("invalid").is_err());
        assert!(LatencyConfig::from_str("1-2-3").is_err());
    }

    #[test]
    fn test_latency_sampling() {
        let config = LatencyConfig {
            base_ms: 100,
            jitter_ms: 10,
            distribution: Distribution::Uniform,
        };

        let mut rng = StdRng::seed_from_u64(42);

        // Sample should be within range
        for _ in 0..100 {
            let sample = config.sample(&mut rng);
            assert!(
                sample >= 90 && sample <= 110,
                "Sample {} out of range",
                sample
            );
        }
    }

    #[test]
    fn test_packet_loss_deterministic() {
        let config = FaultConfig::new(42).with_packet_loss(0.5);

        // With same seed, results should be deterministic
        let mut drops = 0;
        for _ in 0..1000 {
            if config.should_drop_packet() {
                drops += 1;
            }
        }

        // Reset and test again
        let config2 = FaultConfig::new(42).with_packet_loss(0.5);
        let mut drops2 = 0;
        for _ in 0..1000 {
            if config2.should_drop_packet() {
                drops2 += 1;
            }
        }

        assert_eq!(drops, drops2, "Deterministic packet loss failed");
    }

    #[test]
    fn test_disabled_config() {
        let config = FaultConfig::disabled();
        assert!(!config.is_active());
        assert!(!config.should_drop_packet());
        assert_eq!(config.bandwidth_delay(1000), 0);
    }

    #[test]
    fn test_bandwidth_delay() {
        let config = FaultConfig::new(42).with_bandwidth_limit(1000); // 1 KB/s

        // 1000 bytes at 1 KB/s = 1 second = 1000ms
        assert_eq!(config.bandwidth_delay(1000), 1000);

        // 500 bytes at 1 KB/s = 0.5 seconds = 500ms
        assert_eq!(config.bandwidth_delay(500), 500);
    }

    #[tokio::test]
    async fn test_inject_latency() {
        let config = FaultConfig::new(42).with_latency(LatencyConfig {
            base_ms: 10,
            jitter_ms: 0,
            distribution: Distribution::Uniform,
        });

        let start = std::time::Instant::now();
        let latency = config.inject_latency().await;
        let elapsed = start.elapsed().as_millis();

        assert_eq!(latency, 10);
        assert!(
            elapsed >= 10 && elapsed < 50,
            "Elapsed {} not in range",
            elapsed
        );
        assert_eq!(config.metrics().latency_events(), 1);
    }

    #[test]
    fn test_builder() {
        // No faults configured
        let config = FaultConfigBuilder::new().build();
        assert!(config.is_none());

        // With latency
        let config = FaultConfigBuilder::new()
            .latency(Some("50-20".to_string()))
            .seed(Some(42))
            .build();
        assert!(config.is_some());
        assert!(config.unwrap().is_active());

        // With packet loss
        let config = FaultConfigBuilder::new().packet_loss(Some(0.1)).build();
        assert!(config.is_some());
    }
}
