//! Compute kernel abstraction layer
//!
//! This module provides a trait-based system for compute kernels with runtime dispatch.
//! Kernels can have multiple implementations (CPU, CUDA, etc.) that are selected at runtime
//! based on device availability and configuration.
//!
//! # Design
//!
//! The kernel system is designed around two main concepts:
//!
//! 1. **Kernel traits** - Define the interface for specific operations (e.g., `Matmul`, `Orthogonalize`)
//! 2. **Kernel dispatcher** - Selects the best available implementation at runtime
//!
//! # Example
//!
//! ```ignore
//! use psyche_modeling::kernels::{KernelDispatcher, Orthogonalize};
//!
//! let dispatcher = KernelDispatcher::new(KernelConfig::default());
//! let orthogonalizer = dispatcher.orthogonalizer();
//!
//! // Use the kernel
//! let g = orthogonalizer.polar_express(&gradient, 5);
//! ```

mod orthogonalize;

pub use orthogonalize::{CpuOrthogonalize, Orthogonalize};

use tch::Device;

/// Configuration for kernel selection
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Preferred device for computation
    pub device: Device,
    /// Whether to allow CPU fallbacks when GPU kernels are unavailable
    pub allow_cpu_fallback: bool,
    /// Enable verbose logging of kernel selection
    pub verbose: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            allow_cpu_fallback: true,
            verbose: false,
        }
    }
}

impl KernelConfig {
    /// Create a config for CPU computation
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            ..Default::default()
        }
    }

    /// Create a config for CUDA computation
    ///
    /// Note: CUDA kernels are a future feature. Currently falls back to CPU.
    pub fn cuda(device_index: usize) -> Self {
        Self {
            device: Device::Cuda(device_index),
            allow_cpu_fallback: true,
            ..Default::default()
        }
    }
}

/// Kernel dispatcher that selects implementations at runtime
pub struct KernelDispatcher {
    config: KernelConfig,
    orthogonalizer: Box<dyn Orthogonalize>,
}

impl KernelDispatcher {
    /// Create a new kernel dispatcher with the given configuration
    pub fn new(config: KernelConfig) -> Self {
        let orthogonalizer = Self::select_orthogonalizer(&config);

        Self {
            config,
            orthogonalizer,
        }
    }

    /// Get the orthogonalization kernel
    pub fn orthogonalizer(&self) -> &dyn Orthogonalize {
        self.orthogonalizer.as_ref()
    }

    /// Get the current configuration
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    /// Select the best orthogonalization kernel for the config
    fn select_orthogonalizer(config: &KernelConfig) -> Box<dyn Orthogonalize> {
        // Currently only CPU implementation available
        // Future: Add CUDA implementation selection here
        if config.verbose {
            tracing::info!("Selected CPU orthogonalization kernel (Polar Express)");
        }
        Box::new(CpuOrthogonalize::new())
    }
}

impl std::fmt::Debug for KernelDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelDispatcher")
            .field("config", &self.config)
            .field("orthogonalizer", &self.orthogonalizer.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_dispatcher_creation() {
        let config = KernelConfig::default();
        let dispatcher = KernelDispatcher::new(config);
        assert_eq!(dispatcher.orthogonalizer().name(), "cpu_polar_express");
    }
}
