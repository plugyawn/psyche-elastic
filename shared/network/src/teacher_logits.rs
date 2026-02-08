//! Compressed teacher logits for in-place distillation.
//!
//! Tier-0 clients produce teacher logits during their forward pass.
//! These are compressed to top-k per token and broadcast to student (tier>0) clients
//! for knowledge distillation loss computation.

use psyche_core::BatchId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Compressed teacher logits: only top-k indices and values per token.
///
/// Size estimate: batch=4, seq=2048, top_k=32 → ~1MB
/// (4 * 2048 * 32 * (2 + 2) bytes = 1,048,576 bytes)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CompressedTeacherLogits {
    /// Vocabulary indices of top-k logits, flattened: [batch × seq × top_k].
    /// Stored as u16 (supports vocab up to 65535; use u32 variant for larger).
    pub top_indices: Vec<u16>,

    /// Raw f16 bits of top-k logit values, flattened: [batch × seq × top_k].
    /// Stored as u16 (IEEE 754 half-precision bit representation).
    pub top_values_f16: Vec<u16>,

    pub batch_size: u16,
    pub seq_len: u16,
    pub top_k: u16,

    /// Temperature used when producing these logits (for KL divergence computation).
    pub temperature: f32,
}

impl CompressedTeacherLogits {
    /// Expected number of elements in top_indices / top_values_f16.
    pub fn expected_len(&self) -> usize {
        (self.batch_size as usize) * (self.seq_len as usize) * (self.top_k as usize)
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), TeacherLogitsError> {
        let expected = self.expected_len();
        if self.top_indices.len() != expected {
            return Err(TeacherLogitsError::ShapeMismatch {
                field: "top_indices",
                expected,
                actual: self.top_indices.len(),
            });
        }
        if self.top_values_f16.len() != expected {
            return Err(TeacherLogitsError::ShapeMismatch {
                field: "top_values_f16",
                expected,
                actual: self.top_values_f16.len(),
            });
        }
        if self.top_k == 0 {
            return Err(TeacherLogitsError::ZeroTopK);
        }
        Ok(())
    }
}

/// A teacher logit blob with metadata for network transmission.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransmittableTeacherLogits {
    pub step: u32,
    pub batch_id: BatchId,
    pub logits: CompressedTeacherLogits,
}

impl TransmittableTeacherLogits {
    /// Deterministic SHA-256 hash for witness verification.
    /// Mirrors the pattern from `TransmittableDistroResult::comptue_hash()`.
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.step.to_be_bytes());
        hasher.update(self.batch_id.0.start.to_be_bytes());
        hasher.update(self.batch_id.0.end.to_be_bytes());
        hasher.update(self.logits.batch_size.to_be_bytes());
        hasher.update(self.logits.seq_len.to_be_bytes());
        hasher.update(self.logits.top_k.to_be_bytes());
        hasher.update(self.logits.temperature.to_be_bytes());
        // Hash indices
        for idx in &self.logits.top_indices {
            hasher.update(idx.to_be_bytes());
        }
        // Hash values
        for val in &self.logits.top_values_f16 {
            hasher.update(val.to_be_bytes());
        }
        hasher.finalize().into()
    }

    /// Serialized size estimate in bytes.
    pub fn estimated_size(&self) -> usize {
        let n = self.logits.expected_len();
        // 2 bytes per index + 2 bytes per value + metadata overhead
        n * 4 + 64
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TeacherLogitsError {
    #[error("{field} length mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("top_k must be > 0")]
    ZeroTopK,
}

#[cfg(test)]
mod tests {
    use super::*;
    use psyche_core::ClosedInterval;

    fn make_test_logits(batch_size: u16, seq_len: u16, top_k: u16) -> CompressedTeacherLogits {
        let n = (batch_size as usize) * (seq_len as usize) * (top_k as usize);
        CompressedTeacherLogits {
            top_indices: (0..n).map(|i| (i % 1000) as u16).collect(),
            top_values_f16: vec![0x3C00; n], // f16 representation of 1.0
            batch_size,
            seq_len,
            top_k,
            temperature: 2.0,
        }
    }

    #[test]
    fn test_validate_ok() {
        let logits = make_test_logits(4, 128, 32);
        assert!(logits.validate().is_ok());
    }

    #[test]
    fn test_validate_shape_mismatch() {
        let mut logits = make_test_logits(4, 128, 32);
        logits.top_indices.pop();
        assert!(logits.validate().is_err());
    }

    #[test]
    fn test_validate_zero_top_k() {
        let logits = CompressedTeacherLogits {
            top_indices: vec![],
            top_values_f16: vec![],
            batch_size: 1,
            seq_len: 1,
            top_k: 0,
            temperature: 2.0,
        };
        assert!(logits.validate().is_err());
    }

    #[test]
    fn test_hash_determinism() {
        let logits = make_test_logits(2, 64, 16);
        let transmittable = TransmittableTeacherLogits {
            step: 42,
            batch_id: BatchId(ClosedInterval::new(0, 4)),
            logits,
        };
        let hash1 = transmittable.compute_hash();
        let hash2 = transmittable.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_changes_with_step() {
        let logits = make_test_logits(2, 64, 16);
        let t1 = TransmittableTeacherLogits {
            step: 1,
            batch_id: BatchId(ClosedInterval::new(0, 4)),
            logits: logits.clone(),
        };
        let t2 = TransmittableTeacherLogits {
            step: 2,
            batch_id: BatchId(ClosedInterval::new(0, 4)),
            logits,
        };
        assert_ne!(t1.compute_hash(), t2.compute_hash());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let logits = make_test_logits(2, 32, 8);
        let transmittable = TransmittableTeacherLogits {
            step: 10,
            batch_id: BatchId(ClosedInterval::new(0, 2)),
            logits,
        };
        let bytes = postcard::to_allocvec(&transmittable).unwrap();
        let decoded: TransmittableTeacherLogits = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.step, transmittable.step);
        assert_eq!(decoded.logits.batch_size, transmittable.logits.batch_size);
        assert_eq!(decoded.logits.seq_len, transmittable.logits.seq_len);
        assert_eq!(decoded.logits.top_k, transmittable.logits.top_k);
        assert_eq!(decoded.logits.top_indices.len(), transmittable.logits.top_indices.len());
        assert_eq!(decoded.logits.top_values_f16.len(), transmittable.logits.top_values_f16.len());
        // Hash should match after roundtrip
        assert_eq!(decoded.compute_hash(), transmittable.compute_hash());
    }

    #[test]
    fn test_estimated_size() {
        let logits = make_test_logits(4, 2048, 32);
        let transmittable = TransmittableTeacherLogits {
            step: 1,
            batch_id: BatchId(ClosedInterval::new(0, 4)),
            logits,
        };
        // 4 * 2048 * 32 * 4 + 64 = 1,048,640
        assert_eq!(transmittable.estimated_size(), 1_048_640);
    }
}
