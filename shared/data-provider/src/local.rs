use anyhow::{anyhow, bail, Result};
use psyche_core::{BatchId, ClosedInterval, Shuffle, TokenSize};
use rand::seq::SliceRandom;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs;
use tracing::{debug, info, warn};

use crate::{
    file_extensions::DATA_FILE_EXTENSIONS,
    traits::{LengthKnownDataProvider, TokenizedDataProvider},
    TokenizedData,
};

/// Magic number for modded-nanogpt binary format: 20240520
const MODDED_NANOGPT_MAGIC: u32 = 20240520;
/// Header size for modded-nanogpt format
const MODDED_NANOGPT_HEADER_SIZE: usize = 1024;

/// Detected binary data format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// Raw binary tokens (no header)
    Raw,
    /// modded-nanogpt format with 1024-byte header
    ModdedNanogpt { version: u32, token_count: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalDataSplit {
    All,
    Train,
    Validation,
}

impl DataFormat {
    /// Detect the format from the first bytes of a file
    pub fn detect(data: &[u8]) -> Self {
        if data.len() >= 12 {
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == MODDED_NANOGPT_MAGIC {
                let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                let token_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                return DataFormat::ModdedNanogpt {
                    version,
                    token_count,
                };
            }
        }
        DataFormat::Raw
    }

    /// Get the byte offset where actual token data starts
    pub fn header_size(&self) -> usize {
        match self {
            DataFormat::Raw => 0,
            DataFormat::ModdedNanogpt { .. } => MODDED_NANOGPT_HEADER_SIZE,
        }
    }
}

fn is_truthy_env_bool(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "1" | "true" | "yes")
}

fn is_validation_filename(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|x| x.to_str())
        .map(|name| {
            let name = name.to_ascii_lowercase();
            name.contains("val") || name.contains("validation")
        })
        .unwrap_or(false)
}

fn mmap_file(p: &std::path::PathBuf) -> Result<Box<dyn AsRef<[u8]> + Send>> {
    let file = std::fs::File::open(p)?;

    // try to mmap first, only falling back to read if allowed
    match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(mmap) => Ok(Box::new(mmap)),
        Err(e)
            if e.raw_os_error() == Some(22)
                && std::env::var("ALLOW_FAIL_MMAP")
                    .map(|v| is_truthy_env_bool(&v))
                    .unwrap_or(false) =>
        {
            eprintln!("mmap failed (likely under Valgrind), falling back to file read");
            let data = std::fs::read(p)?;
            Ok(Box::new(data))
        }
        Err(e) => Err(e.into()),
    }
}

struct SequencePointer {
    file_index: usize,
    byte_offset: usize,
}

/// Per-file metadata including detected format
#[derive(Debug)]
struct FileMetadata {
    format: DataFormat,
    #[allow(dead_code)]
    header_offset: usize,
}

pub struct LocalDataProvider {
    data_files: Vec<Box<dyn AsRef<[u8]> + Send>>,
    file_metadata: Vec<FileMetadata>,
    sequences: Vec<SequencePointer>,
    seq_len: usize,
    token_size_in_bytes: TokenSize,
}

impl LengthKnownDataProvider for LocalDataProvider {
    fn num_sequences(&self) -> usize {
        self.sequences.len()
    }
}

impl LocalDataProvider {
    /// Get the detected formats for all loaded data files
    pub fn detected_formats(&self) -> Vec<DataFormat> {
        self.file_metadata.iter().map(|m| m.format).collect()
    }

    pub fn new_from_directory(
        dir: impl AsRef<std::path::Path>,
        token_size_in_bytes: TokenSize,
        num_tokens_per_sequence: usize, // num tokens per sequence
        shuffle: Shuffle,
    ) -> Result<Self> {
        Self::new_from_directory_with_split(
            dir,
            token_size_in_bytes,
            num_tokens_per_sequence,
            shuffle,
            LocalDataSplit::All,
        )
    }

    pub fn new_from_directory_with_split(
        dir: impl AsRef<std::path::Path>,
        token_size_in_bytes: TokenSize,
        num_tokens_per_sequence: usize, // num tokens per sequence
        shuffle: Shuffle,
        split: LocalDataSplit,
    ) -> Result<Self> {
        let dir = std::fs::canonicalize(&dir)
            .map_err(|e| anyhow!("Failed to open data directory {:?}: {e}", dir.as_ref()))?;

        // Guardrail: if the dataset was generated via an explicit "fallback" path, refuse to
        // train unless the user opts in. This prevents silent "train on tiny fallback text"
        // failures.
        let fallback_sentinel = dir.join("FALLBACK_USED");
        if fallback_sentinel.exists()
            && !std::env::var("PSYCHE_ALLOW_FALLBACK_DATASET")
                .map(|v| is_truthy_env_bool(&v))
                .unwrap_or(false)
        {
            bail!(
                "Dataset directory {} contains `FALLBACK_USED` (fallback data). Refusing to continue. Set PSYCHE_ALLOW_FALLBACK_DATASET=1 to override.",
                dir.display()
            );
        }

        let mut bin_files = vec![];
        for file in std::fs::read_dir(&dir)
            .map_err(|e| anyhow!("couldn't load training data from {}: {e}", dir.display()))?
            .flatten()
        {
            let file = file.path();
            if let Some(extension) = file.extension().and_then(|s| s.to_str()) {
                if DATA_FILE_EXTENSIONS.contains(&extension) {
                    bin_files.push(file);
                }
            }
        }
        bin_files.sort();

        if split != LocalDataSplit::All {
            bin_files.retain(|path| {
                let is_val = is_validation_filename(path);
                match split {
                    LocalDataSplit::All => true,
                    LocalDataSplit::Train => !is_val,
                    LocalDataSplit::Validation => is_val,
                }
            });
        }

        let data_files = bin_files
            .iter()
            .map(mmap_file)
            .collect::<Result<Vec<_>>>()?;

        if data_files.is_empty() {
            bail!("No {:?} data files in directory {:?}", split, dir);
        }

        info!(
            "Loaded {} files ({}) of training data from directory {}",
            bin_files.len(),
            bin_files
                .iter()
                .map(|f| fs::metadata(f).unwrap().len())
                .sum::<u64>(),
            dir.display()
        );

        // Detect format for each file and store metadata
        let file_metadata: Vec<FileMetadata> = data_files
            .iter()
            .zip(bin_files.iter())
            .map(|(data, path)| {
                let data_slice = data.as_ref().as_ref();
                let format = DataFormat::detect(data_slice);
                let header_offset = format.header_size();

                match format {
                    DataFormat::Raw => {
                        debug!("File {:?}: Raw binary format (no header)", path.file_name());
                    }
                    DataFormat::ModdedNanogpt {
                        version,
                        token_count,
                    } => {
                        info!(
                            "File {:?}: modded-nanogpt format (version={}, tokens={})",
                            path.file_name(),
                            version,
                            token_count
                        );
                    }
                }

                FileMetadata {
                    format,
                    header_offset,
                }
            })
            .collect();

        let deterministic_rng = match shuffle {
            Shuffle::Seeded(random_seed) => Some(ChaCha8Rng::from_seed(random_seed)),
            Shuffle::DontShuffle => None,
        };
        let seq_len_in_bytes = num_tokens_per_sequence * usize::from(token_size_in_bytes);
        let min_len_in_bytes = seq_len_in_bytes + usize::from(token_size_in_bytes); // +1 token for pretraining data!

        let sequences: Vec<SequencePointer> = {
            let mut all_indexes: Vec<_> = data_files
                .iter()
                .zip(file_metadata.iter())
                .enumerate()
                // find every sequence in every file
                .flat_map(|(file_index, (current_tokens, metadata))| {
                    let file_len = current_tokens.as_ref().as_ref().len();
                    let header_offset = metadata.header_offset;

                    // Effective data length after skipping header
                    let data_len = file_len.saturating_sub(header_offset);

                    let pointers: Vec<SequencePointer> =
                        match data_len.checked_sub(min_len_in_bytes) {
                            Some(max_exclusive) => (0..=max_exclusive)
                                .step_by(seq_len_in_bytes)
                                .map(move |offset| SequencePointer {
                                    file_index,
                                    // byte_offset is relative to start of token data (after header)
                                    byte_offset: header_offset + offset,
                                })
                                .collect(),
                            None => {
                                warn!(
                                    file_index,
                                    file_len,
                                    header_offset,
                                    data_len,
                                    min_len_in_bytes,
                                    "Data file too small for seq_len; skipping"
                                );
                                Vec::new()
                            }
                        };
                    pointers.into_iter()
                })
                .collect();
            // and shuffle the whole collection, to avoid bias from a specific file
            if let Some(mut deterministic_rng) = deterministic_rng {
                all_indexes.shuffle(&mut deterministic_rng);
            }
            all_indexes
        };

        if sequences.is_empty() {
            bail!(
                "No sequences found in {}. Ensure files contain at least {} bytes (seq_len={} tokens, token_size={:?}).",
                dir.display(),
                min_len_in_bytes,
                num_tokens_per_sequence,
                token_size_in_bytes
            );
        }

        Ok(Self {
            data_files,
            file_metadata,
            sequences,
            seq_len: num_tokens_per_sequence,
            token_size_in_bytes,
        })
    }

    fn internal_get_samples(&self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        let mut ret: Vec<_> = Vec::new();
        for data_id in data_ids.iter() {
            let SequencePointer {
                byte_offset,
                file_index,
            } = self.sequences.get(data_id as usize).ok_or_else(|| {
                anyhow!(
                    "index {data_id} is out of bounds, we only have {} samples.",
                    self.sequences.len()
                )
            })?;

            let file = &self.data_files[*file_index];
            let data_len = usize::from(self.token_size_in_bytes) * (self.seq_len + 1);
            let data = &file.as_ref().as_ref()[*byte_offset..*byte_offset + data_len];

            let tokens: Vec<i32> = data
                .chunks(self.token_size_in_bytes.into())
                .map(|t| {
                    use TokenSize::*;
                    match self.token_size_in_bytes {
                        TwoBytes => u16::from_le_bytes(t.try_into().unwrap()) as i32,
                        FourBytes => u32::from_le_bytes(t.try_into().unwrap()) as i32,
                    }
                })
                .collect();
            ret.push(TokenizedData::from_input_ids(tokens));
        }
        Ok(ret)
    }
}

impl TokenizedDataProvider for LocalDataProvider {
    async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        self.internal_get_samples(data_ids)
    }
}

pub struct LocalDataProviderIter {
    provider: LocalDataProvider,
    current_index: u64,
}

impl Iterator for LocalDataProviderIter {
    type Item = TokenizedData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.provider.num_sequences() as u64 {
            let result = self
                .provider
                .internal_get_samples(BatchId(ClosedInterval::new(
                    self.current_index,
                    self.current_index,
                )))
                .unwrap()
                .pop()
                .unwrap();
            self.current_index += 1;
            Some(result)
        } else {
            None
        }
    }
}

impl IntoIterator for LocalDataProvider {
    type Item = TokenizedData;
    type IntoIter = LocalDataProviderIter;

    fn into_iter(self) -> Self::IntoIter {
        LocalDataProviderIter {
            provider: self,
            current_index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_format_detect_raw() {
        // Random data that doesn't have the magic number
        let data = vec![0u8; 100];
        assert_eq!(DataFormat::detect(&data), DataFormat::Raw);

        // Empty data
        let empty: [u8; 0] = [];
        assert_eq!(DataFormat::detect(&empty), DataFormat::Raw);

        // Too short for header
        let short = vec![0u8; 8];
        assert_eq!(DataFormat::detect(&short), DataFormat::Raw);
    }

    #[test]
    fn test_data_format_detect_modded_nanogpt() {
        // Create a valid modded-nanogpt header
        let mut data = vec![0u8; 1024];
        // Magic number: 20240520 = 0x01350088
        let magic: u32 = 20240520;
        data[0..4].copy_from_slice(&magic.to_le_bytes());
        // Version: 1
        let version: u32 = 1;
        data[4..8].copy_from_slice(&version.to_le_bytes());
        // Token count: 1000
        let token_count: u32 = 1000;
        data[8..12].copy_from_slice(&token_count.to_le_bytes());

        let detected = DataFormat::detect(&data);
        assert_eq!(
            detected,
            DataFormat::ModdedNanogpt {
                version: 1,
                token_count: 1000
            }
        );
    }

    #[test]
    fn test_data_format_header_size() {
        assert_eq!(DataFormat::Raw.header_size(), 0);
        assert_eq!(
            DataFormat::ModdedNanogpt {
                version: 1,
                token_count: 1000
            }
            .header_size(),
            1024
        );
    }
}
