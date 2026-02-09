# DEVLOG

## Unreleased
- Added prefix-only DisTrO apply alignment for MatFormer MLP weights (pad/slice after decode) plus lazy DCT basis generation for unseen chunk sizes; includes targeted unit tests in `shared/modeling/src/distro.rs`.
- Added schema hash handshake: `ModelSchemaInfo` in watcher, client emits local+canonical schema hashes after model init, centralized server stores canonical hash and rejects mismatches via `ClientToServerMessage::Schema`.
- Enabled `LLMTrainingDataLocation::Local` in client init (download or local dir) and added deterministic/local dataset guards in `shared/data-provider/src/local.rs` (sorted file order, skip undersized files, fail on empty sequences, fallback sentinel gate).
- Added `scripts/prepare_tinyshakespeare_bin_dataset.py` to build byte-level `train.bin` + `meta.json`, with explicit fallback opt-in and sentinel file.
- Added reproducible Python runtime bootstrap via `scripts/bootstrap-python-runtime.sh` and pinned deps in `packaging/python-runtime/requirements.lock.txt` + `pyproject.toml`.
- Added macOS Gatekeeper helper `scripts/unquarantine.sh` and wired packaging to include runtime/bootstrap scripts, configs, and optional local dataset in `scripts/package-centralized-mac.sh`.
- Added local-data test config + README in `config/test-tiny-llama-shakespeare`.
- Updated `scripts/psyche-env.sh` to hint bootstrap script when torch import fails.
- Ignored `.uv-cache/` and `/papers` in `.gitignore`.
