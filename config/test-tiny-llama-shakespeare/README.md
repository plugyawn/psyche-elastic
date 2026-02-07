# Tiny LLaMA (local) + Tiny Shakespeare (local)

This config is like `config/test-tiny-llama/`, but uses a small local text dataset
(`data_location = { Local = "./data/tinyshakespeare-bin" }`) instead of dummy data.

To generate the dataset:

```bash
python3 scripts/prepare_tinyshakespeare_bin_dataset.py --out-dir data/tinyshakespeare-bin
```

If the download fails, the script exits non-zero by default (to avoid accidental training on tiny fallback text).

Then run the centralized server + clients pointing at this config's `state.toml`.
