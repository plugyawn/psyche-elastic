# psyche-elastic

<p align="center" width="100%">
    <img src="./psyche-book/src/psyche.jpg">
</p>

**psyche-elastic** is a fork of **Psyche** focused on elastic, heterogeneous training. It is **built on top of Psyche** (PsycheFoundation/psyche) and extends it with MatFormer tiered checkpoints, manifest-based slicing, and practical tooling for mixed‑hardware training.

> All commits authored by `plugyawn` are my contributions.

For canonical Psyche documentation, see https://docs.psyche.network (upstream). This README adds the elastic workflow and fork‑specific details.

---

## What This Unlocks

- **Heterogeneous training**: smaller devices train smaller tiers; large devices train full tiers; all contribute to shared weights.
- **Selective downloads**: tiered checkpoints pull only the files needed for a given tier (esp. from HF).
- **Safe tier mixing**: schema canonicalization + double‑slicing protection prevents mismatched runs.
- **Operational clarity**: explicit manifests + metadata give reliable tier detection and validation.

---

## MatFormer Workflow (End‑to‑End)

```mermaid
flowchart LR
  A[Universal checkpoint<br/>(tier 0)] --> B[export_matformer_tiers.py]
  B --> C[matformer_manifest.json]
  B --> D[Tier dirs<br/>-tier1/-tier2/...]
  C --> E[Client load strategy]
  D --> E
  E -->|auto| F[Sliced if complete; else universal]
  E -->|sliced| G[Require sliced or fail]
  E -->|universal| H[Always full]
```

Key points:
- Tier slices are **prefixes** of the FFN width: tier 1 = 1/2, tier 2 = 1/4, etc.
- Clients at smaller tiers update only prefix weights; shared prefixes receive gradients from all tiers.
- Helper mode (stochastic suffix sampling) is **wired but currently disabled** until sparse alignment/rotation is finalized.

---

## Quickstart

### Prereqs
- Rust toolchain
- Python (3.11/3.12 recommended) with **PyTorch** installed
- `tmux` (optional, for local testnet UI)

### Environment

Use the repo helper to point `tch-rs` to your Python torch install:

```bash
source scripts/psyche-env.sh
```

### Local Testnet (fastest path)

```bash
just local-testnet \
  --num-clients 3 \
  --config-path ./config/consilience-match-llama2-20m-fineweb-pretrain-dev/ \
  --client-matformer-tiers 0,1,2
```

If you don’t have `just`, use:

```bash
cargo run -p psyche-centralized-local-testnet -- start \
  --num-clients 3 \
  --config-path ./config/consilience-match-llama2-20m-fineweb-pretrain-dev/ \
  --client-matformer-tiers 0,1,2
```

---

## Tiered Checkpoints & Manifests

### Export Tier Slices

```bash
python scripts/export_matformer_tiers.py --src checkpoints/my-model --tiers 1 2
```

Produces:
- `checkpoints/my-model-tier1/`, `-tier2/` (sliced weights)
- `matformer_manifest.json` in the **universal** directory
- Tier configs with:
  - `matformer_tier`
  - `matformer_base_intermediate_size`
  - `intermediate_size` (sliced width)

### Manifest (Schema v1)

```json
{
  "schema_version": 1,
  "matformer_base_intermediate_size": 1024,
  "common_files": ["tokenizer.json"],
  "tiers": [
    {"tier": 1, "intermediate_size": 512, "files": ["../my-model-tier1/config.json", "../my-model-tier1/model.safetensors"]}
  ],
  "sha256": {"...": "..."}
}
```

Paths are **relative to the manifest location**; the loader normalizes for HF and rejects absolute paths.

### Load Strategy

- `auto` (default): use sliced if complete, otherwise fall back to universal.
- `sliced`: require the tier slice; fail fast if missing.
- `universal`: always load full checkpoint.

The loader prevents **double‑slicing** if a checkpoint is already tiered.

---

## How It Works (High Level)

```mermaid
graph TD
  C[Client joins] --> T[Select tier + load strategy]
  T --> L[Load checkpoint
(manifest-aware)]
  L --> M[Model runs at tier]
  M --> G[Gradients on prefix weights]
  G --> A[Aggregate by parameter name]
  A --> U[Update shared prefix]
```

---

## What Changed in This Fork

Key contributions (all `plugyawn` commits):

- MatFormer tier slices, manifests, and hub selective downloads
- Metadata inference + schema canonicalization for mixed tiers
- Helper‑mode infrastructure (temporarily disabled)
- Heterogeneous gradient aggregation normalization
- NanoGPT architecture + MatFormer support
- Optimizer and kernel work (Muon, Polar Express) with tests
- System metrics logging + fault injection tools
- Packaging and local testnet improvements

---

## Tests

```bash
source scripts/psyche-env.sh
cargo test -p psyche-client
```

(Additional tests live under `shared/` and `psyche-book`.)

---

## Notes & Limitations

- **Tensor parallelism with tier > 0** is not supported yet.
- **Helper mode** (suffix sampling) is disabled until sparse alignment/rotation is complete.
- Tier export currently assumes a **single `.safetensors` shard**.

---

## Upstream

This repo is built on top of the upstream **Psyche** project:
- Upstream: https://github.com/PsycheFoundation/psyche
- Docs: https://docs.psyche.network
