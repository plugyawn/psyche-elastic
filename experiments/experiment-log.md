# Experiment Log

## 2026-01-30 04:40 UTC — SSSL (1-bit signSGD, disjoint batches)
- config: `config/test-nanogpt-distro-shakespeare-sssl`
- command: `.cargo-target/release/psyche-centralized-local-testnet start --headless --headless-exit-after-secs 300 --server-port 20100 --num-clients 4 --config-path ./config/test-nanogpt-distro-shakespeare-sssl --client-matformer-tiers 0,1,1,1 --client-matformer-helper-fractions 0,0,0,0 --write-distro-data ./logs/distro-sssl --tui false`
- data assignment: disjoint across trainers (default coordinator behavior)
- metrics (S_sum vs L): sign agreement overall 0.6930; MLP prefix 0.5135
- metrics (applied update vs L): sign agreement overall 0.8524; MLP prefix 0.6720
- counts: total_all=580,117,248; total_prefix=53,477,376; steps=17
- loss (approx): tier0 ~10.79→8.43 by step ~17; tier1 ~10.88→8.47

## 2026-01-30 04:41 UTC — SSSL (FP, disjoint batches)
- config: `config/test-nanogpt-distro-shakespeare-sssl-fp`
- command: `.cargo-target/release/psyche-centralized-local-testnet start --headless --headless-exit-after-secs 300 --server-port 20101 --num-clients 4 --config-path ./config/test-nanogpt-distro-shakespeare-sssl-fp --client-matformer-tiers 0,1,1,1 --client-matformer-helper-fractions 0,0,0,0 --write-distro-data ./logs/distro-sssl-fp --tui false`
- data assignment: disjoint across trainers (default coordinator behavior)
- metrics (S_sum vs L): sign agreement overall 0.6952; MLP prefix 0.5156
- metrics (applied update vs L): sign agreement overall 0.8484; MLP prefix 0.6460
- counts: total_all=443,619,072; total_prefix=40,894,464; steps=13

## 2026-01-30 04:41 UTC — SSSL (disjoint env var artifact)
- config: `config/test-nanogpt-distro-shakespeare-sssl`
- write_distro_data: `./logs/distro-sssl-disjoint`
- note: `PSYCHE_DONT_SHARE_PREFIX` leaked to all clients; prefix agreement 1.0 is an artifact

## 2026-01-30 04:42 UTC — L-only baseline
- config: `config/test-nanogpt-distro-shakespeare-lonly`
- write_distro_data: `./logs/distro-lonly`
- loss (approx): ~10.88→9.08 by step 18

## 2026-01-30 04:46 UTC — SSSL (1-bit signSGD, same-batch override)
- config: `config/test-nanogpt-distro-shakespeare-sssl`
- command: `PSYCHE_FORCE_SAME_BATCH=1 ./target/release/psyche-centralized-local-testnet start --headless --headless-exit-after-secs 180 --server-port 20102 --num-clients 4 --config-path ./config/test-nanogpt-distro-shakespeare-sssl --client-matformer-tiers 0,1,1,1 --client-matformer-helper-fractions 0,0,0,0 --write-distro-data ./logs/distro-sssl-samebatch --tui false`
- data override: `PSYCHE_FORCE_SAME_BATCH=1` (server returns identical batch for all clients)
- metrics (S_sum vs L on overlapping params):
  - sign agreement overall 0.9969649001936113
  - cosine overall 0.6342529770146774
  - steps_used=15; total_all=417,496,320
- note: MLP prefix agreement not computed for NanoGPT names (prefix alignment only recognizes gate_proj/up_proj/down_proj), so prefix totals are 0

## 2026-01-30 05:02 UTC — SSSL (same-batch, DCT-decoded agreement)
- dataset: tinyshakespeare (same-batch override)
- run logs: `logs/distro-sssl-samebatch-run.log`
- gradients: `logs/distro-sssl-samebatch/`
- analysis: DCT decode via TransformDCT (compression_chunk=64), then align MatFormer prefix
- metrics (S_sum vs L):
  - sign agreement overall 0.686668873875648
  - cosine overall 0.5928188287415342
  - steps_used=15; total_all=511,868,160
- metrics (MLP prefix only):
  - sign agreement 0.5302260504828559
  - cosine 0.1063151044808737
  - total_prefix=47,185,920

## 2026-01-30 05:06 UTC — SSSL (same-batch, DCT-decoded overlap-only)
- analysis: overlap-only in parameter space (prefix slice only, suffix excluded)
- metrics (S_sum vs L, overlap-only):
  - sign agreement overall 0.756396312886845
  - cosine overall 0.6075992356829433
  - steps_used=15; total_overlap_all=464,682,240
- metrics (MLP prefix only):
  - sign agreement 0.5302260504828559
  - cosine 0.1063151044808737
  - total_overlap_mlp_prefix=47,185,920

## 2026-01-30 05:42 UTC — SSSL (same-batch, 100-step overlap-only)
- config: `config/test-nanogpt-distro-shakespeare-sssl-100` (run_id=nanogpt-distro-sssl-100b)
- run logs: `logs/distro-sssl-samebatch-100b-run.log`
- gradients: `logs/distro-sssl-samebatch-100b/`
- analysis: DCT decode via TransformDCT (compression_chunk=64), overlap-only in parameter space
- metrics (S_sum vs L, overlap-only):
  - sign agreement overall 0.7348374531978644
  - cosine overall 0.5435440573093493
  - steps_used=99; total_overlap_all=3,066,902,784
- metrics (MLP prefix only):
  - sign agreement 0.5092931933675953
  - cosine 0.031442353529751506
  - total_overlap_mlp_prefix=311,427,072

## 2026-01-30 06:50 UTC — SSSL raw grads (same-batch, pre-compression)
- run logs: `logs/distro-sssl-rawgrads-run.log`
- raw grads: `logs/raw-grads-sssl-samebatch/` (layer 0 MLP gate/up/down)
- config: temporary 5-step SSSL same-batch run (nanoGPT, Distro)
- analysis: raw float grads, overlap-only alignment on prefix slice (no DCT/TopK/Sign)
- metrics (S_sum vs L, overlap-only across 3 MLP matrices, 4 steps):
  - sign agreement overall 0.788031260172526
  - cosine overall 0.8364424848236094
  - steps_used=4; total_elems=1,572,864

## 2026-01-30 06:58 UTC — DCT-decoded vs raw (same steps, layer 0 MLP)
- run logs: `logs/distro-sssl-rawgrads-run.log`
- distro results: `logs/distro-sssl-rawgrads-samebatch/`
- analysis: DCT decode + overlap-only, same steps as raw (1–4)
- metrics (S_sum vs L, overlap-only across 3 MLP matrices, 4 steps):
  - sign agreement overall 0.5782890319824219
  - cosine overall 0.26171875274115003
  - steps_used=4; total_elems=1,572,864
- comparison: raw on same steps had sign 0.788031260172526 / cosine 0.8364424848236094
