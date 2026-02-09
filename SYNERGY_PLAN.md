# MatFormer Synergy Plan (Tier0 + Tier>0)

## Goal
Make tier>0 compute **improve** tier0 training (not just “do no harm”) under:
- low tier>0 VRAM (cannot hold full model)
- WAN-friendly bandwidth (top-k payloads)

## Mechanism (What We Implemented)
1. **Tier0 suffix gate (addendum capacity)**
   - Tier0 gradually ramps in FFN suffix capacity so the full model starts close to the prefix-only submodel.
   - Optional learnable per-layer scalar multiplies the scheduled ramp.

2. **In-place distillation (same-batch teacher -> student)**
   - Tier0 produces compressed teacher logits for student-assigned batches.
   - Tier>0 trains with `CE + beta * KD` (default), where KD is a WAN-friendly **top-k + tail-bucket KL**
     (teacher sends top-k logits + per-token `logsumexp`).

3. **KD strengthening (bounded, WAN-safe)**
   - Add a bounded scale factor so KD does not vanish when the teacher is high-entropy:
     `KD *= 1 / clamp(mean_q_topk_mass, floor, 1.0)` (default `floor=0.05`).

4. **Coupled schedule (synergy flags)**
   - `--matformer-synergy-phase-a-steps`: prefix-only warmup (suffix gate beta=0, distill beta=0)
   - `--matformer-synergy-ramp-steps`: ramp both suffix gate and distillation together
   - These override the individual `--matformer-*-start-step` and `--matformer-*-warmup-steps` settings.

## Important Correctness Properties
- If distillation is active and teacher logits are missing, the student blocks and eventually errors.
- If teacher payload is invalid, distillation does **not** silently fall back to CE (loss becomes `None` and
  training fails loudly). Confidence gating is the only intended “KD off” path.

## Telemetry (What To Watch)
Look for `distillation stats` logs (tier>0):
- `ce`, `kd_raw`, `kd_scale`, `kd`, `beta_requested`, `beta`
- `q_topk_mass`, `p_topk_mass` (teacher vs student mass on teacher head)

## How To Run (Example)
Centralized local-testnet (2 clients, tiers 0 and 1):
```bash
bash -lc 'source scripts/psyche-env.sh && cargo run -p psyche-centralized-local-testnet -- start \
  --headless --headless-exit-after-secs 120 \
  --num-clients 2 \
  --config-path ./config/test \
  --client-matformer-tiers 0,1 \
  --matformer-distillation-beta-max 0.2 \
  --matformer-distillation-min-teacher-topk-mass 0.02 \
  --matformer-distillation-kd-q-topk-mass-floor 0.05 \
  --matformer-synergy-phase-a-steps 200 \
  --matformer-synergy-ramp-steps 500 \
  --tui false'
```

## Tests
Run in the repo-local torch env:
```bash
bash -lc 'source scripts/psyche-env.sh && cargo test -p psyche-modeling -p psyche-client'
```

