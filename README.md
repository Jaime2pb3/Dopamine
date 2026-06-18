# Dopamine

**A 16 MB neocortical language model with *isomeric test-time training* — designed, built, and trained end-to-end on a single RTX 3060.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Compute](https://img.shields.io/badge/trained%20on-RTX%203060%2012GB-76b900)

Dopamine is a from-scratch language model built for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) (*train the smallest LM you can that fits in 16 MB*). It combines a **neocortical 3-phase architecture** with an original inference-time learning objective — **Isomeric TTT** — that adapts the model *while it reads the validation set*, without ever using the target tokens as a supervision signal.

The whole project runs under one hard constraint: **everything fits and trains on a single consumer GPU (RTX 3060, 12 GB).** H100-scale settings are available through environment-variable overrides.

---

## What's new here

Dopamine makes two distinct contributions — one architectural, one in the learning objective.

### 1. Isomeric Test-Time Training (original contribution)

Canonical test-time training (Sun et al.) adapts a model at inference using a self-supervised next-token cross-entropy loss. Dopamine replaces that loss with **isomeric coherence** over a `ResonanceBuffer`:

```
L_iso = | P_t(R ∪ {p_new}) − P_regime |  +  β · | Δ²P_t |
```

| Term | Meaning |
|------|---------|
| `R` | Detached slots of the ResonanceBuffer — the model's prior experiential fingerprint, shape `(K, D)` |
| `p_new` | `perceptual_flow.mean(dim=(0,1))` — the current candidate flow, with gradient, shape `(D,)` |
| `P_t` | Mean polarization (pairwise cosine distance) over the buffer |
| `P_regime` | EMA window of the historical `P_t` |
| `Δ²P_t` | Acceleration of polarization |

Instead of *"learn to predict the next token,"* the PerceptionAgent learns to *"stay coherent with the experiential fingerprint you've already laid down."*

**Key property — the loss is blind to the target `y`.** The gradient flows only through `p_new` into the PerceptionAgent weights (`c_q`, `c_k`, `c_v`, `proj`, `percept_gate`). Soma, Dendrite, embedding, and Axon stay frozen during adaptation. This is real inference-time adaptation with no label leakage.

**Challenge-legal by construction.** The eval loop is *score-first*: each chunk is graded **before** it is used for adaptation, so the model only test-time-trains on tokens it has already been scored on.

```
1. Score bpb on chunk_i          (torch.no_grad)
2. Update the polarization monitor from the ResonanceBuffer
3. ttt_trainer.adapt(chunk_i)    (K inner steps of L_iso)
4. Evaluate chunk_i+1 with the adapted weights
```

### 2. Neocortical 3-phase architecture

A language model organized like cortical processing rather than a plain transformer stack:

- **Phase 1 — Dendritic Encoder (*experience*).** Multi-field receptive attention (local / mid / global); independent agents process the same input in parallel.
- **Phase 1.5 — Perception Agent (*perception*).** Reads similarity between the current experience and past flow; emits a `PerceptualFlow` signal that only exists in the assembly of both.
- **Phase 1.75 — Resonance Buffer (*emergent memory*).** An EMA ring buffer of prior Soma states — **no trainable params**. Memory appears when the current state resembles past flow.
- **Phase 2 — Soma Core (*the medium*).** Holds state without collapsing it; receives experience + perception through U-Net skips across abstraction levels.
- **Phase 3 — Axon Projector + Molecule Gate.** Selective transmission — "think before speaking."

---

## Results

| Build | Compressed size | val_bpb (FineWeb) | Hardware |
|-------|-----------------|-------------------|----------|
| Dopamine (best) | ≈ 16 MB | **≈ 1.232** | RTX 3060 12 GB |

The Isomeric TTT variant is evaluated as an **A/B ablation** (`TTT_ENABLED=1` vs `0`) across 3 seeds. A run counts as a valid improvement when **Δval_bpb > 0.005 nats at p < 0.01**. See [`docs/ttt_iso_es.md`](docs/ttt_iso_es.md) for the full diagnostic notes.

> The same code scales to 8×H100 via env overrides; the RTX 3060 defaults are the point — the architecture and the TTT objective were developed entirely under a severe compute budget.

---

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

You provide the FineWeb data shards and a SentencePiece tokenizer (not included in this repo).

### Smoke run on a single RTX 3060

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=500 \
VAL_LOSS_EVERY=250 \
TTT_ENABLED=1 \
TTT_INNER_STEPS=3 \
TTT_LR=5e-4 \
TTT_BETA=0.1 \
PFV_MONITOR=1 \
torchrun --standalone --nproc_per_node=1 train_dopamine_v3_ttt_iso_3060.py
```

### A/B ablation (TTT on vs off)

```bash
# With Isomeric TTT
TTT_ENABLED=1 RUN_ID=ttt_on  torchrun --standalone --nproc_per_node=1 train_dopamine_v3_ttt_iso_3060.py

# Control (no TTT)
TTT_ENABLED=0 RUN_ID=ttt_off torchrun --standalone --nproc_per_node=1 train_dopamine_v3_ttt_iso_3060.py
```

Compare the final `val_bpb` between runs.

### H100-style override

```bash
TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=512 VAL_BATCH_SIZE=65536 torchrun ... train_dopamine_v3_ttt_iso_3060.py
```

---

## Key hyperparameters (TTT)

| Env var | Default | Role |
|---------|---------|------|
| `TTT_ENABLED` | `1` | Enable / disable Isomeric TTT |
| `TTT_INNER_STEPS` | `3` | K gradient steps per chunk |
| `TTT_LR` | `5e-4` | Inner-loop learning rate (Lion) |
| `TTT_BETA` | `0.1` | Weight of `|Δ²P_t|` in `L_iso` |
| `TTT_RESET_EACH_CHUNK` | `0` | `1` = reset weights between chunks (no accumulated drift) |
| `TTT_WARMUP_CHUNKS` | `2` | Initial chunks that populate `P_history` without adapting |
| `PFV_MONITOR` | `1` | Required — TTT uses the differentiable polarization monitor |

During eval with TTT active, logs report:

```
[TTT-Iso]  chunks_adapted=14  avg_L_iso=0.0423  avg_|∇|=0.1847
```

- `avg_L_iso` falling → the PerceptionAgent is converging toward isomeric coherence
- `avg_|∇|` decreasing → approaching a stable regime
- `avg_L_iso` growing monotonically → `TTT_LR` is too high

---

## Repository layout

```
Dopamine/
├── README.md                              # this file
├── LICENSE                                # MIT
├── requirements.txt
├── train_dopamine_v3_ttt_iso_3060.py      # full model + Isomeric TTT trainer + eval loop
└── docs/
    └── ttt_iso_es.md                      # detailed technical notes (ES): loss, legality, diagnostics, risks
```

Code map (inside the training script):

| Lines | Component |
|-------|-----------|
| 125–225 | `Hyperparameters` (incl. `ttt_*` env vars) |
| 480–560 | `AxonProjector` (rho-mixing) |
| 848–1065 | `IsomericPolarizationMonitor` + differentiable `compute_ttt_loss` |
| 1072–1195 | `ResonanceBuffer` (mean / seq / topk modes) |
| 1221–1315 | `PerceptionAgent` |
| 1352–1560 | `Dopamine` + `forward_with_percept` |
| 1567–1720 | `TTTIsoTrainer` |
| 1826–1920 | `eval_val` (score-first) |
| 2200+ | `main()` |

---

## Background

Dopamine grows out of the **Isomeric Polarization** research program at [TwoQuarks Research](https://twoquarks.com): the idea that divergence among a system's internal realizations is a measurable, actionable signal. The `L_iso` objective is that same principle turned into a learning target.

- Isomeric Polarization framework — https://twoquarks.com/isomeric_polarization.pdf
- Pre-critical structural reorganization (black-box detection) — DOI [10.5281/zenodo.19675750](https://doi.org/10.5281/zenodo.19675750)

---

## Citation

```bibtex
@software{ledesma_dopamine_2026,
  author  = {Ledesma P{\'e}rez, Luis Jaime},
  title   = {Dopamine: A Neocortical Language Model with Isomeric Test-Time Training},
  year    = {2026},
  url     = {https://github.com/Jaime2pb3/Dopamine},
  note    = {TwoQuarks Research}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Luis Jaime Ledesma Pérez** — TwoQuarks Research, Guadalajara, Mexico
[twoquarks.com](https://twoquarks.com) · [research@twoquarks.com](mailto:research@twoquarks.com)
