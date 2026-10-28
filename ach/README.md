# ACH

*"Attention is all you need to learn — but not all you need to think."*

*"Atención es lo único que necesitas para aprender, pero no lo único que necesitas para pensar."*

Luis Jaime Ledesma Pérez · Independent Researcher · TwoQuarks Research · April 2026

---

## Where does ACH get its name?

ACH was named after a neurotransmitter that fascinated us and inspired this architecture: acetylcholine (ACh).

It was the first neurotransmitter discovered and is responsible for transmitting signals between neurons and muscles, as well as regulating key functions in the nervous system [1]. It plays a role in memory and attention by participating in brain circuits related to learning, wakefulness, and mental acuity [2][3]. Notably, one of its functions is to be rapidly broken down within the synapse to prevent constant stimulation [4][5].

[1] https://www.cun.es/diccionario-medico/terminos/acetilcolina
[2] https://psiquiatria.com/glosario/acetilcolina-ach
[3] https://gammaknife.com.ec/acetilcolina-funciones/
[4] https://www.cun.es/diccionario-medico/terminos/acetilcolina
[5] https://www.youtube.com/watch?v=gYLn3wHsSGc&t=5

## So... what is ACH?

ACH is a language model architecture inspired by the neocortex that considers attention an **essential and necessary, but insufficient** mechanism for cognition. It accepts and celebrates the attention paradigm as a brilliant solution for statistical learning from data and expands it with the components that cognition, both biological and artificial, requires to generate a coherent flow that we commonly call *thought*: associative memory, perception as an independent agent, an emergent medium for integration, selective transmission, and architectural self-observation.

ACH does not criticize Transformers. In fact, it incorporates them. This proposal should be understood as a complement, built upon the attention mechanism that Transformers have bequeathed to us.

---

## The thesis

The 2017 article *Attention Is All You Need* presented a mechanism that adapts extraordinarily well to the operation of **absorbing structure from data**. This mechanism is now the fundamental component of all cutting-edge language models. Its success is more than deserved.

However, we often merge two operations that should be treated as different concepts:

- **Learning**: updating internal representations based on exposure to data.
- **Thinking**: integrating, filtering, remembering, and deciding *about* the representations already learned.

Biological cognition uses attention as part of both, but never as the entirety of either. Working memory, hippocampal rehearsal, thalamic filtering, prefrontal deliberation, cortical oscillations for integration: none of these are attention, and all are necessary for the thought process.

ACH takes this distinction seriously and encodes it into its architecture, rather than just fine-tuning it later.

---

## Architecture

ACH is organized into five phases plus an internal observer. The forward pass is not a stack of identical blocks but a sequence of qualitatively distinct operations.

### Phase 1 — Dendritic Encoder (*experience*)

Multi-field receptive attention with three independent agents processing the same input at three scales: local (span 64), mid (span 256), global (full sequence). Each field is followed by a **PlasticDendriticField**, a Hopfield-style associative memory with learned prototypes that performs prototype recall in the forward pass.

**New in v0.3, Geometric Dendrite.** Each field gets non-orthogonal L/R frames with a regularized Gram matrix (inverted in fp32), an axial projection that preserves the orthogonal complement, bounded per-token geodesic transport, and a neutral EMA center. With `GEOMETRIC_DENDRITE_ENABLED=0` the dendritic forward is exactly the v0.2.1 one (tested bit for bit, see below).

### Phase 1.5 — Perception Agent (*perception*)

An independent agent that performs cross-attention between current dendritic experience and past flow stored in the Resonance Buffer. It produces a **PerceptualFlow** signal that exists only in the assembly of present and past.

### Phase 1.75 — Resonance Buffer (*emergent memory*)

A ring buffer of K=64 slots holding EMA snapshots of Soma hidden states from previous forward passes. **No trainable parameters.** Memory is not stored, it flows: it appears when the current state resembles past flow and is silent otherwise.

### Phase 2 — Soma Core (*the medium*)

The vehicle, not the processor. Inspired by Global Workspace Theory (Baars, Dehaene): integration emerges in a medium that lets multiple agents interact. U-Net skip connections preserve access to all abstraction levels.

### Phase 3 — Axon Projector + Cholinergic Gate (*selective transmission*)

Multi-view stress aggregation followed by a per-token cholinergic gate driven by the cross-layer incoherence of the Soma trajectory. Implements *think before speaking*: not every internal state earns the right to be expressed.

### Internal Observer — PfV Monitor (*self-observation*)

A non-trainable module that computes the **Isomeric Polarization** of the model's own internal collections (Resonance slots, dendritic prototypes) at each step: P_t, P_regime, ΔP_t, Δ²P_t, ε, and dead-realization detection.

### Adaptive inference — Isomeric Test-Time Training

At evaluation, the Perception Agent adapts its weights via a loss based on **isomeric coherence** instead of next-token prediction:

```
L_iso = |P_t(R ∪ {p_new}) − P_regime|  +  α·loss_density  +  β·|Δ²P_t|
```

Score-first by construction: each chunk is graded before it is used for adaptation, and the loss never sees the targets.

---

## What distinguishes ACH from a pure transformer

| Capability | Pure Transformer | ACH |
|---|---|---|
| Token mixing via attention | ✓ | ✓ (Dendritic Encoder) |
| Multi-scale receptive fields | partial (depth) | explicit (local/mid/global) |
| Associative memory recall | implicit in weights | explicit (PlasticDendriticField) |
| Episodic memory of own past states | ✗ | ✓ (Resonance Buffer) |
| Perception as separate agent from experience | ✗ | ✓ (Perception Agent) |
| Integration medium distinct from processor | ✗ | ✓ (Soma Core) |
| Selective output gating | ✗ | ✓ (Cholinergic Gate) |
| Architectural self-observation | ✗ | ✓ (PfV Monitor) |
| Adaptive inference on a coherence signal | ✗ | ✓ (Isomeric TTT) |

---

## Results so far (v0.3-geometric, RTX 3060)

One full run, seed 42, 9 layers, dim 512, 8Q/2KV heads, vocab 1024, 22,317,352 params, TTT-Iso on.

| step | val_bpb |
|---|---|
| 0 | 4.0978 |
| 1500 | 1.7599 |
| 3000 | 1.6757 |
| 4500 | 1.6435 |
| **6500** | **1.5811** |

- Compressed artifact: 6418.9 KB model + 43.3 KB code = **6462.2 KB**, well under 16 MB.
- **Deterministic.** A second run with the same seed and different logging cadence gave identical val_bpb and train loss at every comparable step, including the final one.
- **TTT-Iso helps early.** 150-step smoke test: 2.6824 with TTT on vs 2.7116 off.

### Known failure: the cholinergic gate collapsed

This is reported because it matters more than the bpb number.

- `theta` stuck at 0.000 and `k` at 0.50 (the clamp floor) from about step 900 on.
- `mask` hit a floor of 0.840 from about step 3000 and never recovered.
- `percept_gate` decayed monotonically from 0.182 to 0.072.
- `lr_mul` stayed at 1.0 the whole run (no warmdown phase in this run).

Diagnosis: `torch.clamp` gives zero gradient at its edges, so once theta or k touches a bound the state is absorbing. The initial threshold also sat far outside the operating band of the uncertainty signal, and the static conductance channel had more authority than the dynamic ACh branch, so the optimizer neutralized the gate. In short, a val_bpb gain in this run cannot be credited to the gate. The fix (smooth reparametrization, standardized uncertainty, a frozen-base ablation) is in progress and will be published with its own A/B.

---

## Files

```
ach/
├── ach_v0_3_geometric.py        # full model + trainer + TTT-Iso + telemetry
├── test_ach_v0_3_geometric.py   # CPU smoke and invariant tests
├── legacy/
│   └── ach_v0_2_1_persistent.py # frozen baseline for the bit-exact A/B test
└── README.md
```

## Verify without a GPU

```bash
pip install torch numpy sentencepiece
python ach/test_ach_v0_3_geometric.py
```

It checks, on CPU and a tiny config:

1. Geometry OFF with v0.2.1 weights gives a **bit-identical** loss to v0.2.1.
2. Regularized Gram matrices stay finite and non-degenerate.
3. The axial projection changes only the L/R plane and preserves the orthogonal complement.
4. Geodesic transport hits its endpoints exactly and stays finite inside.
5. Calibration accepts non-collinear fields and rejects collapse.
6. Geometry ON: finite loss, telemetry present, every geometry parameter covered by the optimizer and receiving a non-zero gradient.
7. The geometry snapshot is JSON-compatible and finite.

Expected output starts with `OK — ACH v0.3-geometric`.

## Train on a single RTX 3060

You provide the FineWeb shards and the SentencePiece tokenizer (not included). The script looks for a `data/` folder up to three levels above itself, or use the env vars:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=6500 VAL_LOSS_EVERY=250 SEED=42 \
torchrun --standalone --nproc_per_node=1 ach/ach_v0_3_geometric.py
```

Evaluation with TTT-Iso dominates wall time on a 3060 (about 35 min per eval, roughly 94% of the run), so raise `VAL_LOSS_EVERY` for long runs.

## Key environment variables

**Architecture:** `MODEL_DIM`, `NUM_LAYERS`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`, `DENDRITE_LOCAL_SPAN`, `DENDRITE_MID_SPAN`, `RESONANCE_SLOTS`, `RESONANCE_EMA_DECAY`, `RESONANCE_MODE` (`mean`, `seq`, `topk`), `PERCEPTION_HEADS`

**Geometric Dendrite:** `GEOMETRIC_DENDRITE_ENABLED` (0 gives the exact v0.2.1 path), `GEO_METRIC_MAX`, `GEO_TRANSPORT_MAX`, `GEO_TAU_MAX`, `GEO_QUANTIZE`, `GEO_AUX_SCALE`, `GEO_LR`

**Cholinergic Gate:** `ACH_ENABLED`, `ACH_THETA`, `ACH_RHO_K`, `ACH_LAMBDA`, `ACH_BASE_SCALE`, `ACH_FIRST_BLOCK`

**Isomeric TTT:** `TTT_ENABLED` (default 1), `TTT_INNER_STEPS` (3), `TTT_LR` (5e-4), `TTT_BETA` (0.1), `TTT_RESET_EACH_CHUNK`, `TTT_WARMUP_CHUNKS`

**PfV Monitor:** `PFV_MONITOR`, `PFV_DENSITY_WEIGHT`

---

## Status

- First preliminary version: Dopamine v1.0 (April 2026)
- Latest version: **ACH v0.3-geometric** (August 2026)
- Hardware tested: RTX 3060 12 GB
- Research code under active development. The gate fix above is the next milestone.

## Author

**Luis Jaime Ledesma Pérez** · Founder & Researcher · TwoQuarks Research · Guadalajara, México

- Website: [twoquarks.com](https://twoquarks.com)
- Google Scholar: [scholar.google.com/citations?user=FOPg8OgAAAAJ](https://scholar.google.com/citations?user=FOPg8OgAAAAJ)
- GitHub: [github.com/Jaime2pb3](https://github.com/Jaime2pb3)
- LinkedIn: [linkedin.com/in/jaimedeepmind](https://www.linkedin.com/in/jaimedeepmind/)

## How to cite

```bibtex
@misc{ledesma2026_ach,
  author       = {Ledesma P{\'e}rez, Luis Jaime},
  title        = {ACH: A Neocortically-Inspired Architecture for Language Models
                  with Architecture-Native Self-Observation},
  year         = {2026},
  howpublished = {TwoQuarks Research},
  url          = {https://twoquarks.com}
}
```

---

## Closing note

Attention solved a problem that needed solving, and it solved it well. What ACH proposes is not that we replace it, but that we stop pretending it was the whole problem.

The brain learns through plasticity *and* thinks through integration. Both are necessary. Neither is sufficient alone. This codebase is one attempt at building the missing half.
