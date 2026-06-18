# Dopamine v3.0-TTT-Iso — Isomeric Test-Time Training (3060 build)

**Contribución original:** sustituir la SSL loss canónica de TTT (next-token CE) por **coherencia isomérica** sobre el ResonanceBuffer. En vez de "aprende a predecir", el PerceptionAgent aprende a "sé coherente con la huella de experiencia que ya dejaste".

```
L_iso = |P_t(R ∪ {p_new}) - P_regime| + β·|Δ²P_t|
```

donde:
- `R` = slots detached del ResonanceBuffer (huella previa, (K, D))
- `p_new` = `perceptual_flow.mean(dim=(0,1))` (candidato con grad, (D,))
- `P_t` = polarización media vía cosine distance sobre pares
- `P_regime` = EMA window del P_t histórico
- `Δ²P_t` = aceleración de polarización

**Propiedad clave:** `L_iso` es **ciega al target** `y`. El gradiente fluye solo por `p_new` → pesos del PerceptionAgent (c_q, c_k, c_v, proj, percept_gate).

## Legalidad en OpenAI Parameter Golf

Cumple la regla estricta:

> "you are only allowed to test-time train on validation set tokens you've already evaluated your model on, since those tokens have already been graded!"

El flujo dentro de `eval_val` es **score-first**:
1. Medir bpb sobre `chunk_i` (scoring en `torch.no_grad()`)
2. Actualizar `monitor.compute()` con el ResonanceBuffer reciente
3. Aplicar `ttt_trainer.adapt(chunk_i)` → K pasos de L_iso sobre PerceptionAgent
4. Evaluar `chunk_i+1` con pesos adaptados

Los `warmup_chunks` iniciales no adaptan (solo pueblan `P_history`).

## Uso

### Smoke test en RTX 3060 (single GPU)

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

### Comparación A/B (ablación TTT)

Run A (con TTT):
```bash
TTT_ENABLED=1 RUN_ID=ttt_on  torchrun ... train_dopamine_v3_ttt_iso_3060.py
```

Run B (sin TTT, control):
```bash
TTT_ENABLED=0 RUN_ID=ttt_off torchrun ... train_dopamine_v3_ttt_iso_3060.py
```

Comparar `val_bpb` al final. Diferencia > 0.005 nats con p<0.01 (3 seeds) → submission válido.

## Hyperparameters TTT

| Env var | Default | Rol |
|---|---|---|
| `TTT_ENABLED` | 1 | Activar/desactivar TTT-Iso |
| `TTT_INNER_STEPS` | 3 | K pasos de gradiente por chunk |
| `TTT_LR` | 5e-4 | Learning rate del Lion interno |
| `TTT_BETA` | 0.1 | Peso de `|Δ²P_t|` en L_iso |
| `TTT_RESET_EACH_CHUNK` | 0 | 1=resetea pesos entre chunks (no acumula drift) |
| `TTT_WARMUP_CHUNKS` | 2 | Chunks iniciales que pueblan P_history sin adaptar |
| `PFV_MONITOR` | 1 | Requerido: el TTT usa `monitor_resonance.compute_ttt_loss` |

## Diagnósticos en logs

Durante `eval_val` con TTT activo:
```
[TTT-Iso]  chunks_adapted=14  avg_L_iso=0.0423  avg_|∇|=0.1847
```

- `avg_L_iso` bajando → el PerceptionAgent converge hacia coherencia isomérica
- `avg_|∇|` decreciendo → proximidad a un régimen estable
- Si `avg_L_iso` crece monótonamente → TTT_LR demasiado alto

## Arquitectura del archivo

- **Lines 1–68:** header + docstring
- **Lines 125–225:** `Hyperparameters` (incluye `ttt_*` env vars)
- **Lines 480–560:** `AxonProjector` (rho-mixing UP-style, v2.3c)
- **Lines 848–1065:** `IsomericPolarizationMonitor` + `compute_ttt_loss` diferenciable
- **Lines 1072–1195:** `ResonanceBuffer` (modes: mean/seq/topk)
- **Lines 1221–1315:** `PerceptionAgent`
- **Lines 1352–1560:** `Dopamine` + `forward_with_percept`
- **Lines 1567–1720:** `TTTIsoTrainer` (clase nueva)
- **Lines 1826–1920:** `eval_val` reescrito (score-first)
- **Lines 2200+:** `main()` con instanciación de `ttt_trainer`

## Riesgos conocidos

1. **Señal pobre al inicio:** resuelto con `warmup_chunks=2`.
2. **Escalar low-dim:** `P_regime` es escalar, gradient signal limitada. Mitigación opcional: añadir término que use `dim_div` (vector denso del DownQuark).
3. **Compute extra:** K×forward + K×backward per chunk. Con K=3, batch 16K, overhead ≈ 30s en 3060 sobre un val completo. En H100 es negligible.
4. **DDP multi-GPU:** el trainer actual NO sincroniza gradientes entre ranks (single-GPU oriented). Para 8xH100 agregar `dist.all_reduce` sobre `percept_params.grad` antes de `opt.step()`. TODO marcado en código.

## Mejoras futuras

- `L_iso` con señal densa: usar `dim_div` del DownQuark como vector objetivo en vez de solo `P_t` escalar.
- Momentum acumulado del monitor: usar `delta2_P` history para target progresivo en vez de "flat".
- Adaptación selectiva: solo adaptar si `|Δ²P_t| > threshold` (skip chunks estables).
