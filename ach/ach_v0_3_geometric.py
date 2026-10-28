# -*- coding: utf-8 -*-
# ACH v0.3-geometric — Geometric Dendrite + Cholinergic Gate + state-safe TTT
# TwoQuarks — Jaime Ledesma
#
# Derivado de ach_v0_1.py y revisado como implementacion independiente.
# El patcher v0.1 no reproduce estas correcciones; no lo ejecutes sobre este archivo.
#
# CAMBIOS vs base:
#   [ACH-1] Las seis clases de sabor y su bundle: ELIMINADAS.
#   [ACH-2] CholinergicGate: mask por token, incertidumbre
#           aferente desde la trayectoria cross-layer del Soma.
#   [ACH-2b] El bundle de sabores ya no participa en el camino caliente.
#   [ACH-3] _encode_decode(return_trace=True) devuelve (h, x, trace).
#   [ACH-4] MoleculeGate consume la senal ACh en vez de la de estres previa.
#   [ACH-5] ACH_ENABLED=0 conserva una linea base centrada en identidad.
#   [ACH-6] Telemetria JSONL: fases, ACh, gradientes y memoria.
#   [ACH-7] Validacion/TTT aislados: no contaminan el entrenamiento.
#   [ACH-8] Auditoria estricta de cobertura de optimizadores.
#   [ACH-9] Checkpoint final persistente + archivo comprimido recargable.
#   [ACH-10] Tiempo de entrenamiento separado de wall-clock total.
#   [GEO-1] Atencion metrizada de bajo rango compatible con Flash SDP.
#   [GEO-2] Marcos L/R no ortogonales con Gramiano regularizado.
#   [GEO-3] Proyeccion axial STE que conserva el complemento ortogonal.
#   [GEO-4] Transporte geodesico acotado por token y campo.
#   [GEO-5] Centro neutral EMA, calibracion externa y telemetria geometrica.
#   [GEO-6] Ruta A/B exacta con GEOMETRIC_DENDRITE_ENABLED=0.
#
# CONSERVADO: la definicion de polarizacion isomerica y la forma de L_iso.
# AJUSTADO: ResonanceBuffer, PerceptionAgent, TTTIsoTrainer y su integracion de estado.

"""
╔══════════════════════════════════════════════════════════════════╗
║  DOPAMINE  v3.0-TTT-Iso  —  Isomeric Test-Time Training [3060]  ║
║  TwoQuarks — Luis Jaime Ledesma Pérez                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Base: v2.3c (3060 config + PfV Monitor + rho-mixing Axon).     ║
║  Nuevo: Isomeric TTT aplicado al PerceptionAgent en eval.       ║
║                                                                  ║
║  ISOMERIC TTT — propuesta original (Ledesma, 2026):             ║
║                                                                  ║
║    TTT canónico (Sun et al.) usa L_SSL = next-token CE.          ║
║    Aquí sustituimos por coherencia isomérica del ResonanceBuffer:║
║                                                                  ║
║      L_iso = |P_t(R ∪ {p_new}) - P_regime| + β·|Δ²P_t|          ║
║                                                                  ║
║    donde:                                                        ║
║      R         = ResonanceBuffer.slots (K, D) — huella previa    ║
║      p_new     = perceptual_flow.mean(0,1)    — flujo candidato ║
║      P_t       = polarización media (cosine-dist pairs)         ║
║      P_regime  = EMA window del P_t histórico                    ║
║      Δ²P_t     = aceleración de polarización                     ║
║                                                                  ║
║  Integración (LEGAL "score-first TTT"):                         ║
║    1. Mide bpb en chunk_i  (score ya graded)                    ║
║    2. TTT inner loop con chunk_i (K pasos sobre L_iso)          ║
║    3. Evalúa chunk_i+1 con pesos adaptados                      ║
║                                                                  ║
║  Solo el PerceptionAgent se adapta (c_q/c_k/c_v/proj/gate).     ║
║  Soma, Dendrite, embedding, Axon → congelados en TTT.           ║
║                                                                  ║
║  Legalidad (OpenAI Parameter Golf rules, April 2026):           ║
║    ✓ L_iso es ciego al target (no toca y)                       ║
║    ✓ TTT opera solo sobre tokens YA graded                      ║
║    ✓ Compatible con score-first TTT del leaderboard             ║
╠══════════════════════════════════════════════════════════════════╣
║  RTX 3060 12GB VARIANT — defaults ajustados:                    ║
║    • train_batch_tokens = 16384    (vs 65536 en H100)           ║
║    • train_seq_len      = 256      (vs 512 en H100)             ║
║    • val_batch_size     = 16384    (vs 65536 en H100)           ║
║    • iterations default = 1000     (smoke 3060-friendly)        ║
║    • USE_COMPILE default = 0       (eager por default)          ║
║                                                                  ║
║  TTT env vars (nuevos):                                         ║
║    TTT_ENABLED=1           (default ON — activa TTT-Iso en val) ║
║    TTT_INNER_STEPS=3       (K pasos por chunk)                  ║
║    TTT_LR=5e-4             (lr inner loop — conservador)        ║
║    TTT_BETA=0.1            (peso de |Δ²P_t| en L_iso)           ║
║    TTT_RESET_EACH_CHUNK=0  (1=reset a pre-TTT state cada chunk) ║
║    TTT_WARMUP_CHUNKS=2     (chunks con buffer warmup sin adapt) ║
║                                                                  ║
║  Si quieres override a H100-like, usa env vars:                 ║
║    TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=512 ... python ...    ║
╠══════════════════════════════════════════════════════════════════╣
║  Architecture: Neocortical 3-phase model + Emergent Memory      ║
║                                                                  ║
║  Phase 1 — DENDRITIC ENCODER  (Experiencia)                     ║
║    Multi-field receptive attention: local/mid/global.            ║
║    Independent agents processing the same input in parallel.     ║
║                                                                  ║
║  Phase 1.5 — PERCEPTION AGENT  (Percepción)                     ║
║    NEW: Independent agent — reads similarity between             ║
║    current experience (DendriticOut) and past flow               ║
║    (ResonanceBuffer). Generates PerceptualFlow — a signal        ║
║    that only exists in the assembly of both.                     ║
║    High resonance → recognition signal enters Soma.              ║
║    Low resonance  → Soma processes pure experience.              ║
║                                                                  ║
║  Phase 1.75 — RESONANCE BUFFER  (Memoria emergente)             ║
║    NEW: Not stored — flows. EMA ring buffer of Soma              ║
║    hidden states from previous forward passes.                   ║
║    Memory appears when current state resembles past flow.        ║
║    No trainable params — pure observation of the medium.         ║
║                                                                  ║
║  Phase 2 — SOMA CORE  (El Medio)                                ║
║    The vehicle, not the processor. Holds without collapsing.     ║
║    No more nucleus — Soma is now pure medium.                    ║
║    Receives: Experience (Dendritic) + Perception (Perceptual).   ║
║    U-Net skips: all abstraction levels accessible.               ║
║                                                                  ║
║  Phase 3 — CHOLINERGIC GATE + MOLECULE GATE                    ║
║    Transmision local modulada por incoherencia cross-layer.       ║
║                                                                  ║
║  Theory:                                                         ║
║    Consciousness is not output, not processor, not observer.     ║
║    It is the medium that enables agent interaction.              ║
║    Perception + Experience are independent agents.               ║
║    Soma is the vehicle. Memory emerges from resonance.           ║
║    "Observing the observer" = agents accessing their own infra.  ║
║                                                                  ║
║  v2.0 changes vs v1.0:                                          ║
║    [NEW-1] ResonanceBuffer: EMA ring, K=64 slots, decay=0.99    ║
║    [NEW-2] PerceptionAgent: cross-attn Dendritic×Resonance      ║
║    [NEW-3] SomaNucleus removed — Soma is pure medium            ║
║    [NEW-4] SomaBlock simplified: resid_mix + attn + MLP         ║
║    [NEW-5] Forward: x = emb + dendrite + perceptual_flow        ║
║    [NEW-6] ResonanceBuffer updated post-SomaCore each step      ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

class Hyperparameters:
    # Data
    _here           = os.path.dirname(os.path.abspath(__file__))
    _base = _here
    for _ in range(3):
        if os.path.isdir(os.path.join(_base, "data")):
            break
        _base = os.path.dirname(_base)
    data_path       = os.environ.get("DATA_PATH",
                        os.path.join(_base, "data", "datasets", "fineweb10B_sp1024"))
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH",
                        os.path.join(_base, "data", "tokenizers", "fineweb_1024_bpe.model"))
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 42))

    # Validation — RTX 3060 12GB: batch reducido para no OOM en val
    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 16_384))    # 3060: 16K tokens
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 250))       # cada 250 steps
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))       # cada 50 steps

    # Training schedule — 3060-friendly defaults (1000 steps smoke, batch pequeño)
    iterations            = int(os.environ.get("ITERATIONS", 1_000))
    warmdown_iters        = int(os.environ.get("WARMDOWN_ITERS", 300))    # 30% del run
    warmup_steps          = int(os.environ.get("WARMUP_STEPS", 10))
    # 3060: 16K tokens × seq_len=256 = B=64 por micro-batch, con grad_accum=4 → efectivo 64K
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS", 16_384))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN", 256))     # 3060: seq más corta
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 99999.0))

    # Model
    vocab_size   = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers   = int(os.environ.get("NUM_LAYERS", 9))
    model_dim    = int(os.environ.get("MODEL_DIM", 512))
    num_heads    = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))
    mlp_mult     = int(os.environ.get("MLP_MULT", 3))
    rope_base    = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # DendriticEncoder
    dendrite_field_heads = [8, 4, 4]  # v2.3b: [4,2,2]→[8,4,4], head_dim 64→32, +dendritic capacity
    dendrite_local_span  = int(os.environ.get("DENDRITE_LOCAL_SPAN", 64))
    dendrite_mid_span    = int(os.environ.get("DENDRITE_MID_SPAN", 256))

    # ── Geometric Dendrite v0.3 ─────────────────────────────────────
    # Ruta desactivable para A/B. Al estar OFF, el forward dendritico usa
    # exactamente q/k/v, plasticidad y field_mix de ACH v0.2.1.
    geometric_dendrite_enabled = bool(int(os.environ.get(
        "GEOMETRIC_DENDRITE_ENABLED", 1)))
    geo_center_decay       = float(os.environ.get("GEO_CENTER_DECAY", 0.99))
    geo_axis_init_cos      = float(os.environ.get("GEO_AXIS_INIT_COS", 0.35))
    geo_gamma_init         = float(os.environ.get("GEO_GAMMA_INIT", 0.08))
    geo_metric_max         = float(os.environ.get("GEO_METRIC_MAX", 0.30))
    geo_transport_max      = float(os.environ.get("GEO_TRANSPORT_MAX", 0.12))
    geo_tau_max            = float(os.environ.get("GEO_TAU_MAX", 0.35))
    geo_quantize           = bool(int(os.environ.get("GEO_QUANTIZE", 1)))
    geo_quant_step_init    = float(os.environ.get("GEO_QUANT_STEP_INIT", 0.25))
    geo_axis_max_abs_cos   = float(os.environ.get("GEO_AXIS_MAX_ABS_COS", 0.92))
    geo_cross_field_target = float(os.environ.get("GEO_CROSS_FIELD_TARGET", 0.10))
    geo_aux_scale          = float(os.environ.get("GEO_AUX_SCALE", 0.01))
    geo_lr                 = float(os.environ.get("GEO_LR", 1e-3))
    geo_wd                 = float(os.environ.get("GEO_WD", 1e-4))
    geo_eps                = float(os.environ.get("GEO_EPS", 1e-5))

    # CholinergicGate + MoleculeGate
    axon_n_views  = int(os.environ.get("AXON_N_VIEWS", 4))
    axon_t_min    = float(os.environ.get("AXON_T_MIN", 0.20))
    axon_t_max    = float(os.environ.get("AXON_T_MAX", 0.60))
    axon_lambda   = float(os.environ.get("AXON_LAMBDA", 0.12))
    axon_stress_threshold = float(os.environ.get("AXON_STRESS_THRESHOLD", 0.30))

    # ── ACh v0.2 — compuerta colinergica observable ──
    # La conductancia base queda centrada en identidad (gain≈1), no en 0.5.
    ach_enabled       = bool(int(os.environ.get("ACH_ENABLED", 1)))
    ach_theta         = float(os.environ.get("ACH_THETA", 0.5))
    ach_rho_k         = float(os.environ.get("ACH_RHO_K", 2.0))
    ach_base_scale    = float(os.environ.get("ACH_BASE_SCALE", 0.10))
    ach_lambda        = float(os.environ.get("ACH_LAMBDA", os.environ.get("AXON_LAMBDA", 0.12)))
    # Primer bloque cuyo estado entra al trace. Se necesitan >= 3 bloques
    # para que cross_layer_incoherence tenga pares de deltas.
    ach_first_block   = int(os.environ.get("ACH_FIRST_BLOCK", 2))

    # Telemetria: consola compacta + JSONL detallado. Deep observer requiere eager.
    observer_enabled       = bool(int(os.environ.get("OBSERVER_ENABLED", 1)))
    observer_every         = int(os.environ.get("OBSERVER_EVERY", train_log_every))
    observer_sample_tokens = int(os.environ.get("OBSERVER_SAMPLE_TOKENS", 256))
    observer_console       = bool(int(os.environ.get("OBSERVER_CONSOLE", 1)))
    observer_jsonl         = bool(int(os.environ.get("OBSERVER_JSONL", 1)))

    # [NEW] ResonanceBuffer
    resonance_slots     = int(os.environ.get("RESONANCE_SLOTS", 64))
    resonance_ema_decay = float(os.environ.get("RESONANCE_EMA_DECAY", 0.95))  # v2.1: más sensible al flujo reciente
    # v2.2: modo de representación de memoria
    #   "mean"  → compat v2.1, 1 vector por slot (promedio global)
    #   "seq"   → tira de mem_span tokens por slot (preserva factor X replicable)
    #   "topk"  → token más activo del batch (reservoir sampling)
    resonance_mode      = os.environ.get("RESONANCE_MODE", "seq")
    resonance_mem_span  = int(os.environ.get("RESONANCE_MEM_SPAN", 8))

    # [NEW] PerceptionAgent
    perception_heads = int(os.environ.get("PERCEPTION_HEADS", 8))  # v2.1: más granularidad

    # ═════════════════════════════════════════════════════════════
    # [v3.0 NEW] ISOMERIC TEST-TIME TRAINING (TTT-Iso)
    # ═════════════════════════════════════════════════════════════
    # Score-first TTT: después de medir bpb en un chunk de val,
    # adaptar PerceptionAgent con L_iso sobre esos tokens ya graded.
    #
    #   L_iso = |P_t(R ∪ {p_new}) - P_regime| + β·|Δ²P_t|
    #
    # Sólo PerceptionAgent se adapta (Soma/Dendrite/embedding frozen).
    # Cumple regla "test-time train only on already-graded tokens"
    # del OpenAI Parameter Golf Challenge.
    ttt_enabled           = int(os.environ.get("TTT_ENABLED", 1))
    ttt_inner_steps       = int(os.environ.get("TTT_INNER_STEPS", 3))
    ttt_lr                = float(os.environ.get("TTT_LR", 5e-4))
    ttt_beta              = float(os.environ.get("TTT_BETA", 0.1))
    ttt_reset_each_chunk  = int(os.environ.get("TTT_RESET_EACH_CHUNK", 0))
    ttt_warmup_chunks     = int(os.environ.get("TTT_WARMUP_CHUNKS", 2))
    ttt_grad_clip         = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    # 1 = restaurar PerceptionAgent, optimizer TTT, ResonanceBuffer y monitor
    # al terminar cada validacion. Evita fuga de datos de val al entrenamiento.
    ttt_restore_after_eval = bool(int(os.environ.get("TTT_RESTORE_AFTER_EVAL", 1)))

    # Persistencia de artefactos
    artifact_dir           = os.environ.get("ARTIFACT_DIR", "artifacts")
    save_raw_checkpoint    = bool(int(os.environ.get("SAVE_RAW_CHECKPOINT", 1)))
    save_compressed_model  = bool(int(os.environ.get("SAVE_COMPRESSED_MODEL", 1)))

    # Soma freeze (no more nucleus, just init stabilization)
    soma_freeze_steps  = int(os.environ.get("SOMA_FREEZE_STEPS", 100))  # ~4% del run

    # Optimizer
    embed_lr           = float(os.environ.get("EMBED_LR", 0.05))
    matrix_lr          = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr          = float(os.environ.get("SCALAR_LR", 0.04))
    scalar_wd          = float(os.environ.get("SCALAR_WD", 0.01))
    muon_momentum      = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 10))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 2_500))
    lion_lr            = float(os.environ.get("LION_LR", 1e-3))
    lion_beta1         = float(os.environ.get("LION_BETA1", 0.9))
    lion_beta2         = float(os.environ.get("LION_BETA2", 0.99))
    lion_wd            = float(os.environ.get("LION_WD", 0.0))
    beta1              = float(os.environ.get("BETA1", 0.9))
    beta2              = float(os.environ.get("BETA2", 0.95))
    adam_eps           = float(os.environ.get("ADAM_EPS", 1e-8))


# ─────────────────────────────────────────────────────────────
# MUON OPTIMIZER
# ─────────────────────────────────────────────────────────────

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float,
                 backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(
            lr=lr, momentum=momentum,
            backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size  = dist.get_world_size() if distributed else 1
        rank        = dist.get_rank()       if distributed else 0
        for group in self.param_groups:
            params        = group["params"]
            lr            = group["lr"]
            momentum      = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov      = group["nesterov"]
            total_params  = sum(int(p.numel()) for p in params)
            updates_flat  = torch.zeros(total_params,
                device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g     = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────────────────────
# LION OPTIMIZER
# ─────────────────────────────────────────────────────────────

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError(f"Lion lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Lion betas must be in [0,1), got {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            wd           = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["exp_avg"]
                update = m.mul(beta1).add_(g.float(), alpha=1.0 - beta1).sign_()
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)
                m.mul_(beta2).add_(g.float(), alpha=1.0 - beta2)
        return loss


# ─────────────────────────────────────────────────────────────
# PHASE 3 — CHOLINERGIC GATE + MOLECULE GATE
# ─────────────────────────────────────────────────────────────

def cross_layer_incoherence(trace: Tensor) -> Tensor:
    """
    Incoherencia de trayectoria por token desde estados cross-layer.

    trace: (B, T, L, D)
    return: (B, T) en [0, 1]

    El calculo se realiza en float32 para que la geometria angular no dependa
    de la precision bf16 del forward principal. Deltas nulos producen u=0.5
    (neutral), no una falsa certeza.
    """
    if trace.ndim != 4:
        raise ValueError(
            f"trace debe ser (B, T, L, D); recibido {tuple(trace.shape)}")

    B, T, L, _ = trace.shape
    if L < 3:
        return trace.new_full((B, T), 0.5, dtype=torch.float32)

    deltas = trace.float()[:, :, 1:, :] - trace.float()[:, :, :-1, :]
    norms = deltas.norm(dim=-1)
    directions = F.normalize(deltas, dim=-1, eps=1e-6)
    cos_seq = (directions[:, :, 1:, :] * directions[:, :, :-1, :]).sum(dim=-1)

    # Si alguna transicion es casi nula, su direccion no es informativa.
    valid = (norms[:, :, 1:] > 1e-6) & (norms[:, :, :-1] > 1e-6)
    weights = valid.to(cos_seq.dtype)
    denom = weights.sum(dim=-1)
    coh_weighted = (cos_seq * weights).sum(dim=-1) / denom.clamp_min(1.0)
    coh = torch.where(denom > 0, coh_weighted, torch.zeros_like(coh_weighted))
    return ((1.0 - coh) * 0.5).clamp(0.0, 1.0)


class CholinergicGate(nn.Module):
    """
    ACh v0.2: compuerta por token centrada en identidad.

        c_base = 1 + base_scale * tanh(conductance_base)
        rho_t  = sigmoid(k * (u_t - theta))
        mask_t = c_base * (1 - lambda * rho_t * sigmoid(w_supp))

    La medicion u_t queda detached. La respuesta a esa medicion
    (conductance_base, rho_k, theta, w_supp) si es entrenable.
    """
    def __init__(self, dim: int, lam: float = 0.12,
                 theta_init: float = 0.5, rho_k_init: float = 2.0,
                 ach_enabled: bool = True, base_scale: float = 0.10,
                 freeze_conductance: bool = False):
        super().__init__()
        if not (0.0 <= lam < 1.0):
            raise ValueError(f"lam debe estar en [0,1), recibido {lam}")
        if not (0.0 <= base_scale < 1.0):
            raise ValueError(f"base_scale debe estar en [0,1), recibido {base_scale}")
        self.dim = dim
        self.lam = float(lam)
        self.base_scale = float(base_scale)
        self.ach_enabled = ach_enabled

        # Cero -> gain exactamente 1.0 al inicio.
        self.conductance_base = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.conductance_base.requires_grad_(not freeze_conductance)
        self.rho_k = nn.Parameter(torch.tensor(float(rho_k_init)))
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))
        self.w_supp = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

        # Tensores de diagnostico: no sincronizan CUDA dentro del forward.
        self.register_buffer("last_u_mean", torch.tensor(-1.0), persistent=False)
        self.register_buffer("last_u_std", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_rho", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_mask_mean", torch.tensor(1.0), persistent=False)
        self.register_buffer("last_mask_min", torch.tensor(1.0), persistent=False)
        self.register_buffer("last_mask_max", torch.tensor(1.0), persistent=False)

    def _base_conductance(self, dtype: torch.dtype) -> Tensor:
        raw = self.conductance_base.to(dtype)
        return 1.0 + self.base_scale * torch.tanh(raw)

    def forward(self, h: Tensor, trace: Tensor | None = None) -> Tensor:
        c_base = self._base_conductance(h.dtype)

        if not self.ach_enabled or trace is None:
            mask = c_base.view(1, 1, -1)
            with torch.no_grad():
                self.last_u_mean.fill_(-1.0)
                self.last_u_std.zero_()
                self.last_rho.zero_()
                self.last_mask_mean.copy_(c_base.float().mean())
                self.last_mask_min.copy_(c_base.float().min())
                self.last_mask_max.copy_(c_base.float().max())
            return h * mask

        with torch.no_grad():
            u = cross_layer_incoherence(trace)

        k = self.rho_k.clamp(min=0.5, max=8.0).to(h.dtype)
        th = self.theta.clamp(min=0.0, max=1.0).to(h.dtype)
        rho = torch.sigmoid(k * (u.to(h.dtype) - th)).unsqueeze(-1)

        supp = torch.sigmoid(self.w_supp.to(h.dtype))
        mask = c_base.view(1, 1, -1) * (
            1.0 - self.lam * rho * supp.view(1, 1, -1))
        mask = mask.clamp(min=0.05, max=1.5)

        with torch.no_grad():
            self.last_u_mean.copy_(u.mean())
            self.last_u_std.copy_(u.std(unbiased=False))
            self.last_rho.copy_(rho.float().mean())
            self.last_mask_mean.copy_(mask.float().mean())
            self.last_mask_min.copy_(mask.float().min())
            self.last_mask_max.copy_(mask.float().max())
        return h * mask

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        c_base = self._base_conductance(torch.float32)
        supp = torch.sigmoid(self.w_supp.float())
        return {
            "u_mean": float(self.last_u_mean.item()),
            "u_std": float(self.last_u_std.item()),
            "rho_mean": float(self.last_rho.item()),
            "mask_mean": float(self.last_mask_mean.item()),
            "mask_min": float(self.last_mask_min.item()),
            "mask_max": float(self.last_mask_max.item()),
            "theta": float(self.theta.clamp(0.0, 1.0).item()),
            "rho_k": float(self.rho_k.clamp(0.5, 8.0).item()),
            "base_mean": float(c_base.mean().item()),
            "supp_mean": float(supp.mean().item()),
            "lambda": self.lam,
        }


class MoleculeGate(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        rank = min(32, dim // 8)
        self.gate_down = CastedLinear(dim, rank, bias=False)
        self.gate_up = CastedLinear(rank, vocab_size, bias=False)
        nn.init.normal_(self.gate_down.weight, std=0.01)
        nn.init.zeros_(self.gate_up.weight)

    def forward(self, logits: Tensor, h: Tensor, stress: Tensor | float,
                stress_threshold: float) -> Tensor:
        # Camino completamente tensorial: sin if/min sobre escalares CUDA.
        s = torch.as_tensor(stress, device=logits.device, dtype=logits.dtype)
        th = torch.as_tensor(stress_threshold, device=logits.device,
                             dtype=logits.dtype)
        weight = ((s - th) / (1.0 - th + 1e-6)).clamp(0.0, 1.0)
        correction = self.gate_up(torch.tanh(self.gate_down(h)))
        return logits + weight * correction


class RuntimeObserver:
    """Telemetria de bajo acoplamiento; almacena solo reducciones detached."""
    def __init__(self, enabled: bool = True, sample_tokens: int = 256):
        self.enabled = bool(enabled)
        self.sample_tokens = max(int(sample_tokens), 8)
        self.active = False
        self.step = -1
        self.phase = ""
        self._stats: dict[str, object] = {}

    def arm(self, step: int, phase: str) -> None:
        if not self.enabled:
            return
        self.active = True
        self.step = int(step)
        self.phase = str(phase)
        self._stats = {}

    @torch.no_grad()
    def tensor(self, name: str, x: Tensor) -> None:
        if not self.active:
            return
        z = x.detach().float()
        flat = z.reshape(-1)
        if flat.numel() > self.sample_tokens:
            idx = torch.linspace(0, flat.numel() - 1, self.sample_tokens,
                                 device=flat.device).long()
            sample = flat.index_select(0, idx)
        else:
            sample = flat
        self._stats[name] = {
            "mean": z.mean(),
            "std": z.std(unbiased=False),
            "rms": z.square().mean().sqrt(),
            "absmax": z.abs().max(),
            "p10": torch.quantile(sample, 0.10),
            "p50": torch.quantile(sample, 0.50),
            "p90": torch.quantile(sample, 0.90),
        }

    @torch.no_grad()
    def trace(self, name: str, trace: Tensor) -> None:
        if not self.active:
            return
        tf = trace.detach().float()
        block_rms = tf.square().mean(dim=(0, 1, 3)).sqrt()
        deltas = tf[:, :, 1:, :] - tf[:, :, :-1, :]
        delta_rms = deltas.square().mean(dim=(0, 1, 3)).sqrt()
        dirs = F.normalize(deltas, dim=-1, eps=1e-6)
        if dirs.size(2) >= 2:
            curvature = (dirs[:, :, 1:, :] * dirs[:, :, :-1, :]).sum(-1)
            curvature_mean = curvature.mean(dim=(0, 1))
        else:
            curvature_mean = tf.new_empty((0,))
        self._stats[name] = {
            "block_rms": block_rms,
            "delta_rms": delta_rms,
            "delta_cos": curvature_mean,
        }

    @torch.no_grad()
    def logits(self, name: str, logits: Tensor) -> None:
        if not self.active:
            return
        z = logits.detach().float().reshape(-1, logits.size(-1))
        z = z[:self.sample_tokens]
        p = z.softmax(dim=-1)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
        top2 = z.topk(k=min(2, z.size(-1)), dim=-1).values
        margin = top2[:, 0] - top2[:, 1] if top2.size(-1) == 2 else top2[:, 0]
        self._stats[name] = {
            "entropy_mean": entropy.mean(),
            "entropy_std": entropy.std(unbiased=False),
            "top1_margin_mean": margin.mean(),
            "logit_absmax": z.abs().max(),
        }

    def finish(self) -> None:
        self.active = False

    def snapshot(self, clear: bool = True) -> dict:
        def convert(v):
            if isinstance(v, Tensor):
                if v.numel() == 1:
                    return float(v.item())
                return [float(x) for x in v.detach().cpu().reshape(-1).tolist()]
            if isinstance(v, dict):
                return {k: convert(x) for k, x in v.items()}
            return v

        out = {
            "step": self.step,
            "phase": self.phase,
            "tensors": convert(self._stats),
        }
        if clear:
            self._stats = {}
        return out


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE PRIMITIVES
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device, dtype):
        if (self._cos_cached is None or
                self._seq_len_cached != seq_len or
                self._cos_cached.device != device):
            t     = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)


class GQAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, rope_base: float,
                 qk_gain_init: float = 1.5):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = dim // n_heads
        kv_dim          = n_kv_heads * self.head_dim
        self.c_q    = CastedLinear(dim, dim,    bias=False)
        self.c_k    = CastedLinear(dim, kv_dim, bias=False)
        self.c_v    = CastedLinear(dim, kv_dim, bias=False)
        self.proj   = CastedLinear(dim, dim,    bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_gain = nn.Parameter(
            torch.full((n_heads,), qk_gain_init, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        self.fc   = CastedLinear(dim, mult*dim, bias=False)
        self.proj = CastedLinear(mult*dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


# ─────────────────────────────────────────────────────────────
# PHASE 1 — PLASTIC DENDRITIC ENCODER  (Experiencia)
# Unchanged from v1 — independent agents, multi-scale.
# ─────────────────────────────────────────────────────────────

class PlasticDendriticField(nn.Module):
    def __init__(self, dim: int, n_proto: int):
        super().__init__()
        self.dim     = dim
        self.n_proto = n_proto
        self.prototypes = nn.Parameter(torch.randn(n_proto, dim) * 0.02)
        self.log_beta = nn.Parameter(
            torch.tensor(math.log(math.sqrt(dim)), dtype=torch.float32))

    def forward(self, h: Tensor) -> Tensor:
        B, T, D = h.shape
        dtype    = h.dtype
        h_norm  = F.normalize(h.float(), dim=-1)
        P       = F.normalize(self.prototypes.float(), dim=-1)
        beta    = self.log_beta.float().exp()
        sim     = h_norm @ P.T
        weights = torch.softmax(beta * sim, dim=-1)
        recall  = weights @ self.prototypes.float()
        alpha   = sim.max(dim=-1, keepdim=True).values.clamp(0.0, 1.0)
        return ((1.0 - alpha) * h.float() + alpha * recall).to(dtype)


def _inverse_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    return math.log(math.expm1(x))


def _logit(x: float) -> float:
    x = min(max(float(x), 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


class GeometricFrameBank(nn.Module):
    """Tres marcos L/R no ortogonales: local, mid y global.

    Cada campo mantiene:
      - centro neutral mu_N como EMA no entrenable;
      - dos direcciones aprendibles L/R en el espacio completo;
      - dos direcciones aprendibles L/R en head-space para metrizar q/k/v;
      - escalas y pasos de cuantizacion axiales;
      - intensidad de metrica, transporte y geodesia acotadas.

    El Gramiano se regulariza antes de invertirlo. La proyeccion conserva
    exactamente el complemento ortogonal del plano L/R; solo modifica las
    coordenadas dentro del plano.
    """
    n_fields = 3
    n_axes = 2

    def __init__(self, dim: int, head_dim: int, *, enabled: bool,
                 center_decay: float, axis_init_cos: float,
                 gamma_init: float, metric_max: float,
                 transport_max: float, tau_max: float,
                 quantize: bool, quant_step_init: float,
                 axis_max_abs_cos: float, cross_field_target: float,
                 eps: float):
        super().__init__()
        if not (0.0 <= center_decay < 1.0):
            raise ValueError('geo center_decay must be in [0,1)')
        if not (-0.95 < axis_init_cos < 0.95):
            raise ValueError('geo axis_init_cos must be in (-0.95,0.95)')
        if dim < 2 or head_dim < 2:
            raise ValueError('geometric frames require dim/head_dim >= 2')

        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.enabled = bool(enabled)
        self.center_decay = float(center_decay)
        self.metric_max = float(metric_max)
        self.transport_max = float(transport_max)
        self.tau_max = float(tau_max)
        self.quantize = bool(quantize)
        self.axis_max_abs_cos = float(axis_max_abs_cos)
        self.cross_field_target = float(cross_field_target)
        self.eps = float(eps)

        self.axes = nn.Parameter(self._init_frames(dim, axis_init_cos))       # (3,2,D)
        self.head_axes = nn.Parameter(self._init_frames(head_dim, axis_init_cos))

        self.gamma_projector = CastedLinear(dim, self.n_fields, bias=True)
        nn.init.zeros_(self.gamma_projector.weight)
        nn.init.constant_(self.gamma_projector.bias, _logit(gamma_init))

        self.metric_raw = nn.Parameter(torch.full(
            (self.n_fields,), _logit(0.25)))
        self.transport_raw = nn.Parameter(torch.full(
            (self.n_fields,), _logit(0.25)))
        self.eta_raw = nn.Parameter(torch.full((self.n_fields,), _logit(0.25)))
        self.disagreement_gain_raw = nn.Parameter(torch.full(
            (self.n_fields,), _inverse_softplus(2.0)))
        self.residual_gain_raw = nn.Parameter(torch.full(
            (self.n_fields,), _inverse_softplus(2.0)))
        self.log_coord_scale = nn.Parameter(torch.zeros(self.n_fields, self.n_axes))
        self.register_buffer('quant_step', torch.full(
            (self.n_fields, self.n_axes), float(quant_step_init)))

        self.register_buffer('center', torch.zeros(self.n_fields, dim))
        self.register_buffer('center_initialized',
                             torch.zeros(self.n_fields, dtype=torch.bool))

        self._last_gamma = None
        self._last_tau = None
        self._last_disagreement = None
        self._last_projection_residual = None

        if not self.enabled:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _init_frames(dim: int, target_cos: float) -> Tensor:
        frames = []
        sin = math.sqrt(max(1.0 - target_cos * target_cos, 1e-8))
        for _ in range(GeometricFrameBank.n_fields):
            q, _ = torch.linalg.qr(torch.randn(dim, 2), mode='reduced')
            left = q[:, 0]
            right = target_cos * q[:, 0] + sin * q[:, 1]
            frames.append(torch.stack([left, right], dim=0))
        return torch.stack(frames, dim=0)

    def _normalized_axes(self, head_space: bool = False) -> Tensor:
        raw = self.head_axes if head_space else self.axes
        return F.normalize(raw.float(), dim=-1, eps=self.eps)

    def _gram(self, field_idx: int, head_space: bool = False) -> tuple[Tensor, Tensor]:
        axes = self._normalized_axes(head_space)[field_idx]              # (2,d)
        gram = axes @ axes.T
        eye = torch.eye(self.n_axes, device=gram.device, dtype=gram.dtype)
        # La inversión geométrica debe ejecutarse en FP32.
        # pinv tolera Gram singular o casi singular bajo cuantización.
        with torch.autocast(
                device_type=gram.device.type,
                enabled=False):
            gram_fp32 = gram.float()
            eye_fp32 = torch.eye(
                gram_fp32.size(-1),
                device=gram_fp32.device,
                dtype=torch.float32,
            )
            regularization = max(float(self.eps), 1e-4)
            gram_inv = torch.linalg.pinv(
                gram_fp32 + regularization * eye_fp32,
                hermitian=True,
                rtol=1e-5,
            ).to(dtype=gram.dtype)
        return axes, gram_inv

    def _dual_projection(self, z: Tensor, field_idx: int,
                         head_space: bool = False) -> tuple[Tensor, Tensor]:
        axes, gram_inv = self._gram(field_idx, head_space=head_space)
        cov = torch.einsum('...d,ad->...a', z.float(), axes)
        coords = cov @ gram_inv
        plane = torch.einsum('...a,ad->...d', coords, axes)
        return plane, coords

    @staticmethod
    def _preserve_rms(reference: Tensor, transformed: Tensor,
                      eps: float) -> Tensor:
        ref_rms = reference.float().square().mean(dim=-1, keepdim=True).sqrt()
        new_rms = transformed.float().square().mean(dim=-1, keepdim=True).sqrt()
        return transformed * (ref_rms / new_rms.clamp_min(eps)).to(transformed.dtype)

    def gamma_prior(self, x: Tensor) -> Tensor:
        logits = self.gamma_projector(F.rms_norm(x, (x.size(-1),)))
        return torch.sigmoid(logits.float())                            # (B,T,3)

    def metric_transform(self, z: Tensor, field_idx: int,
                         gamma: Tensor, *, values: bool = False) -> Tensor:
        """Low-rank metric/transport transform without materializing M.

        z: (B,H,T,head_dim), gamma: (B,T). For q/k this is a dynamic
        square-root metric. For v it is the message transport operator.
        """
        if not self.enabled:
            return z
        plane, _ = self._dual_projection(z, field_idx, head_space=True)
        if values:
            strength = self.transport_max * torch.sigmoid(
                self.transport_raw[field_idx])
        else:
            strength = self.metric_max * torch.sigmoid(
                self.metric_raw[field_idx])
        g = gamma[:, None, :, None].to(z.dtype)
        out = z + strength.to(z.dtype) * g * plane.to(z.dtype)
        if values:
            return self._preserve_rms(z, out, self.eps)
        return F.rms_norm(out, (out.size(-1),), eps=self.eps)

    def effective_center(self, h: Tensor, field_idx: int) -> Tensor:
        stored = self.center[field_idx].float()
        batch_center = h.detach().float().mean(dim=(0, 1))
        init = self.center_initialized[field_idx]
        return torch.where(init, stored, batch_center)

    def project(self, h: Tensor, field_idx: int) -> tuple[Tensor, Tensor]:
        dtype = h.dtype
        center = self.effective_center(h, field_idx)
        z = h.float() - center
        plane, coords = self._dual_projection(z, field_idx, head_space=False)
        z_perp = z - plane

        if self.quantize:
            scale = self.log_coord_scale[field_idx].float().exp().clamp(0.1, 10.0)
            delta = self.quant_step[field_idx].float().clamp_min(1e-4)
            scaled = coords * scale
            discrete = torch.round(scaled / delta) * delta
            scaled_q = scaled + (discrete - scaled).detach()             # STE
            coords_q = scaled_q / scale
        else:
            coords_q = coords

        axes = self._normalized_axes(False)[field_idx]
        plane_q = torch.einsum('...a,ad->...d', coords_q, axes)
        projected = center + z_perp + plane_q
        residual = (projected - h.float()).norm(dim=-1) / \
                   z.norm(dim=-1).clamp_min(self.eps)
        return projected.to(dtype), residual.clamp(0.0, 4.0)

    def geodesic_transport(self, h: Tensor, projected: Tensor,
                           field_idx: int, gamma: Tensor) -> tuple[Tensor, Tensor]:
        center = self.effective_center(h, field_idx)
        z = h.float() - center
        p = projected.float() - center
        r0 = z.norm(dim=-1, keepdim=True)
        r1 = p.norm(dim=-1, keepdim=True)
        u = z / r0.clamp_min(self.eps)
        q = p / r1.clamp_min(self.eps)
        dot = (u * q).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps,
                                                      1.0 - self.eps)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        tangent = (q - dot * u) / sin_theta.clamp_min(self.eps)

        eta = torch.sigmoid(self.eta_raw[field_idx])
        tau = (self.tau_max * eta * gamma.float()).clamp(0.0, 1.0).unsqueeze(-1)
        direction = torch.cos(tau * theta) * u + torch.sin(tau * theta) * tangent

        # Cerca de theta=0, SLERP se vuelve numericamente innecesario.
        lerp_direction = F.normalize((1.0 - tau) * u + tau * q,
                                     dim=-1, eps=self.eps)
        direction = torch.where((theta < 1e-4).expand_as(direction),
                                lerp_direction, direction)
        log_r = ((1.0 - tau) * torch.log(r0.clamp_min(self.eps)) +
                 tau * torch.log(r1.clamp_min(self.eps)))
        out = center + torch.exp(log_r) * direction

        # Estados degenerados usan interpolacion lineal segura.
        degenerate = ((r0 < self.eps) | (r1 < self.eps)).expand_as(out)
        linear = h.float() + tau * (projected.float() - h.float())
        out = torch.where(degenerate, linear, out)
        out = torch.where((tau <= self.eps).expand_as(out), h.float(), out)
        out = torch.where((tau >= 1.0 - self.eps).expand_as(out),
                          projected.float(), out)
        return out.to(h.dtype), tau.squeeze(-1)

    def posterior_gamma(self, prior: Tensor, field_outputs: list[Tensor],
                        projections: list[Tensor], residuals: list[Tensor]) -> tuple[Tensor, Tensor]:
        stack = torch.stack([x.float() for x in field_outputs], dim=2)   # B,T,3,D
        total = stack.sum(dim=2, keepdim=True)
        other = (total - stack) / float(self.n_fields - 1)
        disagreement = 0.5 * (1.0 - F.cosine_similarity(stack, other, dim=-1))
        disagreement = disagreement.clamp(0.0, 1.0)
        residual = torch.stack(residuals, dim=-1).tanh()

        axes = self._normalized_axes(False)
        axis_cos = (axes[:, 0] * axes[:, 1]).sum(dim=-1)
        confidence = torch.sqrt((1.0 - axis_cos.square()).clamp_min(self.eps))

        prior_logit = torch.logit(prior.clamp(1e-5, 1.0 - 1e-5))
        gd = F.softplus(self.disagreement_gain_raw).view(1, 1, -1)
        gr = F.softplus(self.residual_gain_raw).view(1, 1, -1)
        logits = prior_logit + gd * (disagreement - 0.15) + gr * (residual - 0.10)
        gamma = torch.sigmoid(logits) * confidence.view(1, 1, -1)
        return gamma.clamp(0.0, 1.0), disagreement

    @torch.no_grad()
    def update_centers(self, field_outputs: list[Tensor]) -> None:
        if not self.training:
            return
        currents = torch.stack([
            h.detach().float().mean(dim=(0, 1)) for h in field_outputs
        ], dim=0)
        # DDP usa broadcast_buffers=False en este modelo. Sin esta reduccion,
        # cada rank construiria una geometria neutral distinta.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(currents, op=dist.ReduceOp.SUM)
            currents.div_(dist.get_world_size())
        for i, current in enumerate(currents):
            initialized = self.center_initialized[i].to(current.dtype)
            ema = (self.center[i].float() * self.center_decay +
                   current * (1.0 - self.center_decay))
            updated = initialized * ema + (1.0 - initialized) * current
            self.center[i].copy_(updated)
            self.center_initialized[i].fill_(True)

    def regularization_loss(self) -> Tensor:
        axes = self._normalized_axes(False)
        head_axes = self._normalized_axes(True)
        cos_full = (axes[:, 0] * axes[:, 1]).sum(dim=-1)
        cos_head = (head_axes[:, 0] * head_axes[:, 1]).sum(dim=-1)
        max_cos = self.axis_max_abs_cos
        independence = F.relu(cos_full.abs() - max_cos).square().mean()
        independence = independence + 0.5 * F.relu(
            cos_head.abs() - max_cos).square().mean()

        # Coherencia de subespacios, no de ejes individuales: invariante a
        # rotaciones y permutaciones L/R dentro de cada campo.
        qs = []
        for i in range(self.n_fields):
            q, _ = torch.linalg.qr(axes[i].T, mode='reduced')           # D,2
            qs.append(q)
        overlaps = []
        for i, j in ((0, 1), (1, 2), (0, 2)):
            c = qs[i].T @ qs[j]
            overlaps.append(c.square().sum() / self.n_axes)
        overlap = torch.stack(overlaps)
        coherence = (overlap - self.cross_field_target).square().mean()
        return independence + 0.25 * coherence

    @torch.no_grad()
    def calibrate(self, neutral: Tensor, left: Tensor, right: Tensor) -> dict:
        """Calibra mu_N y ejes L/R desde estados empiricos.

        Entradas: (N,3,D) o (N,D). En el segundo caso la misma calibracion
        se replica en los tres campos. Las direcciones siguen siendo params y
        pueden continuar entrenandose despues de calibrar.
        """
        def canonical(name: str, x: Tensor) -> Tensor:
            if x.ndim == 2:
                x = x[:, None, :].expand(-1, self.n_fields, -1)
            if x.ndim != 3 or x.size(1) != self.n_fields or x.size(2) != self.dim:
                raise ValueError(
                    f'{name} must be (N,3,{self.dim}) or (N,{self.dim}); got {tuple(x.shape)}')
            if x.size(0) < 2 or not torch.isfinite(x).all():
                raise ValueError(f'{name} requires >=2 finite samples')
            return x.to(device=self.axes.device, dtype=torch.float32)

        n = canonical('neutral', neutral)
        l = canonical('left', left)
        r = canonical('right', right)
        mu_n = n.median(dim=0).values
        raw_l = l.median(dim=0).values - mu_n
        raw_r = r.median(dim=0).values - mu_n
        if (raw_l.norm(dim=-1) < self.eps).any() or (raw_r.norm(dim=-1) < self.eps).any():
            raise RuntimeError('degenerate geometric calibration: near-zero axis')
        axis_l = F.normalize(raw_l, dim=-1, eps=self.eps)
        axis_r = F.normalize(raw_r, dim=-1, eps=self.eps)
        cos_lr = (axis_l * axis_r).sum(dim=-1)
        if (cos_lr.abs() > 0.985).any():
            raise RuntimeError('degenerate geometric calibration: nearly collinear L/R axes')

        self.center.copy_(mu_n)
        self.center_initialized.fill_(True)
        self.axes[:, 0].copy_(axis_l)
        self.axes[:, 1].copy_(axis_r)

        for i in range(self.n_fields):
            axes, gram_inv = self._gram(i, head_space=False)
            cov_l = torch.einsum('nd,ad->na', l[:, i] - mu_n[i], axes)
            cov_r = torch.einsum('nd,ad->na', r[:, i] - mu_n[i], axes)
            coords_l = cov_l @ gram_inv
            coords_r = cov_r @ gram_inv
            typical = torch.stack([
                coords_l[:, 0].abs().median(),
                coords_r[:, 1].abs().median(),
            ]).clamp_min(1e-3)
            self.log_coord_scale[i].copy_(-typical.log())

        return self.snapshot()

    @torch.no_grad()
    def snapshot(self) -> dict:
        axes = self._normalized_axes(False)
        head_axes = self._normalized_axes(True)
        axis_cos = (axes[:, 0] * axes[:, 1]).sum(dim=-1)
        head_cos = (head_axes[:, 0] * head_axes[:, 1]).sum(dim=-1)
        out = {
            'enabled': self.enabled,
            'axis_cos_LR': [float(x) for x in axis_cos.cpu()],
            'head_axis_cos_LR': [float(x) for x in head_cos.cpu()],
            'gram_det': [float(x) for x in (1.0 - axis_cos.square()).cpu()],
            'center_initialized': [bool(x) for x in self.center_initialized.cpu()],
            'metric_strength': [float(x) for x in (
                self.metric_max * torch.sigmoid(self.metric_raw)).cpu()],
            'transport_strength': [float(x) for x in (
                self.transport_max * torch.sigmoid(self.transport_raw)).cpu()],
            'eta': [float(x) for x in torch.sigmoid(self.eta_raw).cpu()],
            'quant_step': [[float(v) for v in row] for row in
                           self.quant_step.cpu()],
        }
        if self._last_gamma is not None:
            out['gamma_mean'] = [float(x) for x in
                                 self._last_gamma.float().mean(dim=(0, 1)).cpu()]
        if self._last_tau is not None:
            out['tau_mean'] = [float(x) for x in
                               self._last_tau.float().mean(dim=(0, 1)).cpu()]
        return out


class DendriticEncoder(nn.Module):
    def __init__(self, dim: int, field_heads: list[int],
                 local_span: int, mid_span: int,
                 n_kv_heads: int, rope_base: float,
                 n_proto: int = 16, *,
                 geometric_enabled: bool = True,
                 geo_center_decay: float = 0.99,
                 geo_axis_init_cos: float = 0.35,
                 geo_gamma_init: float = 0.08,
                 geo_metric_max: float = 0.30,
                 geo_transport_max: float = 0.12,
                 geo_tau_max: float = 0.35,
                 geo_quantize: bool = True,
                 geo_quant_step_init: float = 0.25,
                 geo_axis_max_abs_cos: float = 0.92,
                 geo_cross_field_target: float = 0.10,
                 geo_eps: float = 1e-5):
        super().__init__()
        self.dim         = dim
        self.field_heads = field_heads
        self.n_heads     = sum(field_heads)
        self.n_kv_heads  = n_kv_heads
        self.head_dim    = dim // self.n_heads
        if self.head_dim * self.n_heads != dim:
            raise ValueError(f'dim={dim} must be divisible by total field heads={self.n_heads}')
        self.local_span  = local_span
        self.mid_span    = mid_span
        kv_dim = n_kv_heads * self.head_dim
        self.q_local  = CastedLinear(dim, field_heads[0] * self.head_dim, bias=False)
        self.q_mid    = CastedLinear(dim, field_heads[1] * self.head_dim, bias=False)
        self.q_global = CastedLinear(dim, field_heads[2] * self.head_dim, bias=False)
        self.c_k    = CastedLinear(dim, kv_dim, bias=False)
        self.c_v    = CastedLinear(dim, kv_dim, bias=False)
        self.proj   = CastedLinear(dim, dim,    bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.plastic_local  = PlasticDendriticField(dim, n_proto)
        self.plastic_mid    = PlasticDendriticField(dim, n_proto)
        self.plastic_global = PlasticDendriticField(dim, n_proto)
        self.field_mix = nn.Parameter(torch.ones(3, dtype=torch.float32) / 3.0)
        self.geometry = GeometricFrameBank(
            dim, self.head_dim,
            enabled=geometric_enabled,
            center_decay=geo_center_decay,
            axis_init_cos=geo_axis_init_cos,
            gamma_init=geo_gamma_init,
            metric_max=geo_metric_max,
            transport_max=geo_transport_max,
            tau_max=geo_tau_max,
            quantize=geo_quantize,
            quant_step_init=geo_quant_step_init,
            axis_max_abs_cos=geo_axis_max_abs_cos,
            cross_field_target=geo_cross_field_target,
            eps=geo_eps,
        )
        self._mask_cache: dict = {}

    def _make_span_mask(self, T: int, span: int, device, dtype) -> Tensor:
        key = (T, span, str(device), str(dtype))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        causal = torch.ones(T, T, device=device, dtype=torch.bool).tril()
        if span < T:
            idx = torch.arange(T, device=device)
            dist_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)) >= -span
            causal = causal & dist_mask
        mask = torch.zeros(T, T, device=device, dtype=dtype)
        mask.masked_fill_(~causal, float('-inf'))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        k_base = self.c_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_base = self.c_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k_base = F.rms_norm(k_base, (k_base.size(-1),))
        cos, sin = self.rotary(T, x.device, x.dtype)

        gamma_prior = self.geometry.gamma_prior(x) if self.geometry.enabled else None
        q_projs  = [self.q_local, self.q_mid, self.q_global]
        plastics = [self.plastic_local, self.plastic_mid, self.plastic_global]
        spans    = [self.local_span, self.mid_span, T]
        field_outputs: list[Tensor] = []

        for field_idx, (q_proj, plastic, span, nh) in enumerate(
                zip(q_projs, plastics, spans, self.field_heads)):
            q = q_proj(x).reshape(B, T, nh, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            k_exp = k_base
            v_exp = v_base

            if self.geometry.enabled:
                g = gamma_prior[:, :, field_idx]
                q = self.geometry.metric_transform(q, field_idx, g, values=False)
                k_exp = self.geometry.metric_transform(k_base, field_idx, g, values=False)
                v_exp = self.geometry.metric_transform(v_base, field_idx, g, values=True)

            # La geometria vive en head-space semantico; RoPE se aplica despues
            # para conservar equivariancia posicional.
            q = apply_rope(q, cos, sin)
            k_exp = apply_rope(k_exp, cos, sin)

            if span < T:
                attn_mask = self._make_span_mask(T, span, x.device, q.dtype)
                y = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=attn_mask, is_causal=False,
                    enable_gqa=(nh != self.n_kv_heads))
            else:
                y = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=None, is_causal=True,
                    enable_gqa=(nh != self.n_kv_heads))

            field_h = y.transpose(1, 2).contiguous().reshape(B, T, nh * self.head_dim)
            if field_h.size(-1) < D:
                pad = torch.zeros(B, T, D - field_h.size(-1),
                                  device=x.device, dtype=x.dtype)
                field_h_full = torch.cat([field_h, pad], dim=-1)
            else:
                field_h_full = field_h
            field_outputs.append(plastic(field_h_full))

        if self.geometry.enabled:
            projections, residuals = [], []
            for i, h_field in enumerate(field_outputs):
                p, r = self.geometry.project(h_field, i)
                projections.append(p)
                residuals.append(r)
            gamma, disagreement = self.geometry.posterior_gamma(
                gamma_prior, field_outputs, projections, residuals)
            transported, taus = [], []
            for i, (h_field, projected) in enumerate(zip(field_outputs, projections)):
                out, tau = self.geometry.geodesic_transport(
                    h_field, projected, i, gamma[:, :, i])
                transported.append(out)
                taus.append(tau)
            self.geometry._last_gamma = gamma.detach()
            self.geometry._last_tau = torch.stack(taus, dim=-1).detach()
            self.geometry._last_disagreement = disagreement.detach()
            self.geometry._last_projection_residual = torch.stack(
                residuals, dim=-1).detach()
            self.geometry.update_centers(field_outputs)
            field_outputs = transported

        mix   = F.softmax(self.field_mix.to(x.dtype), dim=0)
        mixed = sum(mix[i] * field_outputs[i] for i in range(3))
        return self.proj(mixed)

    def geometry_aux_loss(self) -> Tensor:
        if not self.geometry.enabled:
            return self.field_mix.new_zeros(())
        return self.geometry.regularization_loss()

    @torch.no_grad()
    def geometry_snapshot(self) -> dict:
        return self.geometry.snapshot()

    @torch.no_grad()
    def calibrate_geometry(self, neutral: Tensor, left: Tensor,
                           right: Tensor) -> dict:
        return self.geometry.calibrate(neutral, left, right)


# ─────────────────────────────────────────────────────────────
# [NEW] RESONANCE BUFFER  (Memoria emergente)
#
# No es un almacén — es un registro del flujo.
# La memoria aparece cuando el estado actual resuena con lo
# que ya ha pasado por el Soma.
#
# Implementación:
#   - K slots de tamaño (D,) — inicializados en cero
#   - Buffer circular: nuevo estado se escribe en ptr actual
#   - Actualización: EMA del hidden state del Soma
#   - Sin params entrenables — pura observación del medio
#   - filled: flag que indica si el buffer tiene contenido real
#
# Filosofía:
#   La memoria no se "guarda" — resuena.
#   Si no hay match con el flujo previo, no hay señal.
#   Si hay match, el PerceptionAgent lo detecta y lo amplifica.
# ─────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════
# [v2.3 NEW] ISOMERIC POLARIZATION MONITOR
#
# Adaptación del framework PfV (Ledesma, "Polarization from
# Views — Isomeric Polarization") aplicado como AUTO-OBSERVADOR
# INTERNO del modelo.
#
# Cada conjunto de realizaciones (slots del ResonanceBuffer,
# prototipos del PlasticDendriticField) es un ℛ_t del paper.
# Computamos:
#
#   P_t     = Agg({ d(Φ(r^i), Φ(r^j)) })_{i<j}    (instantánea)
#   P_reg   = E_{t ∈ window}[P_t]                   (régimen)
#   ΔP_t    = P_t − P_{t−1}                         (velocidad)
#   Δ²P_t   = ΔP_t − ΔP_{t−1}                       (aceleración)
#   ε_t     = quantile(d_pairs, q)                  (umbral isomérico)
#   iso_mask = { (i,j) : d_ij ≤ ε_t }              (pares isómeros)
#
# Observable Φ = identity (slots y prototipos son vectores en 𝒴).
# Divergencia d = distancia coseno ∈ [0, 2].
# Agregador Agg = mean (default) o trimmed_mean (robusto).
#
# Señales útiles:
#   P_t → 0    : colapso — realizaciones redundantes (mode=mean bug)
#   P_t alto   : incoherencia — ruido en las realizaciones
#   Δ²P_t alto : transición de régimen — momento crítico del flujo
#   n_isomers↑ : plasticidad estructural indicada (poda/sustitución)
#   dead > τ   : ramas inactivas — candidatas a renacimiento
#
# Sin params entrenables. Sin .item() en el hot path.
# Overhead: O(K²·D) por compute — trivial para K∈[16,64], D=512.
# ═════════════════════════════════════════════════════════════

class IsomericPolarizationMonitor(nn.Module):
    """
    PfV monitor aplicado a un conjunto de m realizaciones.

    [v3.1 PATCH] Señal densa en compute_ttt_loss:
        L_iso = loss_regime (1D) + α·loss_density (K-D) + β·loss_accel (1D)
    El nuevo término loss_density expande el rango del gradiente de
    1 dimensión escalar a K dimensiones vectoriales (K = n_realizations).
    Controla el peso con env var PFV_DENSITY_WEIGHT (default 1.0).
    PFV_DENSITY_WEIGHT=0.0 desactiva el fix (comportamiento v3.0 original).

    Args:
        n_realizations: m, número de realizaciones en ℛ_t
        window:         tamaño del ring buffer de P_history (para régimen)
        agg:            'mean' o 'trimmed_mean'  (Agg del paper)
        epsilon_quantile: q ∈ (0,1). ε_t = quantile(d_pairs, q).
                         q pequeño (0.1) → ε estricto, pocos isómeros
                         q grande  (0.5) → ε laxo, muchos isómeros
        activation_ema: EMA decay para activation_count
        density_weight: [v3.1] peso del término vectorial loss_density
    """
    def __init__(self, n_realizations: int, window: int = 50,
                 agg: str = 'mean', epsilon_quantile: float = 0.1,
                 activation_ema: float = 0.95,
                 density_weight: float = None):
        super().__init__()
        assert agg in ('mean', 'trimmed_mean')
        assert 0.0 < epsilon_quantile < 1.0
        self.n_realizations   = n_realizations
        self.window           = window
        self.agg              = agg
        self.epsilon_quantile = epsilon_quantile
        self.activation_ema   = activation_ema
        # [v3.1] density_weight por env var si no se pasa explícito
        if density_weight is None:
            density_weight = float(os.environ.get("PFV_DENSITY_WEIGHT", 1.0))
        self.density_weight   = density_weight

        # Ring buffer de P_t — para P_regime, ΔP_t, Δ²P_t
        self.register_buffer('P_history', torch.zeros(window))
        self.register_buffer('P_ptr',     torch.zeros(1, dtype=torch.long))
        self.register_buffer('P_fill',    torch.zeros(1, dtype=torch.float32))

        # Ring buffer de ΔP_t — para Δ²P_t
        self.register_buffer('dP_history', torch.zeros(window))

        # EMA de activación por realización — para detectar dead realizations
        self.register_buffer('activation_count', torch.zeros(n_realizations))

    @torch.no_grad()
    def compute(self, realizations: Tensor) -> dict:
        """
        realizations: (K, D) o (K, L, D)

        Returns dict con tensores escalares y máscaras:
            P_t, P_regime, delta_P, delta2_P, epsilon, n_isomers,
            iso_mask, dead_mask, dead_count
        """
        K = realizations.size(0)
        # Reducir (K, L, D) → (K, D) con mean sobre la dimensión temporal
        if realizations.ndim == 3:
            r = realizations.float().mean(dim=1)
        else:
            r = realizations.float()

        # Pairwise cosine distance
        r_norm = F.normalize(r, dim=-1)
        sim    = r_norm @ r_norm.T                                      # (K, K)
        d      = 1.0 - sim                                              # (K, K) ∈ [0, 2]

        # Triángulo superior estricto (pares i<j, sin diagonal)
        mask_upper = torch.triu(
            torch.ones(K, K, device=r.device, dtype=torch.bool), diagonal=1)
        d_pairs = d[mask_upper]                                         # (K*(K-1)/2,)

        # P_t = Agg(d_pairs)
        if self.agg == 'trimmed_mean':
            sorted_d, _ = d_pairs.sort()
            n_trim = max(1, int(len(sorted_d) * 0.1))
            P_t = sorted_d[n_trim:-n_trim].mean() if len(sorted_d) > 2*n_trim \
                  else sorted_d.mean()
        else:
            P_t = d_pairs.mean()

        # ε adaptativo — quantile robusto a outliers
        epsilon = torch.quantile(d_pairs, self.epsilon_quantile)

        # iso_mask: pares con d ≤ ε (dentro del upper triangle)
        iso_mask = (d <= epsilon) & mask_upper
        n_isomers = iso_mask.sum().float()

        # ── Update ring buffers sin .item() ──
        # one-hot del ptr actual para escribir en la posición correcta
        ptr_mask = F.one_hot(self.P_ptr, num_classes=self.window).squeeze(0).float()

        # ΔP_t antes de escribir (usa P_{t-1} del ring)
        # prev_ptr apunta al último P_t escrito (que es P_{t-1})
        prev_ptr   = (self.P_ptr - 1) % self.window
        prev_mask  = F.one_hot(prev_ptr, num_classes=self.window).squeeze(0).float()
        prev_P     = (self.P_history * prev_mask).sum()
        prev_dP    = (self.dP_history * prev_mask).sum()

        delta_P = P_t - prev_P

        # Δ²P_t = ΔP_t − ΔP_{t−1}
        delta2_P = delta_P - prev_dP

        # Escribir P_t y ΔP_t en la posición actual
        new_P_hist  = self.P_history  * (1.0 - ptr_mask) + P_t * ptr_mask
        new_dP_hist = self.dP_history * (1.0 - ptr_mask) + delta_P * ptr_mask
        self.P_history.copy_(new_P_hist)
        self.dP_history.copy_(new_dP_hist)
        self.P_ptr.add_(1).remainder_(self.window)
        self.P_fill.add_(1.0)

        # P_regime — mean de history sobre los slots rellenados
        effective_n = torch.clamp(self.P_fill, min=1.0, max=float(self.window))
        P_regime    = self.P_history.sum() / effective_n

        # Dead realizations — activación < 10% de la mediana global.
        # Criterio robusto: una realización está "muerta" si su activation
        # es despreciable comparada con lo típico del conjunto activo.
        if self.activation_count.numel() > 0 and self.activation_count.sum() > 0:
            median   = torch.median(self.activation_count)
            dead_mask = self.activation_count < 0.1 * median
        else:
            dead_mask = torch.zeros_like(self.activation_count, dtype=torch.bool)
        dead_count = dead_mask.sum().float()

        return {
            'P_t':      P_t,
            'P_regime': P_regime,
            'delta_P':  delta_P,
            'delta2_P': delta2_P,
            'epsilon':  epsilon,
            'n_isomers':  n_isomers,
            'iso_mask':   iso_mask,
            'dead_mask':  dead_mask,
            'dead_count': dead_count,
        }

    @torch.no_grad()
    def update_activations(self, scores: Tensor) -> None:
        """
        scores: (K,) — señal de activación por realización (ej: max attention,
                softmax max, top-1 count normalizado). EMA update.
        """
        assert scores.shape == self.activation_count.shape, \
            f"expected {self.activation_count.shape}, got {scores.shape}"
        self.activation_count.mul_(self.activation_ema).add_(
            scores.float(), alpha=1.0 - self.activation_ema)

    # ═════════════════════════════════════════════════════════════
    # [v3.1 PATCH] LOSS DIFERENCIABLE — Isomeric TTT con señal densa
    # ═════════════════════════════════════════════════════════════
    def compute_ttt_loss(self,
                          realizations_frozen: Tensor,
                          realization_new: Tensor,
                          beta: float = 0.1,
                          return_diag: bool = False):
        """
        Loss diferenciable para Test-Time Training isomérico — v3.1.

        Descomposición:
            loss_regime  = |P_t(R ∪ {p_new}) - P_regime|         [1D, legado]
            loss_density = mean |d(p_new, R) - d_slot_typical|   [K-D, NUEVO]
            loss_accel   = |Δ²P_t|                                [1D, legado]
            L_iso = loss_regime + α·loss_density + β·loss_accel

        α (density_weight) configurable vía env var PFV_DENSITY_WEIGHT.
        α=0 reproduce comportamiento v3.0 original.
        """
        # Normalizar realizations_frozen a 2D (K, D)
        if realizations_frozen.ndim == 3:
            Rf = realizations_frozen.detach().float().mean(dim=1)  # (K, D)
        else:
            Rf = realizations_frozen.detach().float()              # (K, D)

        pn = realization_new.float()                                # (D,)
        K  = Rf.size(0)

        Rf_norm = F.normalize(Rf, dim=-1)                          # (K, D)
        pn_norm = F.normalize(pn, dim=-1)                          # (D,)

        # ─── Término 1: loss_regime (formulación original) ───
        R_aug      = torch.cat([Rf, pn.unsqueeze(0)], dim=0)       # (K+1, D)
        R_aug_norm = F.normalize(R_aug, dim=-1)
        sim_aug    = R_aug_norm @ R_aug_norm.T                     # (K+1, K+1)
        d_aug      = 1.0 - sim_aug

        mask_upper_aug = torch.triu(
            torch.ones_like(d_aug, dtype=torch.bool), diagonal=1)
        d_pairs_aug = d_aug[mask_upper_aug]

        if self.agg == 'trimmed_mean':
            sorted_d, _ = d_pairs_aug.sort()
            n_trim = max(1, int(len(sorted_d) * 0.1))
            P_t_aug = sorted_d[n_trim:-n_trim].mean() \
                      if len(sorted_d) > 2 * n_trim else sorted_d.mean()
        else:
            P_t_aug = d_pairs_aug.mean()

        P_regime = (self.P_history.sum() /
                    torch.clamp(self.P_fill, min=1.0,
                                max=float(self.window))).detach()
        loss_regime = (P_t_aug - P_regime).abs()

        # ─── Término 2: loss_density — SEÑAL K-DIMENSIONAL [v3.1 NUEVO] ───
        # d_pn_to_R[i] = d(p_new, R_i) — vector (K,) con grad
        d_pn_to_R = 1.0 - (Rf_norm @ pn_norm)                      # (K,)

        # Target: d_slot_typical[i] = mean_{j≠i} d(R_i, R_j) — detached
        with torch.no_grad():
            sim_RR         = Rf_norm @ Rf_norm.T                   # (K, K)
            d_RR           = 1.0 - sim_RR
            mask_off_diag  = 1.0 - torch.eye(K, device=Rf.device)
            denom          = max(K - 1, 1)
            d_slot_typical = (d_RR * mask_off_diag).sum(dim=1) / denom

        loss_density = (d_pn_to_R - d_slot_typical).abs().mean()

        # ─── Término 3: loss_accel (formulación original) ───
        prev_ptr  = (self.P_ptr - 1) % self.window
        prev2_ptr = (self.P_ptr - 2) % self.window
        m1 = F.one_hot(prev_ptr,  num_classes=self.window).squeeze(0).float()
        m2 = F.one_hot(prev2_ptr, num_classes=self.window).squeeze(0).float()
        P_prev  = (self.P_history * m1).sum().detach()
        P_prev2 = (self.P_history * m2).sum().detach()

        fill_frac = torch.clamp(self.P_fill / 2.0, max=1.0).detach()
        delta2_P  = P_t_aug - 2.0 * P_prev + P_prev2
        loss_accel = fill_frac * delta2_P.abs()

        # ─── Total ───
        loss = (loss_regime
                + self.density_weight * loss_density
                + beta * loss_accel)
        # Garantizar scalar — P_fill puede venir como shape (1,)
        if loss.dim() > 0:
            loss = loss.squeeze()

        if return_diag:
            return loss, {
                'loss_regime':  float(loss_regime.detach().mean().item()),
                'loss_density': float(loss_density.detach().item()),
                'loss_accel':   float(loss_accel.detach().mean().item()),
                'P_t_aug':      float(P_t_aug.detach().item()),
                'P_regime':     float(P_regime.mean().item()),
                'd_pn_mean':    float(d_pn_to_R.detach().mean().item()),
                'd_pn_std':     float(d_pn_to_R.detach().std().item()),
            }
        return loss



# ─────────────────────────────────────────────────────────────
# [NEW] RESONANCE BUFFER  (Memoria emergente)  — v2.3
# ─────────────────────────────────────────────────────────────

class ResonanceBuffer(nn.Module):
    """
    Memoria emergente — registro del flujo previo del Soma.

    v2.2 — tres modos de representación, seleccionables por env:

      mode="mean"   (default, compat v2.1):
        Cada slot = (D,). EMA del mean(dim=(0,1)) del hidden state.
        Colapsa batch y secuencia → 1 vector por slot.
        Ventaja: barato, estable. Desventaja: pierde estructura.

      mode="seq":
        Cada slot = (L, D) donde L = mem_span.
        Guarda una "tira" de estados temporales → preserva factor X replicable.
        Mejor match con la nota original: "el factor X puede replicarse
        cuando existe una situación similar a la conocida".

      mode="topk":
        Cada slot = (D,). En vez de mean, se escoge el hidden state del
        token con mayor norma (más activo) dentro del batch. Reservoir
        sampling del flujo real, no promedio difuso.

    Sin params entrenables. Sin .item() — no rompe el pipeline async de CUDA.
    """
    def __init__(self, dim: int, n_slots: int = 64, ema_decay: float = 0.99,
                 mode: str = "mean", mem_span: int = 8):
        super().__init__()
        assert mode in ("mean", "seq", "topk")
        self.dim       = dim
        self.n_slots   = n_slots
        self.ema_decay = ema_decay
        self.mode      = mode
        self.mem_span  = mem_span if mode == "seq" else 1

        # Buffer dim varía según mode
        if mode == "seq":
            slots_shape = (n_slots, self.mem_span, dim)
        else:
            slots_shape = (n_slots, dim)

        self.register_buffer("slots",    torch.zeros(*slots_shape))
        self.register_buffer("slot_ptr", torch.zeros(1, dtype=torch.long))
        # [PERF-3] fill_count en float evita .item() de control flow.
        # warmup_weight ∈ [0,1] crece suavemente hasta 1.0 después de n_slots updates.
        self.register_buffer("fill_count", torch.zeros(1, dtype=torch.float32))

    @torch.no_grad()
    def update(self, h: Tensor) -> None:
        """
        h: (B, T, D) — hidden states del Soma post-processing.
        Update puramente tensorial, sin .item() / .bool() branching.
        """
        # Construir la "muestra" a guardar según el modo
        if self.mode == "mean":
            # Compat v2.1 — un vector por slot
            sample = h.detach().float().mean(dim=(0, 1))           # (D,)
            target = sample                                         # (D,) matches slots[ptr]

        elif self.mode == "topk":
            # Escoger el token más activo (mayor norma) del batch — un representante real
            h_flat = h.detach().float().reshape(-1, h.size(-1))    # (B*T, D)
            norms  = h_flat.norm(dim=-1)                            # (B*T,)
            idx    = norms.argmax()                                 # escalar
            target = h_flat[idx]                                    # (D,)

        else:  # "seq"
            # Guardar una "tira" de estados: downsample uniforme sobre T → L posiciones
            h_mean_batch = h.detach().float().mean(dim=0)           # (T, D)  promediar batch
            T = h_mean_batch.size(0)
            L = self.mem_span
            # Índices equiespaciados para muestrear L tokens de T
            idx = torch.linspace(0, T - 1, L, device=h.device).long()
            target = h_mean_batch.index_select(0, idx)              # (L, D)

        # [PERF-3] One-hot del slot actual — sin .item()
        ptr_mask_flat = torch.nn.functional.one_hot(
            self.slot_ptr, num_classes=self.n_slots
        ).squeeze(0).float()                                        # (K,)
        # Broadcast al shape correcto del buffer
        if self.mode == "seq":
            ptr_mask = ptr_mask_flat.view(self.n_slots, 1, 1)       # (K, 1, 1)
            target_bcast = target.unsqueeze(0)                      # (1, L, D)
        else:
            ptr_mask = ptr_mask_flat.view(self.n_slots, 1)          # (K, 1)
            target_bcast = target.unsqueeze(0)                      # (1, D)

        # Warmup suave: primer pase por el buffer usa decay=0, ya lleno usa decay=ema_decay.
        # Reemplaza la rama if self.filled.item() sin sincronización.
        warmup_weight = torch.clamp(self.fill_count / float(self.n_slots), max=1.0)
        effective_decay = self.ema_decay * warmup_weight             # escalar tensor, sin .item()

        old = self.slots.float()
        new_slot_val = effective_decay * old + (1.0 - effective_decay) * target_bcast
        # Aplica solo al slot actual; el resto queda intacto
        updated = ptr_mask * new_slot_val + (1.0 - ptr_mask) * old
        self.slots.copy_(updated.to(self.slots.dtype))

        # Avanza puntero y contador sin sync
        self.slot_ptr.add_(1).remainder_(self.n_slots)
        self.fill_count.add_(1.0)

    def get_slots(self) -> Tensor:
        """Todos los slots, incluidos los aun vacios; shape fija para el hot path."""
        if self.mode == "seq":
            return self.slots.reshape(-1, self.dim)
        return self.slots

    def valid_mask(self) -> Tensor:
        """Mascara tensorial (K_eff,) de slots realmente poblados."""
        n_valid = self.fill_count.clamp(min=0.0, max=float(self.n_slots)).long()
        slot_ids = torch.arange(self.n_slots, device=self.slots.device)
        valid = slot_ids < n_valid
        if self.mode == "seq":
            valid = valid.repeat_interleave(self.mem_span)
        return valid

    @torch.no_grad()
    def get_active_slots(self) -> Tensor:
        """Solo memoria poblada. Se usa fuera del hot path (PfV/TTT/logging)."""
        n_valid = min(int(self.fill_count.item()), self.n_slots)
        if n_valid <= 0:
            return self.get_slots()[:0]
        if self.mode == "seq":
            return self.slots[:n_valid].reshape(-1, self.dim)
        return self.slots[:n_valid]

    def resonance_score(self, h: Tensor) -> float:
        """
        Diagnóstico: máxima similitud coseno entre estado actual y slots.
        Usa .item() porque es solo para logging, no está en el hot path.
        """
        h_mean = h.detach().float().mean(dim=(0, 1))               # (D,)
        h_norm = F.normalize(h_mean.unsqueeze(0), dim=-1)          # (1, D)
        slots_flat = self.get_slots().float()                      # (K_eff, D)
        # Si está al inicio con todos ceros, la similitud es 0 — no rompe nada.
        s_norm = F.normalize(slots_flat, dim=-1)
        sims   = (h_norm @ s_norm.T).squeeze(0)                    # (K_eff,)
        return float(sims.max().item())


# ─────────────────────────────────────────────────────────────
# [NEW] PERCEPTION AGENT  (Percepción)
#
# Agente independiente — detecta similitud entre lo que pasa
# ahora (DendriticOut / Experiencia) y lo que ha fluido antes
# (ResonanceBuffer / Memoria).
#
# Genera PerceptualFlow — señal que SOLO existe en el ensamble.
# No es Dendritic. No es Resonance. Es la interacción entre ambos.
#
# Implementación:
#   Cross-attention donde:
#     Q = DendriticOut  (qué hay ahora — experiencia actual)
#     K,V = ResonanceBuffer.slots  (qué ha fluido antes)
#   Output: PerceptualFlow (B, T, D)
#
#   percept_gate: vector aprendido, sigmoid → [0,1]
#     Inicia cerca de cero — percepción silenciosa al arrancar.
#     Se activa progresivamente donde aporta.
#
# Alta resonancia → atención fuerte sobre slots → señal rica
# Baja resonancia → atención distribuida uniforme → señal débil
# ─────────────────────────────────────────────────────────────

class PerceptionAgent(nn.Module):
    """
    Percepción: cross-attention entre Experiencia actual y Memoria emergente.

    Q = DendriticOut   (B, T, D)  — experiencia presente
    K = ResonanceSlots (K, D)     — memoria fluida
    V = ResonanceSlots (K, D)     — contenido a recuperar

    Output = PerceptualFlow (B, T, D) — solo existe en el ensamble.
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads

        # Proyecciones
        self.c_q    = CastedLinear(dim, dim, bias=False)
        self.c_k    = CastedLinear(dim, dim, bias=False)
        self.c_v    = CastedLinear(dim, dim, bias=False)
        self.proj   = CastedLinear(dim, dim, bias=False)

        # Normas
        self.query_norm = RMSNorm()
        self.key_norm   = RMSNorm()

        # Gate: cuánto del flujo perceptual entra al Soma
        # Inicia en -1.5 → sigmoid(-1.5) ≈ 0.18
        # No silencioso, no dominante — percepción activa pero contenida.
        # Aprende a abrirse o cerrarse según lo que el Soma necesite.
        self.percept_gate = nn.Parameter(
            torch.full((dim,), -1.5, dtype=torch.float32))

        # Init proj near zero — percepción silenciosa al inicio
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, dendritic_out: Tensor, resonance_slots: Tensor,
                resonance_valid: Tensor | None = None) -> Tensor:
        """
        dendritic_out:    (B, T, D)
        resonance_slots:  (K_eff, D)
        resonance_valid:  (K_eff,) bool. Slots vacios quedan fuera del softmax.
        """
        B, T, D = dendritic_out.shape
        K = resonance_slots.size(0)

        q = self.c_q(self.query_norm(dendritic_out))
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))

        slots_norm = self.key_norm(resonance_slots.unsqueeze(0)).squeeze(0)
        k_proj = self.c_k(slots_norm)
        v_proj = self.c_v(slots_norm)
        k = k_proj.reshape(K, self.n_heads, self.head_dim).transpose(0, 1)
        v = v_proj.reshape(K, self.n_heads, self.head_dim).transpose(0, 1)
        k = F.rms_norm(k, (k.size(-1),))
        k = k.unsqueeze(0).expand(B, -1, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1, -1)

        attn_mask = None
        memory_gain = dendritic_out.new_tensor(1.0)
        if resonance_valid is not None:
            valid = resonance_valid.to(device=dendritic_out.device, dtype=torch.bool)
            has_memory = valid.any()
            # SDPA no tolera una fila totalmente -inf. Habilitamos el slot 0
            # como fallback y anulamos la salida con memory_gain cuando no hay memoria.
            fallback = torch.arange(K, device=valid.device).eq(0)
            safe_valid = valid | (fallback & ~has_memory)
            attn_mask = torch.zeros(K, device=dendritic_out.device,
                                    dtype=dendritic_out.dtype)
            attn_mask = attn_mask.masked_fill(~safe_valid, float("-inf"))
            attn_mask = attn_mask.view(1, 1, 1, K)
            memory_gain = has_memory.to(dendritic_out.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        out = self.proj(out)
        gate = torch.sigmoid(self.percept_gate.to(dendritic_out.dtype))
        return out * gate.view(1, 1, -1) * memory_gain


# ─────────────────────────────────────────────────────────────
# PHASE 2 — SOMA BLOCK  (El Medio — simplificado)
#
# v2.0: SomaNucleus eliminado.
# El Soma ya no es un banco de prototipos — es el medio puro.
# No decide, no almacena, no transforma. Sostiene.
#
# Pipeline por bloque:
#   0. resid_mix: mezcla aprendida x + x0 [ARCH-3]
#   1. Self-attention (tokens entre sí — contexto)
#   2. MLP (síntesis)
#
# El Soma recibe el flujo combinado (Dendritic + Perceptual)
# desde el forward del modelo — ya integrado antes de entrar.
# ─────────────────────────────────────────────────────────────

class SomaBlock(nn.Module):
    """
    Bloque del Soma — el medio.
    Sin SomaNucleus. Sin cross-attn a prototipos.
    Recibe experiencia + percepción ya integradas.
    Solo procesa el flujo — no lo almacena ni lo decide.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float = 1.5):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn      = GQAttention(dim, n_heads, n_kv_heads, rope_base, qk_gain_init)
        self.mlp       = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        # [ARCH-3] resid_mix: highway al embedding original
        self.resid_mix  = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """
        x:  (B, T, dim) — estado actual (experiencia + percepción integradas)
        x0: (B, T, dim) — embedding original (highway)
        """
        mix = self.resid_mix.to(dtype=x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x   = x + self.attn_scale.to(x.dtype) * self.attn(self.attn_norm(x))
        x   = x + self.mlp_scale.to(x.dtype)  * self.mlp(self.mlp_norm(x))
        return x


# ─────────────────────────────────────────────────────────────
# NEOCORTICAL LM — Modelo completo v2.0
# ─────────────────────────────────────────────────────────────

class Dopamine(nn.Module):
    """
    Dopamine v2.0 — Consciousness as Medium.

    Phase 1   — DendriticEncoder:    Experiencia (qué pasa ahora)
    Phase 1.5 — PerceptionAgent:     Percepción  (resonancia con el pasado)
    Phase 1.75— ResonanceBuffer:     Memoria emergente (flujo previo del Soma)
    Phase 2   — SomaCore:            El Medio (sostiene sin decidir)
    Phase 3   — CholinergicGate:       Transmisión selectiva

    Forward pipeline:
      x = embed(input) → x0 (highway)
      dendritic = DendriticEncoder(x)
      perceptual = PerceptionAgent(dendritic, resonance.slots)
      x = x + dendritic + perceptual   ← Soma recibe experiencia + percepción
      x = SomaCore(x, x0)             ← El medio procesa el flujo combinado
      resonance.update(x)              ← Memoria actualiza con lo que fluyó
      h = CholinergicGate(final_norm(x), trace)
      logits = MoleculeGate(h @ embed.T)
    """
    def __init__(self, args: Hyperparameters):
        super().__init__()
        d = args.model_dim

        self.logit_softcap = args.logit_softcap
        self.args          = args

        # Embedding (weight-tied)
        self.tok_emb    = nn.Embedding(args.vocab_size, d)
        self.input_norm = RMSNorm()

        # Phase 1: DendriticEncoder (Experiencia)
        self.dendrite = DendriticEncoder(
            dim         = d,
            field_heads = args.dendrite_field_heads,
            local_span  = args.dendrite_local_span,
            mid_span    = args.dendrite_mid_span,
            n_kv_heads  = args.num_kv_heads,
            rope_base   = args.rope_base,
            n_proto     = 16,
            geometric_enabled=args.geometric_dendrite_enabled,
            geo_center_decay=args.geo_center_decay,
            geo_axis_init_cos=args.geo_axis_init_cos,
            geo_gamma_init=args.geo_gamma_init,
            geo_metric_max=args.geo_metric_max,
            geo_transport_max=args.geo_transport_max,
            geo_tau_max=args.geo_tau_max,
            geo_quantize=args.geo_quantize,
            geo_quant_step_init=args.geo_quant_step_init,
            geo_axis_max_abs_cos=args.geo_axis_max_abs_cos,
            geo_cross_field_target=args.geo_cross_field_target,
            geo_eps=args.geo_eps,
        )
        self.dendrite_norm = RMSNorm()

        # [NEW] Phase 1.75: ResonanceBuffer (Memoria emergente — sin params)
        self.resonance = ResonanceBuffer(
            dim       = d,
            n_slots   = args.resonance_slots,
            ema_decay = args.resonance_ema_decay,
            mode      = args.resonance_mode,
            mem_span  = args.resonance_mem_span,
        )

        # [NEW] Phase 1.5: PerceptionAgent (Percepción)
        self.perception = PerceptionAgent(
            dim     = d,
            n_heads = args.perception_heads,
        )
        self.percept_norm = RMSNorm()  # norma del flujo antes de entrar al Soma

        # Phase 2: SomaCore (El Medio — sin nucleus)
        self.n_enc = args.num_layers // 2
        self.n_dec = args.num_layers - self.n_enc
        self.blocks = nn.ModuleList([
            SomaBlock(d, args.num_heads, args.num_kv_heads,
                      args.mlp_mult, args.rope_base, qk_gain_init=1.5)
            for _ in range(args.num_layers)
        ])
        n_skip = min(self.n_enc, self.n_dec)
        self.skip_weights = nn.Parameter(
            torch.ones(n_skip, d, dtype=torch.float32))
        self.final_norm = RMSNorm()

        # Phase 3: CholinergicGate + MoleculeGate
        self.ach_first_block = min(max(args.ach_first_block, 0),
                                   args.num_layers - 3)
        self.axon = CholinergicGate(
            dim         = d,
            lam         = args.ach_lambda,
            theta_init  = args.ach_theta,
            rho_k_init  = args.ach_rho_k,
            ach_enabled = args.ach_enabled,
            base_scale  = args.ach_base_scale,
        )
        self.molecule_gate = MoleculeGate(d, args.vocab_size)
        self.observer = RuntimeObserver(
            enabled=args.observer_enabled,
            sample_tokens=args.observer_sample_tokens,
        )

        # ── [v2.3] PfV Monitors — auto-observadores internos ──
        # Sin params entrenables. Sin gradiente. Overhead O(K²·D) trivial.
        # Se activan con env PFV_MONITOR=1 (default ON).
        # Cada monitor trackea un conjunto de realizaciones ℛ_t:
        self.pfv_enabled = bool(int(os.environ.get("PFV_MONITOR", "1")))
        if self.pfv_enabled:
            # K_eff del ResonanceBuffer depende del modo: seq → n_slots*mem_span
            K_resonance = (args.resonance_slots * args.resonance_mem_span
                           if args.resonance_mode == "seq"
                           else args.resonance_slots)
            self.monitor_resonance = IsomericPolarizationMonitor(
                n_realizations   = K_resonance,
                window           = int(os.environ.get("PFV_WINDOW", 50)),
                agg              = os.environ.get("PFV_AGG", "mean"),
                epsilon_quantile = float(os.environ.get("PFV_EPS_Q", 0.1)),
            )
            # Un monitor por campo dendrítico (local / mid / global)
            n_proto = 16
            self.monitor_plastic_local  = IsomericPolarizationMonitor(n_realizations=n_proto)
            self.monitor_plastic_mid    = IsomericPolarizationMonitor(n_realizations=n_proto)
            self.monitor_plastic_global = IsomericPolarizationMonitor(n_realizations=n_proto)

        # Weight init
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)

    def arm_observer(self, step: int, phase: str) -> None:
        self.observer.arm(step, phase)

    def observer_snapshot(self, clear: bool = True) -> dict:
        return self.observer.snapshot(clear=clear)

    @torch.no_grad()
    def pfv_snapshot(self) -> dict:
        """
        Retorna dict con stats PfV de todos los conjuntos de realizaciones.
        Se llama desde el training loop cada val_loss_every steps.
        No requiere gradiente — no afecta el modelo.
        """
        if not self.pfv_enabled:
            return {}
        out = {}
        # ResonanceBuffer: nunca medir ceros aun no poblados como realizaciones.
        slots = self.resonance.get_active_slots()
        if slots.size(0) >= 2:
            out["resonance"] = self.monitor_resonance.compute(slots)
        else:
            z = self.tok_emb.weight.new_zeros(())
            out["resonance"] = {
                "P_t": z, "P_regime": z, "delta_P": z, "delta2_P": z,
                "epsilon": z, "n_isomers": z, "dead_count": z,
                "iso_mask": torch.zeros(1, 1, device=z.device, dtype=torch.bool),
                "dead_mask": torch.zeros(1, device=z.device, dtype=torch.bool),
            }
        # PlasticDendriticField — prototipos por escala
        out["plastic_local"]  = self.monitor_plastic_local.compute(
            self.dendrite.plastic_local.prototypes)
        out["plastic_mid"]    = self.monitor_plastic_mid.compute(
            self.dendrite.plastic_mid.prototypes)
        out["plastic_global"] = self.monitor_plastic_global.compute(
            self.dendrite.plastic_global.prototypes)
        return out

    def _encode_decode(self, input_ids: Tensor,
                       return_trace: bool = False):
        """
        v2.0 forward pipeline:
          1. Embed + normalize → x0 (highway)
          2. DendriticEncoder → dendritic_out (Experiencia)
          3. PerceptionAgent(dendritic_out, resonance.slots) → perceptual_flow
          4. x = x + dendritic_out + perceptual_flow  (Soma recibe todo)
          5. SomaCore (U-Net) → h (el flujo procesado por el medio)
          6. resonance.update(h) — la memoria registra lo que fluyó
          7. Return final_norm(h), h
        """
        x  = self.tok_emb(input_ids)
        x  = self.input_norm(x)
        x0 = x                                         # highway [ARCH-3]
        self.observer.tensor("phase.embed", x)

        # Phase 1: Experiencia — qué hay ahora
        dendritic_out = self.dendrite(self.dendrite_norm(x))   # (B, T, D)
        self.observer.tensor("phase.dendritic", dendritic_out)
        geo = self.dendrite.geometry
        if geo.enabled and geo._last_gamma is not None:
            self.observer.tensor("geometry.gamma", geo._last_gamma)
            self.observer.tensor("geometry.tau", geo._last_tau)
            self.observer.tensor("geometry.disagreement", geo._last_disagreement)
            self.observer.tensor("geometry.projection_residual",
                                 geo._last_projection_residual)
            with torch.no_grad():
                axes = geo._normalized_axes(False)
                axis_cos = (axes[:, 0] * axes[:, 1]).sum(dim=-1)
            self.observer.tensor("geometry.axis_cos_lr", axis_cos)
            self.observer.tensor("geometry.gram_det", 1.0 - axis_cos.square())

        # Phase 1.5: Percepción — resonancia con lo que ha fluido
        resonance_slots = self.resonance.get_slots()           # (K, D)
        resonance_valid = self.resonance.valid_mask()
        perceptual_flow = self.perception(
            dendritic_out, resonance_slots, resonance_valid)       # (B, T, D)
        self.observer.tensor("phase.perceptual", perceptual_flow)
        self.observer.tensor("memory.valid_fraction", resonance_valid.float().mean())

        # Soma recibe: experiencia actual + reconocimiento de lo familiar
        # percept_norm estabiliza el flujo combinado antes del Soma
        x = x + dendritic_out + self.percept_norm(perceptual_flow)
        self.observer.tensor("phase.soma_input", x)

        # Phase 2: SomaCore — el medio procesa el flujo
        skips = []
        traces = [] if return_trace else None
        n_total = self.n_enc + self.n_dec
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
            if return_trace and i >= self.ach_first_block:
                traces.append(x)
        for i in range(self.n_dec):
            if skips:
                sw = self.skip_weights[i].to(dtype=x.dtype)
                x  = x + sw[None, None, :] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)
            if return_trace and (self.n_enc + i) >= self.ach_first_block:
                traces.append(x)
        if return_trace:
            expected = n_total - self.ach_first_block
            if len(traces) != expected:
                raise RuntimeError(
                    f"trace incompleto: esperados {expected}, "
                    f"obtenidos {len(traces)}")
            trace_tensor = torch.stack(traces, dim=2)
            self.observer.trace("soma.trace", trace_tensor)

        # [ARCH-v2.2] La memoria registra el flujo POST-norm — estado estabilizado
        # en escala, lo que hace el cross-attn del PerceptionAgent más fiel al
        # espacio de representación real del Soma. Antes: pre-norm (inconsistente).
        h = self.final_norm(x)
        self.observer.tensor("phase.soma_output", h)
        self.resonance.update(h)
        if return_trace:
            return h, x, trace_tensor
        return h, x

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                use_axon: bool = True) -> Tensor:
        if use_axon and self.axon.ach_enabled:
            h, _, trace = self._encode_decode(input_ids, return_trace=True)
        else:
            h, _ = self._encode_decode(input_ids)
            trace = None

        self.observer.tensor("phase.pre_gate", h)
        if use_axon:
            h = self.axon(h, trace)
        self.observer.tensor("phase.post_gate", h)

        h_flat = h.reshape(-1, h.size(-1))
        logits = F.linear(h_flat, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if use_axon:
            logits = self.molecule_gate(logits, h_flat,
                                        self.axon.last_u_mean,
                                        self.args.axon_stress_threshold)

        self.observer.logits("output.logits", logits)
        targets = target_ids.reshape(-1)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        geo_aux = self.dendrite.geometry_aux_loss()
        if self.training and self.dendrite.geometry.enabled and self.args.geo_aux_scale > 0.0:
            loss = ce_loss + self.args.geo_aux_scale * geo_aux
        else:
            loss = ce_loss
        self.observer.tensor("output.ce_loss", ce_loss)
        self.observer.tensor("geometry.aux_loss", geo_aux)
        self.observer.tensor("output.loss", loss)
        self.observer.finish()
        return loss

    def get_logits(self, input_ids: Tensor) -> Tensor:
        if self.axon.ach_enabled:
            h, _, trace = self._encode_decode(input_ids, return_trace=True)
        else:
            h, _ = self._encode_decode(input_ids)
            trace = None
        self.observer.tensor("phase.pre_gate", h)
        h = self.axon(h, trace)
        self.observer.tensor("phase.post_gate", h)
        h_flat = h.reshape(-1, h.size(-1))
        logits = F.linear(h_flat, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        logits = self.molecule_gate(logits, h_flat,
                                    self.axon.last_u_mean,
                                    self.args.axon_stress_threshold)
        self.observer.logits("output.logits", logits)
        self.observer.finish()
        return logits

    # ═════════════════════════════════════════════════════════════
    # [v3.0 TTT-Iso] HOOK PARA EXTRAER perceptual_flow CON GRAD
    # ═════════════════════════════════════════════════════════════
    def forward_with_percept(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Variante de forward que expone el perceptual_flow con grad.
        Usada por TTTIsoTrainer.

        Returns:
            dendritic_out:   experiencia actual (B, T, D)
            perceptual_flow: (B, T, D) con grad — entrada del TTT-Iso
        """
        x  = self.tok_emb(input_ids)
        x  = self.input_norm(x)
        x0 = x

        dendritic_out = self.dendrite(self.dendrite_norm(x))
        resonance_slots = self.resonance.get_slots()
        resonance_valid = self.resonance.valid_mask()
        perceptual_flow = self.perception(
            dendritic_out, resonance_slots, resonance_valid)

        return dendritic_out, perceptual_flow


# ═════════════════════════════════════════════════════════════════
# [v3.0 NEW] ISOMERIC TEST-TIME TRAINING — TTTIsoTrainer
#
# Adapta solo el PerceptionAgent usando L_iso como loss SSL.
# Se ejecuta DESPUÉS de medir bpb sobre un chunk de val
# (score-first TTT, legal según reglas de OpenAI Parameter Golf).
#
# Flujo por chunk:
#   1. Forward del chunk con grad solo en PerceptionAgent.
#   2. Extraer perceptual_flow (B, T, D) → p_new = mean(0,1) → (D,)
#   3. Tomar resonance.slots detached como R (huella previa).
#   4. L_iso = monitor.compute_ttt_loss(R, p_new, beta)
#   5. backward + step (K veces)
#
# Gradiente fluye solo por: c_q, c_k, c_v, proj, percept_gate
# del PerceptionAgent. Todo lo demás queda frozen.
#
# Invariantes protegidos:
#   - No se toca el target `y` — loss ciega al label.
#   - ResonanceBuffer.update() NO se llama durante TTT —
#     si lo hiciéramos, cada inner step movería el target del
#     siguiente y el loop colapsaría.
#   - Muon/Lion/AdamW del training principal quedan intactos:
#     aquí usamos un optimizer temporal (Lion dedicado).
# ═════════════════════════════════════════════════════════════════

class TTTIsoTrainer:
    """TTT-Iso aislado por sesion de evaluacion."""
    def __init__(self, base_model: "Dopamine", lr: float = 5e-4,
                 inner_steps: int = 3, beta: float = 0.1,
                 reset_each_chunk: bool = False, grad_clip: float = 1.0):
        self.base_model = base_model
        self.inner_steps = inner_steps
        self.beta = beta
        self.reset_each_chunk = reset_each_chunk
        self.grad_clip = float(grad_clip)
        self.percept_params = list(base_model.perception.parameters())
        self.opt = Lion(self.percept_params, lr=lr,
                        betas=(0.9, 0.99), weight_decay=0.0)
        self.n_chunks_adapted = 0
        self.n_chunks_session = 0
        self.last_loss = float("nan")
        self._session_model_state = None
        self._session_opt_state = None

    def begin_session(self) -> None:
        # Snapshot del estado ENTRENADO actual, no del init del modelo.
        self._session_model_state = {
            k: v.detach().clone()
            for k, v in self.base_model.perception.state_dict().items()
        }
        self._session_opt_state = copy.deepcopy(self.opt.state_dict())
        self.n_chunks_session = 0

    def _reset_to_session_base(self) -> None:
        if self._session_model_state is None:
            raise RuntimeError("TTT session no inicializada")
        self.base_model.perception.load_state_dict(self._session_model_state)
        self.opt.load_state_dict(copy.deepcopy(self._session_opt_state))

    def end_session(self, restore: bool = True) -> None:
        if restore and self._session_model_state is not None:
            self.base_model.perception.load_state_dict(self._session_model_state)
            self.opt.load_state_dict(copy.deepcopy(self._session_opt_state))
        self._session_model_state = None
        self._session_opt_state = None

    def _freeze_all_except_perception(self) -> dict[str, bool]:
        prev = {}
        for name, p in self.base_model.named_parameters():
            prev[name] = p.requires_grad
            p.requires_grad_(name.startswith("perception."))
        return prev

    def _restore_grad_state(self, prev: dict[str, bool]) -> None:
        for name, p in self.base_model.named_parameters():
            if name in prev:
                p.requires_grad_(prev[name])

    def adapt(self, input_ids: Tensor) -> dict:
        if self._session_model_state is None:
            self.begin_session()
        if self.reset_each_chunk:
            self._reset_to_session_base()

        prev_grad_state = self._freeze_all_except_perception()
        was_training = self.base_model.perception.training
        self.base_model.perception.train()
        diag = {
            "losses": [], "grad_norms": [], "loss_regime": [],
            "loss_density": [], "loss_accel": [], "nonfinite": False,
        }

        try:
            for _ in range(self.inner_steps):
                self.opt.zero_grad(set_to_none=True)
                dev_type = input_ids.device.type
                with torch.autocast(device_type=dev_type,
                                    dtype=torch.bfloat16,
                                    enabled=(dev_type == "cuda")):
                    _, perceptual_flow = self.base_model.forward_with_percept(input_ids)

                p_new = perceptual_flow.float().mean(dim=(0, 1))
                R_frozen = self.base_model.resonance.get_active_slots().detach().float()
                if R_frozen.size(0) < 2:
                    # No existe geometria relacional suficiente para TTT.
                    break
                loss, parts = self.base_model.monitor_resonance.compute_ttt_loss(
                    realizations_frozen=R_frozen,
                    realization_new=p_new,
                    beta=self.beta,
                    return_diag=True,
                )
                if not torch.isfinite(loss):
                    diag["nonfinite"] = True
                    break

                loss.backward()
                if self.grad_clip > 0:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        self.percept_params, self.grad_clip)
                else:
                    grads = [p.grad.float().norm().square()
                             for p in self.percept_params if p.grad is not None]
                    grad_norm_t = torch.stack(grads).sum().sqrt() if grads else loss.new_zeros(())

                if not torch.isfinite(grad_norm_t):
                    diag["nonfinite"] = True
                    break

                diag["losses"].append(float(loss.detach().item()))
                diag["grad_norms"].append(float(grad_norm_t.detach().item()))
                for key in ("loss_regime", "loss_density", "loss_accel"):
                    diag[key].append(float(parts[key]))
                self.opt.step()

            if diag["nonfinite"]:
                # No dejar una adaptacion parcial corrupta.
                self._reset_to_session_base()
            elif diag["losses"]:
                self.last_loss = diag["losses"][-1]
                self.n_chunks_adapted += 1
                self.n_chunks_session += 1
        finally:
            self.base_model.perception.train(was_training)
            self._restore_grad_state(prev_grad_state)

        return diag


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens   = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np    = np.fromfile(file, dtype="<u2",
                               count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files    = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens   = load_data_shard(self.files[self.file_idx])
        self.pos      = 0

    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos+k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank       = rank
        self.world_size = world_size
        self.device     = device
        self.stream     = TokenStream(pattern)

    def next_batch(self, global_tokens: int,
                   seq_len: int, grad_accum: int) -> tuple[Tensor, Tensor]:
        local_tokens  = global_tokens // (self.world_size * grad_accum)
        per_rank_span = local_tokens + 1
        chunk         = self.stream.take(per_rank_span * self.world_size)
        start         = self.rank * per_rank_span
        local         = chunk[start:start+per_rank_span].to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True))


# ─────────────────────────────────────────────────────────────
# VALIDATION + BPB
# ─────────────────────────────────────────────────────────────

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable+1]


def build_sentencepiece_luts(sp, vocab_size: int, device):
    import sentencepiece as spm_module
    sp_vocab = int(sp.vocab_size())
    table_size           = max(sp_vocab, vocab_size)
    base_bytes_np        = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_np       = np.ones((table_size,),  dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
            torch.tensor(is_boundary_np,       dtype=torch.bool,  device=device))


def eval_val(args, model, rank, world_size, device, grad_accum,
             val_tokens, base_bytes_lut, leading_space_lut, boundary_lut,
             ttt_trainer=None, log_fn=None, base_model=None):
    """FineWeb BPB con TTT score-first y restauracion transaccional de estado."""
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum)
    local_batch_seqs = max(local_batch_tokens // args.train_seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)

    base = base_model if base_model is not None else (
        ttt_trainer.base_model if ttt_trainer is not None else
        (model.module if hasattr(model, "module") else model))
    ttt_active = (ttt_trainer is not None) and bool(args.ttt_enabled)
    warmup_chunks_left = int(args.ttt_warmup_chunks) if ttt_active else 0
    ttt_diag_all = {
        "losses": [], "grad_norms": [], "loss_regime": [],
        "loss_density": [], "loss_accel": [], "nonfinite": 0,
    }

    # Todo estado mutable que el forward de validacion toca se restaura al final.
    resonance_state = {k: v.detach().clone()
                       for k, v in base.resonance.state_dict().items()}
    monitor_state = None
    if getattr(base, "pfv_enabled", False):
        monitor_state = {k: v.detach().clone()
                         for k, v in base.monitor_resonance.state_dict().items()}
    was_training = model.training
    model.eval()
    if ttt_active:
        ttt_trainer.begin_session()

    try:
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            raw = val_tokens[bs * args.train_seq_len:(be * args.train_seq_len) + 1]
            raw = raw.to(device=device, dtype=torch.int64, non_blocking=True)
            x = raw[:-1].reshape(-1, args.train_seq_len)
            y = raw[1:].reshape(-1, args.train_seq_len)

            with torch.no_grad():
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    batch_loss = model(x, y, use_axon=True).detach()

            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            token_cnt += n
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (leading_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(torch.int16)
            byte_cnt += tb.to(torch.float64).sum()

            if ttt_active:
                active_slots = base.resonance.get_active_slots()
                enough_geometry = active_slots.size(0) >= 2
                if enough_geometry:
                    with torch.no_grad():
                        _ = base.monitor_resonance.compute(active_slots)
                if warmup_chunks_left <= 0 and enough_geometry:
                    diag = ttt_trainer.adapt(x)
                    for key in ("losses", "grad_norms", "loss_regime",
                                "loss_density", "loss_accel"):
                        ttt_diag_all[key].extend(diag[key])
                    ttt_diag_all["nonfinite"] += int(diag["nonfinite"])
                else:
                    warmup_chunks_left -= 1

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(byte_cnt, op=dist.ReduceOp.SUM)

        val_loss = float(loss_sum / token_cnt)
        bpt = val_loss / math.log(2.0)
        tpb = float(token_cnt / byte_cnt)

        if ttt_active and ttt_diag_all["losses"] and log_fn is not None:
            def avg(key):
                xs = ttt_diag_all[key]
                return sum(xs) / max(len(xs), 1)
            log_fn(
                f"  [TTT-Iso] chunks={ttt_trainer.n_chunks_session}"
                f"  inner_updates={ttt_trainer.n_chunks_session * ttt_trainer.inner_steps}"
                f"  L={avg('losses'):.4f}"
                f" [reg={avg('loss_regime'):.4f} den={avg('loss_density'):.4f}"
                f" acc={avg('loss_accel'):.4f}]"
                f"  |grad|={avg('grad_norms'):.4f}"
                f"  nonfinite={ttt_diag_all['nonfinite']}"
            )
        return val_loss, float(bpt * tpb)
    finally:
        restore = bool(args.ttt_restore_after_eval)
        if ttt_active:
            ttt_trainer.end_session(restore=restore)
        if restore:
            base.resonance.load_state_dict(resonance_state)
            if monitor_state is not None:
                base.monitor_resonance.load_state_dict(monitor_state)
        model.train(was_training)


# ─────────────────────────────────────────────────────────────
# PERSISTENT CHECKPOINTS + COMPRESSED MODEL LOADER
# ─────────────────────────────────────────────────────────────

def _serializable_hparams(args) -> dict:
    """Extrae hiperparametros simples de la clase/configuracion."""
    out = {}
    simple = (str, int, float, bool, type(None))
    for name in dir(args):
        if name.startswith("_"):
            continue
        value = getattr(args, name)
        if isinstance(value, simple):
            out[name] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, simple) for v in value):
            out[name] = list(value)
    return out


def save_raw_checkpoint(model: nn.Module, args, step: int,
                        artifact_dir: str | Path) -> tuple[Path, float]:
    """Guarda un state_dict exacto en CPU, util para reanudar o evaluar."""
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ach_{args.run_id}_step{step:06d}_raw.pt"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "format": "ach-v0.3-geometric-raw-state-dict",
        "run_id": args.run_id,
        "step": int(step),
        "hparams": _serializable_hparams(args),
        "model_state_dict": state,
    }
    torch.save(payload, path)
    return path, path.stat().st_size / 1000.0


def save_compressed_archive(model_bytes: bytes, args, step: int,
                            artifact_dir: str | Path) -> tuple[Path, float]:
    """Persiste el byte-stream zlib producido por quantize_and_compress."""
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ach_{args.run_id}_step{step:06d}_compressed.pt.zlib"
    path.write_bytes(model_bytes)
    return path, path.stat().st_size / 1000.0


def load_compressed_state_dict(path: str | Path) -> dict[str, Tensor]:
    """Reconstruye un state_dict desde el archivo .pt.zlib de ACH."""
    raw = zlib.decompress(Path(path).read_bytes())
    buffer = io.BytesIO(raw)
    try:
        obj = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        obj = torch.load(buffer, map_location="cpu")

    state: dict[str, Tensor] = {}
    state.update(obj.get("passthrough", {}))

    scales = obj.get("scales", {})
    for name, q in obj.get("quantized", {}).items():
        scale = scales[name].float()
        if q.ndim == 2:
            state[name] = q.float() * scale[:, None]
        else:
            state[name] = q.float() * scale

    for name, packed in obj.get("svd", {}).items():
        U = packed["U_q"].float() * packed["U_s"].float()[:, None]
        Vt = packed["Vt_q"].float() * packed["Vt_s"].float()[:, None]
        S = packed["S_k"].float()
        state[name] = (U * S[None, :]) @ Vt

    return state


def load_compressed_into(model: nn.Module, path: str | Path,
                         strict: bool = True):
    """Carga el archivo comprimido dentro de una instancia Dopamine compatible."""
    state = load_compressed_state_dict(path)
    return model.load_state_dict(state, strict=strict)


def load_v021_checkpoint_into_geometric(model: nn.Module, path: str | Path):
    """Carga un checkpoint raw ACH v0.2.1 verificando la migracion.

    Se permiten como missing solamente parametros/buffers bajo
    ``dendrite.geometry``. Cualquier otra diferencia aborta.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    incompat = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in incompat.missing_keys
                   if not k.startswith("dendrite.geometry.")]
    if bad_missing or incompat.unexpected_keys:
        raise RuntimeError(
            f"legacy migration failed: missing={bad_missing}, "
            f"unexpected={incompat.unexpected_keys}")
    return incompat


# ─────────────────────────────────────────────────────────────
# QUANTIZATION + RESONANCE DISTILLATION
# ─────────────────────────────────────────────────────────────

def distill_resonance(model: nn.Module) -> str:
    """
    Post-training: snapshot del ResonanceBuffer como constantes Python.

    [v3.1 PATCH] Soporta los tres modos del buffer:
      mode="mean" | "topk":  slots shape (K, D)       → SVD directo
      mode="seq":            slots shape (K, L, D)    → flatten → SVD → reshape

    La destilación SVD opera sobre una matriz 2D. Si el buffer es seq,
    colapsamos (K, L, D) → (K, L*D), destilamos, y guardamos metadatos
    para reconstruir (K, L, D) en inferencia.
    """
    slots = model.resonance.slots.detach().float().cpu().numpy()

    # Detectar modo por dimensionalidad del tensor
    if slots.ndim == 2:
        K, D = slots.shape
        L = 1
        mat = slots
        seq_mode = False
    elif slots.ndim == 3:
        K, L, D = slots.shape
        mat = slots.reshape(K, L * D)
        seq_mode = True
    else:
        raise ValueError(
            f"ResonanceBuffer.slots ndim={slots.ndim} unsupported "
            f"(expected 2 for mean/topk, or 3 for seq)."
        )

    # SVD para compresión eficiente
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    top_k = min(16, K)
    U_k  = U[:, :top_k]
    S_k  = S[:top_k]
    Vt_k = Vt[:top_k, :]
    var_explained = float((S_k**2).sum() / max((S**2).sum(), 1e-12))

    filled = bool(model.resonance.fill_count.item() > 0)
    ptr    = int(model.resonance.slot_ptr.item())

    lines = [
        "# ResonanceBuffer destilado — snapshot del flujo final (v3.1)",
        f"# Mode: {'seq' if seq_mode else 'mean/topk'}  |  Slots: {K}"
        + (f"×{L}×{D}" if seq_mode else f"×{D}"),
        f"# EMA decay: {model.resonance.ema_decay}",
        f"# Buffer filled: {filled}  |  Last ptr: {ptr}",
        f"# Varianza explicada con top_{top_k}: {var_explained:.4f}",
        "import numpy as np",
        "",
        f"RESONANCE_N_SLOTS  = {K}",
        f"RESONANCE_SPAN     = {L}",
        f"RESONANCE_DIM      = {D}",
        f"RESONANCE_TOP_K    = {top_k}",
        f"RESONANCE_SEQ_MODE = {seq_mode}",
        "",
        "# Bases principales del flujo (top_k × (L*D))",
        f"RESONANCE_BASES  = np.array({Vt_k.tolist()}, dtype=np.float32)",
        f"RESONANCE_SCALES = np.array({S_k.tolist()}, dtype=np.float32)",
        f"RESONANCE_COEFS  = np.array({U_k.tolist()}, dtype=np.float32)",
        "",
        "def reconstruct_slots():",
        '    """Reconstruye slots desde constantes destiladas."""',
        "    flat = (RESONANCE_COEFS * RESONANCE_SCALES[None, :]) @ RESONANCE_BASES",
        "    if RESONANCE_SEQ_MODE:",
        "        return flat.reshape(RESONANCE_N_SLOTS, RESONANCE_SPAN, RESONANCE_DIM)",
        "    return flat",
    ]
    return "\n".join(lines)



# ─────────────────────────────────────────────────────────────
# COMPRESSION STRATEGY v2.0
#
# Objetivo: ~14MB compressed (margen para el challenge)
#
# Pipeline por tensor:
#   Pequeños (<64K elems)   → fp16 passthrough
#   Medianos (64K-256K)     → int8 per-row (standard)
#   Grandes  (>256K elems)  → SVD low-rank + int8
#     SVD retiene componentes hasta 90% varianza explicada
#     o hasta rank_cap (whichever menor)
#     Guarda U_k (m×k) + S_k (k,) + Vt_k (k×n) en int8
#     Overhead: k*(m+n+1) vs m*n original
#     Break-even: k < m*n/(m+n+1) — siempre vale para rank bajo
#
# Layers grandes en Dopamine v2:
#   blocks[i].attn.c_q/k/v/proj: 512×512 = 262K → SVD
#   blocks[i].mlp.fc: 512×1536 = 786K → SVD agresivo
#   blocks[i].mlp.proj: 1536×512 = 786K → SVD agresivo
#   dendrite.c_k/v: 512×128 = 65K → int8 standard
#   perception.c_q/k/v/proj: 512×512 → SVD
# ─────────────────────────────────────────────────────────────

def _svd_compress_tensor(t32: torch.Tensor,
                          variance_target: float = 0.90,
                          rank_cap: int = 96) -> dict:
    """
    SVD low-rank compression de un tensor 2D.
    Retiene componentes hasta `variance_target` de varianza explicada.
    Limita rank a rank_cap para controlar overhead.

    Returns dict con U_k, S_k, Vt_k en int8 + sus scales fp16.
    """
    m, n = t32.shape
    # SVD completo
    try:
        U, S, Vt = torch.linalg.svd(t32, full_matrices=False)  # economy SVD
    except Exception:
        return None  # fallback a int8 normal si falla

    # Varianza acumulada
    var_total = (S ** 2).sum()
    var_cum   = (S ** 2).cumsum(0)
    # Mínimo rank para alcanzar variance_target
    rank_needed = int((var_cum / var_total >= variance_target).nonzero()[0].item()) + 1
    rank = min(rank_needed, rank_cap, min(m, n))

    U_k  = U[:, :rank]   # (m, rank)
    S_k  = S[:rank]      # (rank,)
    Vt_k = Vt[:rank, :]  # (rank, n)

    var_explained = float((S_k**2).sum() / var_total)

    # Comprimir los factores en int8
    def _to_int8(x):
        clip  = float(torch.quantile(x.abs().flatten(), 0.9999).item())
        scale = torch.tensor(clip / 127.0 if clip > 0 else 1.0, dtype=torch.float32)
        if x.ndim == 2:
            clip_r  = torch.quantile(x.abs(), 0.9999, dim=1)
            clipped = torch.clamp(x, -clip_r[:,None], clip_r[:,None])
            scale_r = (clip_r / 127.0).clamp_min(1.0/127.0)
            q = torch.clamp(torch.round(clipped / scale_r[:,None]), -127, 127).to(torch.int8)
            return q, scale_r.to(torch.float16)
        else:
            q = torch.clamp(torch.round(x.clamp(-clip,clip)/scale), -127,127).to(torch.int8)
            return q, scale.to(torch.float16)

    U_q,  U_s  = _to_int8(U_k)
    Vt_q, Vt_s = _to_int8(Vt_k)
    # S se guarda en fp16 directamente — es un vector pequeño (rank,)
    return {
        "type":          "svd",
        "shape":         (m, n),
        "rank":          rank,
        "var_explained": var_explained,
        "U_q":           U_q,   "U_s":   U_s,
        "S_k":           S_k.to(torch.float16),
        "Vt_q":          Vt_q,  "Vt_s":  Vt_s,
    }


def quantize_and_compress(model: nn.Module, code_path: str,
                           svd_threshold: int = 200_000,
                           variance_target: float = 0.90,
                           rank_cap: int = 96) -> dict:
    """
    Compresión en tres niveles:
      - Pequeños (<64K)     → fp16 passthrough
      - Medianos (64K-200K) → int8 per-row
      - Grandes  (>200K)    → SVD low-rank + int8

    svd_threshold: numel mínimo para activar SVD
    variance_target: fracción de varianza a preservar con SVD
    rank_cap: rank máximo permitido en SVD
    """
    sd       = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    quant    = {}
    scales   = {}
    svd_data = {}
    passthrough = {}
    stats    = {"svd": 0, "int8": 0, "fp16": 0, "total_orig_kb": 0.0}

    for name, t in sd.items():
        orig_kb = t.numel() * t.element_size() / 1000
        stats["total_orig_kb"] += orig_kb

        if not t.is_floating_point() or t.numel() <= 65_536:
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
            stats["fp16"] += 1
            continue

        t32 = t.float()

        # Intento SVD para tensores grandes 2D
        if t32.ndim == 2 and t32.numel() > svd_threshold:
            result = _svd_compress_tensor(t32, variance_target, rank_cap)
            if result is not None:
                svd_data[name] = result
                stats["svd"] += 1
                continue

        # Int8 per-row para el resto
        if t32.ndim == 2:
            clip    = torch.quantile(t32.abs(), 0.9999, dim=1)
            clipped = torch.clamp(t32, -clip[:,None], clip[:,None])
            scale   = (clip / 127.0).clamp_min(1.0/127.0)
            q       = torch.clamp(
                torch.round(clipped / scale[:,None]), -127, 127).to(torch.int8)
            quant[name]  = q
            scales[name] = scale.to(torch.float16)
        else:
            clip  = float(torch.quantile(t32.abs().flatten(), 0.9999).item())
            scale = torch.tensor(clip/127.0 if clip > 0 else 1.0)
            q     = torch.clamp(
                torch.round(t32.clamp(-clip,clip)/scale), -127, 127).to(torch.int8)
            quant[name]  = q
            scales[name] = scale
        stats["int8"] += 1

    obj = {
        "quantized":   quant,
        "scales":      scales,
        "svd":         svd_data,
        "passthrough": passthrough,
        "compression": {
            "svd_threshold":    svd_threshold,
            "variance_target":  variance_target,
            "rank_cap":         rank_cap,
            "n_svd":            stats["svd"],
            "n_int8":           stats["int8"],
            "n_fp16":           stats["fp16"],
        }
    }

    buf             = io.BytesIO()
    torch.save(obj, buf)
    model_bytes     = zlib.compress(buf.getvalue(), level=9)
    code_bytes      = Path(code_path).read_bytes()
    code_compressed = zlib.compress(code_bytes, level=9)
    total_bytes     = len(model_bytes) + len(code_compressed)

    return {
        "model_compressed_kb": len(model_bytes) / 1000,
        "code_compressed_kb":  len(code_compressed) / 1000,
        "total_kb":            total_bytes / 1000,
        "fits_16mb":           total_bytes < 16_000_000,
        "fits_14mb":           total_bytes < 14_000_000,
        "n_svd_tensors":       stats["svd"],
        "n_int8_tensors":      stats["int8"],
        "model_bytes":         model_bytes,
    }


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def build_optimizer_groups(model: nn.Module) -> tuple[list[Tensor], list[Tensor], list[Tensor], dict]:
    """Particion exhaustiva y disjunta de parametros entrenables."""
    emb, matrices, scalars = [], [], []
    names = {"embedding": [], "matrix": [], "scalar": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "tok_emb.weight":
            emb.append(p); names["embedding"].append(name)
        elif p.ndim == 2 and name != "skip_weights" and not name.endswith("resid_mix"):
            matrices.append(p); names["matrix"].append(name)
        else:
            scalars.append(p); names["scalar"].append(name)

    memberships: dict[int, list[str]] = {}
    for group_name, params in (("embedding", emb), ("matrix", matrices), ("scalar", scalars)):
        for p in params:
            memberships.setdefault(id(p), []).append(group_name)
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in memberships]
    duplicate = {n: memberships[id(p)] for n, p in model.named_parameters()
                 if p.requires_grad and len(memberships.get(id(p), [])) != 1}
    if missing or duplicate:
        raise RuntimeError(f"optimizer coverage invalida: missing={missing}, duplicate={duplicate}")
    return emb, matrices, scalars, names


@torch.no_grad()
def collect_parameter_stats(model: nn.Module) -> dict:
    groups = {
        "dendrite": model.dendrite,
        "geometry": model.dendrite.geometry,
        "perception": model.perception,
        "soma": model.blocks,
        "ach": model.axon,
        "molecule": model.molecule_gate,
        "embedding": model.tok_emb,
    }
    out = {}
    for name, module in groups.items():
        sum_sq = None
        absmax = None
        count = 0
        for p in module.parameters():
            x = p.detach().float()
            v = x.square().sum()
            m = x.abs().max()
            sum_sq = v if sum_sq is None else sum_sq + v
            absmax = m if absmax is None else torch.maximum(absmax, m)
            count += x.numel()
        if count:
            out[name] = {
                "param_rms": float((sum_sq / count).sqrt().item()),
                "param_absmax": float(absmax.item()),
            }
    return out


def collect_gradient_stats(model: nn.Module) -> dict:
    prefixes = {
        "geometry": "dendrite.geometry.",
        "dendrite": "dendrite.", "perception": "perception.",
        "soma": "blocks.", "ach": "axon.",
        "molecule": "molecule_gate.", "embedding": "tok_emb.",
    }
    accum = {k: [] for k in prefixes}
    per_param = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        gnorm = g.norm()
        per_param[name] = float(gnorm.item())
        for group, prefix in prefixes.items():
            if name.startswith(prefix):
                accum[group].append(gnorm.square())
                break
    grouped = {}
    for group, vals in accum.items():
        grouped[group] = float(torch.stack(vals).sum().sqrt().item()) if vals else 0.0
    return {"groups": grouped, "parameters": per_param}


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum  = max(1, 8 // world_size)
    grad_scale  = 1.0 / grad_accum

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (enable_flash_sdp, enable_math_sdp,
                                      enable_mem_efficient_sdp, enable_cudnn_sdp)
    enable_flash_sdp(False)
    enable_cudnn_sdp(False)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(True)

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master else None
    telemetry_file = (f"logs/{args.run_id}_telemetry.jsonl"
                      if master and args.observer_jsonl else None)

    def log0(msg: str, console: bool = True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    def emit_telemetry(payload: dict) -> None:
        if not telemetry_file:
            return
        with open(telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    log0(code, console=False)
    log0("="*80, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    import sentencepiece as spm
    sp         = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb_lut, ls_lut, bd_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # Build model
    base_model = Dopamine(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    for name, p in base_model.named_parameters():
        if p.ndim < 2 and p.dtype != torch.float32:
            p.data = p.data.float()

    # [PERF-1] torch.compile — en RTX 3060 el default es OFF.
    # El observer profundo usa side effects Python y no se combina con compile; cuesta
    # proporcionalmente más en GPU pequeña. Además mode="reduce-overhead"
    # con CUDA graphs es incompatible con el Rotary cache → usamos mode="default".
    # Se puede forzar con USE_COMPILE=1 si quieres experimentar.
    use_compile = os.environ.get("USE_COMPILE", "0") == "1"
    if use_compile and base_model.observer.enabled:
        base_model.observer.enabled = False
        log0("observer profundo desactivado: USE_COMPILE=1; ACh/PfV siguen visibles")
    if use_compile:
        compiled_model = torch.compile(
            base_model,
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
        log0(f"torch.compile: enabled (mode=default) — 3060 experimental")
    else:
        compiled_model = base_model  # eager mode
        log0(f"torch.compile: disabled (eager mode) — 3060 default")

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=True: seguro cuando algunos params (log_beta de
        # PlasticDendriticField) pueden saturar el softmax y no recibir grad.
        model = DDP(compiled_model, device_ids=[local_rank],
                    broadcast_buffers=False, find_unused_parameters=True)
    else:
        model = compiled_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"ACH v0.3-geometric — Dopamine TTT-Iso")
    log0(f"params: {n_params:,}  |  layers: {args.num_layers}  |  dim: {args.model_dim}")
    log0(f"heads: {args.num_heads}Q / {args.num_kv_heads}KV  |  mlp_mult: {args.mlp_mult}")
    log0(f"vocab: {args.vocab_size}  |  weight_tying: True  |  RoPE: True")
    log0(f"DendriticEncoder: field_heads={args.dendrite_field_heads}  "
         f"local={args.dendrite_local_span}  mid={args.dendrite_mid_span}  n_proto=16")
    log0(f"GeometricDendrite: enabled={args.geometric_dendrite_enabled}  "
         f"metric_max={args.geo_metric_max}  transport_max={args.geo_transport_max}  "
         f"tau_max={args.geo_tau_max}  quantize={args.geo_quantize}  "
         f"geo_lr={args.geo_lr}")
    log0(f"ResonanceBuffer: {args.resonance_slots} slots × {args.model_dim}d  "
         f"EMA_decay={args.resonance_ema_decay}  mode={args.resonance_mode}"
         f"{'  span='+str(args.resonance_mem_span) if args.resonance_mode=='seq' else ''}  "
         f"[no trainable params]")
    log0(f"PerceptionAgent: {args.perception_heads} heads  "
         f"cross-attn(Dendritic × Resonance)")
    log0(f"SomaCore: {args.num_layers} blocks — pure medium, no nucleus")
    log0(f"CholinergicGate: λ={args.ach_lambda}  θ0={args.ach_theta}  "
         f"k0={args.ach_rho_k}  base_scale={args.ach_base_scale}  "
         f"trace_from_block={base_model.ach_first_block}")
    log0(f"Observer: enabled={base_model.observer.enabled}  every={args.observer_every}  "
         f"sample_tokens={args.observer_sample_tokens}  jsonl={telemetry_file}")
    log0(f"seed: {args.seed}  |  world_size: {world_size}  |  grad_accum: {grad_accum}")

    # ── Optimizer groups: particion exhaustiva y auditada ──
    embedding_params, matrix_params, scalar_params, optimizer_names = \
        build_optimizer_groups(base_model)
    geometry_params = [p for p in base_model.dendrite.geometry.parameters()
                       if p.requires_grad]
    geometry_ids = {id(p) for p in geometry_params}
    matrix_params_main = [p for p in matrix_params if id(p) not in geometry_ids]
    scalar_params_main = [p for p in scalar_params if id(p) not in geometry_ids]
    log0(f"Optimizer audit: embedding={len(embedding_params)} tensors  "
         f"matrix={len(matrix_params_main)}  scalar/vector={len(scalar_params_main)}  "
         f"geometry={len(geometry_params)}  OK")

    # [OPT-1] Lion for embeddings
    opt_emb = Lion(
        [{"params": embedding_params, "lr": args.lion_lr}],
        betas=(args.lion_beta1, args.lion_beta2),
        weight_decay=args.lion_wd,
    )

    # [OPT-2] Muon for matrices
    opt_muon = Muon(matrix_params_main, lr=args.matrix_lr,
                    momentum=args.muon_momentum_warmup_start,
                    backend_steps=args.muon_backend_steps)

    # [OPT-5] AdamW for scalars
    scalar_groups = []
    if scalar_params_main:
        scalar_groups.append({"params": scalar_params_main,
                              "lr": args.scalar_lr,
                              "weight_decay": args.scalar_wd})
    if geometry_params:
        scalar_groups.append({"params": geometry_params,
                              "lr": args.geo_lr,
                              "weight_decay": args.geo_wd})
    opt_scalar = torch.optim.AdamW(
        scalar_groups,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )

    optimizers = [opt_emb, opt_muon, opt_scalar]
    for opt in optimizers:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]

    def zero_grad_all():
        for o in optimizers: o.zero_grad(set_to_none=True)

    max_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0: return 1.0
        if max_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations-step)/max(args.warmdown_iters,1), 0.0) if ws <= step else 1.0
        step_ms     = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining   = max(max_ms - elapsed_ms, 0.0)
        return remaining / max(warmdown_ms, 1e-9) if remaining <= warmdown_ms else 1.0

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── [v3.0 TTT-Iso] Instanciar trainer de TTT ──
    # Solo se usa en eval_val cuando args.ttt_enabled=1.
    # Opera sobre base_model (sin wrapper DDP/compile) para tener
    # acceso directo a .perception, .resonance, .monitor_resonance.
    if args.ttt_enabled and base_model.pfv_enabled:
        ttt_trainer = TTTIsoTrainer(
            base_model       = base_model,
            lr               = args.ttt_lr,
            inner_steps      = args.ttt_inner_steps,
            beta             = args.ttt_beta,
            reset_each_chunk = bool(args.ttt_reset_each_chunk),
            grad_clip        = args.ttt_grad_clip,
        )
        log0(f"TTT-Iso: ENABLED  lr={args.ttt_lr}  K={args.ttt_inner_steps}"
             f"  β={args.ttt_beta}  reset={args.ttt_reset_each_chunk}"
             f"  warmup_chunks={args.ttt_warmup_chunks}"
             f"  grad_clip={args.ttt_grad_clip}"
             f"  restore_after_eval={args.ttt_restore_after_eval}")
    else:
        ttt_trainer = None
        if args.ttt_enabled and not base_model.pfv_enabled:
            log0(f"TTT-Iso: DISABLED (requiere PFV_MONITOR=1 para monitor_resonance)")
        else:
            log0(f"TTT-Iso: DISABLED (TTT_ENABLED=0)")

    # Warmup
    if args.warmup_steps > 0:
        import copy
        init_state = {k: v.detach().cpu().clone()
                      for k, v in base_model.state_dict().items()}
        init_opt   = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum):
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y, use_axon=True)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            if (ws+1) % 10 == 0 or ws+1 == args.warmup_steps:
                log0(f"warmup {ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── MAIN LOOP ──
    training_ms = 0.0
    wall_t0 = time.perf_counter()
    torch.cuda.synchronize()
    t0   = time.perf_counter()
    step = 0

    while True:
        last_step  = (step == args.iterations)
        should_val = (last_step or
                     (args.val_loss_every > 0 and step % args.val_loss_every == 0))

        if should_val:
            torch.cuda.synchronize()
            training_ms += 1000.0 * (time.perf_counter() - t0)

            if base_model.observer.enabled:
                base_model.arm_observer(step, "validation")
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum,
                val_tokens, bb_lut, ls_lut, bd_lut,
                ttt_trainer=ttt_trainer, log_fn=log0,
                base_model=base_model)

            # [NEW] Diagnósticos de resonancia y percepción
            resonance_filled = base_model.resonance.fill_count.item() > 0
            resonance_norm   = base_model.resonance.slots.float().norm(dim=-1).mean().item()
            percept_gate_act = torch.sigmoid(
                base_model.perception.percept_gate).mean().item()

            ach_diag = base_model.axon.diagnostics()
            axon_info = (f"  ach_u:{ach_diag['u_mean']:.3f}±{ach_diag['u_std']:.3f}"
                         f"  mask:{ach_diag['mask_mean']:.3f}"
                         f"[{ach_diag['mask_min']:.3f},{ach_diag['mask_max']:.3f}]"
                         f"  ρ:{ach_diag['rho_mean']:.3f}"
                         f"  k:{ach_diag['rho_k']:.2f}"
                         f"  θ:{ach_diag['theta']:.3f}")
            resonance_info = (f"  res_filled:{resonance_filled}"
                              f"  res_norm:{resonance_norm:.3f}"
                              f"  percept_gate:{percept_gate_act:.3f}")

            # ── [v2.3] PfV snapshot — auto-observación del modelo ──
            pfv_info = ""
            if base_model.pfv_enabled:
                pfv = base_model.pfv_snapshot()
                if pfv:
                    r = pfv["resonance"]
                    pl = pfv["plastic_local"]
                    pm = pfv["plastic_mid"]
                    pg = pfv["plastic_global"]
                    pfv_info = (
                        f"  PfV[res:P={r['P_t'].item():.3f}"
                        f",ΔP={r['delta_P'].item():+.3f}"
                        f",iso={int(r['n_isomers'].item())}"
                        f",ε={r['epsilon'].item():.3f}]"
                        f" [den:L={pl['P_t'].item():.2f}"
                        f"/M={pm['P_t'].item():.2f}"
                        f"/G={pg['P_t'].item():.2f}]"
                    )

            elapsed_s   = training_ms / 1000.0
            elapsed_min = int(elapsed_s // 60)
            elapsed_sec = elapsed_s % 60
            log0(f"step:{step}/{args.iterations}  val_loss:{val_loss:.4f}  "
                 f"val_bpb:{val_bpb:.4f}  "
                 f"time:{elapsed_min}m{elapsed_sec:.1f}s  "
                 f"train_ms:{training_ms:.0f}{axon_info}{resonance_info}{pfv_info}")

            val_obs = base_model.observer_snapshot(clear=True)
            emit_telemetry({
                "event": "validation", "step": step,
                "val_loss": val_loss, "val_bpb": val_bpb,
                "ach": ach_diag,
                "resonance": {
                    "filled": resonance_filled,
                    "norm_mean": resonance_norm,
                    "percept_gate_mean": percept_gate_act,
                },
                "observer": val_obs,
            })

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        torch.cuda.synchronize()
        elapsed_ms = training_ms + 1000.0*(time.perf_counter()-t0)
        if max_ms is not None and elapsed_ms >= max_ms:
            log0(f"Wallclock limit reached at step {step}")
            break

        mul = lr_mul(step, elapsed_ms)
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * mul

        model.train()
        zero_grad_all()
        capture_observer = (base_model.observer.enabled and args.observer_every > 0
                            and step % args.observer_every == 0)
        if capture_observer:
            base_model.arm_observer(step, "train")

        # Soma freeze: primeros N steps para init estable
        if step == 0:
            for p in base_model.blocks.parameters():
                p.requires_grad_(False)
            log0(f"SomaBlocks frozen for first {args.soma_freeze_steps} steps")
        if step == args.soma_freeze_steps:
            for p in base_model.blocks.parameters():
                p.requires_grad_(True)
            log0(f"SomaBlocks unfrozen at step {step}")

        # Muon momentum warmup
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_momentum_now = ((1 - frac) * args.muon_momentum_warmup_start
                             + frac * args.muon_momentum)
        for g in opt_muon.param_groups:
            g["momentum"] = muon_momentum_now

        for micro in range(grad_accum):
            if distributed:
                model.require_backward_grad_sync = (micro == grad_accum-1)
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, use_axon=True)
            (loss * grad_scale).backward()

        grad_diag = collect_gradient_stats(base_model) if capture_observer else {}
        for o in optimizers: o.step()
        zero_grad_all()

        if capture_observer:
            train_obs = base_model.observer_snapshot(clear=True)
            ach_train = base_model.axon.diagnostics()
            emit_telemetry({
                "event": "train", "step": step,
                "loss": float(loss.detach().item()), "lr_mul": mul,
                "ach": ach_train,
                "geometry": base_model.dendrite.geometry_snapshot(),
                "gradients": grad_diag,
                "parameters": collect_parameter_stats(base_model),
                "observer": train_obs,
            })

        if master and args.train_log_every > 0 and step % args.train_log_every == 0:
            torch.cuda.synchronize()
            elapsed = training_ms + 1000.0*(time.perf_counter()-t0)
            e_s   = elapsed / 1000.0
            e_min = int(e_s // 60)
            e_sec = e_s % 60
            ach_now = base_model.axon.diagnostics()
            grad_ach = (grad_diag.get("groups", {}).get("ach", 0.0)
                        if capture_observer else 0.0)
            grad_geo = (grad_diag.get("groups", {}).get("geometry", 0.0)
                        if capture_observer else 0.0)
            geo_now = base_model.dendrite.geometry_snapshot()
            gamma_mean = (sum(geo_now.get("gamma_mean", [0.0])) / 3.0
                          if geo_now.get("gamma_mean") else 0.0)
            tau_mean = (sum(geo_now.get("tau_mean", [0.0])) / 3.0
                        if geo_now.get("tau_mean") else 0.0)
            extra = (f"  ach_u:{ach_now['u_mean']:.3f}"
                     f"  rho:{ach_now['rho_mean']:.3f}"
                     f"  mask:{ach_now['mask_mean']:.3f}"
                     f"  geo_g:{gamma_mean:.3f}"
                     f"  geo_tau:{tau_mean:.4f}"
                     f"  |g_ach|:{grad_ach:.3e}"
                     f"  |g_geo|:{grad_geo:.3e}") if args.observer_console else ""
            log0(f"step:{step}  loss:{loss.item():.4f}  "
                 f"lr_mul:{mul:.4f}  time:{e_min}m{e_sec:.1f}s{extra}")

        step += 1

    # ── FINAL ──
    torch.cuda.synchronize()
    training_ms += 1000.0 * (time.perf_counter() - t0)
    train_s   = training_ms / 1000.0
    train_min = int(train_s // 60)
    train_sec = train_s % 60
    wall_s    = time.perf_counter() - wall_t0
    wall_min  = int(wall_s // 60)
    wall_sec  = wall_s % 60
    log0(f"\n── Training complete ──")
    log0(f"  Train time : {train_min}m {train_sec:.1f}s  ({train_s:.1f}s; eval excluded)")
    log0(f"  Wall time  : {wall_min}m {wall_sec:.1f}s  ({wall_s:.1f}s; eval included)")
    log0(f"  Steps done : {step}")
    log0(f"  train ms/step: {training_ms / max(step, 1):.1f}ms")

    if master:
        raw_ckpt = None
        if args.save_raw_checkpoint:
            raw_ckpt, raw_kb = save_raw_checkpoint(
                base_model, args, step, args.artifact_dir)
            log0(f"\n── Persistent Checkpoint ──")
            log0(f"  raw checkpoint : {raw_ckpt}  ({raw_kb:.1f} KB)")

        size_info = quantize_and_compress(base_model, __file__)
        compressed_path = None
        compressed_kb = 0.0
        if args.save_compressed_model:
            compressed_path, compressed_kb = save_compressed_archive(
                size_info["model_bytes"], args, step, args.artifact_dir)

        log0(f"\n── Artifact Size ──")
        log0(f"  model compressed : {size_info['model_compressed_kb']:.1f} KB")
        log0(f"  code compressed  : {size_info['code_compressed_kb']:.1f} KB")
        log0(f"  TOTAL            : {size_info['total_kb']:.1f} KB")
        log0(f"  fits 16MB        : {size_info['fits_16mb']}")
        log0(f"  fits 14MB        : {size_info['fits_14mb']}")
        log0(f"  SVD tensors      : {size_info['n_svd_tensors']}")
        log0(f"  int8 tensors     : {size_info['n_int8_tensors']}")
        if telemetry_file:
            log0(f"  telemetry JSONL  : {telemetry_file}")
        if compressed_path is not None:
            log0(f"  compressed file : {compressed_path}  ({compressed_kb:.1f} KB)")
        if raw_ckpt is not None:
            log0(f"  raw checkpoint  : {raw_ckpt}")

        log0(f"\n── ACh + Molecule diagnostics ──")
        log0(f"  last_u_mean      : {base_model.axon.last_u_mean.item():.4f}")
        log0(f"  last_mask_mean   : {base_model.axon.last_mask_mean.item():.4f}")
        log0(f"  field_mix        : "
             f"{F.softmax(base_model.dendrite.field_mix, dim=0).tolist()}")

        log0(f"\n── Resonance + Perception diagnostics ──")
        log0(f"  resonance filled : {base_model.resonance.fill_count.item() > 0}")
        log0(f"  resonance ptr    : {base_model.resonance.slot_ptr.item()}")
        slot_norms = base_model.resonance.slots.float().norm(dim=-1)
        log0(f"  slot_norm mean   : {slot_norms.mean().item():.4f}")
        log0(f"  slot_norm std    : {slot_norms.std().item():.4f}")
        percept_gate = torch.sigmoid(base_model.perception.percept_gate)
        log0(f"  percept_gate μ   : {percept_gate.mean().item():.4f}")
        log0(f"  percept_gate max : {percept_gate.max().item():.4f}")

        log0(f"\n── SomaCore diagnostics ──")
        attn_scales = [
            float(p.data.abs().mean().item())
            for n, p in base_model.blocks.named_parameters()
            if 'attn_scale' in n
        ]
        mlp_scales = [
            float(p.data.abs().mean().item())
            for n, p in base_model.blocks.named_parameters()
            if 'mlp_scale' in n
        ]
        log0(f"  attn_scale |μ|   : {sum(attn_scales)/max(len(attn_scales),1):.4f}")
        log0(f"  mlp_scale  |μ|   : {sum(mlp_scales)/max(len(mlp_scales),1):.4f}")

        # Distilación del ResonanceBuffer
        resonance_py   = distill_resonance(base_model)
        resonance_path = f"logs/resonance_distilled_{args.run_id}.py"
        with open(resonance_path, "w") as f:
            f.write(resonance_py)
        resonance_compressed_kb = len(
            __import__('zlib').compress(resonance_py.encode(), level=9)) / 1000
        log0(f"  resonance_distilled : {resonance_path}")
        log0(f"  resonance_compressed: {resonance_compressed_kb:.1f} KB")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
