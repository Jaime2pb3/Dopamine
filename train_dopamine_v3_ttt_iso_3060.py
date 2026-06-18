# -*- coding: utf-8 -*-
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
║  Phase 3 — AXON PROJECTOR + MOLECULE GATE                       ║
║    Unchanged — selective transmission, think before speaking.    ║
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

import glob
import io
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

    # AxonProjector + MoleculeGate
    axon_n_views  = int(os.environ.get("AXON_N_VIEWS", 4))
    axon_t_min    = float(os.environ.get("AXON_T_MIN", 0.20))
    axon_t_max    = float(os.environ.get("AXON_T_MAX", 0.60))
    axon_lambda   = float(os.environ.get("AXON_LAMBDA", 0.12))
    axon_stress_threshold = float(os.environ.get("AXON_STRESS_THRESHOLD", 0.30))

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
# SIX QUARKS — Isomeric Polarization
# ─────────────────────────────────────────────────────────────

class DownQuark:
    """NTVM — mean pairwise divergence across views"""
    def compute(self, phi_list: list[Tensor]) -> tuple[float, Tensor]:
        m = len(phi_list)
        if m < 2:
            return 0.0, torch.zeros(phi_list[0].size(-1))
        diffs = []
        scalar_dists = []
        for i in range(m):
            for j in range(i+1, m):
                diff = (phi_list[i] - phi_list[j]).abs()
                diffs.append(diff)
                cos = F.cosine_similarity(phi_list[i], phi_list[j], dim=-1).mean()
                scalar_dists.append(float(1.0 - cos.item()))
        dim_div = torch.stack(diffs).mean(0).squeeze(0)
        return float(np.mean(scalar_dists)), dim_div


class StrangeQuark:
    """VPEC — inter-metric disagreement"""
    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2: return 0.0
        stds = []
        for i in range(m):
            for j in range(i+1, m):
                a, b = phi_list[i], phi_list[j]
                d1 = float(1.0 - F.cosine_similarity(a, b, dim=-1).mean().item())
                d2 = float((a-b).norm(dim=-1).mean().item() /
                           (a.norm(dim=-1).mean().item() + 1e-9))
                va = float(a.var(dim=-1).mean().item())
                vb = float(b.var(dim=-1).mean().item())
                d3 = abs(va-vb) / (max(va, vb) + 1e-9)
                stds.append(float(np.std([d1, d2, d3])))
        return float(np.mean(stds)) if stds else 0.0


class UpQuark:
    """SDRBD — Sarle Bimodality Coefficient"""
    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 4: return 0.0
        dists = []
        for i in range(m):
            for j in range(i+1, m):
                cos = F.cosine_similarity(phi_list[i], phi_list[j], dim=-1).mean()
                dists.append(float(1.0 - cos.item()))
        if len(dists) < 3: return 0.0
        arr = np.array(dists)
        n = len(arr)
        mu = arr.mean()
        if arr.std() < 1e-9: return 0.0
        skew = float(np.mean((arr - mu)**3) / (arr.std()**3 + 1e-9))
        kurt = float(np.mean((arr - mu)**4) / (arr.std()**4 + 1e-9))
        bc   = (skew**2 + 1) / (kurt + 3 * (n-1)**2 / ((n-2)*(n-3) + 1e-9))
        return float(bc)


class CharmQuark:
    """ETSV — exponential tail separation"""
    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2: return 0.0
        sims = []
        for i in range(m):
            for j in range(i+1, m):
                cos = float(F.cosine_similarity(
                    phi_list[i], phi_list[j], dim=-1).mean().item())
                sims.append(cos)
        if not sims: return 0.0
        arr = np.array(sims)
        hi  = arr[arr > arr.mean()]
        lo  = arr[arr <= arr.mean()]
        if len(hi) == 0 or len(lo) == 0: return 0.0
        return float(abs(hi.mean() - lo.mean()))


class TopQuark:
    """MEVR — max eigenvalue variance ratio"""
    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2: return 0.0
        try:
            stacked = torch.stack([p.squeeze(0).float() for p in phi_list])
            cov = torch.cov(stacked.T)
            eigs = torch.linalg.eigvalsh(cov)
            eigs = eigs.abs()
            total = eigs.sum().item()
            if total < 1e-9: return 0.0
            return float(eigs.max().item() / total)
        except Exception:
            return 0.0


class BottomQuark:
    """CSCD — cosine similarity collapse detector"""
    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2: return 0.0
        sims = []
        for i in range(m):
            for j in range(i+1, m):
                cos = float(F.cosine_similarity(
                    phi_list[i], phi_list[j], dim=-1).mean().item())
                sims.append(abs(cos))
        return float(np.mean(sims)) if sims else 0.0


class QuarkBundle:
    """Aggregates all six quarks into scalar S and per-dim divergence."""
    def __init__(self):
        self.down   = DownQuark()
        self.strange = StrangeQuark()
        self.up     = UpQuark()
        self.charm  = CharmQuark()
        self.top    = TopQuark()
        self.bottom = BottomQuark()
        self._history: list[float] = []

    def reset(self):
        self._history.clear()

    def observe(self, phi_list: list[Tensor]) -> tuple[float, Tensor]:
        s_down, dim_div = self.down.compute(phi_list)
        s_strange = self.strange.compute(phi_list)
        s_up      = self.up.compute(phi_list)
        s_charm   = self.charm.compute(phi_list)
        s_top     = self.top.compute(phi_list)
        s_bottom  = self.bottom.compute(phi_list)
        S = float(np.mean([s_down, s_strange, s_up, s_charm, s_top, s_bottom]))
        self._history.append(S)
        return S, dim_div


# ─────────────────────────────────────────────────────────────
# PHASE 3 — AXON PROJECTOR + MOLECULE GATE
# ─────────────────────────────────────────────────────────────

class AxonProjector(nn.Module):
    """
    Isomeric Polarization Axon — v2.3c (UP-style rho-mixing)

    El axon se polariza segun la compresion isometrica del modelo:

        rho(S) = sigmoid(k * (S - S_threshold))   in [0, 1]

        c_base     = sigmoid(conductance_base)      <- sin estres: paso libre
        c_stressed = c_base * (1 - lam * div_norm)  <- bajo estres: suprime dims redundantes

        mask = (1 - rho) * c_base + rho * c_stressed

    S bajo  (representaciones dispersas)  -> rho~0 -> mask~c_base    -> axon abierto
    S alto  (representaciones comprimidas) -> rho~1 -> mask~c_stressed -> axon polarizado

    conductance_base recibe gradientes reales. QuarkBundle corre en no_grad
    y solo produce escalares S y dim_div (sin parametros propios).

    Analogo directo de UP: Q_eff = (1-rho)*Q0 + rho*Q1
    donde Q0=canal_base, Q1=canal_stress_modulado.
    """
    def __init__(self, dim: int, n_views: int, t_min: float, t_max: float,
                 lam: float, stress_threshold: float):
        super().__init__()
        self.dim              = dim
        self.n_views          = n_views
        self.t_min            = t_min
        self.t_max            = t_max
        self.lam              = lam
        self.stress_threshold = stress_threshold

        # conductance_base: sigmoid(~0)=0.5, gradientes activos desde paso 0
        self.conductance_base = nn.Parameter(
            torch.zeros(dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))

        # rho_k escala la transicion — aprendible, inicia suave (k=2.0)
        self.rho_k = nn.Parameter(torch.tensor(2.0))

        self.quarks         = QuarkBundle()
        self.last_S         = 0.0
        self.last_rho       = 0.0
        self.last_mask_mean = 1.0

    def reset_quarks(self):
        self.quarks.reset()

    def forward(self, h: Tensor) -> Tensor:
        h_flat = h.reshape(-1, self.dim)
        h_mean = h_flat.mean(dim=0, keepdim=True)

        # Paso 1: senal isometrica — QuarkBundle sin gradiente (no tiene params)
        with torch.no_grad():
            temps    = np.linspace(self.t_min, self.t_max, self.n_views)
            phi_list = []
            for t in temps:
                noise = torch.randn_like(h_mean) * t * 0.25
                r     = h_mean + noise
                phi_list.append(F.layer_norm(r, r.shape[-1:]))
            S, dim_div = self.quarks.observe(phi_list)

        self.last_S = S

        # Paso 2: rho isometrico — polarizacion segun compresion
        k   = self.rho_k.clamp(min=0.5, max=8.0)
        S_t = torch.tensor(S, dtype=h.dtype, device=h.device)
        rho = torch.sigmoid(k * (S_t - self.stress_threshold))
        self.last_rho = float(rho.item())

        # Paso 3: dos canales de conductance (gradientes reales via conductance_base)
        cb = self.conductance_base.to(h.dtype)

        # Canal Q0 — base, sin modulacion
        c_base = torch.sigmoid(cb)

        # Canal Q1 — suprime dimensiones con alta divergencia (redundantes bajo compresion)
        dim_div_t    = dim_div.to(device=h.device, dtype=h.dtype)
        dim_div_norm = dim_div_t / (dim_div_t.max() + 1e-9)
        c_stressed   = c_base * (1.0 - self.lam * dim_div_norm)
        c_stressed   = c_stressed.clamp(min=0.05)

        # Paso 4: mezcla isometrica UP-style
        mask = (1.0 - rho) * c_base + rho * c_stressed

        self.last_mask_mean = float(mask.mean().item())
        return h * mask.unsqueeze(0).unsqueeze(0)


class MoleculeGate(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        rank = min(32, dim // 8)
        self.gate_down = CastedLinear(dim, rank,        bias=False)
        self.gate_up   = CastedLinear(rank, vocab_size, bias=False)
        nn.init.normal_(self.gate_down.weight, std=0.01)
        nn.init.zeros_(self.gate_up.weight)

    def forward(self, logits: Tensor, h: Tensor, S: float,
                stress_threshold: float) -> Tensor:
        if S <= stress_threshold:
            return logits
        correction   = self.gate_up(torch.tanh(self.gate_down(h)))
        stress_weight = min((S - stress_threshold) / (1.0 - stress_threshold + 1e-9), 1.0)
        return logits + stress_weight * correction


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


class DendriticEncoder(nn.Module):
    def __init__(self, dim: int, field_heads: list[int],
                 local_span: int, mid_span: int,
                 n_kv_heads: int, rope_base: float,
                 n_proto: int = 16):
        super().__init__()
        self.dim         = dim
        self.field_heads = field_heads
        self.n_heads     = sum(field_heads)
        self.n_kv_heads  = n_kv_heads
        self.head_dim    = dim // self.n_heads
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
        # [PERF-2] Cache de span masks — no recrear en cada forward.
        self._mask_cache: dict = {}

    def _make_span_mask(self, T: int, span: int, device, dtype) -> Tensor:
        # [PERF-2] Cacheado por (T, span, device, dtype). El mask es determinista
        # y el seq_len es fijo durante el run → se construye una vez por campo.
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
        mask.masked_fill_(~causal, float("-inf"))
        # (1, 1, T, T) listo para broadcast sobre (B, n_heads, T, T)
        mask = mask.unsqueeze(0).unsqueeze(0)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        k = self.c_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, x.dtype)
        k = apply_rope(k, cos, sin)
        q_projs  = [self.q_local,  self.q_mid,  self.q_global]
        plastics = [self.plastic_local, self.plastic_mid, self.plastic_global]
        spans    = [self.local_span, self.mid_span, T]
        field_outputs = []
        for q_proj, plastic, span, nh in zip(q_projs, plastics, spans, self.field_heads):
            q = q_proj(x).reshape(B, T, nh, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            q = apply_rope(q, cos, sin)
            k_exp = k
            v_exp = v
            if span < T:
                # [PERF-2] Mask cacheado, ya incluye unsqueeze(0,0) y dtype correcto.
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
            field_h_plastic = plastic(field_h_full)
            field_outputs.append(field_h_plastic)
        mix   = F.softmax(self.field_mix.to(x.dtype), dim=0)
        mixed = sum(mix[i] * field_outputs[i] for i in range(3))
        return self.proj(mixed)


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

    Args:
        n_realizations: m, número de realizaciones en ℛ_t
        window:         tamaño del ring buffer de P_history (para régimen)
        agg:            'mean' o 'trimmed_mean'  (Agg del paper)
        epsilon_quantile: q ∈ (0,1). ε_t = quantile(d_pairs, q).
                         q pequeño (0.1) → ε estricto, pocos isómeros
                         q grande  (0.5) → ε laxo, muchos isómeros
        activation_ema: EMA decay para activation_count
    """
    def __init__(self, n_realizations: int, window: int = 50,
                 agg: str = 'mean', epsilon_quantile: float = 0.1,
                 activation_ema: float = 0.95):
        super().__init__()
        assert agg in ('mean', 'trimmed_mean')
        assert 0.0 < epsilon_quantile < 1.0
        self.n_realizations   = n_realizations
        self.window           = window
        self.agg              = agg
        self.epsilon_quantile = epsilon_quantile
        self.activation_ema   = activation_ema

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
    # [v3.0 TTT-Iso] LOSS DIFERENCIABLE — Isomeric TTT
    # ═════════════════════════════════════════════════════════════
    def compute_ttt_loss(self,
                          realizations_frozen: Tensor,
                          realization_new: Tensor,
                          beta: float = 0.1) -> Tensor:
        """
        Loss diferenciable para Test-Time Training isomérico.

        El gradiente fluye SOLO a través de `realization_new`. Los
        `realizations_frozen` se tratan como contexto detached (huella previa).

        Args:
            realizations_frozen: (K, D) — slots detached del ResonanceBuffer
                                 (o lo que sirva como conjunto de referencia).
            realization_new:     (D,) — candidato con grad (p. ej. el
                                 mean del perceptual_flow del PerceptionAgent).
            beta:                peso de |Δ²P_t| en la loss.

        Returns:
            escalar con grad:
                L_iso = |P_t(R ∪ {p_new}) - P_regime| + β·|Δ²P_t|

        Nota: este método NO actualiza buffers — es puro cálculo.
              P_regime y Δ²P_t provienen del historial ya poblado por
              forward passes previos (via .compute()).
        """
        K = realizations_frozen.size(0)
        Rf = realizations_frozen.detach().float()      # (K, D)
        pn = realization_new.float()                    # (D,)  con grad

        # Conjunto aumentado R ∪ {p_new} — solo pn lleva grad
        R_aug = torch.cat([Rf, pn.unsqueeze(0)], dim=0) # (K+1, D)

        # Pairwise cosine distance sobre R_aug
        R_norm  = F.normalize(R_aug, dim=-1)
        sim     = R_norm @ R_norm.T                     # (K+1, K+1)
        d       = 1.0 - sim

        mask_upper = torch.triu(
            torch.ones_like(d, dtype=torch.bool), diagonal=1)
        d_pairs = d[mask_upper]                         # ((K+1)K/2,)

        # P_t sobre el conjunto aumentado
        if self.agg == 'trimmed_mean':
            sorted_d, _ = d_pairs.sort()
            n_trim = max(1, int(len(sorted_d) * 0.1))
            if len(sorted_d) > 2 * n_trim:
                P_t_aug = sorted_d[n_trim:-n_trim].mean()
            else:
                P_t_aug = sorted_d.mean()
        else:
            P_t_aug = d_pairs.mean()

        # P_regime detached — target estable del régimen actual
        effective_n = torch.clamp(self.P_fill, min=1.0,
                                   max=float(self.window))
        P_regime = (self.P_history.sum() / effective_n).detach()

        # Término 1: coherencia con el régimen
        loss_regime = (P_t_aug - P_regime).abs()

        # Término 2: penalizar aceleración — |Δ²P_t|
        # Δ²P_t = (P_t_aug - P_{t-1}) - (P_{t-1} - P_{t-2})
        #       = P_t_aug - 2·P_{t-1} + P_{t-2}
        prev_ptr  = (self.P_ptr - 1) % self.window
        prev2_ptr = (self.P_ptr - 2) % self.window
        m1 = F.one_hot(prev_ptr,  num_classes=self.window).squeeze(0).float()
        m2 = F.one_hot(prev2_ptr, num_classes=self.window).squeeze(0).float()
        P_prev  = (self.P_history * m1).sum().detach()
        P_prev2 = (self.P_history * m2).sum().detach()

        # Gated por fill para no penalizar aceleración con historial vacío
        fill_frac = torch.clamp(self.P_fill / 2.0, max=1.0).detach()
        delta2_P  = P_t_aug - 2.0 * P_prev + P_prev2
        loss_accel = fill_frac * delta2_P.abs()

        return loss_regime + beta * loss_accel


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
        """
        Retorna slots en shape (K_eff, D):
          - mode="mean" / "topk": (n_slots, D)
          - mode="seq":            (n_slots * mem_span, D)  ← aplanado para el PerceptionAgent
        """
        if self.mode == "seq":
            return self.slots.reshape(-1, self.dim)
        return self.slots

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

    def forward(self, dendritic_out: Tensor, resonance_slots: Tensor) -> Tensor:
        """
        dendritic_out:    (B, T, D)
        resonance_slots:  (K_eff, D)  — K_eff = n_slots (mean/topk) o n_slots*mem_span (seq)

        Returns: perceptual_flow (B, T, D)
        """
        B, T, D = dendritic_out.shape
        K = resonance_slots.size(0)

        # Query desde experiencia actual
        q = self.c_q(self.query_norm(dendritic_out))               # (B, T, D)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))

        # [PERF-4] Proyectar K/V sobre (K, D) ANTES de expandir al batch.
        # Los slots son idénticos en todo el batch → proyectar (B, K, D) era
        # desperdicio de B× FLOPs. Ahora proyectamos una vez y expandimos con
        # broadcast (expand no copia memoria).
        slots_norm = self.key_norm(resonance_slots.unsqueeze(0)).squeeze(0)  # (K, D)
        k_proj = self.c_k(slots_norm)                                        # (K, D)
        v_proj = self.c_v(slots_norm)                                        # (K, D)

        k = k_proj.reshape(K, self.n_heads, self.head_dim).transpose(0, 1)   # (n_heads, K, hd)
        v = v_proj.reshape(K, self.n_heads, self.head_dim).transpose(0, 1)   # (n_heads, K, hd)
        k = F.rms_norm(k, (k.size(-1),))

        # Broadcast al batch — view, no alloca.
        k = k.unsqueeze(0).expand(B, -1, -1, -1)                             # (B, n_heads, K, hd)
        v = v.unsqueeze(0).expand(B, -1, -1, -1)

        # Cross-attention: tokens de experiencia atienden a slots de memoria.
        # No causal — la memoria no tiene orden temporal secuencial propio.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False)              # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)    # (B, T, D)
        out = self.proj(out)                                        # (B, T, D)

        # Gate: percepción controla su propia integración
        gate = torch.sigmoid(self.percept_gate.to(dendritic_out.dtype))  # (D,)
        return out * gate.unsqueeze(0).unsqueeze(0)                       # (B, T, D)


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
    Phase 3   — AxonProjector+Gate:  Transmisión selectiva

    Forward pipeline:
      x = embed(input) → x0 (highway)
      dendritic = DendriticEncoder(x)
      perceptual = PerceptionAgent(dendritic, resonance.slots)
      x = x + dendritic + perceptual   ← Soma recibe experiencia + percepción
      x = SomaCore(x, x0)             ← El medio procesa el flujo combinado
      resonance.update(x)              ← Memoria actualiza con lo que fluyó
      h = AxonProjector(final_norm(x))
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

        # Phase 3: AxonProjector + MoleculeGate
        self.axon = AxonProjector(
            dim              = d,
            n_views          = args.axon_n_views,
            t_min            = args.axon_t_min,
            t_max            = args.axon_t_max,
            lam              = args.axon_lambda,
            stress_threshold = args.axon_stress_threshold,
        )
        self.molecule_gate = MoleculeGate(d, args.vocab_size)

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
        # ResonanceBuffer
        slots = self.resonance.get_slots()                         # (K_eff, D)
        out["resonance"] = self.monitor_resonance.compute(slots)
        # PlasticDendriticField — prototipos por escala
        out["plastic_local"]  = self.monitor_plastic_local.compute(
            self.dendrite.plastic_local.prototypes)
        out["plastic_mid"]    = self.monitor_plastic_mid.compute(
            self.dendrite.plastic_mid.prototypes)
        out["plastic_global"] = self.monitor_plastic_global.compute(
            self.dendrite.plastic_global.prototypes)
        return out

    def _encode_decode(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
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

        # Phase 1: Experiencia — qué hay ahora
        dendritic_out = self.dendrite(self.dendrite_norm(x))   # (B, T, D)

        # Phase 1.5: Percepción — resonancia con lo que ha fluido
        resonance_slots = self.resonance.get_slots()           # (K, D)
        perceptual_flow = self.perception(dendritic_out, resonance_slots)  # (B, T, D)

        # Soma recibe: experiencia actual + reconocimiento de lo familiar
        # percept_norm estabiliza el flujo combinado antes del Soma
        x = x + dendritic_out + self.percept_norm(perceptual_flow)

        # Phase 2: SomaCore — el medio procesa el flujo
        skips = []
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.n_dec):
            if skips:
                sw = self.skip_weights[i].to(dtype=x.dtype)
                x  = x + sw[None, None, :] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)

        # [ARCH-v2.2] La memoria registra el flujo POST-norm — estado estabilizado
        # en escala, lo que hace el cross-attn del PerceptionAgent más fiel al
        # espacio de representación real del Soma. Antes: pre-norm (inconsistente).
        h = self.final_norm(x)
        self.resonance.update(h)
        return h, x

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                use_axon: bool = True) -> Tensor:
        h, _ = self._encode_decode(input_ids)

        if use_axon:
            h = self.axon(h)

        h_flat = h.reshape(-1, h.size(-1))
        logits = F.linear(h_flat, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if use_axon:
            logits = self.molecule_gate(logits, h_flat,
                                        self.axon.last_S,
                                        self.args.axon_stress_threshold)

        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def get_logits(self, input_ids: Tensor) -> Tensor:
        h, _ = self._encode_decode(input_ids)
        h    = self.axon(h)
        h_flat = h.reshape(-1, h.size(-1))
        logits = F.linear(h_flat, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        logits = self.molecule_gate(logits, h_flat,
                                    self.axon.last_S,
                                    self.args.axon_stress_threshold)
        return logits

    # ═════════════════════════════════════════════════════════════
    # [v3.0 TTT-Iso] HOOK PARA EXTRAER perceptual_flow CON GRAD
    # ═════════════════════════════════════════════════════════════
    def forward_with_percept(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Variante de forward que expone el perceptual_flow con grad.
        Usada por TTTIsoTrainer.

        Returns:
            loss_ce:         cross-entropy (no usada en TTT, solo diagnóstico)
            perceptual_flow: (B, T, D) con grad — entrada del TTT-Iso
        """
        x  = self.tok_emb(input_ids)
        x  = self.input_norm(x)
        x0 = x

        dendritic_out = self.dendrite(self.dendrite_norm(x))
        resonance_slots = self.resonance.get_slots()
        perceptual_flow = self.perception(dendritic_out, resonance_slots)

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
    """
    Trainer especializado para Isomeric TTT sobre PerceptionAgent.

    Uso (dentro de eval_val, score-first):
        ttt = TTTIsoTrainer(model, lr=5e-4, inner_steps=3, beta=0.1)
        for chunk in val_chunks:
            bpb_chunk = score_chunk(model, chunk)   # ← gradea primero
            ttt.adapt(chunk)                         # ← luego adapta
    """
    def __init__(self, base_model: "Dopamine",
                 lr: float = 5e-4,
                 inner_steps: int = 3,
                 beta: float = 0.1,
                 reset_each_chunk: bool = False):
        self.base_model       = base_model
        self.inner_steps      = inner_steps
        self.beta             = beta
        self.reset_each_chunk = reset_each_chunk

        # Sólo optimizamos PerceptionAgent
        self.percept_params = list(base_model.perception.parameters())
        # Lion ligero para TTT — sign-based, robusto a ruido
        self.opt = Lion(self.percept_params, lr=lr,
                        betas=(0.9, 0.99), weight_decay=0.0)

        # Snapshot del estado inicial para reset opcional
        self._init_state = None
        if reset_each_chunk:
            self._init_state = {
                k: v.detach().clone()
                for k, v in base_model.perception.state_dict().items()
            }

        # Stats
        self.n_chunks_adapted = 0
        self.last_loss = float('nan')

    def _freeze_all_except_perception(self) -> dict:
        """Congela todo el modelo excepto PerceptionAgent. Retorna estado previo."""
        prev = {}
        for name, p in self.base_model.named_parameters():
            prev[name] = p.requires_grad
            # Solo PerceptionAgent descongelado
            p.requires_grad_(name.startswith("perception."))
        return prev

    def _restore_grad_state(self, prev: dict) -> None:
        """Restaura requires_grad a como estaba antes."""
        for name, p in self.base_model.named_parameters():
            if name in prev:
                p.requires_grad_(prev[name])

    def adapt(self, input_ids: Tensor) -> dict:
        """
        Ejecuta K inner steps de TTT-Iso sobre el chunk dado.

        Args:
            input_ids: (B, T) — tokens YA graded (bpb ya medido).

        Returns:
            dict con diagnósticos del TTT (last_loss, grad_norm, etc.)
        """
        if self.reset_each_chunk and self._init_state is not None:
            self.base_model.perception.load_state_dict(self._init_state)

        # Congelar todo excepto PerceptionAgent
        prev_grad_state = self._freeze_all_except_perception()

        # Activar train mode solo en PerceptionAgent (dropout no aplica aquí
        # porque no hay dropout, pero por higiene).
        self.base_model.perception.train()

        diag = {'losses': [], 'grad_norms': []}

        try:
            for step in range(self.inner_steps):
                self.opt.zero_grad(set_to_none=True)

                # Forward con grad
                with torch.autocast(device_type="cuda",
                                     dtype=torch.bfloat16, enabled=True):
                    _, perceptual_flow = self.base_model.forward_with_percept(
                        input_ids)

                # p_new = mean del flujo perceptual sobre batch y tiempo
                p_new = perceptual_flow.float().mean(dim=(0, 1))   # (D,)

                # R = slots detached del ResonanceBuffer
                R_frozen = self.base_model.resonance.get_slots().detach().float()

                # L_iso — diferenciable a través de p_new
                loss = self.base_model.monitor_resonance.compute_ttt_loss(
                    realizations_frozen = R_frozen,
                    realization_new     = p_new,
                    beta                = self.beta,
                )
                diag['losses'].append(float(loss.item()))

                loss.backward()

                # Grad norm diagnóstico
                total_grad = 0.0
                for p in self.percept_params:
                    if p.grad is not None:
                        total_grad += float(p.grad.norm().item() ** 2)
                diag['grad_norms'].append(float(total_grad ** 0.5))

                self.opt.step()

            self.last_loss = diag['losses'][-1] if diag['losses'] else float('nan')
            self.n_chunks_adapted += 1

        finally:
            # Devolver el modelo a eval y restaurar requires_grad
            self.base_model.perception.eval()
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
             ttt_trainer=None, log_fn=None):
    """
    Evaluación FineWeb val con BPB.

    Si `ttt_trainer` es provisto y TTT está habilitado, aplica Isomeric TTT
    en formato SCORE-FIRST: cada chunk se gradea (bpb contribuye) ANTES
    de ser usado para adaptar PerceptionAgent. Esto cumple la regla legal
    de Parameter Golf: "TTT only on tokens already graded".

    Args:
        ttt_trainer: TTTIsoTrainer o None (sin TTT → eval clásico).
        log_fn:      función de logging opcional para diagnósticos TTT.
    """
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum)
    local_batch_seqs   = local_batch_tokens  // args.train_seq_len
    total_seqs  = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start   = (total_seqs * rank)      // world_size
    seq_end     = (total_seqs * (rank+1))  // world_size

    loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    token_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt  = torch.zeros((), device=device, dtype=torch.float64)

    ttt_active = (ttt_trainer is not None) and bool(args.ttt_enabled)
    warmup_chunks_left = int(args.ttt_warmup_chunks) if ttt_active else 0
    chunk_idx = 0

    # Diagnóstico agregado
    ttt_diag_all = {'losses': [], 'grad_norms': []}

    model.eval()
    # OJO: NO podemos usar inference_mode() si hay TTT — necesitamos grad.
    # Pero el scoring mismo sí debe ir en no_grad para no allocar memoria.
    for bs in range(seq_start, seq_end, local_batch_seqs):
        be  = min(bs + local_batch_seqs, seq_end)
        raw = val_tokens[bs*args.train_seq_len:(be*args.train_seq_len)+1]
        raw = raw.to(device=device, dtype=torch.int64, non_blocking=True)
        x   = raw[:-1].reshape(-1, args.train_seq_len)
        y   = raw[1:].reshape(-1, args.train_seq_len)

        # ── FASE 1: SCORING ── (sin grad, contribuye a bpb)
        with torch.no_grad():
            with torch.autocast(device_type="cuda",
                                 dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, use_axon=True).detach()

        n = float(y.numel())
        loss_sum  += batch_loss.to(torch.float64) * n
        token_cnt += n
        prev_ids = x.reshape(-1)
        tgt_ids  = y.reshape(-1)
        tb  = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        tb += (leading_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(torch.int16)
        byte_cnt += tb.to(torch.float64).sum()

        # ── FASE 2: TTT-Iso ── (con grad, solo tokens ya graded)
        if ttt_active:
            # Unwrap DDP si aplica — TTT opera sobre base_model directo
            base = model.module if hasattr(model, 'module') else model
            # Si el modelo fue compilado, el base_model subyacente se
            # pasó al trainer al construirlo; `model` aquí es el wrapper.

            # Poblar P_history del monitor ANTES del TTT — asegura que
            # P_regime no sea 0 cuando compute_ttt_loss lo consulta.
            # El scoring previo ya actualizó el ResonanceBuffer via forward,
            # así que aquí solo hay que correr el monitor sobre los slots.
            with torch.no_grad():
                _ = ttt_trainer.base_model.monitor_resonance.compute(
                    ttt_trainer.base_model.resonance.get_slots())

            if warmup_chunks_left <= 0:
                diag = ttt_trainer.adapt(x)
                ttt_diag_all['losses'].extend(diag['losses'])
                ttt_diag_all['grad_norms'].extend(diag['grad_norms'])
            else:
                warmup_chunks_left -= 1

        chunk_idx += 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,  op=dist.ReduceOp.SUM)
        dist.all_reduce(token_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_cnt,  op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum / token_cnt)
    bpt      = val_loss / math.log(2.0)
    tpb      = float(token_cnt / byte_cnt)

    # Log TTT diagnostics si hubo adaptación
    if ttt_active and ttt_diag_all['losses'] and log_fn is not None:
        avg_loss  = sum(ttt_diag_all['losses'])     / len(ttt_diag_all['losses'])
        avg_gnorm = sum(ttt_diag_all['grad_norms']) / len(ttt_diag_all['grad_norms'])
        log_fn(f"  [TTT-Iso]  chunks_adapted={ttt_trainer.n_chunks_adapted}"
               f"  avg_L_iso={avg_loss:.4f}"
               f"  avg_|∇|={avg_gnorm:.4f}")

    model.train()
    return val_loss, float(bpt * tpb)


# ─────────────────────────────────────────────────────────────
# QUANTIZATION + RESONANCE DISTILLATION
# ─────────────────────────────────────────────────────────────

def distill_resonance(model: nn.Module) -> str:
    """
    Post-training: snapshot del ResonanceBuffer como constantes Python.

    En v1 destilábamos el nucleus (prototipos fijos).
    En v2 destilamos los slots del ResonanceBuffer —
    el estado final de la memoria emergente al terminar training.

    Esto es diferente: no es el DNA del modelo, es su último estado
    de flujo. Una fotografía del medio en su momento final.
    """
    slots = model.resonance.slots.detach().float().cpu().numpy()  # (K, D)
    K, D  = slots.shape

    # SVD para compresión eficiente
    U, S, Vt = np.linalg.svd(slots, full_matrices=False)
    top_k = min(16, K)
    U_k  = U[:, :top_k]
    S_k  = S[:top_k]
    Vt_k = Vt[:top_k, :]
    var_explained = float((S_k**2).sum() / (S**2).sum())

    filled = bool(model.resonance.fill_count.item() > 0)
    ptr    = int(model.resonance.slot_ptr.item())

    lines = [
        "# ResonanceBuffer destilado — snapshot del flujo final",
        f"# Slots: {K}×{D}  |  EMA decay: {model.resonance.ema_decay}",
        f"# Buffer filled: {filled}  |  Last ptr: {ptr}",
        f"# Varianza explicada con top_{top_k}: {var_explained:.4f}",
        "import numpy as np",
        "",
        f"RESONANCE_N_SLOTS  = {K}",
        f"RESONANCE_DIM      = {D}",
        f"RESONANCE_TOP_K    = {top_k}",
        "",
        "# Bases principales del flujo (top_k × D)",
        f"RESONANCE_BASES  = np.array({Vt_k.tolist()}, dtype=np.float32)",
        f"RESONANCE_SCALES = np.array({S_k.tolist()}, dtype=np.float32)",
        f"RESONANCE_COEFS  = np.array({U_k.tolist()}, dtype=np.float32)",
        "",
        "def reconstruct_slots():",
        '    """Reconstruye los slots de resonancia desde constantes destiladas."""',
        "    return (RESONANCE_COEFS * RESONANCE_SCALES[None, :]) @ RESONANCE_BASES",
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

    def log0(msg: str, console: bool = True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

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
    # Los graph breaks del AxonProjector (quarks con .item()) cuestan
    # proporcionalmente más en GPU pequeña. Además mode="reduce-overhead"
    # con CUDA graphs es incompatible con el Rotary cache → usamos mode="default".
    # Se puede forzar con USE_COMPILE=1 si quieres experimentar.
    use_compile = os.environ.get("USE_COMPILE", "0") == "1"
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
    log0(f"Dopamine v2.1 — Consciousness as Medium")
    log0(f"params: {n_params:,}  |  layers: {args.num_layers}  |  dim: {args.model_dim}")
    log0(f"heads: {args.num_heads}Q / {args.num_kv_heads}KV  |  mlp_mult: {args.mlp_mult}")
    log0(f"vocab: {args.vocab_size}  |  weight_tying: True  |  RoPE: True")
    log0(f"DendriticEncoder: field_heads={args.dendrite_field_heads}  "
         f"local={args.dendrite_local_span}  mid={args.dendrite_mid_span}  n_proto=16")
    log0(f"ResonanceBuffer: {args.resonance_slots} slots × {args.model_dim}d  "
         f"EMA_decay={args.resonance_ema_decay}  mode={args.resonance_mode}"
         f"{'  span='+str(args.resonance_mem_span) if args.resonance_mode=='seq' else ''}  "
         f"[no trainable params]")
    log0(f"PerceptionAgent: {args.perception_heads} heads  "
         f"cross-attn(Dendritic × Resonance)")
    log0(f"SomaCore: {args.num_layers} blocks — pure medium, no nucleus")
    log0(f"AxonProjector: n_views={args.axon_n_views}  "
         f"λ={args.axon_lambda}  stress_θ={args.axon_stress_threshold}")
    log0(f"seed: {args.seed}  |  world_size: {world_size}  |  grad_accum: {grad_accum}")

    # ── Optimizer groups ──
    # DendriticEncoder matrices → Muon
    dendrite_matrix = [
        p for n, p in base_model.dendrite.named_parameters()
        if p.ndim == 2 and 'prototypes' not in n
    ]
    prototype_params = [
        p for n, p in base_model.dendrite.named_parameters()
        if 'prototypes' in n
    ]
    # [NEW] PerceptionAgent matrices → Muon
    perception_matrix = [
        p for n, p in base_model.perception.named_parameters()
        if p.ndim == 2
    ]
    # SomaBlock matrices → Muon
    block_matrix = [
        p for n, p in base_model.blocks.named_parameters()
        if p.ndim == 2 and "scale" not in n
    ]
    # MoleculeGate matrices → Muon
    gate_matrix = [
        p for n, p in base_model.molecule_gate.named_parameters()
        if p.ndim == 2
    ]
    matrix_params = (dendrite_matrix + prototype_params + perception_matrix
                     + block_matrix + gate_matrix)

    # Scalars → AdamW
    log_beta_params = [
        p for n, p in base_model.dendrite.named_parameters()
        if 'log_beta' in n
    ]
    resid_mix_params = [
        p for n, p in base_model.blocks.named_parameters()
        if 'resid_mix' in n
    ]
    q_gain_params = [
        p for n, p in base_model.blocks.named_parameters()
        if 'q_gain' in n
    ]
    # [NEW] PerceptionAgent gate → Adam scalar
    percept_gate_params = [base_model.perception.percept_gate]

    scalar_params = (
        [p for n, p in base_model.blocks.named_parameters()
         if (p.ndim < 2 or "scale" in n)
         and 'resid_mix' not in n
         and 'q_gain' not in n]
        + resid_mix_params
        + q_gain_params
        + [base_model.skip_weights]
        + [base_model.axon.conductance_base]
        + [base_model.axon.rho_k]
        + [base_model.dendrite.field_mix]
        + log_beta_params
        + percept_gate_params
    )

    # [OPT-1] Lion for embeddings
    opt_emb = Lion(
        [{"params": [base_model.tok_emb.weight], "lr": args.lion_lr}],
        betas=(args.lion_beta1, args.lion_beta2),
        weight_decay=args.lion_wd,
    )

    # [OPT-2] Muon for matrices
    opt_muon = Muon(matrix_params, lr=args.matrix_lr,
                    momentum=args.muon_momentum_warmup_start,
                    backend_steps=args.muon_backend_steps)

    # [OPT-5] AdamW for scalars
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.scalar_wd,
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
        )
        log0(f"TTT-Iso: ENABLED  lr={args.ttt_lr}  K={args.ttt_inner_steps}"
             f"  β={args.ttt_beta}  reset={args.ttt_reset_each_chunk}"
             f"  warmup_chunks={args.ttt_warmup_chunks}")
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

            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum,
                val_tokens, bb_lut, ls_lut, bd_lut,
                ttt_trainer=ttt_trainer, log_fn=log0)

            # [NEW] Diagnósticos de resonancia y percepción
            resonance_filled = base_model.resonance.fill_count.item() > 0
            resonance_norm   = base_model.resonance.slots.float().norm(dim=-1).mean().item()
            percept_gate_act = torch.sigmoid(
                base_model.perception.percept_gate).mean().item()

            axon_info = (f"  axon_S:{base_model.axon.last_S:.3f}"
                         f"  mask_μ:{base_model.axon.last_mask_mean:.3f}"
                         f"  ρ:{base_model.axon.last_rho:.3f}"
                         f"  k:{base_model.axon.rho_k.item():.2f}")
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

        for o in optimizers: o.step()
        zero_grad_all()

        if master and args.train_log_every > 0 and step % args.train_log_every == 0:
            torch.cuda.synchronize()
            elapsed = training_ms + 1000.0*(time.perf_counter()-t0)
            e_s   = elapsed / 1000.0
            e_min = int(e_s // 60)
            e_sec = e_s % 60
            log0(f"step:{step}  loss:{loss.item():.4f}  "
                 f"lr_mul:{mul:.4f}  time:{e_min}m{e_sec:.1f}s")

        step += 1

    # ── FINAL ──
    torch.cuda.synchronize()
    training_ms += 1000.0 * (time.perf_counter() - t0)
    total_s   = training_ms / 1000.0
    total_min = int(total_s // 60)
    total_sec = total_s % 60
    log0(f"\n── Training complete ──")
    log0(f"  Total time : {total_min}m {total_sec:.1f}s  ({total_s:.1f}s)")
    log0(f"  Steps done : {step}")
    log0(f"  ms/step    : {training_ms / max(step, 1):.1f}ms")

    if master:
        size_info = quantize_and_compress(base_model, __file__)
        log0(f"\n── Artifact Size ──")
        log0(f"  model compressed : {size_info['model_compressed_kb']:.1f} KB")
        log0(f"  code compressed  : {size_info['code_compressed_kb']:.1f} KB")
        log0(f"  TOTAL            : {size_info['total_kb']:.1f} KB")
        log0(f"  fits 16MB        : {size_info['fits_16mb']}")
        log0(f"  fits 14MB        : {size_info['fits_14mb']}")
        log0(f"  SVD tensors      : {size_info['n_svd_tensors']}")
        log0(f"  int8 tensors     : {size_info['n_int8_tensors']}")

        log0(f"\n── Axon + Molecule diagnostics ──")
        log0(f"  last_S           : {base_model.axon.last_S:.4f}")
        log0(f"  last_mask_mean   : {base_model.axon.last_mask_mean:.4f}")
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
