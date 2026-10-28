#!/usr/bin/env python3
"""Structural and numerical smoke tests for ACH v0.3-geometric."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_args(m, geometric: bool):
    a = m.Hyperparameters()
    a.vocab_size = 64
    a.num_layers = 5
    a.model_dim = 64
    a.num_heads = 4
    a.num_kv_heads = 2
    a.mlp_mult = 2
    a.dendrite_field_heads = [4, 2, 2]
    a.dendrite_local_span = 8
    a.dendrite_mid_span = 16
    a.resonance_slots = 4
    a.resonance_ema_decay = 0.9
    a.resonance_mode = "seq"
    a.resonance_mem_span = 2
    a.perception_heads = 4
    a.ach_enabled = True
    a.ach_theta = 0.5
    a.ach_rho_k = 2.0
    a.ach_lambda = 0.12
    a.ach_base_scale = 0.10
    a.ach_first_block = 1
    a.observer_enabled = True
    a.observer_sample_tokens = 32
    a.train_seq_len = 8
    a.val_batch_size = 16
    a.ttt_enabled = 0
    a.geometric_dendrite_enabled = geometric
    a.geo_center_decay = 0.9
    a.geo_axis_init_cos = 0.35
    a.geo_gamma_init = 0.08
    a.geo_metric_max = 0.30
    a.geo_transport_max = 0.12
    a.geo_tau_max = 0.35
    a.geo_quantize = True
    a.geo_quant_step_init = 0.25
    a.geo_axis_max_abs_cos = 0.92
    a.geo_cross_field_target = 0.10
    a.geo_aux_scale = 0.01
    a.geo_eps = 1e-5
    return a


def finite_positive(x: float) -> bool:
    return bool(torch.isfinite(torch.tensor(x))) and x > 0.0


def main() -> int:
    root = Path(__file__).resolve().parent
    geo = load_module(root / "ach_v0_3_geometric.py", "ach_geo")
    # v0.2.1 is the frozen baseline: geometry OFF must reproduce it bit for bit.
    legacy_path = Path(os.environ.get(
        "ACH_V021_PATH", str(root / "legacy" / "ach_v0_2_1_persistent.py")))
    legacy = load_module(legacy_path, "ach_legacy")

    torch.manual_seed(17)
    os.environ["PFV_MONITOR"] = "1"

    # 1) Exact A/B route: legacy weights + geometry OFF must be bit-identical.
    legacy_model = legacy.Dopamine(small_args(legacy, geometric=False)).float().eval()
    geo_off = geo.Dopamine(small_args(geo, geometric=False)).float().eval()
    incompat = geo_off.load_state_dict(legacy_model.state_dict(), strict=False)
    assert not incompat.unexpected_keys
    assert incompat.missing_keys and all(
        k.startswith("dendrite.geometry.") for k in incompat.missing_keys)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        loss_legacy = legacy_model(x, y)
        loss_geo_off = geo_off(x, y)
    assert torch.equal(loss_legacy, loss_geo_off), (
        f"geometry OFF changed baseline: {loss_legacy} vs {loss_geo_off}")

    # 2) Frame invariant: regularized Gram matrices stay finite/nondegenerate.
    frame = geo.GeometricFrameBank(
        dim=16, head_dim=4, enabled=True, center_decay=0.9,
        axis_init_cos=0.35, gamma_init=0.08, metric_max=0.3,
        transport_max=0.12, tau_max=1.0, quantize=True,
        quant_step_init=0.25, axis_max_abs_cos=0.92,
        cross_field_target=0.10, eps=1e-5,
    ).float()
    snap = frame.snapshot()
    assert all(v > 0.05 for v in snap["gram_det"]), snap

    # 3) Projection changes only the L/R plane; orthogonal complement is preserved.
    frame.center.zero_()
    frame.center_initialized.fill_(True)
    h = torch.randn(3, 5, 16)
    projected, residual = frame.project(h, 0)
    plane_h, _ = frame._dual_projection(h, 0, head_space=False)
    plane_p, _ = frame._dual_projection(projected, 0, head_space=False)
    orth_h = h - plane_h
    orth_p = projected - plane_p
    assert torch.allclose(orth_h, orth_p, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(residual).all()

    # 4) Geodesic endpoints and finite interior.
    frame.eta_raw.data.fill_(20.0)
    out0, tau0 = frame.geodesic_transport(h, projected, 0, torch.zeros(3, 5))
    out1, tau1 = frame.geodesic_transport(h, projected, 0, torch.ones(3, 5))
    assert torch.equal(out0, h)
    assert torch.allclose(out1, projected, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(out1).all() and torch.isfinite(tau0).all() and torch.isfinite(tau1).all()

    # 5) Empirical calibration accepts non-collinear fields and rejects collapse.
    neutral = torch.randn(12, 3, 16) * 0.05
    left = neutral + 0.4 * torch.nn.functional.normalize(
        torch.randn(1, 3, 16), dim=-1)
    right_dir = torch.nn.functional.normalize(torch.randn(1, 3, 16), dim=-1)
    right = neutral + 0.4 * right_dir
    cal = frame.calibrate(neutral, left, right)
    assert all(cal["center_initialized"])
    assert all(v > 1e-3 for v in cal["gram_det"])

    # 6) End-to-end geometry ON: finite loss, telemetry, optimizer coverage,
    # and nonzero gradients in every active parameter family.
    torch.manual_seed(19)
    model = geo.Dopamine(small_args(geo, geometric=True)).float().train()
    model.arm_observer(step=1, phase="geo-smoke")
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    loss.backward()
    assert finite_positive(float(loss.detach()))

    _, _, _, names = geo.build_optimizer_groups(model)
    all_names = set(names["matrix"]) | set(names["scalar"]) | set(names["embedding"])
    for required in (
        "dendrite.geometry.axes",
        "dendrite.geometry.head_axes",
        "dendrite.geometry.gamma_projector.weight",
        "dendrite.geometry.gamma_projector.bias",
        "dendrite.geometry.metric_raw",
        "dendrite.geometry.transport_raw",
        "dendrite.geometry.eta_raw",
        "dendrite.geometry.log_coord_scale",
    ):
        assert required in all_names, f"missing optimizer coverage: {required}"

    gradient_families = {
        "axes": model.dendrite.geometry.axes,
        "head_axes": model.dendrite.geometry.head_axes,
        "gamma_weight": model.dendrite.geometry.gamma_projector.weight,
        "metric": model.dendrite.geometry.metric_raw,
        "transport": model.dendrite.geometry.transport_raw,
        "eta": model.dendrite.geometry.eta_raw,
        "coord_scale": model.dendrite.geometry.log_coord_scale,
    }
    grad_norms = {}
    for name, param in gradient_families.items():
        assert param.grad is not None, f"no gradient: {name}"
        norm = float(param.grad.detach().float().norm())
        assert finite_positive(norm), f"invalid gradient {name}: {norm}"
        grad_norms[name] = norm

    obs = model.observer_snapshot(clear=True)
    keys = set(obs.get("tensors", {}))
    required_telemetry = {
        "geometry.gamma", "geometry.tau", "geometry.disagreement",
        "geometry.projection_residual", "geometry.axis_cos_lr",
        "geometry.gram_det", "geometry.aux_loss",
    }
    assert required_telemetry <= keys, sorted(required_telemetry - keys)

    # 7) Geometry snapshot is JSON-compatible and finite.
    snapshot = model.dendrite.geometry_snapshot()
    assert snapshot["enabled"] is True
    for key in ("axis_cos_LR", "gram_det", "metric_strength",
                "transport_strength", "eta", "gamma_mean", "tau_mean"):
        assert key in snapshot
        vals = torch.tensor(snapshot[key], dtype=torch.float32)
        assert torch.isfinite(vals).all(), (key, snapshot[key])

    print("OK — ACH v0.3-geometric")
    print(f"baseline exact: loss={loss_legacy.item():.8f}")
    print(f"geometry-on loss={loss.item():.8f}")
    print("geometry grad norms:")
    for name, value in grad_norms.items():
        print(f"  {name:16s} {value:.6e}")
    print("geometry snapshot:")
    print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
