"""Fit Chinchilla-style scaling law L(N,D) = E + A*N^-alpha + B*D^-beta.

Consumes per-run JSONL logs of (step, tokens_seen, val_loss) and per-run
model_config.json (to count non-embedding params N), fits the parametric
form via L-BFGS in log-space, and writes a diagnostic plot.

Usage:
    python scripts/fit_scaling_laws.py \
        --runs checkpoints/scaling_10m checkpoints/scaling_30m checkpoints/scaling_80m \
        --out docs/figures/scaling.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def count_non_embedding_params(cfg: dict) -> int:
    d = cfg["d_model"]
    L = cfg["n_layers"]
    n_h = cfg["n_heads"]
    n_kv = cfg.get("n_kv_heads", n_h)
    d_ff = cfg["d_ff"]
    head_dim = d // n_h
    attn = d * (n_h * head_dim) + 2 * d * (n_kv * head_dim) + (n_h * head_dim) * d
    ffn = 3 * d * d_ff
    norms = 2 * d
    return L * (attn + ffn + norms) + d  # + final norm


def load_run(run_dir: Path) -> tuple[int, np.ndarray, np.ndarray]:
    cfg_path = run_dir / "model_config.json"
    log_path = run_dir / "train_log.jsonl"
    if not cfg_path.exists() or not log_path.exists():
        raise FileNotFoundError(f"missing model_config.json or train_log.jsonl in {run_dir}")
    cfg = json.loads(cfg_path.read_text())
    N = count_non_embedding_params(cfg)
    tokens, losses = [], []
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        if "val_loss" in rec and "tokens_seen" in rec:
            tokens.append(rec["tokens_seen"])
            losses.append(rec["val_loss"])
    return N, np.array(tokens, dtype=np.float64), np.array(losses, dtype=np.float64)


def fit(points: list[tuple[float, float, float]]) -> dict:
    N = np.array([p[0] for p in points])
    D = np.array([p[1] for p in points])
    L = np.array([p[2] for p in points])

    def loss_fn(theta):
        log_E, log_A, log_B, alpha, beta = theta
        pred = np.exp(log_E) + np.exp(log_A) * N ** (-alpha) + np.exp(log_B) * D ** (-beta)
        return float(np.sum((np.log(pred) - np.log(L)) ** 2))

    x0 = np.array([np.log(1.5), np.log(400.0), np.log(400.0), 0.34, 0.28])
    res = minimize(loss_fn, x0, method="L-BFGS-B")
    log_E, log_A, log_B, alpha, beta = res.x
    return {
        "E": float(np.exp(log_E)),
        "A": float(np.exp(log_A)),
        "B": float(np.exp(log_B)),
        "alpha": float(alpha),
        "beta": float(beta),
        "loss": float(res.fun),
        "n_points": len(points),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/scaling.png"))
    args = ap.parse_args()

    points = []
    per_run = []
    for run in args.runs:
        N, D, L = load_run(run)
        per_run.append((run.name, N, D, L))
        for d, l in zip(D, L):
            if d > 0 and np.isfinite(l):
                points.append((float(N), float(d), float(l)))

    if len(points) < 5:
        raise SystemExit(f"need at least 5 (N,D,L) points to fit; got {len(points)}")

    fit_result = fit(points)
    print("Fitted scaling law:")
    print(f"  L(N,D) = {fit_result['E']:.3f} + {fit_result['A']:.1f}*N^-{fit_result['alpha']:.3f}"
          f" + {fit_result['B']:.1f}*D^-{fit_result['beta']:.3f}")
    print(f"  Chinchilla reference: alpha=0.34, beta=0.28")
    print(f"  Fit log-MSE: {fit_result['loss']:.4f} over {fit_result['n_points']} points")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, N, D, L in per_run:
        ax.plot(D, L, marker="o", ms=3, label=f"{name} (N={N/1e6:.0f}M)")
        E, A, B, a, b = (fit_result[k] for k in ("E", "A", "B", "alpha", "beta"))
        D_grid = np.geomspace(D.min(), D.max(), 100)
        ax.plot(D_grid, E + A * N ** (-a) + B * D_grid ** (-b), "--", alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Tokens seen (D)"); ax.set_ylabel("Val loss")
    ax.set_title(f"Scaling fit: alpha={fit_result['alpha']:.3f}, beta={fit_result['beta']:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
