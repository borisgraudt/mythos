"""
loop.py — Main training loop.

Paper-grade training instrumentation:
  - proper gradient accumulation (one optimizer step per `accum_steps` micro-batches)
  - gradient-norm logging (training-stability analysis)
  - throughput in tokens/sec
  - Model FLOPs Utilization (MFU) estimate (efficiency analysis)
  - tokens-seen counter (for Chinchilla-style scaling curves)
  - optional Weights & Biases integration
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .trainer import Trainer

logger = logging.getLogger(__name__)


# ─── FLOPs / MFU helpers ────────────────────────────────────────────────────

def _model_flops_per_token(model) -> float:
    """Approximate forward+backward FLOPs per training token.
    Standard 6·N rule (Kaplan 2020 / Chinchilla). Over-counts slightly for
    GQA / tied-embeddings but is the community default for MFU reporting."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 6.0 * n_params


_GPU_PEAK_TFLOPS = {
    "A100": 312.0,      # bf16 tensor cores
    "H100": 989.0,
    "H200": 989.0,
    "V100": 125.0,
    "RTX 4090": 165.0,
    "RTX 3090": 142.0,
}

def _detect_gpu_peak_tflops(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    for key, tflops in _GPU_PEAK_TFLOPS.items():
        if key in name:
            return tflops
    return None


# ─── Main loop ──────────────────────────────────────────────────────────────

def training_loop(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    max_steps: int,
    save_dir: Path,
    save_every: int = 1000,
    eval_every: int = 500,
    log_every: int = 50,
    resume_step: int = 0,
    accum_steps: int = 1,
    wandb_run=None,
) -> dict:
    """Run `max_steps` optimizer updates with `accum_steps` gradient accumulation
    micro-batches each. Logs loss, LR, grad-norm, tokens/sec, and MFU."""

    logger.info("Starting training for %d optimizer steps "
                "(×%d grad-accum = %d micro-batches)",
                max_steps, accum_steps, max_steps * accum_steps)
    logger.info("Device: %s", trainer.device)
    logger.info("Mixed precision (bfloat16): %s", trainer.bfloat16)

    # FLOPs / MFU setup
    flops_per_token = _model_flops_per_token(trainer.model)
    peak_tflops     = _detect_gpu_peak_tflops(trainer.device)
    if peak_tflops:
        logger.info("GPU peak bf16: %.0f TFLOPs — MFU will be reported", peak_tflops)

    train_losses: list[float] = []
    tokens_seen = 0
    start_time  = time.time()
    last_log_t  = start_time
    last_log_tokens = 0

    pbar = tqdm(total=max_steps, initial=resume_step, desc="Training", ncols=110)
    train_iter = iter(train_loader)

    save_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = save_dir / "train_log.jsonl"
    jsonl_f = jsonl_path.open("a", buffering=1)

    for step in range(resume_step, max_steps):
        step_loss = 0.0

        # ── Gradient accumulation micro-batches ─────────────────────────────
        for micro in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            step_loss    += trainer.train_step(batch)
            tokens_seen  += batch["input_ids"].numel() * accum_steps

        step_loss /= accum_steps

        # ── Optimizer step (once per `accum_steps` micro-batches) ───────────
        grad_norm = trainer.optimizer_step()
        train_losses.append(step_loss)

        # ── Logging ─────────────────────────────────────────────────────────
        if (step + 1) % log_every == 0:
            now = time.time()
            dt = now - last_log_t
            recent_tokens = tokens_seen - last_log_tokens
            tok_per_s = recent_tokens / max(dt, 1e-6)

            mfu = None
            if peak_tflops:
                achieved_tflops = tok_per_s * flops_per_token / 1e12
                mfu = achieved_tflops / peak_tflops

            lr = trainer.scheduler.get_last_lr()[0]
            postfix = {
                "loss": f"{step_loss:.4f}",
                "lr":   f"{lr:.2e}",
                "gnorm": f"{grad_norm:.2f}",
                "tok/s": f"{tok_per_s:,.0f}",
            }
            if mfu is not None:
                postfix["mfu"] = f"{mfu:.1%}"
            pbar.set_postfix(postfix)

            if wandb_run is not None:
                metrics = {
                    "train/loss":       step_loss,
                    "train/lr":         lr,
                    "train/grad_norm":  grad_norm,
                    "throughput/tok_per_s": tok_per_s,
                    "progress/tokens":  tokens_seen,
                    "progress/step":    step + 1,
                }
                if mfu is not None:
                    metrics["throughput/mfu"] = mfu
                wandb_run.log(metrics, step=step + 1)

            last_log_t = now
            last_log_tokens = tokens_seen

        # ── Evaluation ──────────────────────────────────────────────────────
        if val_loader is not None and (step + 1) % eval_every == 0:
            val_loss = trainer.evaluate(val_loader)
            val_ppl  = float(torch.tensor(val_loss).exp())
            logger.info("Step %d | val_loss: %.4f | val_ppl: %.2f",
                        step + 1, val_loss, val_ppl)
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss, "val/perplexity": val_ppl},
                              step=step + 1)
            jsonl_f.write(json.dumps({
                "step": step + 1,
                "tokens_seen": tokens_seen,
                "train_loss": step_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }) + "\n")
            trainer.maybe_save_checkpoint(val_loss, save_dir, save_every)
        elif (step + 1) % save_every == 0:
            avg_loss = sum(train_losses[-save_every:]) / min(len(train_losses), save_every)
            logger.info("Step %d | train_loss: %.4f | tokens: %.2fM",
                        step + 1, avg_loss, tokens_seen / 1e6)
            trainer.maybe_save_checkpoint(avg_loss, save_dir, save_every)

        pbar.update(1)

    pbar.close()
    jsonl_f.close()

    # ── Final checkpoint ────────────────────────────────────────────────────
    final_loss = val_loader and trainer.evaluate(val_loader) or float("inf")
    trainer.maybe_save_checkpoint(final_loss, save_dir, save_every, is_final=True)

    elapsed = time.time() - start_time
    logger.info("Training complete in %.2f hours", elapsed / 3600)
    logger.info("Total tokens: %.2fB", tokens_seen / 1e9)
    logger.info("Avg throughput: %.0f tok/s", tokens_seen / max(elapsed, 1))

    # Dump run summary (for papers)
    summary = {
        "elapsed_hours":    elapsed / 3600,
        "total_tokens":     tokens_seen,
        "final_step":       max_steps,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss":   final_loss if val_loader else None,
        "avg_tok_per_s":    tokens_seen / max(elapsed, 1),
        "accum_steps":      accum_steps,
    }
    (save_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    if wandb_run is not None:
        wandb_run.summary.update(summary)

    return {"train_losses": train_losses, **summary}
