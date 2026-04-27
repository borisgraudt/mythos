# Training Guide — Mythos 500M

End-to-end recipe from a fresh A100 instance to a published HuggingFace release
with paper-grade monitoring (Weights & Biases, MFU, grad-norm, throughput).

---

## TL;DR — time and cost

| Run | Hardware | Tokens | Wall-clock | Cost |
|---|---|---:|---:|---:|
| Debug (150 M) | MacBook M2 | 21 M | ~4 h | $0 |
| **Production (500 M)** | **RunPod A100 40 GB** | **~1.6 B** | **~10 h** | **~$25** |
| Ablation (150 M on medium) | A100 40 GB | ~1.6 B | ~3 h | ~$7 |

A100 at ~40 K tok/s, bf16, proper gradient accumulation (bs=4 × accum=8 × seq=512).

---

## 1. Provision the cloud box (5 min)

**RunPod** → *Deploy* → *Pods* → pick `NVIDIA A100 80GB` or `40GB`:

- Template: **PyTorch 2.5 + CUDA 12.1** (community template)
- Disk: **100 GB volume**
- Expose SSH + Jupyter
- Estimated: $1.89 / hr (40 GB Community) – $2.89 / hr (80 GB Secure)

SSH in and verify GPU:

```bash
nvidia-smi                     # confirm A100 + >38 GB free
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 2. Clone and install (2 min)

```bash
git clone https://github.com/borisgraudt/mythos
cd mythos
pip install -e ".[dev]"         # installs wandb, transformers, etc.
```

Log into the three services you'll need:

```bash
huggingface-cli login           # for final upload
wandb login                     # paste key from https://wandb.ai/authorize
```

---

## 3. Prepare the medium dataset on the cloud box (~30 min)

```bash
python scripts/prepare_data.py --medium
```

Produces `data/medium/` with:
- `raw/wikipedia/` — 100 K articles, ~400 MB
- `tokenizer/` — 32 K-vocab BPE
- `encoded/` — binary shards, ~1.5 B tokens
- `splits/{train,val,test}.txt`

---

## 4. Launch training with full monitoring

```bash
python scripts/train.py \
    --mode full \
    --model configs/model/base_500m.yaml \
    --training configs/training/base.yaml \
    --data data/medium \
    --wandb \
    --wandb_project mythos \
    --run_name 500m_run1
```

### What you'll see

**Terminal (tqdm progress bar):**

```
Training:  12%| 12000/100000 [1:14:03<9:03:00, 2.45s/it]
          loss=3.8124 lr=2.88e-04 gnorm=0.87 tok/s=43,120 mfu=31.4%
```

**W&B dashboard** (link printed on start) with live plots:
- `train/loss`, `train/lr`, `train/grad_norm`
- `val/loss`, `val/perplexity` (every 500 steps)
- `throughput/tok_per_s`, `throughput/mfu`
- `progress/tokens`, `progress/step`

These are the **exact curves you need for the paper**.

---

## 5. Expected loss trajectory

Use these as sanity checks. If you diverge → lower `learning_rate` to 2e-4.

| Step | Train loss | Val PPL | Tokens seen |
|---:|---:|---:|---:|
| 500 | ~7.8 | — | 8 M |
| 5 000 | ~4.2 | ~70 | 82 M |
| 25 000 | ~3.3 | ~30 | 410 M |
| 60 000 | ~2.9 | ~20 | 983 M |
| 100 000 | ~2.7 | ~15 | 1.64 B |

**Target:** val perplexity < 20 on Wikipedia val split. If you hit that, you have
a publishable result for a "from-scratch 500M on 1.6 B tokens" paper.

---

## 6. Evaluate

```bash
python scripts/eval.py \
    --checkpoint checkpoints/base_500m_500m_run1/final.pt \
    --data data/medium --split val \
    --seq_len 512 --batch_size 8
```

For paper: also run on **external benchmarks** — WikiText-2, LAMBADA:

```bash
# (Requires manual download — see docs/RESEARCH.md)
python scripts/eval.py --checkpoint <ckpt> --text wikitext-2-raw/wiki.test.raw \
    --tokenizer data/medium/tokenizer/tokenizer.json
```

---

## 7. Release to HuggingFace

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/base_500m_500m_run1/final.pt \
    --tokenizer data/medium/tokenizer/tokenizer.json \
    --repo bgraudt/mythos
```

Uploads LLaMA-compatible weights. After a minute, `bgraudt/mythos` page shows:
- "Try it" inference widget (with HF Pro / Inference API)
- Native `AutoModelForCausalLM.from_pretrained("bgraudt/mythos")`
- Works with vLLM, TGI, llama.cpp, Ollama

Also upload GGUF for Ollama users:

```bash
python scripts/export_gguf.py \
    --checkpoint checkpoints/base_500m_500m_run1/final.pt \
    --tokenizer data/medium/tokenizer/tokenizer.json \
    --quantize q4_k_m
```

---

## 8. What you now have for a paper

Everything the run produces feeds directly into paper sections:

| Paper section | Source |
|---|---|
| §3 Architecture | `docs/ARCHITECTURE.md` + `src/core/` |
| §4 Training data | `docs/RESEARCH.md` + `data/pipelines/` |
| §5 Optimization | W&B `train/lr`, `train/grad_norm` curves |
| §6 Compute efficiency | W&B `throughput/mfu`, `throughput/tok_per_s` |
| §7 Scaling | W&B `progress/tokens` vs `val/perplexity` |
| §8 Results | `eval.py` outputs on WikiText-2 / LAMBADA |
| Appendix A — Hyperparameters | `checkpoints/.../training_config.json` |
| Appendix B — Reproducibility | `run_summary.json` + this guide |

Export W&B run as PDF (W&B UI → Share → Export) → drop straight into `docs/`.

---

## 9. Troubleshooting

**OOM on A100 40 GB**
Lower `batch_size` to 2, raise `gradient_accumulation` to 16 — same effective batch, half the activation memory.

**MFU < 20 %**
Check that `bfloat16: true` in `configs/training/base.yaml` and that `torch.compile` isn't disabled. On A100 you should see 30–45 %.

**Loss spikes at step 2 000**
That's the warmup-peak transition. If it recovers within 500 steps, ignore. If it diverges, restart with `learning_rate: 2.0e-4`.

**wandb disabled due to sign-in failure**
Run `wandb login --relogin` or pass `WANDB_MODE=offline` to record locally and sync later.

---

## 10. Post-run checklist

- [ ] A100 instance is stopped (don't keep paying)
- [ ] `checkpoints/base_500m_*/final.pt` copied off the pod (use `rsync`)
- [ ] W&B run is public or shared with collaborators
- [ ] `bgraudt/mythos` HF repo live
- [ ] `docs/results.md` updated with final numbers
- [ ] Loss-curve PNG exported from W&B → `docs/assets/loss_curve.png`
- [ ] `run_summary.json` committed
