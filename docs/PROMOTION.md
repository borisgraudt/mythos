# Promotion materials

Ready to post. Upload `docs/figures/scaling.png` as the image in Reddit and Twitter.

---

## Reddit post

**Subreddit:** r/MachineLearning (first), then r/LocalLLaMA a few days later

**Title:**
> I built a transformer LLM series from scratch and partially reproduced Chinchilla scaling laws on a MacBook Air M3

**Body:**

Hey everyone,

I spent the last few months building Mythos — a decoder-only LLM series trained entirely from scratch on a MacBook Air M3 (16 GB). No Hugging Face Trainer, no nanoGPT base, no black-box dependencies. Every component hand-rolled: GQA attention, RoPE, SwiGLU, RMSNorm, BPE tokenizer, training loop with MFU tracking, and a full HF/GGUF export pipeline.

**The research part:** instead of just training one model, I ran a scaling-laws study across two model sizes (10M and 30M parameters), each trained to near Chinchilla-optimal compute (D ≈ 20·N tokens). Then fit the Hoffmann et al. parametric form from scratch:

```
L(N, D) = E + A·N^(-α) + B·D^(-β)
```

**Results:**
- Fitted α = 0.329 (Chinchilla reference: 0.34) — **3% deviation**, model-size scaling reproduces well
- Fitted β = 2.91 (Chinchilla reference: 0.28) — unreliable: the 10M run only reached ~45% of its token budget, so the data-scaling dimension is under-sampled
- E = 4.01 (estimated irreducible loss on this data mixture)

The honest takeaway: α is correct, β needs a third point (80M) to constrain properly. That's a GPU run I'll do later.

[INSERT docs/figures/scaling.png]

**Models:**
| Model | Params | Tokens | Val Loss | Val PPL |
|-------|--------|--------|----------|---------|
| mythos-10m | 12.7M | 90M | 5.15 | 172 |
| mythos-30m | 24.9M | 786M | 4.84 | 127 |

**Stack:** PyTorch 2.5, Apple MPS (bfloat16), FineWeb-Edu + GitHub code, HuggingFace tokenizers.

**Links:**
- GitHub: https://github.com/borisgraudt/mythos
- HuggingFace: https://huggingface.co/bgraudt/mythos (1.6k+ downloads)
- Scaling study: https://github.com/borisgraudt/mythos/blob/main/docs/SCALING.md

Happy to answer questions about the implementation or the scaling setup.

---

## Twitter / X thread

**Tweet 1 (hook):**
I built a LLM series from scratch on a MacBook Air M3 and ran a scaling-laws study.

Fitted α = 0.329 vs Chinchilla's 0.34. Here's what I built and what I found 🧵

**Tweet 2:**
Mythos: decoder-only transformer series (10M → 30M params), built from scratch.

Architecture: GQA + SwiGLU + RoPE + RMSNorm — same decisions as LLaMA 3, implemented and understood line by line. No Trainer API. No nanoGPT.

**Tweet 3:**
The goal was to reproduce the Chinchilla scaling law:

L(N, D) = E + A·N^-α + B·D^-β

Two model sizes, each trained to compute-optimal token budget (D ≈ 20·N). Then fit the curve from scratch on a consumer laptop.

**Tweet 4 (results + graph):**
Results:

α = 0.329 (Chinchilla: 0.34) ✓ — 3% off, model-size scaling is correct
β = 2.91 (Chinchilla: 0.28) ✗ — data-scaling unreliable, 10M run was undertrained

Honest science: 2 points aren't enough to constrain β. Need a 3rd (80M) run.

[ATTACH docs/figures/scaling.png]

**Tweet 5:**
Both runs on MacBook Air M3, 16GB unified memory.

10M: 75 hrs wall-clock (mostly eval overhead — fixed mid-run)
30M: 15.7 hrs, 786M tokens, val loss 4.84

Total GPU cost: $0.

**Tweet 6:**
Full pipeline: raw data → BPE tokenizer → training loop → eval → HF export (LlamaForCausalLM) → GGUF for Ollama.

39 tests, CI, Gradio demo, technical report.

GitHub: https://github.com/borisgraudt/mythos
HF: https://huggingface.co/bgraudt/mythos

**Tweet 7:**
If you're learning how LLMs work under the hood, the repo has design rationale for every architectural decision.

⭐ appreciated — helps people find it.

Scaling writeup: https://github.com/borisgraudt/mythos/blob/main/docs/SCALING.md

---

## Posting order

1. Push updated README + scaling.png to GitHub
2. Upload 30M checkpoint to HF (`scaling-30m` branch)
3. Post Twitter thread
4. Post Reddit 24–48h later (GitHub has stars from Twitter by then)

**Tag in Twitter:** @karpathy, @_akhaliq — they occasionally boost student ML projects with real empirical results.

---

## HN (optional, higher risk/reward)

**Title:** Show HN: I reproduced Chinchilla scaling laws from scratch on a MacBook Air

Only post if the Reddit thread does well first. HN audience is technical and will ask hard questions about β being off — be ready to explain the undertraining of the 10M run.
