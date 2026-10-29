# EP65 — Mixed Precision Training & Memory Optimization

**Deep Learning Series · Episode 65 · Module 14 — Scaling Deep Learning**

> “Half the memory, twice as fast.”  
> We measured what is really true.

## What this episode measures

- **FP32 / FP16 / BF16** bit layouts and real number casts
- **Real gradients** of a trained network cast to FP16 at three loss scales
- **The real PyTorch GradScaler** (scale trace + skipped steps)
- **Five training precisions** × two learning rates × 3 seeds
- **State memory** (weights + gradients + Adam) measured after one real step
- **Activation memory** measured with `saved_tensors_hooks` (fp32 vs autocast vs checkpointing)
- **Step times** on this machine (and real peak memory + speed on CUDA GPU if available)

No number in the video is typed by hand. Everything comes from `ep65_results.json`.

## Quick start

```bash
pip install torch numpy matplotlib
python ep65.py
