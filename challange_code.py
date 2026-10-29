"""EP65 - Mixed precision and memory: compare FP32, FP16 and BF16 on real numbers, measure gradient underflow and the loss scaler, count the memory of AMP and of activation checkpointing, and check what is really faster on YOUR machine.
    python ep65_mixed_precision.py     # 3-6 minutes on a laptop CPU; writes the precision/ folder (dashboard.png) and ep65_results.json
pip install torch numpy matplotlib
# a feature this machine or this torch version lacks is skipped and reported, not faked. If you have a CUDA GPU, the script also measures real peak memory and speed on it.
"""
import json, os, platform, time, warnings
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
warnings.filterwarnings("ignore")
# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SEEDS = 3                            # every training number below is a mean over this many independent runs
STEPS, BATCH, WIDTH, DEPTH = 150, 128, 256, 6    # the training runs: steps, batch, width and depth of the network
LRS = {"normal": 1e-3, "small": 2e-5}            # Adam learning rates: a normal one, and a small one (small updates are what low precision loses)
MODES = ["fp32", "bf16_amp", "fp16_scaler", "fp16_noscaler", "bf16_pure"]    # AMP = automatic mixed precision; "pure" = the weights themselves are 16-bit
SCALE0, GROWTH = 2 ** 12, 15         # the loss scaler: starting scale, and the number of clean steps before it doubles (the PyTorch defaults are 2**16 and 2000)
LOG2_SCALES = [0, 8, 16]             # loss scales (as powers of two) used when we look at the gradients
ACT_WIDTH, ACT_DEPTH, ACT_BATCH = 512, 12, 2048  # the network whose memory and speed we measure: 12 blocks, a big batch
SEGMENTS = [1, 2, 3, 4, 6, 12]       # activation checkpointing: cut the 12 blocks into this many pieces
TIME_BATCH, REPS = 512, 3            # the speed test: batch, timed repeats (the median is used)
MM_N = 1024                          # the matrix-multiplication speed test: N x N matrices
OUT = Path(__file__).with_name("precision")      # dashboard.png ends up here
# ── 2. NUMBER FORMATS + DATA: what each format can hold, shown by casting real numbers ──
DT = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
def formats():
    out = {}
    for nm, dt in DT.items():
        f = torch.finfo(dt); mant = int(round(-np.log2(f.eps))); acc = torch.ones((), dtype=dt)
        for _ in range(1000): acc = acc + torch.tensor(1e-3, dtype=dt)                # add 0.001 a thousand times to 1.0: the answer should be 2.0
        out[nm] = dict(bits=f.bits, exp=f.bits - 1 - mant, mant=mant, max=float(f.max), tiny=float(f.tiny), sub=float(f.tiny * f.eps), eps=float(f.eps),
                       big=float(torch.tensor(70000.0).to(dt)), small=float(torch.tensor(1e-8).to(dt)), one=float(torch.tensor(1 + 1 / 256).to(dt)), acc=float(acc))
    return out
def data(seed, n=2048):              # ten noisy clusters in 64 dimensions: tiny on purpose, so every run takes seconds
    c = torch.randn(10, 64, generator=torch.Generator().manual_seed(123)); g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n + 1024,), generator=g); x = c[y] + 2.5 * torch.randn(n + 1024, 64, generator=g)
    return x[:n], y[:n], x[n:], y[n:]
def make(width=WIDTH, depth=DEPTH):
    torch.manual_seed(0); layers = [nn.Linear(64, width), nn.ReLU()]
    for _ in range(depth - 1): layers += [nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(width, 10))
def blocks(width=ACT_WIDTH, depth=ACT_DEPTH):    # the memory test network: `depth` blocks of Linear + ReLU
    torch.manual_seed(0); return [nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(depth)]
def nbytes(ts): return sum(t.numel() * t.element_size() for t in ts)
def ms(v): return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
def ac(mode, dev="cpu"):             # the autocast context of a mode: matmuls run in 16 bits, sums and losses stay in 32
    if mode in ("fp32", "bf16_pure"): return nullcontext()
    return torch.autocast(dev, dtype=torch.bfloat16 if mode.startswith("bf16") else torch.float16)
# ── 3. MEASURE: gradient underflow, memory (state + activations), checkpointing, speed ──
def train(mode, seed, lr):           # one training run; evaluation is always in fp32, so only the TRAINING precision differs
    x, y, xt, yt = data(seed); net = make(); g = torch.Generator().manual_seed(seed); trace = []; bad = False
    if mode == "bf16_pure": net = net.to(torch.bfloat16)
    opt = torch.optim.Adam(net.parameters(), lr=lr); sc = torch.amp.GradScaler("cpu", init_scale=SCALE0, growth_interval=GROWTH) if mode == "fp16_scaler" else None
    for s in range(STEPS):
        b = torch.randint(0, len(x), (BATCH,), generator=g); xb = x[b].to(torch.bfloat16) if mode == "bf16_pure" else x[b]; opt.zero_grad()
        with ac(mode): loss = nn.functional.cross_entropy(net(xb), y[b])
        bad = bad or not bool(torch.isfinite(loss))
        if sc: sc.scale(loss).backward(); sc.step(opt); sc.update(); trace.append(sc.get_scale())
        else: loss.backward(); opt.step()
    net = net.float().eval()
    with torch.no_grad(): lo, hi = net(xt), net(x); res = dict(loss=float(nn.functional.cross_entropy(lo, yt)), acc=float((lo.argmax(1) == yt).float().mean()), train_loss=float(nn.functional.cross_entropy(hi, y)))
    sk = int(sum(b < a for a, b in zip([SCALE0] + trace, trace))) if trace else 0     # a smaller scale = the step had inf or nan gradients and was skipped
    return net, dict(res, bad=bad, trace=trace, skipped=sk)
def underflow(net, seed):            # take the REAL gradients of a trained network and ask: what happens to them in 16 bits, at different loss scales?
    x, y, _, _ = data(seed); net.zero_grad(); nn.functional.cross_entropy(net(x[:512]), y[:512]).backward()
    g = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).abs(); g = g[g > 0]; rows = {}
    def look(h): return dict(zero=float((h == 0).float().mean()), sub=float(((h > 0) & (h < 2 ** -14)).float().mean()), inf=float(torch.isinf(h).float().mean()))
    for k in LOG2_SCALES: rows[str(k)] = look((g * 2.0 ** k).to(torch.float16).float())
    lg = np.log2(g.numpy()); edges = list(range(-44, 9)); cnt = np.histogram(lg, bins=edges)[0]
    return dict(n=int(g.numel()), fp16=rows, bf16=look(g.to(torch.bfloat16).float()), median_log2=float(np.median(lg)), hist=dict(edges=edges, counts=[int(c) for c in cnt]))
def saved(fn, *a, keep=()):          # bytes autograd keeps alive for the backward pass (each storage counted once, the weights themselves not counted)
    seen, tot, skip = set(), [0], {t.untyped_storage().data_ptr() for t in keep}
    def pack(t):
        k = t.untyped_storage().data_ptr()
        if k not in seen and k not in skip: seen.add(k); tot[0] += t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t): out = fn(*a)
    return tot[0], out
def state(mode):                     # bytes of weights + gradients + Adam's two numbers per weight, after one real step
    net = make(); xb = torch.randn(32, 64); yb = torch.randint(0, 10, (32,))
    if mode == "bf16_pure": net = net.to(torch.bfloat16); xb = xb.to(torch.bfloat16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with ac(mode): loss = nn.functional.cross_entropy(net(xb), yb)
    loss.backward(); opt.step(); n = sum(p.numel() for p in net.parameters())
    return dict(n=n, params=nbytes(net.parameters()), grads=nbytes(p.grad for p in net.parameters()), opt=nbytes(t for st in opt.state.values() for t in st.values() if torch.is_tensor(t) and t.ndim > 0))
def act_memory(mode, S):             # activation memory of the 12-block network: S = 0 keeps everything, S > 0 checkpoints it in S pieces
    bl = blocks(); ps = [p for b in bl for p in b.parameters()]; x = torch.randn(ACT_BATCH, ACT_WIDTH).requires_grad_(True)
    segs = [nn.Sequential(*[bl[i] for i in idx]) for idx in np.array_split(range(ACT_DEPTH), S or 1)]
    with ac(mode):
        if not S: full, _ = saved(lambda h: nn.Sequential(*bl)(h), x, keep=ps); return dict(kept=full, rebuild=0, peak=full)
        def fwd(h):
            for s in segs: h = checkpoint(s, h, use_reentrant=True)
            return h
        kept, _ = saved(fwd, x, keep=ps); h = x.detach().requires_grad_(True); rebuild = 0
        for s in segs:
            b, o = saved(s, h, keep=ps); rebuild = max(rebuild, b); h = o.detach().requires_grad_(True)
    return dict(kept=kept, rebuild=rebuild, peak=kept + rebuild)        # what stays between the passes + the one piece that is rebuilt while going backward
def timeit(fn, reps=REPS):
    t = time.perf_counter(); fn(); first = time.perf_counter() - t
    if first > 6: return first                                         # too slow on this machine: one timing is enough
    ts = []
    for _ in range(reps): t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))
def step_time(mode, S=0, dev="cpu", batch=TIME_BATCH):             # one training step (forward, backward, Adam) of the 12-block network, in seconds
    bl = [b.to(dev) for b in blocks()]; head = nn.Linear(ACT_WIDTH, 10).to(dev); ps = [p for b in bl for p in b.parameters()] + list(head.parameters()); opt = torch.optim.Adam(ps, lr=1e-3)
    sc = torch.amp.GradScaler(dev) if mode == "fp16_scaler" else None; x = torch.randn(batch, ACT_WIDTH, device=dev).requires_grad_(True); y = torch.randint(0, 10, (batch,), device=dev)
    segs = [nn.Sequential(*[bl[i] for i in idx]) for idx in np.array_split(range(ACT_DEPTH), S or 1)]
    def step():
        opt.zero_grad(); h = x
        with ac(mode, dev):
            for s in segs: h = checkpoint(s, h, use_reentrant=True) if S else s(h)
            loss = nn.functional.cross_entropy(head(h), y)
        (sc.scale(loss) if sc else loss).backward(); sc.step(opt) if sc else opt.step(); sc and sc.update()
        if dev == "cuda": torch.cuda.synchronize()
    return step
def matmul_gflops(name):
    a = torch.randn(MM_N, MM_N).to(DT[name]); b = torch.randn(MM_N, MM_N).to(DT[name]); t = timeit(lambda: a @ b); return 2 * MM_N ** 3 / t / 1e9
def gpu_bench(best):                 # optional: the same network on a CUDA GPU, with real peak memory (skipped and reported when there is no GPU)
    if not torch.cuda.is_available(): return dict(status="skipped: no CUDA GPU on this machine (run the script on a GPU to fill this in)")
    try:
        rows = {}
        for nm, mode, S in (("fp32", "fp32", 0), ("AMP bf16", "bf16_amp", 0), ("AMP fp16 + scaler", "fp16_scaler", 0), ("AMP bf16 + checkpointing", "bf16_amp", best)):
            if "bf16" in mode and not torch.cuda.is_bf16_supported(): continue
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); fn = step_time(mode, S, "cuda", ACT_BATCH); fn(); t = timeit(fn, 5); rows[nm] = dict(ms=t * 1e3, peak_mb=torch.cuda.max_memory_allocated() / 1e6)
        return dict(status="ok", name=torch.cuda.get_device_name(0), rows=rows)
    except Exception as e: return dict(status=f"failed: {type(e).__name__}: {str(e)[:120]}")
# ── 4. DASHBOARD ──────────────────────────────────────────────────────────
def dashboard(R, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#050815", "axes.facecolor": "#0d1117", "axes.edgecolor": "#6272a4", "text.color": "#f8f8f2", "axes.labelcolor": "#f8f8f2",
                         "xtick.color": "#f8f8f2", "ytick.color": "#f8f8f2", "axes.grid": True, "grid.alpha": 0.15}); fig, ax = plt.subplots(2, 2, figsize=(13, 7.4))
    a = ax[0, 0]; h = R["grad"]["hist"]; e = np.array(h["edges"][:-1]); a.bar(e, h["counts"], width=1.0, color="#8be9fd", alpha=0.75, label="real gradients (log2 of size)")
    for x0, c, t in ((-24, "#ff5555", "fp16 smallest"), (-14, "#ffb86c", "fp16 normal from"), (16, "#ff79c6", "fp16 largest")): a.axvline(x0, color=c, ls="--"); a.text(x0 + 0.3, max(h["counts"]) * 0.9, t, color=c, fontsize=7, rotation=90, va="top")
    a.set_title("Gradients against the fp16 range (loss scaling shifts them right)", fontsize=10); a.set_xlabel("log2 |gradient|"); a.legend(fontsize=7)
    a = ax[0, 1]; k = R["train"]["normal"]; ms_ = [m for m in MODES if k[m]["status"] == "ok"]; a.bar(range(len(ms_)), [k[m]["loss"][0] for m in ms_], yerr=[k[m]["loss"][1] for m in ms_], color="#50fa7b", capsize=3, ecolor="#f8f8f2")
    a.set_xticks(range(len(ms_)), ms_, fontsize=7, rotation=12); a.set_title(f"Test loss after {R['steps']} steps, by training precision ({R['seeds']} runs)", fontsize=10)
    a = ax[1, 0]; sw = R["act"]["sweep"]; xs = [s["S"] for s in sw]; a.plot(xs, [s["fp32"]["peak"] / 1e6 for s in sw], "-o", color="#ff5555", label="fp32"); a.plot(xs, [s["bf16"]["peak"] / 1e6 for s in sw], "-o", color="#50fa7b", label="bf16 autocast")
    a.set_title("Activation memory, MB (0 = keep everything, otherwise checkpointed in S pieces)", fontsize=9); a.set_xlabel("pieces S"); a.legend(fontsize=7)
    a = ax[1, 1]; sp = R["speed"]["step_ms"]; a.bar(range(len(sp)), list(sp.values()), color="#ffb86c"); a.set_xticks(range(len(sp)), list(sp), fontsize=7, rotation=12); a.set_title(f"One training step, ms, on this CPU ({R['cores']} cores)", fontsize=10)
    fig.suptitle("Mixed precision: what is really smaller, what is really faster", fontsize=14, color="#8be9fd"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
# ── 5. RUN TRAINING: every mode, at both learning rates, over several seeds ──
def run_training():
    out = {}
    for key, lr in LRS.items():
        out[key] = {}
        for m in MODES:
            try:
                rs = [train(m, s, lr)[1] for s in range(SEEDS)]
                out[key][m] = dict(status="ok", loss=ms([r["loss"] for r in rs]), acc=ms([r["acc"] for r in rs]), train_loss=ms([r["train_loss"] for r in rs]), bad=any(r["bad"] for r in rs), skipped=ms([r["skipped"] for r in rs]),
                                   trace=rs[0]["trace"])
            except Exception as e: out[key][m] = dict(status=f"skipped: {type(e).__name__}: {str(e)[:100]}")
    return out
# ── 6. RUN: measure everything, write one results file ────────────────────
def run():
    OUT.mkdir(exist_ok=True); torch.set_num_threads(max(1, min(4, os.cpu_count() or 1))); fm = formats(); tr = run_training(); net, _ = train("fp32", 0, LRS["normal"]); gr = underflow(net, 0)
    stt = {m: state(m) for m in ("fp32", "bf16_amp", "bf16_pure")}; sweep = []
    for S in [0] + SEGMENTS: sweep.append(dict(S=S, fp32=act_memory("fp32", S), bf16=act_memory("bf16_amp", S)))
    best = min(SEGMENTS, key=lambda S: next(s for s in sweep if s["S"] == S)["fp32"]["peak"]); full = sweep[0]["fp32"]["peak"]
    sp = {}; t0 = step_time("fp32"); sp["fp32"] = timeit(t0) * 1e3; sp["bf16_amp"] = timeit(step_time("bf16_amp")) * 1e3
    try: sp["fp16_scaler"] = timeit(step_time("fp16_scaler")) * 1e3
    except Exception as e: sp["fp16_scaler"] = None
    ck = {S: timeit(step_time("fp32", S)) * 1e3 for S in (best,)}
    mm = {}
    for nm in DT:
        try: mm[nm] = matmul_gflops(nm)
        except Exception: mm[nm] = None
    a = stt["fp32"]; n = a["n"]
    R = dict(ver={p: version(p) for p in ("torch", "numpy", "matplotlib")}, cores=os.cpu_count(), cpu=platform.processor() or platform.machine(), seeds=SEEDS, steps=STEPS, lrs=LRS, modes=MODES, scale0=SCALE0, growth=GROWTH, log2_scales=LOG2_SCALES,
             formats=fm, grad=gr, train=tr, state={m: v for m, v in stt.items()}, n_params=n,
             act=dict(width=ACT_WIDTH, depth=ACT_DEPTH, batch=ACT_BATCH, sweep=sweep, best=best, full_mb=full / 1e6, best_mb=next(s for s in sweep if s["S"] == best)["fp32"]["peak"] / 1e6, segments=SEGMENTS,
                      n_params=sum(p.numel() for b in blocks() for p in b.parameters())),
             speed=dict(step_ms={k: v for k, v in sp.items() if v is not None}, ckpt_ms=ck[best], ckpt_S=best, matmul_gflops={k: v for k, v in mm.items() if v is not None}, batch=TIME_BATCH),
             gpu=gpu_bench(best), arith=dict(bpp_fp32=4 + 4 + 8, bpp_pure_bf16=2 + 2 + 4, bpp_classic=2 + 2 + 4 + 8))
    dashboard(R, OUT / "dashboard.png"); Path(__file__).with_name("ep65_results.json").write_text(json.dumps(R)); return R

if __name__ == "__main__":
    R = run(); f = lambda m: f"{m[0]:.3f} ± {m[1]:.3f}"
    print(f"{R['cores']} CPU core(s); torch {R['ver']['torch']}; {R['seeds']} runs per training number")
    for nm, v in R["formats"].items(): print(f"  {nm}: {v['bits']} bits ({v['exp']} exponent, {v['mant']} mantissa), largest {v['max']:.3g}, smallest normal {v['tiny']:.3g} | 70000 -> {v['big']} | 1e-8 -> {v['small']} | 1+1/256 -> {v['one']} | 1.0 + 0.001 x1000 -> {v['acc']:.4f}")
    for k, v in R["grad"]["fp16"].items(): print(f"  gradients x 2^{k} in fp16: {v['zero'] * 100:.2f}% flushed to zero, {v['sub'] * 100:.2f}% subnormal, {v['inf'] * 100:.2f}% overflow  (bf16: {R['grad']['bf16']['zero'] * 100:.2f}% zero)")
    for k, d in R["train"].items():
        for m, v in d.items(): print(f"  lr {R['lrs'][k]:g} {m:<14}", f"test loss {f(v['loss'])}, accuracy {f(v['acc'])}, skipped steps {v['skipped'][0]:.1f}" if v["status"] == "ok" else v["status"])
    for m, v in R["state"].items(): print(f"  state, {m}: {sum(v[k] for k in ('params', 'grads', 'opt')) / v['n']:.1f} bytes per weight (weights {v['params'] / v['n']:.0f} + gradients {v['grads'] / v['n']:.0f} + Adam {v['opt'] / v['n']:.0f})")
    for s in R["act"]["sweep"]: print(f"  activations, {s['S']} pieces: fp32 {s['fp32']['peak'] / 1e6:.1f} MB | bf16 autocast {s['bf16']['peak'] / 1e6:.1f} MB")
    print("  step time, ms:", ", ".join(f"{k} {v:.0f}" for k, v in R["speed"]["step_ms"].items()), f"| fp32 + checkpointing in {R['speed']['ckpt_S']} pieces {R['speed']['ckpt_ms']:.0f}", "| matmul GFLOP/s:", ", ".join(f"{k} {v:.1f}" for k, v in R["speed"]["matmul_gflops"].items()))
    print("  GPU:", R["gpu"]["status"], *(f"{k}: {v['ms']:.0f} ms, {v['peak_mb']:.0f} MB" for k, v in R["gpu"].get("rows", {}).items()), sep=" | ")
