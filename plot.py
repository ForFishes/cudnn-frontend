"""#2546 convergence A/B plots from the three p3 runs (compaction off x2, on x1)."""
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = "/root/paddlejob/share-storage/gpfs/system-public/shenliang/eb_dsl481/output"
OUT = "/root/paddlejob/share-storage/gpfs/system-public/shenliang/pr2520_whl/conv_plot"
KEYS = [("loss", "LM loss"), ("global_grad_norm", "global grad norm"),
        ("global_grad_norm_expert_cur_dp", "expert grad norm"), ("indexer_loss", "indexer loss")]
RUNS = [("p3_on1", "compaction ON", "#2a78d6", "-"),
        ("p3_off1", "compaction OFF (run 1)", "#eb6834", "-"),
        ("p3_off2", "compaction OFF (run 2)", "#1baf7a", "--")]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(tag):
    d = {}
    for f in glob.glob(f"{O}/conv3_{tag}/node0/workerlog.*"):
        for line in open(f, errors="ignore"):
            if "logging_callback" in line and "global_step:" in line:
                m = dict(re.findall(r"(\w+): ([-\d.e+]+)", line))
                d[int(m["global_step"])] = {k: float(m[k]) for k, _ in KEYS}
    steps = np.array(sorted(d))
    return steps, {k: np.array([d[s][k] for s in steps]) for k, _ in KEYS}


def smooth(x, w=50):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), out])


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("step", color=INK2, fontsize=9)


data = {t: load(t) for t, *_ in RUNS}
steps = data["p3_off1"][0]
assert all(len(data[t][0]) == 5000 for t, *_ in RUNS)

# Figure 1: the curves (50-step moving average)
fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), facecolor=SURF)
for ax, (k, name) in zip(axes.flat, KEYS):
    style(ax, f"{name} (50-step moving average)")
    for tag, label, color, ls in RUNS:
        ax.plot(steps, smooth(data[tag][1][k]), color=color, ls=ls, lw=2 if ls == "-" else 1.6, label=label)
    if k != "loss":
        ax.set_yscale("log")
axes.flat[0].legend(frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("PR #2546 convergence A/B: DSA stage3, 8L/32E C9_T8 widths, 64k, 1 node, real optimizer",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{OUT}/curves.png", dpi=130, facecolor=SURF)

# Figure 2: per-step |difference| vs the off-vs-off noise (200-step moving average)
fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), facecolor=SURF)
for ax, (k, name) in zip(axes.flat, KEYS):
    style(ax, f"|Δ {name}| (200-step moving average, log scale)")
    on, off1, off2 = (data[t][1][k] for t in ("p3_on1", "p3_off1", "p3_off2"))
    ax.plot(steps, smooth(np.abs(off2 - off1), 200), color="#eb6834", lw=2, label="noise: OFF run 2 − OFF run 1")
    ax.plot(steps, smooth(np.abs(on - off1), 200), color="#2a78d6", lw=2, label="ON − OFF run 1")
    ax.plot(steps, smooth(np.abs(on - off2), 200), color="#2a78d6", lw=1.6, ls="--", label="ON − OFF run 2")
    ax.set_yscale("log")
axes.flat[0].legend(frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("Step-wise difference vs run-to-run noise (same code, same seed: fp32 atomics in dKV / indexer bwd)",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{OUT}/diff_vs_noise.png", dpi=130, facecolor=SURF)
print("ok")
