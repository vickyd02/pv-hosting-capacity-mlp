"""Run the experiment grid over several seeds, then summarize to CSV and make the plots.

Each run writes results/<name>_seed<k>.json and is skipped if that file already exists, so the
sweep can be stopped and restarted.

  python src/sweep.py                  # all experiments, seeds 0 1 2
  python src/sweep.py --seeds 0 --only repro alpha
  python src/sweep.py --plot-only      # rebuild results/summary.csv and the figures
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import numpy as np

EXPERIMENTS = {
    "repro": {"repro_1layer": [], "repro_3layer": ["--hidden", "615", "615", "615"]},
    "ablation": {"nopv_inputs": ["--state", "nopv"]},
    "alpha": {f"alpha{a}": ["--alpha", str(a)] for a in [2, 5, 10]},
    "lasso": {f"lam{l}": ["--inputs", "all", "--lam", str(l)]
              for l in [0, 0.03, 0.1, 0.3, 1]},
    "vonly": {"vonly_repro": ["--data", "data/hc_dataset_vonly.npz"],
              "vonly_allbus": ["--data", "data/hc_dataset_vonly.npz", "--inputs", "all"]},
    "combined": {"alpha5_lam0.1": ["--inputs", "all", "--alpha", "5", "--lam", "0.1"]},
}

# Paper Table 4: average error (%) with voltages from non-pilot buses, per test set
PAPER_TABLE4 = {10: [11.84, 15.013, 15.44], 15: [9.83, 10.12, 10.9], 20: [5.83, 5.57, 5.99],
                25: [1.098, 0.88, 2.09]}
PAPER_PILOT = [0.7136, 0.5915, 0.5755]            # Table 2

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--only", nargs="+", default=list(EXPERIMENTS))
parser.add_argument("--out", default="results")
parser.add_argument("--plot-only", action="store_true")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))

if not args.plot_only:
    for group in args.only:
        for name, flags in EXPERIMENTS[group].items():
            if "--data" in flags and not os.path.exists(flags[flags.index("--data") + 1]):
                print(f"skipping {name}: {flags[flags.index('--data') + 1]} not found")
                break
            for s in args.seeds:
                path = os.path.join(args.out, f"{name}_seed{s}.json")
                if os.path.exists(path):
                    continue
                print(f"running {name} seed {s}", flush=True)
                subprocess.run([sys.executable, os.path.join(here, "train.py"), *flags,
                                "--seed", str(s), "--json", path], check=True)

# ---------------- summary ----------------
runs = {}
for f in sorted(glob.glob(os.path.join(args.out, "*_seed*.json"))):
    name = os.path.basename(f).rsplit("_seed", 1)[0]
    runs.setdefault(name, []).append(json.load(open(f)))

cols = ["avg_err_pct", "max_err_pct", "avg_dev_kw", "rmse_kw", "over_pct", "mean_over_kw"]
sets = ["Test set 1", "Test set 2", "Test set 3", "All"]
lines = ["run,seeds,test_set," + ",".join(f"{c}_mean,{c}_std" for c in cols)
         + ",active_buses_mean"]
for name, rs in runs.items():
    nb = np.mean([r["n_active_buses"] for r in rs])
    for t in sets:
        vals = [f"{np.mean([r['results'][t][c] for r in rs]):.4f},"
                f"{np.std([r['results'][t][c] for r in rs]):.4f}" for c in cols]
        lines.append(f"{name},{len(rs)},{t}," + ",".join(vals) + f",{nb:.1f}")
open(os.path.join(args.out, "summary.csv"), "w").write("\n".join(lines) + "\n")
print(f"wrote {args.out}/summary.csv ({len(runs)} runs)")

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stat(name, key, t="All"):
    v = [r["results"][t][key] for r in runs[name]]
    return np.mean(v), np.std(v)


# Figure: overestimation rate vs average error as alpha grows (pilot-bus inputs)
alpha_runs = [("repro_1layer", 1)] + [(f"alpha{a}", a) for a in [2, 5, 10]]
alpha_runs = [(n, a) for n, a in alpha_runs if n in runs]
if len(alpha_runs) > 1:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for n, a in alpha_runs:
        (e, es), (o, os_) = stat(n, "avg_err_pct"), stat(n, "over_pct")
        ax.errorbar(e, o, xerr=es, yerr=os_, fmt="o", color="C0", capsize=3)
        ax.annotate(f"α={a}", (e, o), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Average error (%)")
    ax.set_ylabel("Test cases overestimated (%)")
    ax.set_title("Asymmetric loss, pilot-bus inputs")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_alpha.png"), dpi=200)

# Figure: average error vs number of buses used
lam_runs = [n for n in runs if n.startswith("lam")]
if lam_runs:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    nb = [np.mean([r["n_active_buses"] for r in runs[n]]) for n in lam_runs]
    er = [stat(n, "avg_err_pct") for n in lam_runs]
    ax.errorbar(nb, [e for e, _ in er], yerr=[s for _, s in er], fmt="o", color="C0",
                capsize=3, label="Group lasso (ours)")
    k = sorted(PAPER_TABLE4)
    ax.plot(k, [np.mean(PAPER_TABLE4[b]) for b in k], "s--", color="C3",
            label="Paper, non-pilot buses (Table 4)")
    ax.plot(10, np.mean(PAPER_PILOT), "*", ms=12, color="C3", label="Paper, pilot buses")
    if "repro_1layer" in runs:
        ax.errorbar(10, stat("repro_1layer", "avg_err_pct")[0],
                    yerr=stat("repro_1layer", "avg_err_pct")[1], fmt="*", ms=12,
                    color="C0", capsize=3, label="Ours, pilot buses")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Buses providing voltage data")
    ax.set_ylabel("Average error (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_lasso.png"), dpi=200)
print("figures saved")
