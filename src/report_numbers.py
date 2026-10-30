"""Print every [TBD] value in the report from results/*_seed*.json (mean over the seeds found).

Usage (from the repo root, after python src/sweep.py):  python src/report_numbers.py
"""
import glob
import json
import os
import numpy as np

PILOT_BUSES = {"17", "25", "31", "38", "45", "54", "61", "74", "84", "95"}
SETS = ["Test set 1", "Test set 2", "Test set 3"]

runs = {}
for f in glob.glob("results/*_seed*.json"):
    name = os.path.basename(f).rsplit("_seed", 1)[0]
    runs.setdefault(name, []).append(json.load(open(f)))


def m(name, key, t="All"):
    if name not in runs:
        return "missing"
    v = [r["results"][t][key] for r in runs[name]]
    return f"{np.mean(v):.2f} (± {np.std(v):.2f}, {len(v)} seeds)"


def per_set(name):
    return ", ".join(f"{t[-1]}: {m(name, 'avg_err_pct', t)}" for t in SETS)


print("== Table 1 (reproduction, 1 layer; replace seed-0 values with these means) ==")
for t in SETS:
    print(f"  {t}: avg {m('repro_1layer', 'avg_err_pct', t)} | max {m('repro_1layer', 'max_err_pct', t)}"
          f" | RMSE {m('repro_1layer', 'rmse_kw', t)}")
print("  3 hidden layers (Table 1 caption):", per_set("repro_3layer"))

print("\n== Discussion: inputs with no PV ==")
print("  avg error, all test sets:", m("nopv_inputs", "avg_err_pct"))

print("\n== Discussion: voltage-only case ==")
print("  pilot buses:", per_set("vonly_repro"))
print("  all buses:  ", per_set("vonly_allbus"))

print("\n== Table 2, asymmetric loss (All) ==")
for a, name in [(1, "repro_1layer"), (2, "alpha2"), (5, "alpha5"), (10, "alpha10")]:
    print(f"  alpha={a}: avg error {m(name, 'avg_err_pct')} | overestimated {m(name, 'over_pct')}")

print("\n== Table 2, group lasso (All) ==")
for lam in ["0", "0.03", "0.1", "0.3", "1"]:
    name = f"lam{lam}"
    if name not in runs:
        print(f"  lambda={lam}: missing")
        continue
    nb = np.mean([r["n_active_buses"] for r in runs[name]])
    buses = set(runs[name][0]["active_buses"])
    overlap = len(buses & PILOT_BUSES) if len(buses) < 132 else "n/a"
    print(f"  lambda={lam}: active buses {nb:.0f} | avg error {m(name, 'avg_err_pct')}"
          f" | overlap with pilot buses (seed {runs[name][0]['args']['seed']}): {overlap}")

print("\n== Combined (alpha=5, lambda=0.1) ==")
name = "alpha5_lam0.1"
if name in runs:
    nb = np.mean([r["n_active_buses"] for r in runs[name]])
    print(f"  active buses {nb:.0f} | avg error {m(name, 'avg_err_pct')}"
          f" | overestimated {m(name, 'over_pct')}")
else:
    print("  missing")
