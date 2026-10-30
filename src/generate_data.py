"""Generate the Monte Carlo hosting-capacity dataset on the IEEE 123-bus feeder, following
Siddique et al. (2026), Sec. 3.1 and Fig. 2.

For each PV penetration level (10%, 20%, ..., 100% of the 85 load buses), draw S random sets of
PV sites. For each set and each of the 24 hours, scale the loads by that hour's loadshape value,
grow every PV unit in 10 kW steps until a violation, and record the HC together with the node
voltage magnitudes and angles at the HC point (Step 8). Voltages with no PV are also saved, for
comparison.

Usage:  python src/generate_data.py --out data/hc_dataset.npz       (all levels, ~8 min)
Sensitivity case without the thermal limit (add --voltage-only):
        python src/generate_data.py --voltage-only --out data/hc_dataset_vonly.npz
In chunks (each level is seeded on its own, so the result is identical):
        python src/generate_data.py --levels 1 2 3 --out data/parts
        python src/generate_data.py --levels 4 5 6 --out data/parts   ... then
        python src/generate_data.py --merge data/parts --out data/hc_dataset.npz
"""
import glob
import os
import argparse
import time
import numpy as np
from feeder import Feeder

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=100, help="PV placements per level (S)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 11)),
                    help="penetration levels to run, in tens of percent")
parser.add_argument("--voltage-only", action="store_true",
                    help="sensitivity case: ignore the line thermal limit")
parser.add_argument("--merge", type=str, default=None, help="merge part files in this folder")
parser.add_argument("--out", type=str, default="data/hc_dataset.npz")
args = parser.parse_args()


def save(path, parts):
    out = {k: np.concatenate([p[k] for p in parts]) for k in ROW_KEYS}
    out.update({k: parts[0][k] for k in ("node_names", "pv_sites", "loadshape")})
    np.savez_compressed(path, **out)
    y = out["y"]
    print(f"saved {path}: {len(y)} rows, HC {y[y > 0].min():.0f}-{y.max():.0f} kW, "
          f"{(y == 0).sum()} rows with HC = 0")


ROW_KEYS = ("Vmag", "Vang", "Vmag0", "Vang0", "y", "pen", "placement", "hour", "site_mask")
if args.merge:
    files = sorted(glob.glob(os.path.join(args.merge, "level_*.npz")))
    save(args.out, [dict(np.load(f)) for f in files])
    raise SystemExit

feeder = Feeder(thermal=not args.voltage_only)
sites_all = np.array(feeder.pv_sites)
n_rows = len(args.levels) * args.scenarios * 24
n_nodes = len(feeder.node_names)

Vmag = np.zeros((n_rows, n_nodes), np.float32)    # at the HC point (model inputs)
Vang = np.zeros((n_rows, n_nodes), np.float32)
Vmag0 = np.zeros((n_rows, n_nodes), np.float32)   # same hour, no PV
Vang0 = np.zeros((n_rows, n_nodes), np.float32)
hc = np.zeros(n_rows)
pen = np.zeros(n_rows)
placement = np.zeros(n_rows, int)                 # placement index within its level
hour = np.zeros(n_rows, int)
site_mask = np.zeros((n_rows, len(sites_all)), bool)

t0, r = time.time(), 0
for lev in args.levels:
    p = lev / 10
    rng = np.random.default_rng([args.seed, lev])
    n_pv = max(1, int(round(p * len(sites_all))))
    for s in range(args.scenarios):
        idx = rng.choice(len(sites_all), n_pv, replace=False)
        sites = list(sites_all[idx])
        for h in range(24):
            feeder.set_hour_mult(feeder.loadshape[h])
            Vmag0[r], Vang0[r] = feeder.voltages()
            _, hc[r] = feeder.hosting_capacity(sites)
            Vmag[r], Vang[r] = feeder.voltages()
            pen[r], placement[r], hour[r] = p, s, h
            site_mask[r, idx] = True
            r += 1
    print(f"penetration {p:.0%} done, {time.time() - t0:.0f} s")

rows = dict(Vmag=Vmag, Vang=Vang, Vmag0=Vmag0, Vang0=Vang0, y=hc, pen=pen, placement=placement,
            hour=hour, site_mask=site_mask)
static = dict(node_names=np.array(feeder.node_names), pv_sites=sites_all,
              loadshape=feeder.loadshape)
if args.out.endswith(".npz"):
    save(args.out, [{**rows, **static}])
else:                                             # chunked run: one part file per level
    os.makedirs(args.out, exist_ok=True)
    n = args.scenarios * 24
    for j, lev in enumerate(args.levels):
        part = {k: v[j * n:(j + 1) * n] for k, v in rows.items()}
        np.savez_compressed(os.path.join(args.out, f"level_{lev:02d}.npz"), **part, **static)
    print(f"saved levels {args.levels} to {args.out}")
