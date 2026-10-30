"""Generate the Monte Carlo hosting-capacity dataset on the IEEE 123-bus feeder.

Each scenario draws a feeder-wide loading level g ~ U(0.3, 1.0) and scales every load by
g times an independent U(0.8, 1.2) factor (loads vary together through the day, plus local
noise). It records the per-unit
voltage at every node with no PV (model inputs), and computes the PV hosting capacity by
bisection (label).

Usage:  python src/generate_data.py --n 5000 --seed 0 --out data/hc_dataset.npz
"""
import argparse
import time
import numpy as np
from feeder import Feeder

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="data/hc_dataset.npz")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
feeder = Feeder()
n_loads, n_nodes = len(feeder.load_names), len(feeder.node_names)

X = np.zeros((args.n, n_nodes))      # node voltages (pu) before PV is added
y = np.zeros(args.n)                 # hosting capacity (kW)
g = rng.uniform(0.3, 1.0, size=(args.n, 1))                      # feeder-wide level
M = g * rng.uniform(0.8, 1.2, size=(args.n, n_loads))            # per-load multipliers

t0 = time.time()
for k in range(args.n):
    feeder.set_loads(M[k])
    X[k] = feeder.voltages()
    y[k] = feeder.hosting_capacity()
    if (k + 1) % 500 == 0:
        print(f"{k + 1}/{args.n} scenarios, {time.time() - t0:.0f} s")

np.savez(args.out, X=X, y=y, load_mult=M,
         node_names=np.array(feeder.node_names), load_names=np.array(feeder.load_names))
print(f"saved {args.out}: X {X.shape}, HC range {y.min():.0f}-{y.max():.0f} kW, "
      f"{(y == 0).sum()} scenarios infeasible with no PV")
