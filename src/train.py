"""Train the MLP hosting-capacity surrogate (Siddique et al. 2026, Sec. 2.5 and 3.3) and the two
modifications.

Examples:
  python src/train.py                          # reproduction: pilot buses, MSE, 1 x 615 ReLU
  python src/train.py --hidden 615 615 615     # 3 hidden layers (paper Table 3)
  python src/train.py --alpha 5                # Modification 1: asymmetric loss
  python src/train.py --inputs all --lam 1e-3  # Modification 2: group lasso over all buses
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn

# Pilot buses, read off the red circles in the paper's Fig. 3. The paper does not list them in
# the text, and some labels are partly hidden by the circles.
PILOT_BUSES = ["17", "25", "31", "38", "45", "54", "61", "74", "84", "95"]

# Test sets by PV penetration (paper Sec. 3.4)
TEST_SETS = {"Test set 1": (0.1, 0.3), "Test set 2": (0.4, 0.6), "Test set 3": (0.7, 1.0)}

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/hc_dataset.npz")
parser.add_argument("--inputs", choices=["pilot", "all"], default="pilot")
parser.add_argument("--state", choices=["hc", "nopv"], default="hc",
                    help="voltages at the HC point (paper Step 8) or with no PV")
parser.add_argument("--alpha", type=float, default=1.0, help="overestimation weight (1 = MSE)")
parser.add_argument("--lam", type=float, default=0.0, help="group-lasso strength")
parser.add_argument("--hidden", type=int, nargs="+", default=[615], help="5 x 123 buses (Table 1)")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1500)
parser.add_argument("--patience", type=int, default=100, help="early stopping on validation")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--json", default=None, help="write metrics to this file")
args = parser.parse_args()
torch.manual_seed(args.seed)

# ---------------- data ----------------
d = np.load(args.data)
mag, ang = (d["Vmag"], d["Vang"]) if args.state == "hc" else (d["Vmag0"], d["Vang0"])
nodes = d["node_names"]
y, pen, placement = d["y"], d["pen"], d["placement"]

cols = np.arange(len(nodes))
if args.inputs == "pilot":
    cols = np.array([i for i, n in enumerate(nodes) if n.split(".")[0] in PILOT_BUSES])
X = np.hstack([mag[:, cols], ang[:, cols]]).astype(np.float64)   # [V | theta], Eq. (9)
feat_bus = np.array([n.split(".")[0] for n in nodes[cols]] * 2)

keep = y > 0                                  # scenarios with HC = 0 are excluded (Sec. 3.1)
X, y, pen, placement = X[keep], y[keep], pen[keep], placement[keep]
varying = X.std(0) > 1e-6                     # drop constant features (source-bus angles)
X, feat_bus = X[:, varying], feat_bus[varying]

# Split by placement, so the 24 hours of one PV layout never straddle train and test.
rng = np.random.default_rng(args.seed)
perm = rng.permutation(placement.max() + 1)
n_pl = len(perm)
tr_pl, va_pl = perm[:int(0.70 * n_pl)], perm[int(0.70 * n_pl):int(0.85 * n_pl)]
tr = np.where(np.isin(placement, tr_pl))[0]
va = np.where(np.isin(placement, va_pl))[0]
te = np.where(~np.isin(placement, np.concatenate([tr_pl, va_pl])))[0]

# Min-max scaling of inputs to [0, 1]; HC divided by a fixed factor (Sec. 3.1)
x_min, x_max = X[tr].min(0), X[tr].max(0)
y_scale = y[tr].max()
to_t = lambda a: torch.tensor(a, dtype=torch.float32)
Xn = to_t((X - x_min) / (x_max - x_min))
yn = to_t(y / y_scale).unsqueeze(1)

buses = np.unique(feat_bus)
groups = [torch.tensor(np.where(feat_bus == b)[0]) for b in buses]   # one group per bus


# ---------------- model ----------------
class MLP(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def asymmetric_mse(pred, target, alpha):
    """Weight alpha on overestimation (pred > target), 1 on underestimation."""
    err = pred - target
    w = torch.where(err > 0, alpha, 1.0)
    return torch.mean(w * err ** 2)


def group_prox(W, lr, lam):
    """Proximal step for the group-lasso penalty: shrinks all first-layer weights of a bus
    together, and sets them exactly to zero once their norm falls below lr * lam."""
    for g in groups:
        norm = torch.norm(W[:, g]).clamp_min(1e-12)
        W[:, g] *= torch.clamp(1 - lr * lam / norm, min=0.0)


def bus_norms(model):
    W = model.net[0].weight.detach()
    return np.array([torch.norm(W[:, g]).item() for g in groups])


model = MLP(X.shape[1], args.hidden)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

t_train = time.time()
best, best_state, wait = np.inf, None, 0
for epoch in range(args.epochs):
    model.train()
    order = tr[torch.randperm(len(tr)).numpy()]
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        opt.zero_grad()
        asymmetric_mse(model(Xn[b]), yn[b], args.alpha).backward()
        opt.step()
        if args.lam > 0:
            with torch.no_grad():
                group_prox(model.net[0].weight, args.lr, args.lam)

    model.eval()
    with torch.no_grad():
        val = asymmetric_mse(model(Xn[va]), yn[va], args.alpha).item()
    if val < best:
        best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait > args.patience:
            break
model.load_state_dict(best_state)
t_train = time.time() - t_train

# ---------------- evaluation (test set, kW) ----------------
model.eval()
with torch.no_grad():
    t0 = time.time()
    pred_all = model(Xn[te]).squeeze(1).numpy() * y_scale
    t_inf = time.time() - t0


def kpis(mask):
    p, t = pred_all[mask], y[te][mask]
    err = p - t
    pe = 100 * np.abs(err) / t                                   # Eq. (15)
    over = err > 0
    return {"avg_err_pct": pe.mean(), "max_err_pct": pe.max(), "avg_dev_kw": np.abs(err).mean(),
            "rmse_kw": np.sqrt((err ** 2).mean()), "over_pct": 100 * over.mean(),
            "mean_over_kw": err[over].mean() if over.any() else 0.0,
            "mean_pred_err_pct": (100 * np.abs(y[tr].mean() - t) / t).mean(), "n": int(mask.sum())}


res = {name: kpis((pen[te] >= lo - 1e-9) & (pen[te] <= hi + 1e-9))
       for name, (lo, hi) in TEST_SETS.items()}
res["All"] = kpis(np.ones(len(te), bool))
norms = bus_norms(model)
active = buses[norms > 0]

print(f"inputs={args.inputs} ({X.shape[1]} features, {len(buses)} buses) state={args.state} "
      f"alpha={args.alpha} lam={args.lam} hidden={args.hidden} epochs={epoch + 1} "
      f"train {t_train:.0f} s")
print(f"{'':12s}{'avg err %':>10s}{'max err %':>10s}{'avg dev kW':>11s}{'RMSE kW':>9s}"
      f"{'over %':>8s}{'mean over kW':>13s}{'mean-pred %':>12s}")
for name, r in res.items():
    print(f"{name:12s}{r['avg_err_pct']:10.3f}{r['max_err_pct']:10.2f}{r['avg_dev_kw']:11.1f}"
          f"{r['rmse_kw']:9.1f}{r['over_pct']:8.1f}{r['mean_over_kw']:13.1f}"
          f"{r['mean_pred_err_pct']:12.2f}")
print(f"inference {1e3 * t_inf:.1f} ms for {len(te)} scenarios")
if args.lam > 0:
    print(f"active buses: {len(active)} / {len(buses)}")
    print("strongest buses:", [str(b) for b in buses[np.argsort(-norms)[:10]]])

if args.json:
    out = {"args": vars(args), "results": res, "epochs": epoch + 1, "train_s": t_train,
           "infer_ms": 1e3 * t_inf, "n_active_buses": int(len(active)),
           "active_buses": list(map(str, active))}
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1, default=float)
