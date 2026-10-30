"""Train the MLP hosting-capacity surrogate.

Examples:
  python src/train.py                                  # baseline: MSE, all node voltages
  python src/train.py --alpha 5                        # asymmetric loss
  python src/train.py --lam 1e-3                       # group lasso on the input layer
  python src/train.py --inputs pilot                   # paper's pilot buses (fill PILOT_BUSES)
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

# Pilot buses from the paper (bus names as in the IEEE 123 OpenDSS model, e.g. "35", "76").
PILOT_BUSES = []

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/hc_dataset.npz")
parser.add_argument("--inputs", choices=["all", "pilot"], default="all")
parser.add_argument("--alpha", type=float, default=1.0, help="overestimation weight (1 = MSE)")
parser.add_argument("--lam", type=float, default=0.0, help="group-lasso strength")
parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--patience", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)

# ---------------- data ----------------
d = np.load(args.data)
X, y, nodes = d["X"], d["y"], d["node_names"]
keep = y > 0                              # drop scenarios that violate limits with no PV
X, y = X[keep], y[keep]
varying = X.std(0) > 1e-6                 # drop nodes whose voltage never changes (source)
X, nodes = X[:, varying], nodes[varying]
if args.inputs == "pilot":
    assert PILOT_BUSES, "fill PILOT_BUSES with the paper's pilot buses"
    cols = [i for i, n in enumerate(nodes) if n.split(".")[0] in PILOT_BUSES]
    X, nodes = X[:, cols], nodes[cols]

rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(y))
n_tr, n_va = int(0.70 * len(y)), int(0.15 * len(y))
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

x_mu, x_sd = X[tr].mean(0), X[tr].std(0)
y_mu, y_sd = y[tr].mean(), y[tr].std()
to_t = lambda a: torch.tensor(a, dtype=torch.float32)
Xn = to_t((X - x_mu) / x_sd)
yn = to_t((y - y_mu) / y_sd).unsqueeze(1)


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


model = MLP(X.shape[1], args.hidden)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

best, best_state, wait = np.inf, None, 0
for epoch in range(args.epochs):
    model.train()
    opt.zero_grad()
    loss = asymmetric_mse(model(Xn[tr]), yn[tr], args.alpha)
    loss.backward()
    opt.step()
    if args.lam > 0:   # proximal step for the group-lasso penalty: shrinks whole input columns
        with torch.no_grad():
            W = model.net[0].weight
            norms = torch.norm(W, dim=0, keepdim=True).clamp_min(1e-12)
            W.mul_(torch.clamp(1 - args.lr * args.lam / norms, min=0.0))

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

# ---------------- evaluation (test set, kW) ----------------
model.eval()
with torch.no_grad():
    t0 = time.time()
    pred = model(Xn[te]).squeeze(1).numpy() * y_sd + y_mu
    t_inf = time.time() - t0
true = y[te]
err = pred - true
mae = np.abs(err).mean()
mape = 100 * np.abs(err / true).mean()
r2 = 1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()
over = err > 0
mean_pred_mae = np.abs(true - y[tr].mean()).mean()

print(f"inputs={X.shape[1]} alpha={args.alpha} lam={args.lam} epochs={epoch + 1}")
print(f"MAE {mae:.1f} kW | MAPE {mape:.2f}% | R2 {r2:.4f} | "
      f"(predict-the-mean MAE {mean_pred_mae:.1f} kW)")
print(f"overestimates {100 * over.mean():.1f}% of test cases, "
      f"mean overestimate {err[over].mean() if over.any() else 0:.1f} kW")
print(f"inference {1e3 * t_inf:.2f} ms for {len(te)} scenarios")

if args.lam > 0:
    col_norm = torch.norm(model.net[0].weight, dim=0).detach().numpy()
    active = col_norm > 0
    print(f"active inputs: {active.sum()} / {len(active)}")
    print("strongest nodes:", list(nodes[np.argsort(-col_norm)[:10]]))
