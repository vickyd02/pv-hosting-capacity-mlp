# Real-time PV Hosting Capacity with an MLP: Reproduction and Extensions

Reproduction of Siddique et al., "Real-time PV hosting capacity analysis of distribution networks using pilot buses data and neural networks," *Applied Energy* 409 (2026), on the IEEE 123-bus feeder, with two modifications: an asymmetric loss that penalizes overestimating hosting capacity (HC), and a group-lasso penalty on the input layer that selects which bus voltages the model needs.

## Setup

```
pip install -r requirements.txt
```

## Data

`data/hc_dataset.npz` holds 5,000 Monte Carlo scenarios: `X` (node voltages in pu with no PV, 5000 x 278), `y` (HC in kW), `load_mult`, `node_names`, `load_names`. To regenerate it (about 2.5 minutes on a laptop):

```
python src/generate_data.py --n 5000 --seed 0 --out data/hc_dataset.npz
```

Each scenario draws a feeder-wide loading level g ~ U(0.3, 1.0) and scales every load by g times an independent U(0.8, 1.2) factor. PV is added at every load bus in proportion to its nominal load (unity power factor), and HC is the largest total PV with all node voltages in [0.95, 1.05] pu and every line under its rating, found by bisection to 1 kW. Scenarios that violate a limit before any PV is added (HC = 0) are dropped during training.

Modeling choices that should be checked against the paper (all in `src/feeder.py`):

- **PV placement:** distributed across all load buses. Pass `pv_buses=[...]` to `Feeder` to use specific buses.
- **Limits:** `V_MAX`, `V_MIN`, `I_MAX` at the top of the file.
- **Line ratings:** the OpenDSS model gives every line the 400 A default, which the main trunk exceeds at nominal load, so each line is rated at max(400 A, 1.25 x its nominal-load current). With these ratings the thermal limit binds in almost every scenario.
- **Regulators:** taps are reset to neutral and re-settled on every power flow, so each HC label does not depend on solve order.

## Training

```
python src/train.py                  # baseline: MSE, all node voltages
python src/train.py --alpha 5        # asymmetric loss, overestimation weight 5
python src/train.py --lam 1          # group lasso (proximal step on first-layer columns)
python src/train.py --inputs pilot   # paper's pilot buses (fill PILOT_BUSES in train.py)
```

The MLP has two hidden layers of 64 ReLU units, standardized inputs and outputs, full-batch Adam (lr 1e-3), and early stopping on validation loss. The split is 70/15/15. Reported metrics are test MAE, MAPE, R2, the share and mean size of overestimates, and inference time.
