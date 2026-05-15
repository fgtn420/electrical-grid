# Implementation Reference

## Module map

```
src/
  data.py       — dataset fetch, scaling, train/test split   [done]
  physics.py    — bounds, tanh projection, energy routing
  model.py      — MLP definition, triple norm computation
  train.py      — training loop, L1 penalty, norm logging
  attack.py     — CW-L1 attack with binary search over c
  evaluate.py   — metrics and plots
```

---

## Libraries

| Library | Purpose |
|---------|---------|
| `ucimlrepo` | Dataset fetch |
| `numpy` | Array ops, type casting |
| `scikit-learn` | `train_test_split`, `StandardScaler`, accuracy score |
| `torch` | Model, autograd, optimizers |
| `matplotlib` | All plots (PNG output) |
| `pandas` | Triple norm CSV logging |

---

## `src/physics.py`

Single source of truth for all physical constraints. Operates in **physical (unscaled) space**.

```python
FEATURE_ORDER   # list of 12 feature names in dataset column order
BOUNDS          # dict[str, (float, float)] — lower/upper per non-p1 feature
                # 11 entries; p1 is derived, not optimised

tanh_project(w, L, U)   -> x     # x = L + (U-L)/2 * (tanh(w) + 1)
arctanh_init(x0, L, U)  -> w0    # inverse; initialises attack from x0 in physical space
enforce_conservation(x) -> x     # sets x[p1] = -(x[p2]+x[p3]+x[p4]) in-place
```

---

## `src/model.py`

```python
class GridMLP(nn.Module):
    # layers: Linear(12,64) -> ReLU -> Linear(64,64) -> ReLU
    #         -> Linear(64,32) -> ReLU -> Linear(32,1)
    # forward returns scalar logit (no sigmoid — BCEWithLogitsLoss in train.py)
    # receives scaled inputs

compute_triple_norm(model) -> list[float]
    # per Linear layer: max_j sum_i |w_ij|
    # returns one value per layer (4 values for this architecture)
```

Both Model A and B use the same class; regularization differs only in the training loop.

---

## `src/train.py`

```python
SEED = 42

train(
    model, X_train, y_train, X_val, y_val,
    # X inputs are already scaled
    l1_lambda=0.0,    # 0.0 for Model A (L2 via weight_decay), >0 for Model B
    weight_decay=0.0, # passed to Adam; non-zero for Model A only
    epochs, lr, batch_size,
) -> pd.DataFrame   # columns: epoch, layer_0_tnorm…layer_3_tnorm, train_loss, val_loss
```

L1 penalty applied manually each step:
```python
l1_loss = l1_lambda * sum(p.abs().sum() for p in model.parameters() if p.ndim >= 2)
# weights only, not biases
```

Optimizer: **Adam** for both models.  
Triple norms logged every epoch; DataFrame saved to `outputs/triple_norms_<model>.csv`.

---

## `src/attack.py`

The attack operates in **physical space**; scaling is applied inside before the forward pass.

```python
attack_sample(
    model, scaler,
    x0_physical,      # shape (12,) — original point in physical space
    c, kappa,
    n_steps, lr,
) -> (delta, converged)
    # delta: shape (12,) in physical space; includes delta[p1] (derived)
    # converged: bool — True if z < -kappa at end

cw_attack(
    model, scaler,
    x0_physical,      # shape (12,)
    kappa,
    c_init, n_binary_steps,   # binary search over c as in CW paper
    n_steps, lr,
) -> (delta, c_final)
    # binary search: find smallest c s.t. attack converges
    # inner loop is attack_sample

run_attack(
    model, scaler,
    X_stable_physical,  # shape (N, 12) — stable test examples in physical space
    kappa,
    c_init, n_binary_steps,
    n_steps, lr,
) -> np.ndarray         # shape (N, 12) — delta per sample; NaN row if no convergence
```

Attack loop per sample:
1. `w0 = arctanh_init(x0, L, U)` — initialise in physical space
2. `x_physical = tanh_project(w, L, U)`
3. `x_physical = enforce_conservation(x_physical)`
4. `x_scaled = scaler.transform(x_physical)` — scale before model
5. Forward → logit `z`; loss = `||delta_physical||_1 + c * max(z + kappa, 0)`
6. Backprop to `w`; Adam step; repeat

`delta_physical = x_physical - x0_physical` — perturbation always measured in physical space.

---

## `src/evaluate.py`

```python
accuracy(model, scaler, X_physical, y) -> float   # standard 0/1 accuracy

stability_margin(deltas)      -> float        # mean ||delta||_1 over converged samples
sparsity_profile(deltas)      -> np.ndarray   # shape (12,) — mean |delta_i|

plot_correlation(df_norms_A, margins_A, df_norms_B, margins_B)
    # X: product of triple norms; Y: per-sample stability margin
    # saved to outputs/correlation.png

plot_robustness_curve(deltas_A, deltas_B, budgets)
    # X: perturbation budget; Y: fraction of samples where ||delta||_1 <= budget
    # saved to outputs/robustness_curve.png

plot_sparsity(deltas_A, deltas_B)
    # bar chart of mean |delta_i| per feature for both models
    # saved to outputs/sparsity.png
```

---

## Data flow summary

```
raw X (physical space)
  └─ StandardScaler (fit on X_train only)
       ├─ X_scaled → model input during training & inference
       └─ X_physical kept for attack init and delta measurement
```

Scaler saved to `outputs/scaler.pkl` so attacks can be re-run without refitting.

---

## Outputs directory

```
outputs/
  scaler.pkl
  triple_norms_A.csv
  triple_norms_B.csv
  model_A.pt
  model_B.pt
  correlation.png
  robustness_curve.png
  sparsity.png
```
