# Adversarial Engineering for Smart Grid Stability

End-to-end PyTorch pipeline for adversarial sensitivity analysis on a simulated smart grid dataset. The goal is to estimate **stability margin**—the smallest physically valid perturbation that flips a stability prediction—using a physics-constrained Carlini–Wagner L1 attack.

---

## 1. Dataset and physical constraints

**Dataset:** [Electrical Grid Stability Simulated Dataset](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data) (UCI)

**Input (12 continuous features):**

| Feature | Description | Bounds |
|---------|-------------|--------|
| `p_1` | Producer power (dependent) | derived |
| `p_2`, `p_3`, `p_4` | Consumer power | `[-2.0, -0.5]` |
| `tau_1` … `tau_4` | Reaction times | `[0.5, 10.0]` |
| `g_1` … `g_4` | Price elasticity | `[0.05, 1.00]` |

**Energy conservation (critical):** The grid is a closed loop; total power must sum to zero.

```
p_1 = -(p_2 + p_3 + p_4)
```

Do not optimize `p_1` directly. Optimize the other 11 variables and set `p_1` in the forward pass from this identity so autograd propagates gradients correctly.

---

## 2. Models and regularization

Train two binary classifiers for stability label `y ∈ {0, 1}`:

- **Model A (control):** L2 regularization (weight decay); dense weights.
- **Model B (experimental):** L1 regularization (Lasso); sparse weights.

**Triple norm (log during training):** For each layer `k`, the layer-wise L1 triple norm is the maximum absolute column sum of the weight matrix:

```
|||W_k|||_1 = max_j  sum_i |w_ij^(k)|
```

---

## 3. Adversarial attack (CW-L1)

Custom **Carlini–Wagner L1** attack (not PGD).

**Objective:** Minimize L1 perturbation size while pushing the pre-activation logit past the decision boundary:

```
min_w  ||delta||_1 + c * max(z + kappa, 0)
```

Run sweeps over confidence `kappa` at `0`, `1σ`, and `2σ` (σ = std of model logits).

**Box constraints:** Optimize unconstrained `w`, map to physical bounds `[L_i, U_i]`:

```
x_i = L_i + (U_i - L_i) / 2 * (tanh(w_i) + 1)
delta = x - x_0
```

**Attack loop:**

1. Initialize `w` so mapped `x ≈ x_0`.
2. Map `w → x` via `tanh` box constraint.
3. Apply physics routing: `x[p_1] = -(x[p_2] + x[p_3] + x[p_4])`.
4. Forward through frozen model; read logit `z`.
5. Compute CW-L1 loss; backprop to `w`; optimizer step (e.g. Adam).

---

## 4. Evaluation and outputs

Compare sparse L1-regularized models with L1 attacks to show alignment between regularization and threat model.

**Required outputs:**

1. **Stability margin:** Mean `||delta||_1` to flip predictions for Model A vs Model B.
2. **Sparsity:** Perturbation vector `delta` showing 1–2 dominant nodes (single points of failure).
3. **Correlation plot:** Stability margin (Y) vs product of layer-wise L1 triple norms (X).
4. **Robustness curve:** Perturbation budget (X) vs attack success rate (Y) for both models.
