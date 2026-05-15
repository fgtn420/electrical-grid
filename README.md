# Adversarial Engineering for Smart Grid Stability

End-to-end PyTorch pipeline for adversarial sensitivity analysis on a simulated smart grid dataset. The goal is to estimate **stability margin**—the smallest physically valid perturbation that flips a stable prediction to unstable—using a physics-constrained Carlini–Wagner L1 attack.

---

## 1. Dataset and physical constraints

**Dataset:** [Electrical Grid Stability Simulated Dataset](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data) (UCI)

**Input (12 continuous features):**

| Feature | Description | Bounds |
|---------|-------------|--------|
| `p_1` | Producer power (dependent) | `[1.5, 6.0]` (derived) |
| `p_2`, `p_3`, `p_4` | Consumer power | `[-2.0, -0.5]` |
| `tau_1` … `tau_4` | Reaction times | `[0.5, 10.0]` |
| `g_1` … `g_4` | Price elasticity | `[0.05, 1.00]` |

**Energy conservation (critical):** The grid is a closed loop; total power must sum to zero.

```
p_1 = -(p_2 + p_3 + p_4)
```

Do not optimize `p_1` directly. Optimize the other 11 variables and compute `p_1` in the forward pass from this identity so autograd propagates gradients correctly. The bounds `[1.5, 6.0]` for `p_1` follow automatically from the consumer bounds.

---

## 2. Models and regularization

Train two binary classifiers for stability label `y ∈ {0, 1}` (1 = stable). Both use the same architecture: 3-layer MLP with hidden sizes `[64, 64, 32]` and ReLU activations. They differ only in regularization:

- **Model A (control):** L2 regularization (weight decay); dense weights.
- **Model B (experimental):** L1 regularization; weights pushed toward zero.

**Triple norm (log during training):** For each layer `k`, the layer-wise L1 triple norm is the maximum absolute column sum of the weight matrix:

```
|||W_k|||_1 = max_j  sum_i |w_ij^(k)|
```

This is the operator norm induced by L1 on vectors. Its product across layers bounds the network's L1 Lipschitz constant, which directly governs the theoretical stability margin.

---

## 3. Adversarial attack (CW-L1)

Custom **Carlini–Wagner L1** attack (not PGD). Attacks are run only on **stable examples** (y = 1), measuring the minimum perturbation needed to flip the prediction to unstable.

**Objective:** Minimize L1 perturbation size while pushing the pre-activation logit below the decision boundary:

```
min_w  ||delta||_1 + c * max(z + kappa, 0)
```

The term `max(z + kappa, 0)` is zero when the attack succeeds with confidence margin `kappa` (i.e., `z < -kappa`), and positive otherwise. Run sweeps over `kappa` at `0`, `1σ`, and `2σ` (σ = std of model logits on the eval set).

**Box constraints:** Optimize unconstrained `w`, map to physical bounds `[L_i, U_i]`:

```
x_i = L_i + (U_i - L_i) / 2 * (tanh(w_i) + 1)
delta = x - x_0
```

**Attack loop:**

1. Initialize `w_0 = arctanh(2*(x_0 - L)/(U - L) - 1)` so that mapped `x ≈ x_0`.
2. Map `w → x` via tanh box constraint.
3. Apply physics routing: `x[p_1] = -(x[p_2] + x[p_3] + x[p_4])`.
4. Forward through frozen model; read logit `z`.
5. Compute CW-L1 loss; backprop to `w`; Adam step.

Note: `delta[p_1]` is included in `||delta||_1` and reflects the induced energy imbalance; it is fully determined by the perturbations to `p_2`–`p_4`.

---

## 4. Evaluation and outputs

Compare L1-regularized Model B against L2-regularized Model A to test the hypothesis that smaller L1 triple norms (induced by L1 regularization) yield larger stability margins and sparser adversarial perturbations.

**Required outputs:**

1. **Stability margin:** Mean `||delta||_1` to flip predictions for Model A vs Model B.
2. **Sparsity:** Perturbation vector `delta` showing 1–2 dominant nodes (single points of failure).
3. **Correlation plot:** Stability margin (Y) vs product of layer-wise L1 triple norms (X).
4. **Robustness curve:** Perturbation budget (X) vs attack success rate (Y) for both models.
