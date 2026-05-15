# CLAUDE.md

## Project

Physics-constrained adversarial sensitivity analysis on the UCI Electrical Grid Stability dataset. Two binary classifiers (L2 vs L1 regularization) are trained, then attacked with a custom Carlini–Wagner L1 attack to measure stability margins. Full specification in `README.md`.

---

## Technical invariants

These must never be violated:

- `p_1 = -(p_2 + p_3 + p_4)` is enforced in the forward pass during the attack. `p_1` is never an optimizable variable.
- All 11 non-`p_1` features are box-constrained via `x_i = L_i + (U_i - L_i)/2 * (tanh(w_i) + 1)`.
- The attack targets only stable examples (y = 1).
- CW-L1 objective: `||delta||_1 + c * max(z + kappa, 0)`.
- L1 regularization is applied as a manual penalty term in the training loss — PyTorch's `weight_decay` is L2 only.
- Triple norm per layer: `max_j sum_i |w_ij|`. Logged every epoch during training.

---

## Code standards

- Keep a modular structure, no god-files, clean separations and organized structure.
- No over-engineering, no error handling, no planning for the future. This is a one-time results-based project.
- If a library exists that cleanly handles a task, use it. Don't reimplement what's already well-solved.
- Minimal inline comments. Only comment why, never what.
- Separate documentation files for bigger parts of the repo for implementation reference.
- No docstrings unless a function's contract is genuinely non-obvious.
- Fixed random seed across all experiments — results must be reproducible.
- Trained model weights are saved to disk so attacks can be re-run without retraining.


## Implementation standards

- Keep all physics logic (energy conservation, bounds) in a single module so it is easy to explain and audit.
- Log triple norms after every epoch during training; save to a CSV for the correlation plot.
- Use a fixed random seed for all experiments (document the seed). Results must be reproducible.
- Save trained model weights so attacks can be re-run without retraining.
- All plots saved as PNG. Axis labels must include units where applicable.
---

## References

All libraries, datasets, papers, and any consulted code must be added to `references.md` as they are introduced. Do this immediately when a new dependency or source is used — not at the end.

- Papers: authors, title, venue, year, one line on what we use it for.
- Libraries: what we use it for (versions pinned at end via `pip freeze`).
- Adapted code: URL and what was taken from it.
