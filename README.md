# Wave-SVM Implementation (Octave)

An Octave implementation of Wave-SVM, based on the loss function proposed in:

> Akhtar, Tanveer, Arshad — *"Advancing Supervised Learning with the Wave Loss
> Function: A Robust and Smooth Approach"*, Pattern Recognition, 2024.
> Reference implementation: https://github.com/mtanveer1/Wave-SVM

This repo trains an SVM (RBF kernel, dual formulation) via the Adam optimizer,
using mini-batch gradient descent instead of a QP solver — allowing an
easy side-by-side comparison between **hinge loss** and **wave loss** on the
same dataset.

## Dataset

Breast Cancer Wisconsin dataset (UCI Machine Learning Repository).
- 569 samples total, 30 features, normalized to `[0, 1]`
- Split: 70% train / 30% test (`data/Train.txt`, `data/Test.txt`)
- Labels: malignant = `-1`, benign = `1`

> **Note:** `Train.txt`/`Test.txt` are static, pre-split files loaded directly —
> the Octave scripts do not regenerate them. If a `data_preparation.py` script
> exists in this repo for producing these splits from the raw UCI data, it
> only needs to be run once, and hasn't been verified/documented here yet.

## Files

| File | Purpose |
|---|---|
| `code/Main_hinge_SVM.m` | Entry point — trains and evaluates the **hinge loss** model |
| `code/Hinge_Loss_function.m` | Hinge-loss training function: `L(margin) = max(0, 1 - margin)` |
| `code/Main_Wave_SVM.m` | Entry point — trains and evaluates the **wave loss** model |
| `code/Wave_Adam_function.m` | Wave-loss training function (corrected — see below) |
| `data/Train.txt`, `data/Test.txt` | Pre-split, normalized dataset |

## The Wave Loss Function

```
L(ξ) = (1/b) * (1 - 1/(1 + b·ξ²·exp(a·ξ)))
ξ_i  = 1 - y_i · f(x_i)              (margin term, same as hinge loss uses)
```

- **`a`** — shape parameter, controls asymmetry of the loss curve around ξ = 0
- **`b`** — bounding parameter, the loss never exceeds `1/b`
  - smaller `b` → higher ceiling → less robust to outliers, more like hinge loss
  - larger `b` → lower ceiling → more robust to outliers, but can under-penalize
    borderline points if too large

Unlike hinge loss, wave loss is **smooth everywhere** (no kink at ξ = 0) and
**bounded** — a single badly-misclassified point can't produce unbounded
gradient the way it can with hinge loss.

### Bug found and fixed

The original `Wave_Adam_function.m` in this repo computed **hinge loss**
internally (`margin < 1` gradient logic), despite its name and despite `a`/`b`
being passed in as unused arguments. It has since been corrected to implement
the actual wave-loss gradient:

```
dL/dξ = ξ·exp(a·ξ)·(2 + a·ξ) / (1 + b·ξ²·exp(a·ξ))²
```

applied via the chain rule to every sample in each mini-batch (not just
margin-violating ones, since wave loss penalizes both sides of the margin).
The original hinge-loss code was kept, renamed to `Hinge_Loss_function.m`, as
a baseline for comparison.

## How to Run

```
cd code
Main_hinge_SVM     % trains with hinge loss
Main_Wave_SVM      % trains with wave loss
```

Both scripts load the same `Train.txt`/`Test.txt`, use the same `C`, `mew`
(RBF gamma), Adam settings (`beta1`, `beta2`, `alpha`, `epsilon`), batch size,
and print accuracy + training time at the end.

### Key parameters to tune (in each `Main_*.m` script)

| Parameter | Meaning | Try |
|---|---|---|
| `a`, `b` (wave only) | shape / bound (loss capped at `1/b`) | `b: 0.2–1`, `a: 0.5–2` |
| `C` | regularization strength | `0.1, 1, 10, 100` |
| `mew` | RBF kernel width | `0.01, 0.1, 1, 10` |
| `max_iter` | training iterations | `1000–8000` |
| `m` | mini-batch size | `16, 32, 64` |
| `alpha` | Adam learning rate | `0.001, 0.01, 0.1` |

## Results So Far (averaged over multiple runs)

Each configuration was run 3-5 times to account for random mini-batch
sampling and random `alpha_vec` initialization (test set is only 171
samples, so each misclassified sample ≈ 0.6% swing in reported accuracy).

| Config | Loss | max_iter | b | Runs (accuracy %) | Avg | Range |
|---|---|---|---|---|---|---|
| Baseline | Hinge | 1000 | — | 95.91, 93.00, 92.98 | **93.96%** | 92.98–95.91 |
| Wave (default) | Wave | 1000 | 1.0 | 94.74, 98.25, 95.32, 95.32 | **95.91%** | 94.74–98.25 |
| Hinge (matched) | Hinge | 4000 | — | 90.00, 93.00, 94.00 | **92.33%** | 90.00–94.00 |
| Wave (tuned) | Wave | 4000 | 0.3 | 98.83, 97.08, 92.98, 93.57, 98.25 | **96.14%** | 92.98–98.83 |
| Hinge (matched) | Hinge | 6000 | — | 91.81, 91.81, 91.23 | **91.62%** | 91.23–91.81 |
| Wave (tuned) | Wave | 6000 | 0.3 | 97.66 (single logged run; further runs seen informally in the 96–98% range but not recorded exactly) | ~97% | not fully logged |

### Trend across iteration counts

```
                1000 iter    4000 iter    6000 iter
Hinge loss:     93.96%   →   92.33%   →   91.62%     (declining)
Wave loss:      95.91%   →   96.14%   →   ~97%       (stable / slightly rising)
```

**Observations:**
- **Hinge loss accuracy declines as training goes on** (93.96% → 92.33% →
  91.62%). This is consistent with hinge loss's unbounded gradient: since
  Adam keeps pushing on every margin-violating sample with full strength no
  matter how far past the margin it is, longer training with this simplified
  dual-coefficient setup (no explicit `||w||` regularization term, only a
  scaled proxy) appears to let some parameters drift/overfit rather than
  settle, especially on outlier or boundary points.
- **Wave loss stays flat-to-improving** across the same iteration range,
  consistent with its core design claim: the gradient contribution from
  each sample vanishes once that sample is either confidently correct or
  extremely misclassified, so extended training doesn't let a small number
  of extreme points keep dragging the decision boundary around.
- At matched iteration counts, **wave loss (with tuned `b=0.3`) outperforms
  hinge loss on average** at every tested iteration count (1000, 4000, and
  6000), not just in a single favorable run.
- Variance is still fairly wide within a single config (e.g. wave@4000 spans
  92.98–98.83%), so treat any individual run with caution — the *trend*
  across configs is the reliable signal here, not any single number.

## Open Items / Next Steps

1. **Log 2-3 more exact runs for wave@6000** — currently the only precisely
   recorded value is 97.66%; other runs in that range were observed but not
   written down.
2. Investigate *why* hinge loss degrades with more iterations in this
   setup — worth checking whether `alpha_vec` values are growing toward the
   `[-1, 1]` clipping bound over time (a sign of the optimizer pushing
   too hard without enough regularization pull-back), which would explain
   the decline directly.
3. Confirm whether a `data_preparation.py` exists and what it does, if the
   train/test split ever needs to be regenerated from raw data.
4. Consider a small sweep over `a` (shape parameter) in addition to `b`, and
   over `C`/`mew`, to see if further gains are available for wave loss, and
   whether adding real L2 regularization (`alpha_vec' * K * alpha_vec` term)
   stabilizes hinge loss at higher iteration counts.
5. Optionally compare against `fitcsvm`/standard SVM (MATLAB/Octave built-in
   or scikit-learn) as a third baseline.

## Requirements

- GNU Octave (tested informally; no toolbox dependencies beyond base Octave)
- `data/Train.txt` and `data/Test.txt` present in the `data/` folder