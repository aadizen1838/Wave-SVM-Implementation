# Wave-SVM Implementation

Binary classification using Wave-SVM (Support Vector Machine with Wave Loss Function) and Adam optimizer.

## Results

**Accuracy: 95.91%**
- Correct Predictions: 164/171
- Training Time: 0.7 seconds
- Dataset: Breast Cancer (569 samples, 30 features)

## Files

- `Main_Wave_SVM.m` - Main executable script
- `Wave_Adam_function.m` - Training algorithm with Adam optimizer
- `Train.txt` - Training data (70%)
- `Test.txt` - Test data (30%)

## How to Run

### Requirements
- GNU Octave (or MATLAB)

### Steps

1. Open Octave
2. Navigate to project folder:
   ```octave
   cd /path/to/Wave-SVM-Implementation
   ```

3. Load data and run:
   ```octave
   load Train.txt
   load Test.txt
   Main_Wave_SVM
   ```

## Algorithm

**Wave-SVM** combines:
- **Wave Loss Function** - Robust to outliers, smooth for optimization
- **Adam Optimizer** - Adaptive learning rates with momentum
- **RBF Kernel** - Non-linear classification boundaries

## Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| C | 1 | Regularization strength |
| mew | 1 | Kernel parameter (gamma) |
| alpha | 0.01 | Learning rate |
| max_iter | 1000 | Training iterations |

To improve accuracy, try different values for `C` and `mew` in `Main_Wave_SVM.m`

## Implementation

- Language: MATLAB/Octave
- Environment: GNU Octave (open-source MATLAB alternative)
- License: MIT

## References

Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository
