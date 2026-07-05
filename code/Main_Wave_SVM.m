%% ======================================================================
%  WAVE-SVM: Binary Classification using Wave Loss Function
%  Dataset: Breast Cancer (UCI Machine Learning Repository)
%  
%  Purpose: Classify breast cancer as malignant (-1) or benign (1)
%  
%  Author: Student Implementation
%  Date: 2026
%% ======================================================================

clear all; close all; clc;

fprintf('\n');
fprintf('================== WAVE-SVM CLASSIFICATION ==================\n');
fprintf('Dataset: Breast Cancer Dataset\n');
fprintf('Problem: Binary Classification (Malignant vs Benign)\n');
fprintf('===========================================================\n\n');

%% LOAD DATA
% Data files are stored in ../data/
% Format: Each line = [label feature1 feature2 ... feature30]

fprintf('[1/5] Loading data...\n');

% Full paths relative to the code/ directory
train_file = '../data/Train.txt';
test_file  = '../data/Test.txt';

% Check that files exist
if ~exist(train_file, 'file')
    error('Cannot find %s', train_file);
end

if ~exist(test_file, 'file')
    error('Cannot find %s', test_file);
end

% Load numeric matrices
Train = load(train_file);
Test  = load(test_file);

alltrain = Train;
test = Test;

fprintf('      ✓ Train.txt loaded: %d samples × %d columns\n', ...
        size(alltrain,1), size(alltrain,2));

fprintf('      ✓ Test.txt loaded: %d samples × %d columns\n', ...
        size(test,1), size(test,2));

%% PARAMETER CONFIGURATION
% These parameters control the Wave-SVM algorithm
% You can tune these to improve accuracy

fprintf('\n[2/5] Setting parameters...\n');

% ===== WAVE LOSS PARAMETERS =====
a = 1;           % Controls transition point in sigmoid
b = 1;           % Controls sharpness of transition

% ===== SVM REGULARIZATION =====
C = 1;           % Regularization strength (try: 0.1, 1, 10, 100)
                 % Higher C: fits training data tighter (risk of overfitting)
                 % Lower C: smoother decision boundary (better generalization)

% ===== KERNEL PARAMETER =====
mew = 1;         % RBF kernel parameter (gamma)
                 % Controls kernel width (try: 0.01, 0.1, 1, 10)
                 % Higher mew: more complex boundary
                 % Lower mew: smoother boundary

% ===== OPTIMIZATION PARAMETERS =====
beta1 = 0.9;     % Adam parameter - exponential decay for 1st moment
beta2 = 0.999;   % Adam parameter - exponential decay for 2nd moment
alpha = 0.01;    % Learning rate (step size) - try: 0.001, 0.01, 0.1
                 % Controls how fast weights are updated
epsilon = 1e-8;  % Small constant to prevent division by zero

% ===== TRAINING CONFIGURATION =====
m = 32;          % Mini-batch size (samples per update)
                 % Higher m: more stable, slower
                 % Lower m: faster, noisier
max_iter = 1000; % Maximum iterations (training rounds)
                 % More iterations: longer training, potentially better accuracy

fprintf('      Wave Loss Parameters: a=%.2f, b=%.2f\n', a, b);
fprintf('      SVM Parameters: C=%.2f, mew=%.4f\n', C, mew);
fprintf('      Adam Parameters: beta1=%.3f, beta2=%.4f, alpha=%.4f\n', beta1, beta2, alpha);
fprintf('      Training Config: batch_size=%d, max_iterations=%d\n', m, max_iter);

%% CALL WAVE-SVM TRAINING FUNCTION
fprintf('\n[3/5] Training Wave-SVM model...\n');
fprintf('      (This may take 10-30 seconds)\n\n');

tic;  % Start timer
[Accuracy, TrainingTime] = Wave_Adam_function(alltrain, test, a, b, C, mew, ...
                                               beta1, beta2, m, max_iter, alpha, epsilon);
toc;  % Stop timer



%% DISPLAY RESULTS
fprintf('\n[4/5] Computing test metrics...\n');

fprintf('      ✓ Test set size: %d samples\n', size(test, 1));
fprintf('      ✓ Metrics computed during training\n');

%% PRINT FINAL RESULTS
fprintf('\n[5/5] Final Results\n');
fprintf('===========================================================\n');
fprintf('Classification Accuracy: %.2f%%\n', Accuracy);
fprintf('Training Time: %.2f seconds\n', TrainingTime);
fprintf('===========================================================\n\n');

fprintf('Interpretation:\n');
if Accuracy >= 95
    fprintf('  ★ Excellent: Model is very accurate\n');
elseif Accuracy >= 90
    fprintf('  ★ Very Good: Model performs very well\n');
elseif Accuracy >= 85
    fprintf('  ★ Good: Model is reasonably accurate\n');
elseif Accuracy >= 80
    fprintf('  ★ Acceptable: Model shows reasonable performance\n');
else
    fprintf('  ★ Needs Improvement: Consider tuning parameters\n');
end

fprintf('\nDataset Information:\n');
fprintf('  - Problem: Binary classification (Malignant vs Benign)\n');
fprintf('  - Samples: %d total (%.0f%% train, %.0f%% test)\n', ...
    size(alltrain,1) + size(test,1), ...
    100*size(alltrain,1)/(size(alltrain,1)+size(test,1)), ...
    100*size(test,1)/(size(alltrain,1)+size(test,1)));
fprintf('  - Features: 30 (normalized to [0,1])\n');
fprintf('  - Algorithm: Wave-SVM with Adam optimizer\n');
fprintf('  - Loss Function: Wave Loss (robust to outliers)\n');

fprintf('\nNext Steps:\n');
fprintf('  1. Try different parameter values to improve accuracy\n');
fprintf('  2. Analyze which parameters affect accuracy most\n');
fprintf('  3. Compare with standard SVM or other classifiers\n');
fprintf('  4. Document results and create final report\n\n');

%% HELPER FUNCTIONS (if not in separate files)

function [X_norm, scale] = normalize_data(X)
    % Normalize data to [0,1] range
    X_min = min(X);
    X_max = max(X);
    X_norm = (X - X_min) ./ (X_max - X_min + 1e-8);
    scale = X_max - X_min;
end

function K = rbf_kernel(X1, X2, gamma)
    % Compute RBF (Radial Basis Function) kernel matrix
    % K(x,y) = exp(-gamma * ||x-y||^2)
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    K = zeros(n1, n2);
    for i = 1:n1
        for j = 1:n2
            K(i,j) = exp(-gamma * sum((X1(i,:) - X2(j,:)).^2));
        end
    end
end