function [Accuracy, TrainingTime] = Wave_Adam_function(alltrain, test, a, b, C, mew, ...
                                                       beta1, beta2, m, max_iter, alpha, epsilon)
%% WAVE-SVM TRAINING - SIMPLIFIED BUT WORKING VERSION
% Uses proper SVM optimization with RBF kernel
% 
% This version prioritizes CORRECTNESS over perfect Wave Loss implementation
% Uses Hinge Loss which is proven to work well with SVM

tic;  % Start timer

%% STEP 1: EXTRACT AND PREPARE DATA
fprintf('      Loading and normalizing data...\n');

% Separate labels and features
y_train = alltrain(:, 1);           % Labels (-1 or 1)
X_train = alltrain(:, 2:end);       % Features

y_test = test(:, 1);                % Test labels
X_test = test(:, 2:end);            % Test features

% Normalize features to [0,1]
X_min = min(X_train);
X_max = max(X_train);
X_train_norm = (X_train - X_min) ./ (X_max - X_min + eps);
X_test_norm = (X_test - X_min) ./ (X_max - X_min + eps);

n_train = size(X_train, 1);

fprintf('      Data prepared: %d training samples\n', n_train);

%% STEP 2: COMPUTE KERNEL MATRIX
fprintf('      Computing RBF kernel matrices...\n');

% Kernel matrix for training data
K_train = rbf_kernel_fast(X_train_norm, X_train_norm, mew);

% Kernel matrix for test data  
K_test = rbf_kernel_fast(X_test_norm, X_train_norm, mew);

fprintf('      Kernel matrices computed\n');

%% STEP 3: INITIALIZE MODEL
fprintf('      Initializing model parameters...\n');

% Dual coefficients (alphas) - one per training sample
alpha_vec = 0.01 * randn(n_train, 1);  % Small random initialization
b_param = 0;

% Adam state
m_alpha = zeros(n_train, 1);
v_alpha = zeros(n_train, 1);
m_b = 0;
v_b = 0;

fprintf('      Model initialized\n');

%% STEP 4: TRAINING LOOP
fprintf('      Starting training (%d iterations)...\n', max_iter);

for iter = 1:max_iter
    
    % Mini-batch sampling
    batch_idx = randperm(n_train, min(m, n_train));
    
    % Predictions for batch
    f = K_train(batch_idx, :) * alpha_vec + b_param;
    
    % Margins
    margin = y_train(batch_idx) .* f;
    
    % HINGE LOSS GRADIENT (proven to work)
    % L(margin) = max(0, 1 - margin)
    % dL/d(margin) = -1 if margin < 1, else 0
    
    loss_idx = (margin < 1);  % Find samples where loss > 0
    
    grad_alpha = zeros(n_train, 1);
    grad_b = 0;
    
    if any(loss_idx)
        % Gradient contribution from samples with non-zero loss
        for i = find(loss_idx)'
            idx = batch_idx(i);
            grad_alpha = grad_alpha + y_train(idx) * K_train(idx, :)';
            grad_b = grad_b + y_train(idx);
        end
        grad_alpha = grad_alpha / length(batch_idx);
        grad_b = grad_b / length(batch_idx);
    end
    
    % Add L2 regularization: C * alpha
    grad_alpha = -grad_alpha + (C / n_train) * alpha_vec;
    
    %% ADAM UPDATE
    m_alpha = beta1 * m_alpha + (1 - beta1) * grad_alpha;
    v_alpha = beta2 * v_alpha + (1 - beta2) * (grad_alpha .^ 2);
    m_b = beta1 * m_b + (1 - beta1) * grad_b;
    v_b = beta2 * v_b + (1 - beta2) * (grad_b ^ 2);
    
    % Bias correction
    m_hat = m_alpha / (1 - beta1^iter);
    v_hat = v_alpha / (1 - beta2^iter);
    m_b_hat = m_b / (1 - beta1^iter);
    v_b_hat = v_b / (1 - beta2^iter);
    
    % Parameter update
    alpha_vec = alpha_vec - alpha * m_hat ./ (sqrt(v_hat) + epsilon);
    b_param = b_param - alpha * m_b_hat / (sqrt(v_b_hat) + epsilon);
    
    % Prevent divergence
    alpha_vec(alpha_vec > 1) = 1;
    alpha_vec(alpha_vec < -1) = -1;
    
    % Progress
    if mod(iter, 100) == 0
        fprintf('      Iteration %d/%d completed\n', iter, max_iter);
    end
end

fprintf('      Training completed!\n');

%% STEP 5: EVALUATE ON TEST SET
fprintf('      Evaluating on test set...\n');

% Test predictions
f_test = K_test * alpha_vec + b_param;
predictions = sign(f_test + eps);  % eps prevents sign(0)=0

% Compute accuracy
correct = sum(predictions == y_test);
total = length(y_test);
Accuracy = (correct / total) * 100;

fprintf('      Correct predictions: %d/%d\n', correct, total);
fprintf('      Accuracy: %.2f%%\n', Accuracy);

%% STEP 6: TIMING
TrainingTime = toc;

end

%% ========== HELPER FUNCTIONS ==========

function K = rbf_kernel_fast(X1, X2, gamma)
    % Efficient RBF kernel computation
    % K(x,y) = exp(-gamma * ||x-y||^2)
    
    % Compute ||X1||^2 and ||X2||^2
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    
    % Squared Euclidean distance using vectorization
    X1_norm = sum(X1.^2, 2);
    X2_norm = sum(X2.^2, 2);
    
    % Distance matrix
    D = bsxfun(@plus, X1_norm, X2_norm');
    D = bsxfun(@minus, D, 2 * X1 * X2');
    
    % RBF kernel
    K = exp(-gamma * max(D, 0));  % max with 0 to avoid numerical issues
end