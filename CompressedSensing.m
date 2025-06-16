%% Compressed Sensing Demo – toolbox‑free
%  Orthonormal DCT basis + ℓ1 recovery via ISTA
% ----------------------------------------------------------------------
clc; clear; close all;

%% Parameters -----------------------------------------------------------
n       = 256;           % ambient dimension
k       = 10;            % sparsity level
mVals   = 20:5:220;      % #measurements to sweep
lambda  = 5e-4;          % ℓ1 weight for ISTA
maxIter = 1000;          % ISTA iterations

%% Build DCT‑II orthonormal basis  (no toolboxes) -----------------------
Psi = make_dct_matrix(n)';     % so that x = Psi * s

%% Generate a k‑sparse coefficient vector  s_true -----------------------
s_true           = zeros(n,1);
support          = randperm(n,k);
s_true(support)  = randn(k,1);
x_true           = Psi * s_true;          % time‑domain signal

%% Sweep over number of measurements -----------------------------------
err         = zeros(size(mVals));
x_hat_last  = [];

for idx = 1:numel(mVals)
    m  = mVals(idx);

    % Random Gaussian sensing matrix  A  (rows ∼ 𝒩(0,1/m))
    A  = randn(m,n)/sqrt(m);

    y  = A * x_true;                   % noiseless samples
    M  = A * Psi;                      % combined matrix  (m×n)

    % ℓ1 reconstruction via ISTA ---------------------------------------
    s_hat  = ista(M, y, lambda, maxIter);

    x_hat      = Psi * s_hat;
    err(idx)   = norm(x_hat - x_true,2) / norm(x_true,2);

    if idx == numel(mVals)             % save for final plot
        x_hat_last = x_hat;
    end
end

%% Plot  error vs measurements -----------------------------------------
figure;
plot(mVals, err, 'o-', 'LineWidth', 1.4, 'MarkerSize', 6); grid on
xlabel('Number of Measurements');
ylabel('L2 Error');
title('Compressed Sensing Recovery');
set(gca,'FontSize',11);

%% Plot  original vs reconstructed signal ------------------------------
figure;
stem(x_true,'filled','DisplayName','Original'); hold on
stem(x_hat_last,'^','DisplayName','Reconstruction'); grid on
xlabel('Sample Index'); ylabel('Amplitude');
title(sprintf('Signal Reconstruction (m = %d samples)', mVals(end)));
legend('Location','best'); set(gca,'FontSize',11);

% ======================================================================
%                           local functions
% ======================================================================
function C = make_dct_matrix(N)
% make_dct_matrix  Generate the orthonormal DCT‑II matrix (NxN)
%
% The first row uses scale √(1/N); others use √(2/N), matching MATLAB’s
% dctmtx() but without requiring the Image Processing Toolbox.
    C = zeros(N,N);
    for k = 0:N-1
        alpha = sqrt(1/N) * (k==0) + sqrt(2/N) * (k~=0);
        for n = 0:N-1
            C(k+1,n+1) = alpha * cos(pi * k * (2*n+1) / (2*N));
        end
    end
end

function s = ista(M, y, lambda, maxIter)
% ISTA  – Iterative Soft‑Thresholding Algorithm for
%        min ½‖y − M s‖₂² + λ‖s‖₁
    L = norm(M,2)^2;         % Lipschitz constant
    t = 1/L;                 % step size
    s = zeros(size(M,2),1);
    for it = 1:maxIter
        g     = M'*(M*s - y);            % gradient
        s_new = soft(s - t*g, t*lambda); % prox step
        if norm(s_new - s, 2) < 1e-8, s = s_new; break, end
        s = s_new;
    end
end

function z = soft(x, thresh)
% soft  – componentwise soft‑thresholding
    z = sign(x) .* max(abs(x) - thresh, 0);
end
