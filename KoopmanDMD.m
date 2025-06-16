clear; close all; clc;

%% Parameters
dt      = 0.01;           % time step
tSpan   = 0:dt:50;        % sample times
p       = 9;              % max polynomial degree for observables

%% Choose your system (Van-der-Pol)
mu  = 3;
rhs = @(t,x)[ x(2);
               mu*(1 - x(1).^2).*x(2) - x(1) ];
x0  = [2; 0];

%% Simulate & build snapshots
nStates = numel(x0);
X  = zeros(nStates, numel(tSpan));
X(:,1) = x0;
for k = 1:numel(tSpan)-1
    k1 = rhs(tSpan(k)      , X(:,k));
    k2 = rhs(tSpan(k)+dt/2 , X(:,k)+dt/2*k1);
    k3 = rhs(tSpan(k)+dt/2 , X(:,k)+dt/2*k2);
    k4 = rhs(tSpan(k)+dt   , X(:,k)+dt   *k3);
    X(:,k+1) = X(:,k) + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end
X1 = X(:,1:end-1);
X2 = X(:,2:end);

%% Build polynomial observables and compute SVD
[G1, exponentBank] = buildObservables(X1, p);
nFeatures = size(G1,1);
fprintf('Polynomial dictionary dimension = %d features\n', nFeatures);

[U_full,S_full,V_full] = svd(G1,'econ');
singvals = diag(S_full);

% plot singular-value spectrum
figure;
semilogy(singvals,'o-','LineWidth',1.2);
xlabel('Mode index'); ylabel('Singular value (log)');
title('Singular values of G1'); grid on;

%% Sweep truncation rank r = 1:nFeatures
r_list = 1:nFeatures;
err  = zeros(size(r_list));
true_x1 = X(1,1:end-1);
G2 = buildObservables(X2,p);
g0 = G1(:,1);

for ii = 1:length(r_list)
    r = r_list(ii);
    U_r = U_full(:,1:r);
    S_r = S_full(1:r,1:r);
    V_r = V_full(:,1:r);
    A_tilde = U_r' * G2 * V_r / S_r;
    [W,Lambda] = eig(A_tilde);
    Phi = G2 * V_r / S_r * W;
    omega = log(diag(Lambda))/dt;
    b      = Phi \ g0;
    tRecon = tSpan(1:end-1);
    coef   = bsxfun(@times, b, exp(omega * tRecon));
    G_pred = Phi * coef;
    idx_x1 = find(ismember(exponentBank,[1 0],'rows'));
    x1_pred = G_pred(idx_x1,:);
    err(ii) = norm(true_x1 - x1_pred,2) / norm(true_x1,2);
end

% plot error vs r
figure;
plot(r_list, err, 's-','LineWidth',1.2);
xlabel('Truncation rank r'); ylabel('Relative 2-norm error');
title('EDMD reconstruction error vs. r'); grid on;

%% Reconstruction plot at full dictionary rank (r = nFeatures)
r = nFeatures;
U_r = U_full(:,1:r); S_r = S_full(1:r,1:r); V_r = V_full(:,1:r);
A_tilde = U_r' * G2 * V_r / S_r;
[W,Lambda] = eig(A_tilde);
Phi = G2 * V_r / S_r * W;
omega = log(diag(Lambda))/dt;
b      = Phi \ g0;
tRecon = tSpan(1:end-1);
coef   = bsxfun(@times, b, exp(omega * tRecon));
G_pred = Phi * coef;
idx_x1 = find(ismember(exponentBank,[1 0],'rows'));
x1_pred = G_pred(idx_x1,:);

% plot true vs EDMD
figure;
plot(tSpan,    X(1,:),    'k-',  'LineWidth',1.2); hold on;
plot(tRecon,   x1_pred,   'r--', 'LineWidth',1.2);
xlabel('t'); legend('True x_1','EDMD x_1');
title(sprintf('EDMD vs True (p=%d, r=%d)', p, r));
grid on;

%% Helper functions ------------------------------------------------------
function [G, exponentBank] = buildObservables(X, p)
    [nStates,nSnaps] = size(X);
    exponentBank = [];
    for d = 1:p
        exponentBank = [exponentBank; integerPartitions(d,nStates)]; %#ok<AGROW>
    end
    nFeatures = size(exponentBank,1);
    G = ones(nFeatures,nSnaps);
    for f = 1:nFeatures
        alpha = exponentBank(f,:);
        vec   = ones(1,nSnaps);
        for i = 1:nStates
            vec = vec .* (X(i,:).^alpha(i));
        end
        G(f,:) = vec;
    end
end

function combos = integerPartitions(total, dims)
    if dims == 1
        combos = total; return
    end
    combos = [];
    for first = 0:total
        tails  = integerPartitions(total-first, dims-1);
        combos = [combos; first*ones(size(tails,1),1), tails]; %#ok<AGROW>
    end
end
