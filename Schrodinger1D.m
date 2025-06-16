% ========================================================================
% Split–Step Schrödinger + DMD reconstruction demo
% – keeps the t = 0 snapshot and aligns the time vector correctly –
% Jake Wicks · May 2025
% ========================================================================

clear; clc; close all;

%% 1.  Spatial grid and potential ---------------------------------------
N  = 2^12;           % # spatial points
Lx = 8;              % total simulated *time* in seconds
x  = Lx*(2*pi/N)*(-N/2:N/2-1)';     % physical space
V  = 0.5 * x.^2;                     % harmonic potential

%% 2.  Time‑stepping parameters -----------------------------------------
dt = 0.01;
nt = Lx/dt;          % 800 steps  (because Lx = 8)

k   = pi*[0:N/2-1 0 -N/2+1:-1]'/x(end);   % wave numbers
L   = -1i*k.^2*dt/2;
E   = exp(L);                              % linear step operator

%% 3.  Initial wave function --------------------------------------------
x1 =  4;  x2 = -4;  sigma = 0.5;
psi = 2*exp(-(x-x1).^2/(2*sigma^2)) ...
    + 2*exp(-(x-x2).^2/(2*sigma^2));

%% 4.  Pre‑allocate array *including* the initial field -----------------
psiPlot          = zeros(N, nt+1);   % 801 snapshots: t = 0 … 8
psiPlot(:,1)     = psi;              % t = 0
T                = (0:nt)*dt;        % 0, 0.01, …, 8.00  (length 801)

%% 5.  Split–step time integration -------------------------------------
for ktime = 1:nt
    % Linear half‑step in Fourier space
    psi = ifft( E .* fft(psi) );

    % Non‑linear + potential step
    theta = angle(psi) - dt*V - dt*abs(psi).^2;
    psi   = abs(psi) .* exp(1i*theta);

    % Second linear half‑step (Strang split complete)
    psi = ifft( E .* fft(psi) );

    psiPlot(:, ktime+1) = psi;     % store at t = ktime*dt
end

%% 6.  Build data matrices for DMD --------------------------------------
X  = psiPlot(:,1:end-1);   % snapshots  t = 0 … 7.99
X2 = psiPlot(:,2:end);     % snapshots  t = 0.01 … 8.00

%% 7.  Truncated SVD and DMD --------------------------------------------
r = 20;                               % rank truncation
[U,S,V] = svd(X,'econ');
U_r = U(:,1:r);  S_r = S(1:r,1:r);  V_r = V(:,1:r);

A_tilde = U_r' * X2 * V_r / S_r;     % r × r reduced operator
[W,D]   = eig(A_tilde);
lambda  = diag(D);                   % DMD eigenvalues
Phi     = X2 * V_r / S_r * W;        % DMD modes  (N × r)

% Initial amplitudes
b = Phi \ psiPlot(:,1);

% Time‑dynamics matrix (r × (#snapshots))
mSnaps        = size(psiPlot,2);     % 801
time_dynamics = zeros(r, mSnaps);
for k = 1:mSnaps
    time_dynamics(:,k) = (lambda.^(k-1)) .* b;
end

% Reconstruct all snapshots
X_dmd = Phi * time_dynamics;

%% 8.  Compare at the half‑way snapshot ---------------------------------
tidx = floor((mSnaps+1)/2);          % 401  →  t = 4.00 s

figure;
plot(x, real(psiPlot(:,tidx)), 'k', ...
     x, real(X_dmd(:,tidx)),  '--r', 'LineWidth',1.2);
legend('True','DMD');  axis tight;
xlabel('x');  ylabel('Re(\psi)');
title(sprintf('Snapshot at t = %.2f s', T(tidx)));

%% 9.  Space–time plots --------------------------------------------------
figure;
pcolor(T, x, real(psiPlot));  shading interp;  colorbar;
xlabel('t');  ylabel('x');  title('Real part of \psi(x,t)');

figure;
pcolor(T, x, imag(psiPlot));  shading interp;  colorbar;
xlabel('t');  ylabel('x');  title('Imag part of \psi(x,t)');
