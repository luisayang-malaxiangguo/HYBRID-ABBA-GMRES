function [delta, zeta, gamma, Hbar, Qk, Qk1, x_iterate, error_norm, residual_norm] = ...
          AB_hybrid(A, B, b, k, lambda, x_true) 
%  M = A*B, d = b.
%  Run k steps of Arnoldi on M with start d, giving
%     M Q_k = Q_{k+1} Hbar_k,
%  then form:
%     Hbar_k^T Hbar_k = S_k * Delta_k * S_k^T,
%     delta_j = sqrt(Delta_k(j,j)),
%     zeta_j  = (S_k^T * Hbar_k^T * e1)_j / delta_j,
%     gamma_j(lambda) = beta1 * zeta_j * delta_j / (delta_j^2 + lambda),
%
%  and finally
%     y_k^{h,AB}(lambda) = Q_k S_k gamma(lambda),
%     x_k^{h,AB}(lambda) = B * y_k^{h,AB}(lambda).
%
%  INPUT:
%    A        : m x n matrix
%    B        : n x m matrix (back-projector, not necessarily A')
%    b        : m x 1 RHS
%    k        : Arnoldi steps for AB–GMRES
%    lambda   : Tikhonov parameter
%    x_true   : (optional) n x 1 exact solution; if absent, error_norm=NaN
%
%  OUTPUT:
%    delta       : k x 1, projected singular values (sqrt eigs of Hbar' Hbar)
%    zeta        : k x 1, reduced weights
%    gamma       : k x 1, gamma_j(lambda)
%    Hbar        : (k+1) x k Arnoldi Hessenberg matrix
%    Qk          : m x k Arnoldi basis in data space (for y)
%    Qk1         : m x (k+1) Arnoldi basis
%    x_iterate   : n x 1, hybrid AB–GMRES iterate x_k^{h,AB}(lambda)
%    error_norm  : ||x_iterate - x_true||_2 / ||x_true||_2  (NaN if x_true absent)
%    residual_norm : ||b - A*x_iterate||_2 / ||b||_2  (data-space residual)
 
    if nargin < 6
        x_true = [];
    end

    M = A * B;   % m x m
    d = b;       % m x 1

    beta1 = norm(d);
    if beta1 == 0
        error('b is zero; cannot start Arnoldi for AB-hybrid.');
    end

    %  Arnoldi on M with starting vector d 
    m = size(A,1);
    Qk1  = zeros(m, k+1);
    Hbar = zeros(k+1, k);

    Qk1(:,1) = d / beta1;

    for j = 1:k
        v = M * Qk1(:,j);

        % Modified Gram–Schmidt
        for i = 1:j
            Hbar(i,j) = Qk1(:,i)' * v;
            v = v - Hbar(i,j) * Qk1(:,i);
        end

        Hbar(j+1,j) = norm(v);

        if Hbar(j+1,j) == 0
            % breakdown: Krylov subspace invariant
            Qk1(:,j+1) = zeros(m,1);
            Hbar = Hbar(1:j+1,1:j);  % truncate
            k = j;                   % effective dimension
            break;
        else
            Qk1(:,j+1) = v / Hbar(j+1,j);
        end
    end

    Qk   = Qk1(:,1:k);        % m x k
    Qk1  = Qk1(:,1:k+1);      % m x (k+1)
    Hbar = Hbar(1:k+1,1:k);   % (k+1) x k

    %  Projected normal matrix Hbar' Hbar 
    HN = Hbar' * Hbar;        % k x k

    % Spectral decomposition: HN = S_k * Delta_k * S_k^T
    [Sk, Dk] = eig(HN);       % columns of Sk are eigenvectors
    delta = sqrt(max(diag(Dk),0));

    % Sort by descending delta and reorder Sk
    [delta, perm] = sort(delta, 'descend');
    Sk = Sk(:,perm);

    %  zeta_j = (S_k^T Hbar^T e1)_j / delta_j 
    e1 = zeros(k+1,1); 
    e1(1) = 1;
    tmp = Sk' * (Hbar' * e1);   % k x 1

    zeta = zeros(k,1);
    for j = 1:k
        if delta(j) > 0
            zeta(j) = tmp(j) / delta(j);
        else
            zeta(j) = 0;        % convention if delta_j = 0
        end
    end

    %  gamma_j(lambda) = beta1 * zeta_j * delta_j / (delta_j^2 + lambda) 
    gamma = beta1 * zeta .* (delta ./ (delta.^2 + lambda));  % k x 1
  
    yk = Sk * gamma;       % k x 1
    zk     = Qk * yk;      % m x 1
    x_iterate = B * zk;        % n x 1

    residual_norm = norm(b - A * x_iterate) / norm(b);

    if ~isempty(x_true)
        error_norm = norm(x_iterate - x_true) / norm(x_true);
    else
        error_norm = NaN;
    end
end
