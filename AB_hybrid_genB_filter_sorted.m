function [phi_sorted, mu_sorted] = AB_hybrid_genB_filter_sorted(A, B, b, k, lambda)
 
%  Theoretical hybrid AB–GMRES filter factors in the eigenbasis of M = AB,
%  with eigenvalues sorted by decreasing |mu_i|.
%
%  We solve M y = d with
%      M = A*B (m x m),   d = b (m x 1), 
%
%  In the eigenbasis of M, M = S*Lambda*S^{-1}, the exact solution y^dagger
%  has coordinates c = S^{-1} y^dagger, and with d = M y^dagger:
%      tilde_d = S^{-1} d = Lambda c  =>  c_i = tilde_d_i / mu_i.
%
%  The hybrid iterate can be written
%      y_k(λ) = sum_i phi_{i,k}(λ) c_i s_i,
%  with filter factors
%
%    phi_{i,k}(λ) =
%        [ μ_i / (tilde_d_i) ] * sum_{j=1}^k β1 ζ_j δ_j/(δ_j^2 + λ) c_{ij}^{(k)},
%
%  where:
%    μ_i        = eigenvalues of M = A*B,
%    tilde_d    = S^{-1} d = S^{-1} b,
%    β1         = ||d||_2 = ||b||_2,
%    δ_j, ζ_j, c_{ij}^{(k)} defined via Arnoldi/Tikhonov.
%
%  The eigenpairs (μ_i, s_i) are sorted so that |μ_1| ≥ |μ_2| ≥ ... ≥ |μ_m|.
%  Under B = A', M = A*A' is SPD, so μ_i = σ_i(A)^2 and this ordering aligns
%  with the standard SVD ordering (σ_1 ≥ σ_2 ≥ ...).
%
%  INPUT:
%    A      : m x n matrix
%    B      : n x m backprojector
%    b      : m x 1 RHS
%    k      : number of AB-GMRES steps
%    lambda : Tikhonov parameter
%
%  OUTPUT:
%    phi_sorted : m x 1 vector of filter factors in eigenbasis of M = AB,
%                 ordered by decreasing |μ_i|.
%    mu_sorted  : m x 1 vector of eigenvalues μ_i in the same order.

 
    M = A * B;        % m x m
    d = b;            % m x 1
    beta1 = norm(d);  % β1 = ||d||_2

    %  1) Eigen-decomposition of M 
    [S, Lambda] = eig(M);
    mu = diag(Lambda);         % μ_i (possibly complex for general B)

    % Sort by descending |μ_i|
    [~, perm] = sort(abs(mu), 'descend');
    mu_sorted = mu(perm);
    S_sorted  = S(:, perm);

    %  2) Arnoldi data from AB_hybrid 
    % AB_hybrid should do k Arnoldi steps on M = A*B with starting vector d = b
 
    [~, ~, ~, Hbar, Qk, ~, ~, ~, ~] = AB_hybrid(A, B, b, k, lambda);
    % Hbar: (k+1) x k,  Qk: m x k

    %  3) Spectral decomposition of Hbar' * Hbar 
    HN = Hbar' * Hbar;           % k x k
    [Sk, Dk] = eig(HN);          % HN = Sk * Dk * Sk'
    delta = sqrt(max(diag(Dk),0));

    % Sort δ_j in descending order and reorder Sk
    [delta, perm_d] = sort(delta, 'descend');
    Sk = Sk(:, perm_d);

    %  4) ζ_j 
    e1 = zeros(k+1,1);
    e1(1) = 1;
    tmp = Sk' * (Hbar' * e1);    % k x 1

    zeta = zeros(k,1);
    for j = 1:k
        if delta(j) > 0
            zeta(j) = tmp(j) / delta(j);
        else
            zeta(j) = 0;
        end
    end

    %  5) γ_j(λ) 
    gamma = beta1 * zeta .* (delta ./ (delta.^2 + lambda));   % k x 1

    %  6) Couplings c_{ij}^{(k)} 
    %   W_k   = Q_k S_k              (m x k)
    %   C^k   = S_sorted^{-1} W_k    (m x k)
    %        = S_sorted^{-1} Q_k S_k
    %   sum_j γ_j c_{ij}^{(k)} = (C^k * gamma)_i.

    C_base = S_sorted \ Qk;      % m x k, rows correspond to eigenvectors s_i
    C      = C_base * Sk;        % m x k, columns indexed by j

    sum_part = C * gamma;        % m x 1

    %  7) Assemble φ_{i,k}^h(λ) 
    tilde_d = S_sorted \ d;      % m x 1

    m_dim = length(mu_sorted);
    phi_sorted = zeros(m_dim,1);
    for i = 1:m_dim
        if mu_sorted(i) ~= 0 && tilde_d(i) ~= 0
            phi_sorted(i) = mu_sorted(i) * sum_part(i) / tilde_d(i);
        else
            phi_sorted(i) = 0;
        end
    end
end
