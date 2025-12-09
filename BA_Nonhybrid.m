function [x_iterate, error_norm, residual_norm, phi] = ...
    BA_Nonhybrid(A, B, b, k, x_true) 
%  Solve M x ≈ d,   with   M = B*A,   d = B*b using GMRES on M,
%  and compute filter factors φ_i^{(k)} 
%
%  INPUT:
%    A        : m x n forward operator
%    B        : n x m backprojector
%    b        : m x 1 right-hand side
%    k        : requested number of GMRES iterations 
%    x_true   : n x 1 exact solution (for error computation)
%
%  OUTPUT:
%    x_iterate     : n x 1, final GMRES iterate x_k
%    error_norm    : ||x_iterate - x_true||_2 / ||x_true||_2
%    residual_norm : ||b - A*x_iterate||_2 / ||b||_2   (data space-residual)
%    phi           : vector of filter factors φ_i^{(k)} 
%
%  NOTES: 
%   - Filter factors φ_i^{(k)} are  
%         φ_i^{(k)} = 1 - ∏_{j=1}^k (1 - μ_i / Θ_j),
%     where μ_i are eigenvalues of M and Θ_j are harmonic Ritz values of M.
 
    M = B * A;
    n = size(A, 2);

    % Eigenvalues μ_i of M (may be non-symmetric)
    [V_M, D_M] = eig(M); % eigenvectors not needed here
    mu_full = real(diag(D_M));
    % Sort in descending order (mode ordering convention)
    [mu_full, ~] = sort(mu_full, 'descend');
 
    d0    = B * b;                 
    r0    = d0;                    % initial residual (x0 = 0)
    beta0 = norm(r0);
    if beta0 == 0
        error('Initial residual d0 = B*b is zero; GMRES cannot start.');
    end

    Q = zeros(n, k+1);
    H = zeros(k+1, k);
    Q(:,1) = r0 / beta0;
 
    k_eff = 0;                     % effective iterations (in case of breakdown)

    for j = 1:k
        % Arnoldi step: expand K_j(M, d)
        v = M * Q(:,j);
        for i = 1:j
            H(i,j) = Q(:,i)' * v;
            v      = v - H(i,j) * Q(:,i);
        end
        H(j+1,j) = norm(v);

        if H(j+1,j) == 0
            % Happy breakdown: Krylov space invariant
            k_eff = j;
            break;
        end

        Q(:,j+1) = v / H(j+1,j);
        k_eff = j;
    end

    % Small GMRES least-squares problem at j = k_eff
    Hk       = H(1:k_eff+1, 1:k_eff);
    rhs_small = [beta0; zeros(k_eff,1)];
    yk       = Hk \ rhs_small;      % MATLAB solves LS if overdetermined
    xk       = Q(:,1:k_eff) * yk;

    x_iterate = xk;

  
    residual_norm = norm(b - A * x_iterate) / norm(b);
    error_norm    = norm(x_iterate - x_true) / norm(x_true);

    % filter factors φ_i^{(k_eff)} 
    %
    % Harmonic Ritz values Θ_j of M from the Arnoldi data:
    %   Θ_j are eigenvalues of
    %       P = H_k + h_{k+1,k}^2 * (H_k^{-T} e_k e_k^T),
    %   where H_k is the k x k leading block of H and h_{k+1,k} is
    %   the subdiagonal element.
    %
    Hk_square = H(1:k_eff, 1:k_eff);
    hkp1k     = H(k_eff+1, k_eff);

    ek = zeros(k_eff,1); ek(end) = 1;
    P_eig = Hk_square + (hkp1k^2) * (Hk_square' \ (ek*ek'));

    [W_theta, Th_eig] = eig(P_eig); 
    Theta = real(diag(Th_eig)); %harmonic ritz values are eigenvalues of P
    Theta = Theta(:);              % ensure column

    % Use first k_eff eigenvalues μ_i of M
    mu_current = mu_full(1:k_eff);

    % φ_i^{(k_eff)} = 1 - ∏_{j=1}^{k_eff} (1 - μ_i / Θ_j)
    % computed via log-sums for stability
    eps0 = eps;                    % safeguard
    Clog = zeros(k_eff,1);
    for i = 1:k_eff
        factors = 1 - mu_current(i) ./ Theta.';
        % avoid log of non-positive due to roundoff
        factors = max(factors, eps0);
        Clog(i) = sum(log(factors));
    end
    P_final = exp(Clog);
    phi = 1 - P_final;             % non-hybrid filter factors
end
