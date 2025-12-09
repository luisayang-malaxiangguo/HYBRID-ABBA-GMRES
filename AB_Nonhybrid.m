function [x_iterate, error_norm, residual_norm, phi] = ...
    AB_Nonhybrid(A, B, b, k, x_true)

%  Solve        M y ≈ b,    with   M = A*B   (m x m),
%  using        (non-hybrid) GMRES on M,
%  and compute filter factors φ_i^{(k)} 
%
%  In the *matched* case B = A', the nonzero eigenvalues μ_i of
%  M = A*B equal those of A'*A and thus coincide with (σ_i^A)^2.
%
%  The GMRES residual after k_eff steps can be written
%       r_{k_eff} = p_{k_eff}(M) r_0,
%  where p_{k_eff} is a degree-k_eff polynomial with p_{k_eff}(0) = 1.
%  The corresponding (non-hybrid) polynomial filter factors in the
%  eigenbasis of M are
%
%       φ_i^{(k_eff)} = 1 - ∏_{j=1}^{k_eff} (1 - μ_i / Θ_j),
%
%  where μ_i are eigenvalues of M and Θ_j are the harmonic Ritz values
%  of M extracted from the Arnoldi data.
%
%  INPUT:
%    A        : m x n forward operator
%    B        : n x m backprojector
%    b        : m x 1 right-hand side
%    k        : requested number of GMRES iterations
%    x_true   : n x 1 exact solution (for error computation)
%
%  OUTPUT:
%    x_iterate     : n x 1, final AB-GMRES iterate x_k = B*y_k
%    error_norm    : ||x_iterate - x_true||_2 / ||x_true||_2
%    residual_norm : ||b - A*x_iterate||_2 / ||b||_2   (data-space residual)
%    phi           : vector of non-hybrid filter factors φ_i^{(k_eff)}
%                    for the first k_eff eigenvalues μ_i of M = A*B
%
%  NOTES: 
%   - Harmonic Ritz values Θ_j are obtained from the k_eff x k_eff
%     Hessenberg block H_k and the subdiagonal h_{k+1,k} via
%         P = H_k + h_{k+1,k}^2 * (H_k^{-T} e_k e_k^T),
%     and Θ_j are the eigenvalues of P.

    % Build AB operator and its spectrum (for filters) 
    M = A * B;                   % m x m
    m = size(A,1);               % data-space dimension

    % Eigenvalues μ_i of M (may be non-symmetric)
    [~, D_M] = eig(M);
    mu_full = real(diag(D_M));

    % Sort in descending order (mode ordering convention)
    [mu_full, ~] = sort(mu_full, 'descend');

    % GMRES Arnoldi on M y = b, with at most k steps 
    d0    = b;                   % RHS for AB system
    r0    = d0;                  % initial residual (y0 = 0)
    beta0 = norm(r0);
    if beta0 == 0
        error('Initial residual b is zero; GMRES cannot start.');
    end

    Q = zeros(m, k+1);           % Arnoldi basis in data space
    H = zeros(k+1, k);           % Hessenberg
    Q(:,1) = r0 / beta0;

    zk    = zeros(m,1);          % GMRES solution in data space
    k_eff = 0;                   % effective iteration count

    for j = 1:k
        % Arnoldi step: expand K_j(M, b)
        v = M * Q(:,j);
        for i = 1:j
            H(i,j) = Q(:,i)' * v;
            v      = v - H(i,j) * Q(:,i);
        end
        H(j+1,j) = norm(v);

        if H(j+1,j) == 0
            % Happy breakdown: Krylov subspace is invariant
            k_eff = j;
            break;
        end

        Q(:,j+1) = v / H(j+1,j);
        k_eff    = j;
    end

    % Small GMRES least-squares problem at j = k_eff:
    %   minimize || beta0 e1 - \overline H_{k_eff} y ||
    Hk        = H(1:k_eff+1, 1:k_eff);
    rhs= [beta0; zeros(k_eff,1)];
    yk  = Hk \ rhs;         
    zk        = Q(:,1:k_eff) * yk;

    % Map back to x-space: x = B*y
    x_iterate = B * zk;

 
    residual_norm = norm(b - A * x_iterate) / norm(b);
    error_norm    = norm(x_iterate - x_true) / norm(x_true);

    % filter factors φ_i^{(k_eff)} 
    %
    % Harmonic Ritz values Θ_j of M from the Arnoldi data:
    %   Θ_j are eigenvalues of
    %       P = H_k + h_{k+1,k}^2 * (H_k^{-T} e_k e_k^T),
    %   where H_k is the k_eff x k_eff leading block of H and h_{k+1,k}
    %   is subdiagonal element.
    %

    Hk_square = H(1:k_eff, 1:k_eff);
    hkp1k     = H(k_eff+1, k_eff);

    ek = zeros(k_eff,1); 
    ek(end) = 1;

    % P = H_k + h_{k+1,k}^2 * H_k^{-T} e_k e_k^T
    P_eig = Hk_square + (hkp1k^2) * (Hk_square' \ (ek * ek'));

    [~, Th_eig] = eig(P_eig);
    Theta = real(diag(Th_eig));
    Theta = Theta(:);            % ensure column

    % Use first k_eff eigenvalues μ_i of M
    mu_current = mu_full(1:k_eff);

    % φ_i^{(k_eff)} = 1 - ∏_{j=1}^{k_eff} (1 - μ_i / Θ_j)
    % computed via log-sums for better numerical stability
    eps0 = eps;                  % safeguard against log(≤0)
    Clog = zeros(k_eff,1);
    for i = 1:k_eff
        factors = 1 - mu_current(i) ./ Theta.';
        % avoid log of non-positive due to roundoff
        factors = max(factors, eps0);
        Clog(i) = sum(log(factors));
    end
    P_final = exp(Clog);
    phi     = 1 - P_final;       
end
