function phi_theoretical = AB_hybrid_theory_filters(A, B, b, k, lambda)
%  Theoretical hybrid BA–GMRES filter factors in the SVD basis of A
%  in the matched BA case B = A', where
%     M = A' * A,   d = A' * b,
%     mu_i = (sigma_i^A)^2 are eigenvalues of M,
%  and
%     phi_{i,k}^h(λ) =
%       [ μ_i / (v_i^A)' d ] * sum_{j=1}^k β1 ζ_j δ_j/(δ_j^2 + λ) c_{ij}^{(k)},
%  where:
%     μ_i      = (σ_i^A)^2,
%     d        = B*b,
%     β1       = ||d||_2,
%     δ_j      = reduced singular values of \overline H_k,
%     ζ_j      = (S_k^T \overline H_k^T e_1)_j / δ_j,
%     c_{ij}^k = (v_i^A)' w_j^{(k)}, w_j^{(k)} = Q_k S_k e_j.
%
%  INPUT:
%    A      : m x n matrix
%    B      : n x m backprojector (for matched BA, B = A')
%    b      : m x 1 RHS
%    k      : number of BA-GMRES steps
%    lambda : Tikhonov parameter
%
%  OUTPUT:
%    phi_theoretical : r x 1 vector of theoretical filter factors
%                      phi_{i,k}^{h,BA}(\lambda) in the SVD basis of A

    % SVD of A (to get σ_i and v_i^A)
    [UA, SA, ~] = svd(A, 'econ');
    sigma_all = diag(SA);
    r = nnz(sigma_all);          % numerical rank (can add tolerance if needed)
    sigma = sigma_all(1:r);
    UA    = UA(:,1:r);

    % μ_i = (σ_i^A)^2
    mu = sigma.^2;               % r x 1
 
    d     = b;               % n x 1
    beta1 = norm(d);             % β1 = ||d||

    % BA Arnoldi to get Hbar_k and Q_k 
    [~, ~, ~, Hbar, Qk, ~, ~, ~, ~] = AB_hybrid(A, B, b, k, lambda);

    % Spectral decomposition of Hbar_k' * Hbar_k
    HN = Hbar' * Hbar;           % k x k
    [Sk, Dk] = eig(HN);          % HN = Sk * Dk * Sk'
    delta = sqrt(max(diag(Dk),0));

    % Sort by descending δ_j and reorder Sk consistently
    [delta, perm] = sort(delta, 'descend');
    Sk = Sk(:, perm);

    % ζ_j (independent of λ)
    e1  = zeros(k+1,1); 
    e1(1) = 1;
    tmp = Sk' * (Hbar' * e1);    % k x 1 = (S_k^T Hbar^T e1)_j

    zeta = zeros(k,1);
    for j = 1:k
        if delta(j) > 0
            zeta(j) = tmp(j) / delta(j);
        else
            zeta(j) = 0;         % convention if δ_j = 0
        end
    end

    % γ_j(λ) in reduced singular basis
    % γ_j(λ) = β1 ζ_j δ_j / (δ_j^2 + λ)
    gamma = beta1 * zeta .* (delta ./ (delta.^2 + lambda));   % k x 1

    % Couplings c_{ij}^{(k)}
    %   W_k   = Q_k S_k          (n x k)
    %   C^k   = V_A^T W_k        (r x k)
    %        = V_A^T Q_k S_k
    %   sum_j γ_j c_{ij}^{(k)} = (C^k * gamma)_i.
    C_base = UA' * Qk;           % r x k ~ (v_i^A)' q_j
    C      = C_base * Sk;        % r x k, columns correspond to w_j^{(k)}

    sum_part = C * gamma;        % r x 1, ith entry = Σ_j γ_j c_{ij}^{(k)}

    % Assemble φ_{i,k}^h(λ) using μ_i / (v_i^A)' d
    uAd = UA' * d;               % r x 1, (v_i^A)' d

    phi_theoretical = zeros(r,1);
    for i = 1:r
        if mu(i) > 0 && uAd(i) ~= 0
            % φ_i = μ_i / (v_i^A)' d * Σ_j γ_j c_{ij}^{(k)}
            phi_theoretical(i) = mu(i) * sum_part(i) / uAd(i);
        else
            phi_theoretical(i) = 0;
        end
    end
end
