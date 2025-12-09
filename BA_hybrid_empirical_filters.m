function phi_empirical = BA_hybrid_empirical_filters(A,B,b,k,lambda)
%
%  INPUT:
%    A      : m x n
%    b      : m x 1
%    k      : number of BA-GMRES steps
%    lambda : Tikhonov parameter
%
%  OUTPUT:
%    phi    : r x 1 vector of filter factors phi_{i,k}^{h,BA}(\lambda)
%             in the SVD basis of A (r = rank(A))

    [m,n] = size(A);
    %B = A';           % matched case
    % 1) SVD of A
    [UA,SA,VA] = svd(A,'econ');
    sigma = diag(SA);
    r = nnz(sigma);   % numerical rank
    sigma = sigma(1:r);
    VA = VA(:,1:r);
    UA = UA(:,1:r);

    % 2) Use general BA routine to get Hbar, Qk, etc.
    [~, ~, ~,~, ~, ~,xk,~, ~] = BA_hybrid(A,B,b,k,lambda);

   
    % Express xk in SVD basis: xk = sum_i alpha_i v_i^A
    alpha = VA' * xk;    % r x 1

    % Now (u_i^A)^T b and the definition of phi_i:
    coeff_b = UA' * b;   % r x 1, (u_i^A)^T b

    phi_empirical = zeros(r,1);
    for i = 1:r
        if sigma(i) > 0 && coeff_b(i) ~= 0
            % alpha_i = phi_i * ( (u_i^A)^T b / sigma_i )
            phi_empirical(i) = alpha(i) * sigma(i) / coeff_b(i);
        else
            phi_empirical(i) = 0;
        end
    end
end
