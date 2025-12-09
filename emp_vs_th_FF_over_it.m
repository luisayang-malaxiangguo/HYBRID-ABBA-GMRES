%% Evolution of empirical and theoretical filter factors
clear; clc; close all;

%% Problem setup
problemName = 'heat';   % 'shaw', 'heat', 'deriv2', ...
n      = 64;
maxit  = 32;            % maximum number of iterations
lambda = 1e-4;

[A, b_exact, x_true] = generate_test_problem(problemName, n);

rng(0);
noise_level = 1e-2;
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);

b = b_exact + noise;
B = A';                 % matched case

%% SVD of A and coefficients of b 
[UA,SA,VA] = svd(A,'econ');
sigma = diag(SA);
r = nnz(sigma);
sigma = sigma(1:r);
UA    = UA(:,1:r);
VA    = VA(:,1:r);

coeff_b = UA' * b;      % (u_i^A)' b    

%% Preallocate storage for evolution over k
phi_emp_hba_all = zeros(r, maxit);
phi_th_hba_all  = zeros(r, maxit);

phi_emp_hab_all = zeros(r, maxit);
phi_th_hab_all  = zeros(r, maxit);

% If you also want non-hybrid evolution:
phi_emp_ba_all  = zeros(r, maxit);
phi_th_ba_all   = zeros(r, maxit);

phi_emp_ab_all  = zeros(r, maxit);
phi_th_ab_all   = zeros(r, maxit);

%% Loop over iterations k = 1, ..., maxit
for k = 1:maxit
    fprintf('Processing iteration k = %d\n', k);

    %  Hybrid BA–GMRES 
    [~, ~, ~, ~, ~, ~, xk_hba, ~, ~] = BA_hybrid(A, B, b, k, lambda, x_true);
    phi_emp_hba = (VA' * xk_hba) .* sigma ./ coeff_b;   % r×1

    phi_th_hba  = BA_hybrid_theory_filters(A, B, b, k, lambda);  % r_th×1
    len_hba     = min(r, length(phi_th_hba));

    phi_emp_hba_all(1:len_hba, k) = phi_emp_hba(1:len_hba);
    phi_th_hba_all(1:len_hba,  k) = phi_th_hba(1:len_hba);

    %  Hybrid AB–GMRES 
    [~, ~, ~, ~, ~, ~, xk_hab, ~, ~] = AB_hybrid(A, B, b, k, lambda, x_true);
    phi_emp_hab = (VA' * xk_hab) .* sigma ./ coeff_b;   % r×1

    phi_th_hab  = AB_hybrid_theory_filters(A, B, b, k, lambda);  % r_th×1
    len_hab     = min(r, length(phi_th_hab));

    phi_emp_hab_all(1:len_hab, k) = phi_emp_hab(1:len_hab);
    phi_th_hab_all(1:len_hab,  k) = phi_th_hab(1:len_hab);

    %  Non-hybrid BA–GMRES  
    [xk_ba,  ~, ~, phi_ba] = BA_Nonhybrid(A, B, b, k, x_true);
    phi_emp_ba = (VA' * xk_ba) .* sigma ./ coeff_b;

    len_ba = min([r, length(phi_ba), length(phi_emp_ba)]);
    phi_emp_ba_all(1:len_ba, k) = phi_emp_ba(1:len_ba);
    phi_th_ba_all(1:len_ba,  k) = phi_ba(1:len_ba);

    %  Non-hybrid AB–GMRES  
    [xk_ab,  ~, ~, phi_ab] = AB_Nonhybrid(A, B, b, k, x_true);
    phi_emp_ab = (VA' * xk_ab) .* sigma ./ coeff_b;

    len_ab = min([r, length(phi_ab), length(phi_emp_ab)]);
    phi_emp_ab_all(1:len_ab, k) = phi_emp_ab(1:len_ab);
    phi_th_ab_all(1:len_ab,  k) = phi_ab(1:len_ab);

end

%% Plot evolution as heatmaps (i vs k)

kvec = 1:maxit;
ivec = 1:r;

figure;
subplot(2,2,1);
imagesc(kvec, ivec, phi_emp_hba_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Empirical \phi_{i,k}^{HBA}');
colorbar;

subplot(2,2,2);
imagesc(kvec, ivec, phi_th_hba_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Theoretical \phi_{i,k}^{HBA}');
colorbar;

subplot(2,2,3);
imagesc(kvec, ivec, phi_emp_hab_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Empirical \phi_{i,k}^{HAB}');
colorbar;

subplot(2,2,4);
imagesc(kvec, ivec, phi_th_hab_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Theoretical \phi_{i,k}^{HAB}');
colorbar;

%% If you also want non-hybrid evolution, you can add a second figure:
figure;
subplot(2,2,1);
imagesc(kvec, ivec, phi_emp_ba_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Empirical \phi_{i,k}^{BA}');
colorbar;

subplot(2,2,2);
imagesc(kvec, ivec, phi_th_ba_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Theoretical \phi_{i,k}^{BA}');
colorbar;

subplot(2,2,3);
imagesc(kvec, ivec, phi_emp_ab_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Empirical \phi_{i,k}^{AB}');
colorbar;

subplot(2,2,4);
imagesc(kvec, ivec, phi_th_ab_all);
set(gca, 'YDir', 'normal');
xlabel('k'); ylabel('i');
title('Theoretical \phi_{i,k}^{AB}');
colorbar;

i_list = [1, 8, 16, 32];
i_list = i_list(i_list <= r);
K = 1:maxit;

% Hybrid BA–GMRES
figure;
for idx = 1:numel(i_list)
    i = i_list(idx);
    subplot(2, ceil(numel(i_list)/2), idx);
    plot(K, phi_emp_hba_all(i,:), 'o-', ...
         K, phi_th_hba_all(i,:),  'x-', 'LineWidth', 1.2);
    xlabel('k');
    ylabel(sprintf('\\phi_{%d,k}', i));
    title(sprintf('Hybrid BA–GMRES (i = %d)', i));
    grid on;
end
legend('Empirical','Theoretical','Location','Best');

% Hybrid AB–GMRES
figure;
for idx = 1:numel(i_list)
    i = i_list(idx);
    subplot(2, ceil(numel(i_list)/2), idx);
    plot(K, phi_emp_hab_all(i,:), 'o-', ...
         K, phi_th_hab_all(i,:),  'x-', 'LineWidth', 1.2);
    xlabel('k');
    ylabel(sprintf('\\phi_{%d,k}', i));
    title(sprintf('Hybrid AB–GMRES (i = %d)', i));
    grid on;
end
legend('Empirical','Theoretical','Location','Best');

