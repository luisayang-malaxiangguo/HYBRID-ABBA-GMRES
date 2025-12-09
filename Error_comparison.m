%% Compare hybrid and non-hybrid error norms:
%   - error norms ||x_k - x_true|| / ||x_true||
%   - residual norms ||b - A*x_k|| / ||b||

clear; clc; close all;

%%
problemName = 'heat';    % 'shaw', 'heat', 'deriv2', ...
n          = 64;
maxit      = 32;         
lambda     = 1e-4;        
tol        = 0;         
 
[A, b_exact, x_true] = generate_test_problem(problemName, n);

rng(0); % For reproducibility
noise_level = 1e-2; 
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);

b = b_exact + noise; % Noisy right-hand side
B = A';                 

err_hba  = zeros(maxit,1);
res_hba  = zeros(maxit,1);

err_hab  = zeros(maxit,1);
res_hab  = zeros(maxit,1);

err_ba = zeros(maxit,1);
res_ba = zeros(maxit,1);

err_ab = zeros(maxit,1);
res_ab = zeros(maxit,1);

%% 
for i = 1:maxit

    [~, ~, ~, ~, ~, ~, ~, err_k_hba, res_k_hba] = BA_hybrid(A, B, b, i, lambda, x_true);
    err_hba(i) = err_k_hba;
    res_hba(i) = res_k_hba;
 
    [~, ~, ~, ~, ~, ~, ~, err_k_hab, res_k_hab] = AB_hybrid(A, B, b, i, lambda, x_true);
    err_hab(i) = err_k_hab;
    res_hab(i) = res_k_hab;

    [xk_ba, err_k_ba, res_k_ba, phi_ba] =BA_Nonhybrid(A, B, b, i, x_true); 
    err_ba(i) = err_k_ba;
    res_ba(i) = res_k_ba;

    [xk_ab, err_k_ab, res_k_ab, phi_ab] =AB_Nonhybrid(A, B, b, i, x_true); 
    err_ab(i) = err_k_ab;
    res_ab(i) = res_k_ab;


end

%% Plot: error norms (semi-convergence)
figure('Name', 'Convergence History', 'Position', [100 100 1000 400]);

subplot(1,2,1);
semilogy(1:maxit, err_hba,  '--', ...
         1:maxit, err_hab,':', ...
         1:maxit, err_ba, 'x--', ...
         1:maxit, err_ab,'-','LineWidth', 1.8);
xlabel('k');
ylabel('||x_k - x_{true}|| / ||x_{true}||');
legend('Hybrid BA–GMRES','Hybrid AB–GMRES', 'BA–GMRES','AB–GMRES', 'Location','Best');
title(sprintf('Relative Error vs. Iteration, %s, n=%d, \\lambda = %.1e', problemName, n, lambda));
grid on;


subplot(1,2,2);
semilogy(1:maxit, res_hba,  '--', ...
         1:maxit, res_hab,':',...
         1:maxit, res_ba, 'x--', ...
         1:maxit, res_ab,'-','LineWidth', 1.8);
xlabel('k');
ylabel('||b - A x_k|| / ||b||');
legend('Hybrid BA–GMRES','Hybrid AB–GMRES', 'BA–GMRES','AB–GMRES', 'Location','Best');
title('Relative Residual vs. Iteration');
grid on;
