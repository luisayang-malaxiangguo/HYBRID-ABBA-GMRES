%% Compare empirical and theoretical filter factors at a single iteration
clear; clc; close all;

%% Problem setup
problemName = 'heat';   % 'shaw', 'heat', 'deriv2', ...
n = 64;
maxit = 32;      
k = maxit;
lambda = 1e-4;      
 
[A, b_exact, x_true] = generate_test_problem(problemName, n);

rng(0);  
noise_level = 1e-2; 
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);

b = b_exact + noise; % Noisy rhs      
B=A';
%% Empirical
[UA,SA,VA] = svd(A,'econ');
sigma = diag(SA);
r = nnz(sigma);   % numerical rank
sigma = sigma(1:r);
VA = VA(:,1:r);
UA = UA(:,1:r);
 
[~, ~, ~,~, ~, ~,xk_hba,~, ~] = BA_hybrid(A,B,b,k,lambda, x_true);
[~, ~, ~,~, ~, ~,xk_hab,~, ~] = AB_hybrid(A,B,b,k,lambda, x_true);
    

[xk_ba,  ~, ~, phi_ba] = BA_Nonhybrid(A, B, b, k, x_true);
[xk_ab,  ~, ~, phi_ab] = AB_Nonhybrid(A, B, b, k, x_true);

coeff_b = UA' * b;   
phi_empirical_hba = (VA' * xk_hba).* sigma ./ coeff_b;
phi_empirical_hab = (VA' * xk_hab).* sigma ./ coeff_b;
phi_empirical_ba = (VA' * xk_ba).* sigma ./ coeff_b;
phi_empirical_ab = (VA' * xk_ab).* sigma ./ coeff_b;

%% Hybrid  

phi_theoretical_hba = BA_hybrid_theory_filters(A, B, b, k, lambda);
 r_emp_hba = length(phi_empirical_hba);
r_th_hba  = length(phi_theoretical_hba);
r_hba     = min(r_emp_hba, r_th_hba);
phi_emp_hba = phi_empirical_hba(1:r_hba);
phi_th_hba  = phi_theoretical_hba(1:r_hba);
idx_hba = 1:r_hba;


phi_theoretical_hab = AB_hybrid_theory_filters(A, B, b, k, lambda);
r_emp_hab = length(phi_empirical_hab);
r_th_hab  = length(phi_theoretical_hab);
r_hab     = min(r_emp_hab, r_th_hab);
phi_emp_hab = phi_empirical_hab(1:r_hab);
phi_th_hab  = phi_theoretical_hab(1:r_hab);
idx_hab = 1:r_hab;
%% Non-hybrid  

% AB–GMRES
r_emp_ab = length(phi_empirical_ab);
r_th_ab  = length(phi_ab);
r_ab     = min(r_emp_ab, r_th_ab);
phi_emp_ab = phi_empirical_ab(1:r_ab);
phi_th_ab  = phi_ab(1:r_ab);
idx_ab = 1:r_ab;

% BA–GMRES
r_emp_ba = length(phi_empirical_ba);
r_th_ba  = length(phi_ba);
r_ba     = min(r_emp_ba, r_th_ba);
phi_emp_ba = phi_empirical_ba(1:r_ba);
phi_th_ba  = phi_ba(1:r_ba);
idx_ba = 1:r_ba;

%% Plot  
figure('Name', 'Empirical vs Theoretical FF', 'Position', [100 100 1000 400]);

subplot(2,2,1);
plot(idx_hba, phi_emp_hba, '--', ...
     idx_hba, phi_th_hba,  'o-', 'LineWidth', 1.2);
hold off; grid on;
xlabel('i');
ylabel('\phi_{i,k}');
legend('Empirical', 'Theoretical', 'Location','Best');
title(sprintf('Hybrid BA–GMRES, B = A^T, k=%d, \\lambda=%.1e', ...
      k, lambda));
grid on;

subplot(2,2,2);
plot(idx_hab, phi_emp_hab, '--', ...
     idx_hab, phi_th_hab,  'o-', 'LineWidth', 1.2);
xlabel('i');
ylabel('\phi_{i,k}');
legend('Empirical', 'Theoretical', 'Location','Best');
title(sprintf('Hybrid AB–GMRES'));
grid on;

subplot(2,2,3);
plot(idx_ba, phi_emp_ba, '--', ...
     idx_ba, phi_th_ba,  'o-', 'LineWidth', 1.2);
xlabel('i');
ylabel('\phi_{i,k}');
legend('Empirical', 'Theoretical', 'Location','Best');
title(sprintf('BA–GMRES'));
grid on;

subplot(2,2,4);
plot(idx_ab, phi_emp_ab, '--', ...
     idx_ab, phi_th_ab,  'o-', 'LineWidth', 1.2);
xlabel('i');
ylabel('\phi_{i,k}');
legend('Empirical', 'Theoretical', 'Location','Best');
title(sprintf('AB–GMRES'));
grid on;
%%
figure('Name', 'Absolute FF difference', 'Position', [100 100 1000 400]);
subplot(2,2,1);
semilogy(idx_hba, abs(phi_emp_hba - phi_th_hba), 'o', 'LineWidth', 1.2);
xlabel('i');
ylabel('| \phi^{emp}_i - \phi^{th}_i |');
title('Absolute difference Hybrid BA-GMRES');
grid on;

subplot(2,2,2);
semilogy(idx_hab, abs(phi_emp_hab - phi_th_hab), 'o', 'LineWidth', 1.2);
xlabel('i');
ylabel('| \phi^{emp}_i - \phi^{th}_i |');
title('Absolute difference Hybrid AB-GMRES');
grid on;

subplot(2,2,3);
semilogy(idx_ba, abs(phi_emp_ba - phi_th_ba), 'o', 'LineWidth', 1.2);
xlabel('i');
ylabel('| \phi^{emp}_i - \phi^{th}_i |');
title('Absolute difference BA-GMRES');
grid on;

subplot(2,2,4);
semilogy(idx_ab, abs(phi_emp_ab - phi_th_ab), 'o', 'LineWidth', 1.2);
xlabel('i');
ylabel('| \phi^{emp}_i - \phi^{th}_i |');
title('Absolute difference AB-GMRES');
grid on;