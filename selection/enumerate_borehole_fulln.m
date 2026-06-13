%% enumerate_borehole_fulln.m
%  FULL-n exhaustive grammar enumeration for the Borehole/CPT case (n=744).
%
%  Purpose: the earlier ground-truth run (enumerate_grammar_aic.m) subsampled
%  borehole to n=150, where MA1+Constant won -- suspected to be a subsampling
%  artifact (paper's full-n table has Gaussian+Quadratic at 547.20, Markovian
%  at 550.10, nearly tied). This script settles the question by evaluating all
%  39 expressions x 4 means = 156 candidates at the FULL n=744 with exact GP.
%
%  Expected runtime: ~8 s/candidate for SE-like kernels (5 restarts), more for
%  composite kernels -> roughly 0.5-2 h total. Results saved incrementally.
%
%  Run:  >> enumerate_borehole_fulln

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

%% -------- CONFIG (same as enumerate_grammar_aic.m, but no subsampling) ----
N_RESTART = 5;
N_ITER    = -100;
n_means   = 4;
mean_names = {'Zero','Constant','Linear','Quadratic'};
eval_opts = struct('n_restart', N_RESTART, 'n_iter', N_ITER, 'quiet', true);

%% -------- DATA --------
S = load('samedata.mat'); x = S.x_same; y = S.y_same;
[n, D] = size(x);

[exprs, ginfo] = kernel_grammar();
n_expr = numel(exprs);
n_cand = n_expr * n_means;

fprintf('================================================================\n');
fprintf('  BOREHOLE FULL-n EXHAUSTIVE ENUMERATION\n');
fprintf('  n=%d, D=%d | %d expressions x %d means = %d candidates\n', ...
        n, D, n_expr, n_means, n_cand);
fprintf('  %d restarts each -> %d GP fits\n', N_RESTART, n_cand*N_RESTART);
fprintf('================================================================\n');

AIC = NaN(n_expr, n_means);
BIC = NaN(n_expr, n_means);
TT  = zeros(n_expr, n_means);

grand_t0 = tic;
for ei = 1:n_expr
    for mi = 1:n_means
        t0 = tic;
        r = evaluate_expr_aic(exprs(ei), mi, x, y, eval_opts);
        TT(ei,mi) = toc(t0);
        AIC(ei,mi) = r.aic;
        BIC(ei,mi) = r.bic;
    end
    % progress + incremental save after every expression
    [curA, iA] = min(AIC(:)); [eA, mA] = ind2sub(size(AIC), iA);
    fprintf('[%2d/%2d] %-14s done (%.0f s elapsed) | best so far: %s+%s AIC=%.2f\n', ...
            ei, n_expr, exprs(ei).name, toc(grand_t0), ...
            exprs(eA).name, mean_names{mA}, curA);
    save('enumerate_borehole_fulln_results.mat', ...
         'AIC','BIC','TT','exprs','ginfo','mean_names','N_RESTART','n','D');
end
grand_time = toc(grand_t0);

%% -------- SUMMARY --------
[minA, iA] = min(AIC(:)); [eA, mA] = ind2sub(size(AIC), iA);
[minB, iB] = min(BIC(:)); [eB, mB] = ind2sub(size(BIC), iB);

% reference points within this run (same n, same pipeline => comparable)
se_idx  = find(strcmp({exprs.name}, 'SE'),  1);
ma1_idx = find(strcmp({exprs.name}, 'MA1'), 1);
aic_se_quad  = AIC(se_idx, 4);
aic_ma1_best = min(AIC(ma1_idx, :));

fprintf('\n================================================================\n');
fprintf('  SUMMARY (full n=%d)\n', n);
fprintf('================================================================\n');
fprintf('BEST AIC : %-14s + %-9s  AIC=%.2f\n', exprs(eA).name, mean_names{mA}, minA);
fprintf('BEST BIC : %-14s + %-9s  BIC=%.2f\n', exprs(eB).name, mean_names{mB}, minB);
fprintf('Reference: SE+Quadratic (paper opt) AIC=%.2f  (dAIC vs best = %.2f)\n', ...
        aic_se_quad, aic_se_quad - minA);
fprintf('Reference: MA1 best-mean            AIC=%.2f  (dAIC vs best = %.2f)\n', ...
        aic_ma1_best, aic_ma1_best - minA);

% top-10 table
[sortedA, order] = sort(AIC(:));
fprintf('\nTop-10 by AIC:\n');
for k = 1:10
    [ek, mk] = ind2sub(size(AIC), order(k));
    fprintf('  #%2d  %-14s + %-9s  AIC=%.2f\n', ...
            k, exprs(ek).name, mean_names{mk}, sortedA(k));
end

fprintf('\nTotal wall-clock: %.0f s (%.1f h), mean %.1f s/candidate\n', ...
        grand_time, grand_time/3600, grand_time/n_cand);

save('enumerate_borehole_fulln_results.mat', ...
     'AIC','BIC','TT','exprs','ginfo','mean_names','N_RESTART','n','D','grand_time');
fprintf('Saved -> enumerate_borehole_fulln_results.mat\nDONE.\n');
