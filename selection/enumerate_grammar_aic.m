%% enumerate_grammar_aic.m
%  EXHAUSTIVE evaluation of the full compositional grammar on every test case.
%
%  For each case (EX1-EX4 synthetic + Borehole/CPT), fit a GP for ALL
%  (kernel-expression x mean) candidates and record AIC/BIC and wall-clock
%  time. This produces:
%    (1) the ground-truth optimal composite kernel per case (lowest AIC/BIC),
%    (2) the REAL exhaustive cost (replacing the paper's theoretical Table 8),
%    (3) a comparison against the fixed 4-kernel / 16-combination library.
%
%  Results are saved to enumerate_grammar_aic_results.mat.
%
%  Run:  >> enumerate_grammar_aic

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

%% -------- CONFIG --------
N_RESTART = 5;            % multi-restart for robust optimization
N_ITER    = -100;         % minimize: max function evaluations
N_SUB     = 150;          % cap training points used for SELECTION (exact GP is
                          % O(n^3); selection is robust to subsampling). Cases
                          % with n>N_SUB are randomly subsampled for the
                          % exhaustive ground-truth; the full-n cost is probed
                          % separately for the necessity/timing argument.
n_means   = 4;            % Zero, Constant, Linear, Quadratic
mean_names = {'Zero','Constant','Linear','Quadratic'};
eval_opts = struct('n_restart', N_RESTART, 'n_iter', N_ITER, 'quiet', true);

%% -------- DATA (same generators as LLM_Evaluation_Framework) --------
rng(1);
sn = 0.1; n_syn = 100;
x1 = rand(n_syn,2); y1 = x1(:,2) + 0.1*x1(:,1) + sn*randn(n_syn,1);
x2 = rand(n_syn,2); y2 = x2(:,2).^2 + x2(:,1).*x2(:,2) + sn*randn(n_syn,1);
x3 = rand(n_syn,2); y3 = sin(x3(:,2)) + x3(:,1) + sn*randn(n_syn,1);
x4 = rand(n_syn,3); y4 = x4(:,1).*x4(:,2) + x4(:,2).^2 + x4(:,3).^2 + sn*randn(n_syn,1);
S = load('samedata.mat'); x5 = S.x_same; y5 = S.y_same;

cases = struct('name', {'EX1 (linear)','EX2 (polynomial)','EX3 (trigonometric)', ...
                        'EX4 (3D nonlinear)','Borehole (CPT)'}, ...
               'x', {x1,x2,x3,x4,x5}, 'y', {y1,y2,y3,y4,y5});
n_cases = numel(cases);

% Paper's fixed-library optimum (Gaussian == SE) for reference comparison.
paper_opt = struct('kernel', {'SE','SE','SE','SE','SE'}, ...
                   'mean_id', {3,1,4,4,4});   % Lin,Zero,Quad,Quad,Quad

%% -------- ENUMERATE GRAMMAR --------
[exprs, ginfo] = kernel_grammar();
n_expr = numel(exprs);
n_cand = n_expr * n_means;

fprintf('================================================================\n');
fprintf('  EXHAUSTIVE COMPOSITIONAL-KERNEL AIC/BIC ENUMERATION\n');
fprintf('  %d kernel expressions x %d means = %d candidates / case\n', ...
        n_expr, n_means, n_cand);
fprintf('  %d cases, %d restarts each -> %d GP fits total\n', ...
        n_cases, N_RESTART, n_cases*n_cand);
fprintf('================================================================\n');

% storage: AIC(case, expr, mean), BIC, time
AIC = NaN(n_cases, n_expr, n_means);
BIC = NaN(n_cases, n_expr, n_means);
TT  = zeros(n_cases, n_expr, n_means);
results = cell(n_cases,1);

full_n = zeros(n_cases,1);
grand_t0 = tic;
for c = 1:n_cases
    x = cases(c).x; y = cases(c).y; [n_orig,D] = size(x);
    full_n(c) = n_orig;
    % subsample for tractable exact-GP selection ground-truth
    if n_orig > N_SUB
        rng(12345);
        sel = randperm(n_orig, N_SUB);
        x = x(sel,:); y = y(sel);
    end
    n = size(x,1);
    if n_orig > N_SUB
        fprintf('\n--- Case %d: %s  (n=%d of %d subsampled, D=%d) ---\n', ...
                c, cases(c).name, n, n_orig, D);
    else
        fprintf('\n--- Case %d: %s  (n=%d, D=%d) ---\n', c, cases(c).name, n, D);
    end
    case_t0 = tic;
    for ei = 1:n_expr
        for mi = 1:n_means
            t0 = tic;
            r = evaluate_expr_aic(exprs(ei), mi, x, y, eval_opts);
            TT(c,ei,mi) = toc(t0);
            AIC(c,ei,mi) = r.aic;
            BIC(c,ei,mi) = r.bic;
        end
    end
    case_time = toc(case_t0);

    % --- best by AIC and by BIC ---
    a = squeeze(AIC(c,:,:)); b = squeeze(BIC(c,:,:));
    [minA, iA] = min(a(:)); [eA,mA] = ind2sub(size(a), iA);
    [minB, iB] = min(b(:)); [eB,mB] = ind2sub(size(b), iB);

    % --- paper fixed-library optimum AIC for this case ---
    pe = find(strcmp({exprs.name}, paper_opt(c).kernel), 1);
    pm = paper_opt(c).mean_id;
    paper_aic = a(pe, pm);

    fprintf('  exhaustive time: %.1f s (%.2f s/candidate)\n', ...
            case_time, case_time/n_cand);
    fprintf('  BEST AIC : %-14s + %-9s  AIC=%.2f\n', ...
            exprs(eA).name, mean_names{mA}, minA);
    fprintf('  BEST BIC : %-14s + %-9s  BIC=%.2f\n', ...
            exprs(eB).name, mean_names{mB}, minB);
    fprintf('  Paper opt: %-14s + %-9s  AIC=%.2f  (dAIC vs best = %.2f)\n', ...
            paper_opt(c).kernel, mean_names{pm}, paper_aic, paper_aic - minA);

    results{c} = struct('name',cases(c).name,'n',n,'n_orig',n_orig,'D',D, ...
        'best_aic_expr',exprs(eA).name,'best_aic_mean',mean_names{mA},'best_aic',minA, ...
        'best_bic_expr',exprs(eB).name,'best_bic_mean',mean_names{mB},'best_bic',minB, ...
        'paper_expr',paper_opt(c).kernel,'paper_mean',mean_names{pm},'paper_aic',paper_aic, ...
        'case_time',case_time);

    % incremental save so partial progress survives interruption
    save('enumerate_grammar_aic_results.mat', 'AIC','BIC','TT','results', ...
         'exprs','ginfo','mean_names','N_RESTART','N_SUB','full_n','cases');
end
grand_time = toc(grand_t0);

%% -------- FULL-n TIMING PROBE (necessity argument) --------
% Measure the cost of a SINGLE candidate fit at the largest full-n case, to
% quantify how exhaustive exact-GP search scales with data size.
fprintf('\n--- Full-n cost probe ---\n');
[~, big] = max(full_n);
xb = cases(big).x; yb = cases(big).y; nb = size(xb,1);
probe_opts = struct('n_restart', N_RESTART, 'n_iter', N_ITER, 'quiet', true);
tpb = tic; evaluate_expr_aic('SE', 4, xb, yb, probe_opts); t_one_full = toc(tpb);
spc_sub = grand_time/(n_cases*n_cand);
fprintf('  1 candidate (%s, n=%d, %d restarts): %.1f s\n', ...
        cases(big).name, nb, N_RESTART, t_one_full);
fprintf('  => full exhaustive at n=%d: %d candidates -> %.0f s (%.1f h)\n', ...
        nb, n_cand, t_one_full*n_cand, t_one_full*n_cand/3600);
fprintf('  (vs %.1f s at n<=%d subsampled; ratio %.0fx)\n', ...
        spc_sub*n_cand, N_SUB, (t_one_full*n_cand)/(spc_sub*n_cand));

%% -------- SUMMARY --------
fprintf('\n================================================================\n');
fprintf('  SUMMARY\n');
fprintf('================================================================\n');
fprintf('%-22s | %-22s | %-22s | dAIC\n', 'Case', 'Best (AIC)', 'Paper fixed-lib opt');
fprintf('%s\n', repmat('-',1,90));
for c = 1:n_cases
    R = results{c};
    fprintf('%-22s | %-14s+%-7s | %-14s+%-7s | %.2f\n', R.name, ...
        R.best_aic_expr, R.best_aic_mean, R.paper_expr, R.paper_mean, ...
        R.paper_aic - R.best_aic);
end

fprintf('\nTOTAL exhaustive wall-clock: %.1f s (%.1f min) for %d GP fits\n', ...
        grand_time, grand_time/60, n_cases*n_cand*N_RESTART);
fprintf('Mean time per candidate (incl. %d restarts): %.2f s\n', ...
        N_RESTART, grand_time/(n_cases*n_cand));

% Extrapolation: cost grows with library size and restarts.
fprintf('\nScaling (per case, %d restarts, ~%.2f s/candidate):\n', ...
        N_RESTART, grand_time/(n_cases*n_cand));
spc = grand_time/(n_cases*n_cand);
for nk = [4 39 80 150 300]
    fprintf('  %3d kernels x %d means = %4d candidates -> %6.1f s (%.1f min)\n', ...
        nk, n_means, nk*n_means, nk*n_means*spc, nk*n_means*spc/60);
end

save('enumerate_grammar_aic_results.mat', 'AIC','BIC','TT','results', ...
     'exprs','ginfo','mean_names','N_RESTART','N_SUB','full_n', ...
     't_one_full','grand_time');
fprintf('\nSaved -> enumerate_grammar_aic_results.mat\n');
fprintf('DONE.\n');
