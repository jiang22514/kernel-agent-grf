%% test_expr_aic.m
%  End-to-end check of the A2 evaluator: fit composite-kernel GPs on EX1
%  (linear trend) synthetic data and rank a curated candidate subset by AIC.
%  Sanity expectation: smooth-kernel + linear/zero-mean candidates should
%  rank well; the run mainly verifies that composite kernels fit without error
%  and produce finite, ordered AIC/BIC values.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

% --- EX1 data (linear), same generator as LLM_Evaluation_Framework ---
rng(1);
n = 100; sn = 0.1;
x = rand(n,2);
y = x(:,2) + 0.1*x(:,1) + sn*randn(n,1);

% --- curated candidate subset: (expression, mean_id) ---
cands = {
    'SE',     3;   % Gaussian + Linear   (expected strong)
    'SE',     1;   % Gaussian + Zero
    'MA1',    3;   % Matern1/2 + Linear
    'MA5',    3;   % Matern5/2 + Linear
    'RQ',     3;   % Rational Quadratic + Linear
    'PER',    1;   % Periodic + Zero     (should be poor for linear data)
    'SE+LIN', 1;   % (SE + Linear kernel) + Zero mean
    'SE*LIN', 1;   % (SE x Linear kernel) + Zero mean
    'SE+PER', 3;   % composite + Linear
};

opts = struct('n_restart', 3, 'n_iter', -100, 'verbose', false);
mean_names = {'Zero','Constant','Linear','Quadratic'};

fprintf('==============================================================\n');
fprintf('  A2 EVALUATOR TEST on EX1 (linear),  n=%d, restarts=%d\n', n, opts.n_restart);
fprintf('==============================================================\n');
fprintf('%-16s | %-4s | %-7s | %-10s | %-10s | ok\n', ...
        'kernel', 'k', 'nlZ', 'AIC', 'BIC');
fprintf('%s\n', repmat('-', 1, 64));

rows = struct('label',{},'aic',{},'bic',{},'k',{},'nlZ',{},'ok',{});
t0 = tic;
for i = 1:size(cands,1)
    ename = cands{i,1}; mid = cands{i,2};
    r = evaluate_expr_aic(ename, mid, x, y, opts);
    label = sprintf('%s+%s', ename, mean_names{mid});
    fprintf('%-16s | %-4d | %-7.2f | %-10.2f | %-10.2f | %d\n', ...
            label, r.k, r.nlZ, r.aic, r.bic, r.ok);
    rows(end+1) = struct('label',label,'aic',r.aic,'bic',r.bic, ...
                         'k',r.k,'nlZ',r.nlZ,'ok',r.ok); %#ok<SAGROW>
end
elapsed = toc(t0);

% --- ranking by AIC ---
valid = rows(arrayfun(@(s) s.ok && isfinite(s.aic), rows));
[~, order] = sort([valid.aic], 'ascend');
fprintf('\n--- Ranked by AIC (lower = better) ---\n');
for r = 1:numel(order)
    s = valid(order(r));
    fprintf('  #%d  %-16s  AIC=%.2f\n', r, s.label, s.aic);
end

fprintf('\nElapsed: %.1f s for %d candidates (%.2f s/candidate)\n', ...
        elapsed, size(cands,1), elapsed/size(cands,1));
fprintf('DONE.\n');
