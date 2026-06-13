%% probe_borehole_zscore.m
%  Check: with z-scored coordinates, does our exact-GP pipeline reproduce the
%  paper's borehole Table 2 column (Gaussian: Zero 1026.57 / Const 666.28 /
%  Lin 653.14 / Quad 547.20)? Also refit the composite champion.
%
%  Run:  >> probe_borehole_zscore

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

S = load('samedata.mat'); x = S.x_same; y = S.y_same;
xz = (x - mean(x)) ./ std(x);          % z-score each coordinate
n = size(x,1);

opts = struct('n_restart', 5, 'n_iter', -100, 'quiet', true);
paper = [1026.57 666.28 653.14 547.20];
mean_names = {'Zero','Constant','Linear','Quadratic'};

fprintf('--- SE with z-scored x (paper Gaussian column in brackets) ---\n');
for mi = 1:4
    r = evaluate_expr_aic('SE', mi, xz, y, opts);
    fprintf('  SE + %-9s : AIC=%8.2f  [paper %8.2f]\n', mean_names{mi}, r.aic, paper(mi));
    if mi == 4 && r.ok
        ls = exp(r.hyp.cov(1:2)) .* std(x)';   % back to meters
        fprintf('    lengthscales (m): l1=%.2f l2=%.2f  [paper 74.43 / 1.30]\n', ls(1), ls(2));
    end
end

fprintf('\n--- key competitors with z-scored x ---\n');
for nm = {'MA1','RQ'}
    r = evaluate_expr_aic(nm{1}, 2, xz, y, opts);
    fprintf('  %-7s + Constant : AIC=%8.2f\n', nm{1}, r.aic);
end
for nm = {'MA1*LIN','RQ*LIN','SE+MA5'}
    r = evaluate_expr_aic(nm{1}, 2, xz, y, opts);
    fprintf('  %-7s + Constant : AIC=%8.2f\n', nm{1}, r.aic);
end
fprintf('DONE.\n');
