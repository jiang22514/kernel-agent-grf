%% fit_baseline_borehole.m
%  Refit the paper's original choice (SE + Quadratic mean) on the full
%  borehole data with the corrected pipeline, for the side-by-side
%  simulation comparison against the composite champion MA1*LIN + Constant.
%
%  Run:  >> fit_baseline_borehole

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

S = load('samedata.mat'); x = S.x_same; y = S.y_same;
sx = std(x);

opts = struct('n_restart', 5, 'n_iter', -200, 'quiet', true);
r = evaluate_expr_aic('SE', 4, x, y, opts);

fprintf('SE + Quadratic : AIC=%.2f BIC=%.2f nlZ=%.2f (enum best: 531.07)\n', ...
        r.aic, r.bic, r.nlZ);
fprintf('hyp.cov  = %s\n', mat2str(r.hyp.cov', 5));
fprintf('hyp.mean = %s\n', mat2str(r.hyp.mean', 5));
fprintf('hyp.lik  = %.4f (sn=%.4f)\n', r.hyp.lik, exp(r.hyp.lik));

baseline = struct('expr', 'SE', 'mean_id', 4, 'hyp', r.hyp, ...
                  'aic', r.aic, 'bic', r.bic, 'nlZ', r.nlZ, 'sx', sx);
save('baseline_borehole.mat', 'baseline');
fprintf('Saved -> baseline_borehole.mat\nDONE.\n');
