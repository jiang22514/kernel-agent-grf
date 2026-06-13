%% fit_champion_borehole.m
%  Refit the full-n borehole champion (MA1*LIN + Constant) and save its
%  hyperparameters for the simulation stage (the enumeration script stored
%  AIC/BIC but not the fitted hyps).
%
%  Note: evaluate_expr_aic standardizes x by per-dim std (scale only). The
%  saved hyp is therefore in SCALED coordinates; sx is saved alongside so the
%  simulation stage can work consistently in the same scaled frame.
%
%  Run:  >> fit_champion_borehole

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

S = load('samedata.mat'); x = S.x_same; y = S.y_same;
sx = std(x);

opts = struct('n_restart', 5, 'n_iter', -200, 'quiet', true);
r = evaluate_expr_aic('MA1*LIN', 2, x, y, opts);

fprintf('MA1*LIN + Constant : AIC=%.2f BIC=%.2f nlZ=%.2f (enum: 531.07)\n', ...
        r.aic, r.bic, r.nlZ);
fprintf('hyp.cov  = %s\n', mat2str(r.hyp.cov', 5));
fprintf('hyp.mean = %s\n', mat2str(r.hyp.mean', 5));
fprintf('hyp.lik  = %.4f (sn=%.4f)\n', r.hyp.lik, exp(r.hyp.lik));

champion = struct('expr', 'MA1*LIN', 'mean_id', 2, 'hyp', r.hyp, ...
                  'aic', r.aic, 'bic', r.bic, 'nlZ', r.nlZ, 'sx', sx);
save('champion_borehole.mat', 'champion');
fprintf('Saved -> champion_borehole.mat\nDONE.\n');
