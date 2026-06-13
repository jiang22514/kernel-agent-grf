%% fit_singlebest_borehole.m
%  Refit the best SINGLE kernel from the full-n borehole enumeration
%  (RQ + Constant, AIC 557.97, overall rank 34/156 -- every candidate above
%  it is composite) and save its hyperparameters for the simulation figures.
%
%  Run:  >> fit_singlebest_borehole

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

S = load('samedata.mat'); x = S.x_same; y = S.y_same;
sx = std(x);

opts = struct('n_restart', 5, 'n_iter', -200, 'quiet', true);
r = evaluate_expr_aic('RQ', 2, x, y, opts);

fprintf('RQ + Constant : AIC=%.2f BIC=%.2f nlZ=%.2f (enum: 557.97)\n', ...
        r.aic, r.bic, r.nlZ);
fprintf('hyp.cov  = %s  (alpha=%.3f)\n', mat2str(r.hyp.cov', 5), exp(r.hyp.cov(4)));
fprintf('hyp.mean = %s\n', mat2str(r.hyp.mean', 5));
fprintf('hyp.lik  = %.4f (sn=%.4f)\n', r.hyp.lik, exp(r.hyp.lik));

singlebest = struct('expr', 'RQ', 'mean_id', 2, 'hyp', r.hyp, ...
                    'aic', r.aic, 'bic', r.bic, 'nlZ', r.nlZ, 'sx', sx);
save('singlebest_borehole.mat', 'singlebest');
fprintf('Saved -> singlebest_borehole.mat\nDONE.\n');
