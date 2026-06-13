%% probe_borehole_quadmean.m
%  Diagnose why SE+Quadratic at full n=744 gives AIC=4463 in our pipeline
%  while the paper's table reports 547.20.
%
%  Hypothesis: borehole coordinates are in raw meters (large magnitudes), so
%  x.^2 regressor values are huge; with zero-init mean hyps + 0.5 jitter the
%  optimizer fails. Test: least-squares initialization of the mean hyps.
%
%  Run:  >> probe_borehole_quadmean

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

S = load('samedata.mat'); x = S.x_same; y = S.y_same;
[n, D] = size(x);
fprintf('n=%d D=%d\n', n, D);
for d = 1:D
    fprintf('  x%d: min=%.3f max=%.3f std=%.3f\n', d, min(x(:,d)), max(x(:,d)), std(x(:,d)));
end
fprintf('  y : min=%.3f max=%.3f std=%.3f\n\n', min(y), max(y), std(y));

% stored full-n results: SE row, all means
R = load('enumerate_borehole_fulln_results.mat');
se = find(strcmp({R.exprs.name},'SE'),1);
fprintf('Stored full-n AIC, SE row : Zero=%.2f Const=%.2f Lin=%.2f Quad=%.2f\n', ...
        R.AIC(se,1), R.AIC(se,2), R.AIC(se,3), R.AIC(se,4));
fprintf('Paper Table 2 (Gaussian)  : Zero=1026.57 Const=666.28 Lin=653.14 Quad=547.20\n\n');

%% --- refit SE+Quadratic with least-squares-initialized mean hyps ---------
covfunc = {@covSEard}; lik = {@likGauss};
meanfunc = {@meanPoly, 2};

% LS init: y ~ [x, x.^2] (meanPoly basis: x then x.^2, per dim)
PHI = [x, x.^2];
c_ls = PHI \ y;
resid = y - PHI*c_ls;

hyp = struct();
hyp.mean = c_ls(:);
hyp.cov  = [log(std(x))'; log(std(resid))];
hyp.lik  = log(max(std(resid)*0.3, 1e-3));

fprintf('LS-init mean coeffs: %s\n', mat2str(c_ls', 4));
[~, hyp2] = evalc(['minimize(hyp, @gp, -200, @infGaussLik, ' ...
                   'meanfunc, covfunc, lik, x, y)']);
nlZ = gp(hyp2, @infGaussLik, meanfunc, covfunc, lik, x, y);
k = numel(hyp2.cov) + numel(hyp2.mean) + 1;
fprintf('SE+Quadratic (LS-init) : nlZ=%.2f  k=%d  AIC=%.2f  BIC=%.2f\n', ...
        nlZ, k, 2*k+2*nlZ, k*log(n)+2*nlZ);
fprintf('  lengthscales l = [%.2f, %.2f], sf=%.3f, sn=%.4f\n', ...
        exp(hyp2.cov(1)), exp(hyp2.cov(2)), exp(hyp2.cov(3)), exp(hyp2.lik));
fprintf('  (paper: l1=74.43 l2=1.30, AIC=547.20/BIC=568.71)\n\n');

%% --- also refit champion MA1*LIN+Constant with the same robust recipe ----
expr = struct('name','MA1*LIN','op','prod','terms',{{'MA1','LIN'}});
[covfunc2, ~, hyp0c] = build_covfunc_from_expr(expr, D);
hypc = struct('mean', mean(y), 'cov', hyp0c(:), 'lik', log(std(y)*0.3));
[~, hypc2] = evalc(['minimize(hypc, @gp, -200, @infGaussLik, ' ...
                    '{@meanConst}, covfunc2, lik, x, y)']);
nlZc = gp(hypc2, @infGaussLik, {@meanConst}, covfunc2, lik, x, y);
kc = numel(hypc2.cov) + 1 + 1;
fprintf('MA1*LIN+Constant (refit): nlZ=%.2f k=%d AIC=%.2f BIC=%.2f (stored: 531.14)\n', ...
        nlZc, kc, 2*kc+2*nlZc, kc*log(n)+2*nlZc);
fprintf('DONE.\n');
