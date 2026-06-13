%% holdout_borehole.m
%  Held-out predictive validation on the borehole data: does the AIC ranking
%  (in-sample evidence) generalize to unseen data?
%
%  Protocol: 10 random splits, 1/3 train (n=248) / 2/3 test (n=496).
%  Models (the three narrative tiers of the simulation figures):
%    SE      + Quadratic  -- community default
%    RQ      + Constant   -- best single kernel (full-n enumeration)
%    MA1*LIN + Constant   -- grammar champion
%  Metrics on the test set:
%    RMSE  -- point-prediction accuracy (predictive mean)
%    NLPD  -- negative log predictive density (uncertainty quality;
%             uses the full predictive distribution incl. noise)
%  Byproduct: train-subset AIC per split (selection stability at n=248).
%
%  Coordinates are pre-scaled by the TRAIN std (scale only, no centering),
%  mirroring evaluate_expr_aic's internal convention; the test inputs use the
%  same factors so train/test live in one frame.
%
%  Run:  >> holdout_borehole
clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

S = load('samedata.mat'); x = S.x_same; y = S.y_same;
[n, D] = size(x);

models = {
    'SE',      4, 'SE+Quadratic (default)'
    'RQ',      2, 'RQ+Constant (best single)'
    'MA1*LIN', 2, 'MA1*LIN+Constant (champion)'
};
nM = size(models, 1);
NSPLIT = 10;

fit_opts = struct('n_restart', 5, 'n_iter', -150, 'quiet', true, ...
                  'standardize', false);   % we pre-scale outside

RMSE = nan(NSPLIT, nM); NLPD = nan(NSPLIT, nM);
AICt = nan(NSPLIT, nM); SN   = nan(NSPLIT, nM);

t_all = tic;
for s = 1:NSPLIT
    rng(1000 + s);
    perm = randperm(n);
    ntr  = round(n/3);
    itr  = perm(1:ntr); ite = perm(ntr+1:end);
    xtr = x(itr,:); ytr = y(itr);
    xte = x(ite,:); yte = y(ite);

    sx = std(xtr); sx(sx == 0) = 1;
    xtr_s = xtr ./ sx; xte_s = xte ./ sx;

    for m = 1:nM
        expr = models{m,1}; mid = models{m,2};
        fo = fit_opts; fo.seed = 100*s + m;
        r = evaluate_expr_aic(expr, mid, xtr_s, ytr, fo);
        if ~r.ok
            warning('split %d, %s: all restarts failed', s, models{m,3});
            continue;
        end

        covfunc = build_covfunc_from_expr(expr, D);
        switch mid
            case 2, meanfunc = {@meanConst};
            case 4, meanfunc = {@meanSum, {{@meanConst}, {@meanPoly, 2}}};
        end
        [ymu, ys2] = gp(r.hyp, @infGaussLik, meanfunc, covfunc, ...
                        {@likGauss}, xtr_s, ytr, xte_s);

        RMSE(s,m) = sqrt(mean((ymu - yte).^2));
        NLPD(s,m) = mean(0.5*log(2*pi*ys2) + (yte - ymu).^2 ./ (2*ys2));
        AICt(s,m) = r.aic;
        SN(s,m)   = exp(r.hyp.lik);
    end
    fprintf('split %2d/%d done (%.1f s elapsed)\n', s, NSPLIT, toc(t_all));
end

%% ---- summary ----
fprintf('\n=== Held-out validation, borehole (10 splits, train n=%d / test n=%d) ===\n', ...
        round(n/3), n - round(n/3));
fprintf('%-30s  %-16s  %-16s  %-14s\n', 'Model', 'test RMSE', 'test NLPD', 'train AIC');
for m = 1:nM
    fprintf('%-30s  %6.4f +- %.4f  %6.4f +- %.4f  %7.1f +- %5.1f\n', ...
        models{m,3}, mean(RMSE(:,m)), std(RMSE(:,m)), ...
        mean(NLPD(:,m)), std(NLPD(:,m)), mean(AICt(:,m)), std(AICt(:,m)));
end

% per-split winners
[~, iw_r] = min(RMSE, [], 2); [~, iw_n] = min(NLPD, [], 2); [~, iw_a] = min(AICt, [], 2);
fprintf('\nper-split wins (out of %d):\n', NSPLIT);
fprintf('%-30s  RMSE %2d   NLPD %2d   trainAIC %2d\n', models{1,3}, ...
        sum(iw_r==1), sum(iw_n==1), sum(iw_a==1));
fprintf('%-30s  RMSE %2d   NLPD %2d   trainAIC %2d\n', models{2,3}, ...
        sum(iw_r==2), sum(iw_n==2), sum(iw_a==2));
fprintf('%-30s  RMSE %2d   NLPD %2d   trainAIC %2d\n', models{3,3}, ...
        sum(iw_r==3), sum(iw_n==3), sum(iw_a==3));

fprintf('\nfitted noise std sn (mean over splits): %s\n', ...
        mat2str(mean(SN,1,'omitnan'), 3));

save('holdout_borehole_results.mat', 'RMSE', 'NLPD', 'AICt', 'SN', 'models', 'NSPLIT');
fprintf('Saved -> holdout_borehole_results.mat\nDONE in %.1f s.\n', toc(t_all));
