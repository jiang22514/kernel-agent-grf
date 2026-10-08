function res = evaluate_expr_aic_v2(expr, mean_id, x, y, opts)
%EVALUATE_EXPR_AIC_V1  Fit a grammar_v1 composite-kernel GP; return AIC/BIC.
%   Revision pipeline (kernels_v1 + param_v1 + scoring protocol v1).
%   v2: centered horizontal coordinates, real LIN offset, explicit feasible
%   seeds, saved restart parameters, and fixed-noise gradient correction.
%   Based on the retained v1 evaluator; v1 files/results remain unchanged.
%   Differences from evaluate_expr_aic (legacy, kept for reproducibility):
%     - covariance built by build_covfunc_v2 (direction-product Matern/RQ,
%       redundancy-free products; P01/P02/P03 fixed);
%     - initialization by init_cov_hyp_v2 (v1 slot layouts);
%     - k counts IDENTIFIABLE parameters only (param_v1);
%     - per-restart convergence diagnostics recorded (B1-6): final nlZ,
%       iterations used, objective improvement, final gradient norm;
%     - optional nested-mean inheritance (B1-6, scoring_v1.2 / audit P30):
%       opts.parent_hyp contributes ONE start point (plus the embedded-parent
%       floor); the neutral init_cov_hyp_v2 start is always retained, and
%       remaining restarts alternate between the two basins, so a child can
%       escape a bad parent optimum; restarts record their basin;
%     - optional fixed noise: opts.sn_fixed (MPa or data units) freezes the
%       Gaussian noise (excluded from optimization and from k) — the frozen
%       sensitivity scheme of the noise protocol (D2).
%
%   res = evaluate_expr_aic_v2(expr, mean_id, x, y)
%   res = evaluate_expr_aic_v2(expr, mean_id, x, y, opts)
%
%   INPUTS
%     expr     grammar_v1 expression struct or char ('SE+PER','(MA1*LIN)+SE')
%     mean_id  1=Zero, 2=Constant, 3=Linear, 4=Quadratic, 5=Cubic
%     x, y     n-by-D inputs, n-by-1 targets
%     opts     .n_restart (3) .n_iter (-100) .seed (0) .jitter_sd (0.5)
%              .verbose (false) .quiet (true) .standardize (true)
%              .parent_hyp  hyp struct of a fitted simpler mean, same expr
%              .parent_nlZ  its nlZ (enables nested-necessity check report)
%              .sn_fixed    scalar >0: freeze noise at this std
%
%   OUTPUT struct: .expr_name .mean_id .mean_name .aic .bic .nlZ .k .n
%     .hyp .ok .n_success .restarts (per-restart diagnostics)
%     .nested_gap (nlZ - parent_nlZ, when provided) .protocol 'scoring_v1'
%
%   See also: build_covfunc_v2, init_cov_hyp_v2, kernel_grammar

    if nargin < 5, opts = struct(); end
    if ~isfield(opts,'n_restart'), opts.n_restart = 3;    end
    if ~isfield(opts,'n_iter'),    opts.n_iter    = -100; end
    if ~isfield(opts,'seed'),      opts.seed      = 0;    end
    if ~isfield(opts,'jitter_sd'), opts.jitter_sd = 0.5;  end
    if ~isfield(opts,'verbose'),   opts.verbose   = false;end
    if ~isfield(opts,'quiet'),     opts.quiet     = true; end
    if ~isfield(opts,'standardize'), opts.standardize = true; end

    assert(opts.standardize && mean_id<=4, 'linOffset:geometry', 'v2 requires training-only centered geometry and means 1..4');
    if ischar(expr), ename = expr; else, ename = expr.name; end
    [n, D] = size(x);

    % --- rescale coordinates + build covariance/mean（D11：均值族 1..4 且
    %     标准化时，坐标/核/均值与 point_nlz_v1、b65 对照共用同一几何
    %     来源 eval_geometry_v2；其余情形走原路径） ---
    if opts.standardize && mean_id <= 4
        [x, covfunc, meanfunc, n_cov, geometry] = eval_geometry_v2(expr, mean_id, x);
        [~, hyp0_mean, mean_name] = mean_from_id(mean_id, D, x, y);
    else
        if opts.standardize
            sx = std(x); sx(sx == 0) = 1;
            x = x ./ sx;
        end
        [covfunc, n_cov] = build_covfunc_v2(expr, D);
        [meanfunc, hyp0_mean, mean_name] = mean_from_id(mean_id, D, x, y);
    end

    m0 = feval(meanfunc{:}, hyp0_mean(:), x);
    resid = y - m0;
    log_sf0 = log(max(std(resid), 1e-6));
    log_rms_x = log(max(sqrt(mean(x.^2, 1)), 1e-6))';
    hyp0_cov = init_cov_hyp_v2(expr, D, log_sf0, log_rms_x);
    assert(numel(hyp0_cov) == n_cov, 'init_cov_hyp_v2 slot mismatch for %s', ename);

    lik = {@likGauss};
    free_noise = ~(isfield(opts,'sn_fixed') && ~isempty(opts.sn_fixed));
    sn0 = max(0.3*std(resid), 1e-3);
    hyp0 = struct('mean', hyp0_mean(:), 'cov', hyp0_cov(:));
    if free_noise, hyp0.lik = log(sn0); end

    % --- nested-mean inheritance (scoring_v1.2, audit P30 fix):
    %     the parent solution is ONE start point and the floor, NOT a
    %     replacement for the neutral initialization; restarts split between
    %     the neutral and the parent basin so a child can escape a bad
    %     parent optimum ---
    has_parent = isfield(opts,'parent_hyp') && ~isempty(opts.parent_hyp);
    parent_start = [];
    if has_parent
        ph = opts.parent_hyp;
        parent_start = hyp0;
        parent_start.cov = ph.cov(:);
        if numel(ph.mean) <= numel(hyp0.mean)
            parent_start.mean = [ph.mean(:); zeros(numel(hyp0.mean)-numel(ph.mean),1)];
        end
        if free_noise && isfield(ph,'lik'), parent_start.lik = ph.lik; end
    end

    % --- nested floor (B1-6 起点地板): the embedded parent point is a valid
    %     point of the child's parameter space -> evaluate it directly and
    %     never report worse than it (parent's stage-1 nlZ is embedded at
    %     c_extra=0; recorded for the nested-necessity audit trail) ---
    floor_nlZ = [];
    fixed_logsn = [];
    if ~free_noise, fixed_logsn = log(opts.sn_fixed); end
    if has_parent
        hhf = parent_start;
        if ~free_noise, hhf.lik = fixed_logsn; end
        try
            floor_nlZ = gp(hhf, @infGaussLik, meanfunc, covfunc, lik, x, y);
        catch
            floor_nlZ = [];   % 地板点评估失败不致命：按无地板继续
        end
    end

    % Explicit feasible starts (e.g. the exact embedded v1 solution).
    seeds = {};
    if isfield(opts,'seed_hyps'), seeds=opts.seed_hyps; end
    if has_parent, seeds=[{parent_start},seeds]; end
    seed_values=nan(1,numel(seeds));
    for j=1:numel(seeds)
        seeds{j}.cov=seeds{j}.cov(:); seeds{j}.mean=seeds{j}.mean(:);
        assert(numel(seeds{j}.cov)==n_cov && numel(seeds{j}.mean)==numel(hyp0.mean), 'linOffset:seed','Incompatible v2 seed');
        if ~free_noise && isfield(seeds{j},'lik'), seeds{j}=rmfield(seeds{j},'lik'); end
        hs=seeds{j}; if ~free_noise, hs.lik=fixed_logsn; end
        seed_values(j)=gp(hs,@infGaussLik,meanfunc,covfunc,lik,x,y);
        assert(isfinite(seed_values(j)) && isreal(seed_values(j)),'linOffset:seed','Nonfinite direct seed score');
    end
    k = n_cov + numel(hyp0_mean) + double(free_noise);

    best = struct('nlZ', inf, 'hyp', hyp0);
    best_source='none';
    n_success = 0;
    rests = struct('nlZ', {}, 'iter', {}, 'drop', {}, 'gnorm', {}, 'ok', {}, 'basin', {}, 'hyp', {}, 'initial_hyp', {}, 'error', {});
    for r = 1:opts.n_restart
        rng(opts.seed + r);
        pool=[{hyp0},seeds];
        p=mod(r-1,numel(pool))+1; hyp=pool{p}; basin=sprintf('pool_%d',p);
        if r>numel(pool)
            hyp.cov=hyp.cov+opts.jitter_sd*randn(size(hyp.cov));
            hyp.mean=hyp.mean.*(1+0.2*randn(size(hyp.mean)));
            if free_noise, hyp.lik=hyp.lik+opts.jitter_sd*randn(size(hyp.lik)); end
        end
        rr = struct('nlZ', NaN, 'iter', NaN, 'drop', NaN, 'gnorm', NaN, 'ok', false, 'basin', basin, 'hyp', hyp, 'initial_hyp', hyp, 'error', '');
        try
            if free_noise
                obj = @gp;
                iargs = {@infGaussLik, meanfunc, covfunc, lik, x, y};
            else
                obj = @(h, varargin) gp_fixed_lik(h, fixed_logsn, ...
                            meanfunc, covfunc, lik, x, y);
                iargs = {};
            end
            if opts.quiet
                [~] = evalc('[hyp, fX, it] = minimize(hyp, obj, opts.n_iter, iargs{:});');
            else
                [hyp, fX, it] = minimize(hyp, obj, opts.n_iter, iargs{:});
            end
            [nlZ, gn] = nlz_and_gnorm(hyp, fixed_logsn, meanfunc, covfunc, lik, x, y);
            rr.iter = it; rr.drop = fX(1) - fX(end); rr.gnorm = gn;
            if isfinite(nlZ) && isreal(nlZ)
                n_success = n_success + 1;
                rr.nlZ = nlZ; rr.ok = true; rr.hyp = hyp;
                if nlZ < best.nlZ
                    best.nlZ = nlZ; best.hyp = hyp; best_source=sprintf('optimized_restart_%d',r);
                end
                if opts.verbose
                    fprintf('    [%s + %s] restart %d: nlZ=%.3f (it=%d, |g|=%.1e)\n', ...
                            ename, mean_name, r, nlZ, it, gn);
                end
            end
        catch ME
            rr.error=ME.message; rr.hyp=hyp;
            if opts.verbose
                fprintf('    [%s + %s] restart %d FAILED: %s\n', ...
                        ename, mean_name, r, ME.message);
            end
        end
        rests(r) = rr;
    end

    for j=1:numel(seeds)
        if seed_values(j)<best.nlZ, best.nlZ=seed_values(j); best.hyp=seeds{j}; best_source=sprintf('feasible_seed_%d',j); end
    end
    res = struct();
    res.geometry=geometry;
    res.seed_nlZ=seed_values;
    res.seed_hyps=seeds;
    res.kernel_version='lin_horizontal_offset_v2';

    res.expr_name = ename;
    res.mean_id   = mean_id;
    res.mean_name = mean_name;
    res.k         = k;
    res.n         = n;
    res.n_success = n_success;
    res.ok        = n_success > 0 || any(isfinite(seed_values));
    res.hyp       = best.hyp;
    res.restarts  = rests;
    res.protocol  = 'scoring_lin_offset_v2';   % v1.2: neutral+parent dual-basin restarts (audit P30)
    res.floor_used = false;
    if ~isempty(floor_nlZ) && isfinite(floor_nlZ)
        res.floor_nlZ = floor_nlZ;
        if floor_nlZ < best.nlZ
            best.nlZ = floor_nlZ; best.hyp = parent_start; best_source='parent_direct';
            res.floor_used = true;
            res.ok = true; res.hyp = best.hyp;
        end
    end
    if isfield(opts,'parent_nlZ') && ~isempty(opts.parent_nlZ) && res.ok
        res.nested_gap = best.nlZ - opts.parent_nlZ;
    end
    if ~free_noise, res.hyp.lik=fixed_logsn; end
    res.free_noise=free_noise;
    res.best_source=best_source;
    res.seed_used=startsWith(best_source,'feasible_seed') || strcmp(best_source,'parent_direct');
    res.floor_used=res.floor_used || res.seed_used;
    if n_success>0, res.status='optimized';
    elseif res.ok, res.status='feasible_seed_only';
    else, res.status='failed'; end
    if res.ok
        res.nlZ = best.nlZ;
        res.aic = 2*k + 2*best.nlZ;
        res.bic = k*log(n) + 2*best.nlZ;
    else
        res.nlZ = NaN; res.aic = NaN; res.bic = NaN;
    end
end

% ------------------------------------------------------------------------
function [nlZ, gn] = nlz_and_gnorm(hyp, fixed_logsn, meanfunc, covfunc, lik, x, y)
    hh = hyp;
    if ~isempty(fixed_logsn), hh.lik = fixed_logsn; end
    if ~isfield(hh,'lik'), hh.lik = fixed_logsn; end
    % GPML v4.2 training-mode outputs: [nlZ, dnlZ, post] (NOT [post, nlZ, ...])
    [nlZ, dnlZ] = gp(hh, @infGaussLik, meanfunc, covfunc, lik, x, y);
    g = [dnlZ.mean(:); dnlZ.cov(:)];
    if isempty(fixed_logsn) && isfield(dnlZ,'lik'), g = [g; dnlZ.lik(:)]; end
    gn = norm(g);
end

% ------------------------------------------------------------------------
function [nlZ, dnl] = gp_fixed_lik(h, logsn, meanfunc, covfunc, lik, x, y)
    hh = h; hh.lik = logsn;
    if nargout > 1
        [nlZ, dnlZ] = gp(hh, @infGaussLik, meanfunc, covfunc, lik, x, y);
        dnl = rmfield(dnlZ, 'lik');
    else
        nlZ = gp(hh, @infGaussLik, meanfunc, covfunc, lik, x, y);
    end
end

% ------------------------------------------------------------------------
function [meanfunc, hyp0, name] = mean_from_id(mean_id, D, x, y)
% Same trend hierarchy as the legacy evaluator (means include intercept).
    switch mean_id
        case 1
            meanfunc = {@meanZero};  hyp0 = [];      name = 'Zero';
        case 2
            meanfunc = {@meanConst}; hyp0 = mean(y); name = 'Constant';
        case 3
            meanfunc = {@meanSum, {{@meanConst}, {@meanLinear}}};
            hyp0 = ls_coeffs(x, y, 1); name = 'Linear';
        case 4
            meanfunc = {@meanSum, {{@meanConst}, {@meanPoly, 2}}};
            hyp0 = ls_coeffs(x, y, 2); name = 'Quadratic';
        case 5
            meanfunc = {@meanSum, {{@meanConst}, {@meanPoly, 3}}};
            hyp0 = ls_coeffs(x, y, 3); name = 'Cubic';
        otherwise
            error('evaluate_expr_aic_v2:badMean', 'mean_id must be 1..5');
    end
end

% ------------------------------------------------------------------------
function c = ls_coeffs(x, y, deg)
% Least-squares fit of y on [1, x.^1, ..., x.^deg]; [const; poly...] order.
    PHI = ones(size(x,1), 1);
    for j = 1:deg, PHI = [PHI, x.^j]; end %#ok<AGROW>
    c = PHI \ y;
end
