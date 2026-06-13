function res = evaluate_expr_aic(expr, mean_id, x, y, opts)
%EVALUATE_EXPR_AIC  Fit a composite-kernel GP and return AIC/BIC.
%
%   res = evaluate_expr_aic(expr, mean_id, x, y)
%   res = evaluate_expr_aic(expr, mean_id, x, y, opts)
%
%   Fits a Gaussian-process regression model whose signal covariance is the
%   compositional kernel described by `expr` (see kernel_grammar) and whose
%   mean function is selected by `mean_id`, under a Gaussian (white-noise)
%   likelihood. Hyperparameters are optimized by multi-restart conjugate-
%   gradient minimization of the exact negative log marginal likelihood
%   (infGaussLik). AIC/BIC are then computed from the best-restart nlZ.
%
%   Rationale: model SELECTION uses a modest number of observations (n ~ 100),
%   so the exact O(n^3) marginal likelihood is cheap and, crucially, applies
%   to ANY composite covariance without the kernel-specific grid remapping
%   that the Kronecker simulation path requires. The grid/Kronecker engine is
%   used afterwards, for fast simulation with the SELECTED kernel.
%
%   INPUTS
%     expr     kernel expression struct (or string, e.g. 'SE+PER')
%     mean_id  mean function: 1=Zero, 2=Constant, 3=Linear, 4=Quadratic
%     x        n-by-D training inputs
%     y        n-by-1 training targets
%     opts     (optional) struct:
%                .n_restart  number of random restarts        (default 3)
%                .n_iter     max minimize iterations (negative) (default -100)
%                .seed       base RNG seed for restarts        (default 0)
%                .jitter_sd  std of log-hyp restart perturbation (default 0.5)
%                .verbose    print per-restart info            (default false)
%
%   OUTPUT (struct)
%     .expr_name, .mean_id, .mean_name
%     .aic, .bic, .nlZ      best (lowest-nlZ) restart values
%     .k                    number of free hyperparameters (cov+mean+lik)
%     .n                    number of observations
%     .hyp                  best hyperparameter struct
%     .ok                   true if at least one restart succeeded
%     .n_success            number of successful restarts
%
%   See also: build_covfunc_from_expr, kernel_grammar, AIC_exhaustive_verify

    if nargin < 5, opts = struct(); end
    if ~isfield(opts,'n_restart'), opts.n_restart = 3;    end
    if ~isfield(opts,'n_iter'),    opts.n_iter    = -100; end
    if ~isfield(opts,'seed'),      opts.seed      = 0;    end
    if ~isfield(opts,'jitter_sd'), opts.jitter_sd = 0.5;  end
    if ~isfield(opts,'verbose'),   opts.verbose   = false;end
    if ~isfield(opts,'quiet'),     opts.quiet     = false;end  % suppress minimize output

    if ~isfield(opts,'standardize'), opts.standardize = true; end

    if ischar(expr)
        ename = expr;
    else
        ename = expr.name;
    end
    [n, D] = size(x);

    % --- rescale coordinates (divide by per-dim std; NO centering) ---
    % Raw physical units (e.g. meters spanning hundreds) make zero-init
    % log-lengthscales hopeless starting points. Scale-only standardization
    % is a PURE reparameterization for every kernel in the grammar (ARD
    % stationary kernels and per-dim covLINard absorb the scale into their
    % per-dim lengthscales), so the model family -- and hence AIC/BIC -- is
    % identical to raw coordinates. Centering is deliberately avoided: the
    % LIN kernel is not translation invariant, and the coordinate offset
    % acts as its implicit intercept.
    if opts.standardize
        sx = std(x); sx(sx == 0) = 1;
        x = x ./ sx;
    end

    % --- signal covariance from the expression ---
    [covfunc, n_cov] = build_covfunc_from_expr(expr, D);

    % --- mean function (least-squares-initialized hyperparameters) ---
    % Means follow the paper's definitions: Linear/Quadratic/Cubic trends
    % INCLUDE an intercept (meanSum with meanConst). Coefficients are
    % initialized by ordinary least squares on the polynomial basis.
    [meanfunc, hyp0_mean, mean_name] = mean_from_id(mean_id, D, x, y);

    % --- scale-aware covariance initialization ---
    % Signal-std slots start at the std of the LS-detrended residual (crucial
    % e.g. for Zero mean on offset data); lengthscale slots start at 1 (= one
    % coordinate std after standardization).
    m0 = feval(meanfunc{:}, hyp0_mean(:), x);
    resid = y - m0;
    log_sf0 = log(max(std(resid), 1e-6));
    log_rms_x = log(max(sqrt(mean(x.^2, 1)), 1e-6))';   % per-dim, for LIN slots
    hyp0_cov = init_cov_hyp(expr, D, log_sf0, log_rms_x);
    assert(numel(hyp0_cov) == n_cov, 'init_cov_hyp slot mismatch for %s', ename);

    lik = {@likGauss};
    sn0 = max(0.3*std(resid), 1e-3);     % initial noise std
    hyp0 = struct('mean', hyp0_mean(:), ...
                  'cov',  hyp0_cov(:), ...
                  'lik',  log(sn0));

    k = n_cov + numel(hyp0_mean) + 1;   % +1 for the Gaussian noise hyperparameter

    best = struct('nlZ', inf, 'hyp', hyp0);
    n_success = 0;
    for r = 1:opts.n_restart
        rng(opts.seed + r);
        hyp = hyp0;
        if r > 1                      % first restart uses the neutral start
            hyp.cov  = hyp.cov  + opts.jitter_sd*randn(size(hyp.cov));
            % mean coefficients live on natural (not log) scale and may span
            % orders of magnitude (raw-unit coordinates) -> relative jitter
            hyp.mean = hyp.mean .* (1 + 0.2*randn(size(hyp.mean)));
            hyp.lik  = hyp.lik  + opts.jitter_sd*randn(size(hyp.lik));
        end
        try
            if opts.quiet
                [~, hyp] = evalc(['minimize(hyp, @gp, opts.n_iter, @infGaussLik, ' ...
                                  'meanfunc, covfunc, lik, x, y)']);
            else
                hyp = minimize(hyp, @gp, opts.n_iter, @infGaussLik, ...
                               meanfunc, covfunc, lik, x, y);
            end
            nlZ = gp(hyp, @infGaussLik, meanfunc, covfunc, lik, x, y);
            if isfinite(nlZ) && isreal(nlZ)
                n_success = n_success + 1;
                if nlZ < best.nlZ
                    best.nlZ = nlZ; best.hyp = hyp;
                end
                if opts.verbose
                    fprintf('    [%s + %s] restart %d: nlZ=%.3f\n', ...
                            ename, mean_name, r, nlZ);
                end
            end
        catch ME
            if opts.verbose
                fprintf('    [%s + %s] restart %d FAILED: %s\n', ...
                        ename, mean_name, r, ME.message);
            end
        end
    end

    res = struct();
    res.expr_name = ename;
    res.mean_id   = mean_id;
    res.mean_name = mean_name;
    res.k         = k;
    res.n         = n;
    res.n_success = n_success;
    res.ok        = n_success > 0;
    res.hyp       = best.hyp;
    if res.ok
        res.nlZ = best.nlZ;
        res.aic = 2*k + 2*best.nlZ;
        res.bic = k*log(n) + 2*best.nlZ;
    else
        res.nlZ = NaN; res.aic = NaN; res.bic = NaN;
    end
end

% ------------------------------------------------------------------------
function [meanfunc, hyp0, name] = mean_from_id(mean_id, D, x, y)
% Paper-consistent trend hierarchy. Polynomial trends include an intercept
% (meanSum with meanConst), matching the original pipeline, e.g.
% Train_Simulation_same_Test.m: Mean = {'meanSum',{meanConst, meanPoly 2}}.
%
% Initial coefficients via ordinary least squares on the basis
% [1, x_1..x_D, x_1^2..x_D^2, ...]. meanPoly hyp ordering is all degree-1
% coefficients first, then degree-2, etc. (a_11..a_D1, a_12..a_D2, ...),
% which matches the column order of the basis below.
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
            error('evaluate_expr_aic:badMean', 'mean_id must be 1..5');
    end
end

% ------------------------------------------------------------------------
function h = init_cov_hyp(expr, D, logsf, log_rms_x)
% Scale-aware initial covariance hyperparameters (log space) for a grammar
% expression. Lengthscale slots start at 0 (= 1 coordinate std); signal-std
% slots start at logsf (std of the detrended residual); LIN lengthscales are
% set so the linear kernel's magnitude (~ rms(x_d)^2 / l_d^2) matches the
% target variance.
%
% GPML slot layouts:
%   covSEard / covMaternard : [log l_1..D; log sf]            (D+1)
%   covRQard                : [log l_1..D; log sf; log alpha] (D+2)
%   covLINard               : [log l_1..D]                    (D)
%   {covPER,'iso',{covSEiso}}: [log p; log l; log sf]         (3)
    if ischar(expr), expr = parse_expr_str_local(expr); end
    switch expr.op
        case 'base'
            h = prim_hyp0(expr.terms{1}, D, logsf, log_rms_x);
        case 'sum'
            % split total variance evenly: sf_i^2 = sf^2/2
            h = [prim_hyp0(expr.terms{1}, D, logsf - 0.5*log(2), log_rms_x); ...
                 prim_hyp0(expr.terms{2}, D, logsf - 0.5*log(2), log_rms_x)];
        case 'prod'
            a = upper(expr.terms{1}); b = upper(expr.terms{2});
            ha = ~strcmp(a,'LIN'); hb = ~strcmp(b,'LIN');  % has sf slot?
            if ha && hb           % product of variances = total variance
                h = [prim_hyp0(a, D, 0.5*logsf, log_rms_x); ...
                     prim_hyp0(b, D, 0.5*logsf, log_rms_x)];
            elseif ha             % LIN factor neutral (unit magnitude),
                                  % the other factor carries the variance
                h = [prim_hyp0(a, D, logsf, log_rms_x); ...
                     prim_hyp0('LIN', D, 0, log_rms_x)];
            elseif hb
                h = [prim_hyp0('LIN', D, 0, log_rms_x); ...
                     prim_hyp0(b, D, logsf, log_rms_x)];
            else
                error('init_cov_hyp:linlin', 'LIN*LIN excluded by grammar.');
            end
        otherwise
            error('init_cov_hyp:badOp', 'Unknown op "%s".', expr.op);
    end
end

function h = prim_hyp0(tok, D, logsf, log_rms_x)
    switch upper(tok)
        case {'SE','MA1','MA3','MA5'}
            h = [zeros(D,1); logsf];
        case 'RQ'
            h = [zeros(D,1); logsf; 0];
        case 'LIN'
            h = log_rms_x(:) - logsf;   % rms(x_d)^2/l_d^2 ~ sf^2
        case 'PER'
            h = [0; 0; logsf];
        otherwise
            error('init_cov_hyp:badTok', 'Unknown primitive "%s".', tok);
    end
end

function e = parse_expr_str_local(s)
    s = strrep(s, ' ', '');
    ip = strfind(s, '+'); ix = strfind(s, '*');
    if ~isempty(ip)
        e = struct('name', s, 'op', 'sum',  'terms', {{s(1:ip-1), s(ip+1:end)}});
    elseif ~isempty(ix)
        e = struct('name', s, 'op', 'prod', 'terms', {{s(1:ix-1), s(ix+1:end)}});
    else
        e = struct('name', s, 'op', 'base', 'terms', {{s}});
    end
end

% ------------------------------------------------------------------------
function c = ls_coeffs(x, y, deg)
% Least-squares fit of y on [1, x.^1, ..., x.^deg]; returns [const; poly...]
% in meanSum{meanConst, meanPoly} hyperparameter order.
    PHI = ones(size(x,1), 1);
    for j = 1:deg, PHI = [PHI, x.^j]; end %#ok<AGROW>
    c = PHI \ y;
end
