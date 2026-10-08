function terms = expand_expr_terms(expr, D, hyp)
%EXPAND_EXPR_TERMS  Expand a kernel expression into separable Kronecker terms.
%
%   terms = expand_expr_terms(expr, D, hyp)
%
%   Returns the sum-of-separable-terms decomposition used by the grid
%   simulation path:  K(x,x') = sum_k  prod_d  k_{k,d}(x_d, x_d')
%   so that on a Cartesian grid  K_uu = sum_k  kron(K_{k,1}, ..., K_{k,D}).
%
%   Inputs:
%     expr : grammar struct (fields name/op/terms) or char. terms may hold
%            primitive tokens ('SE', ...) or nested expr structs (grammar_v1
%            T3, e.g. '(MA1*LIN)+SE'); char inputs may use parentheses.
%     D    : input dimension
%     hyp  : struct with natural-log hyperparameters per primitive token:
%              .SE  = [log l_1..l_D, log sf]      (also .MA1/.MA3/.MA5)
%              .LIN = [log l_1..l_D]
%              .PER = [log p, log l, log sf]      (per-dim product periodic)
%              .RQ  = [log l_1..l_D, log sf, log alpha]
%            Only tokens present in the expression are required; grammar_v1
%            uses each primitive at most once per expression, so per-token
%            fields are unambiguous. Amplitude placement (which token carries
%            sf) follows the caller: use expand_expr_terms_v1 for the frozen
%            param_v1 mapping.
%
%   Output terms(k) struct array:
%     .f{d} : GPML covariance cell for the 1-D factor along dimension d
%     .h{d} : hyperparameter column vector for that factor
%
%   Simulation-stage realizations (kernels_v1, shared with selection stage;
%   consistency verified to machine precision by test_kernels_v1 /
%   test_build_covfunc_v1):
%     SE / MA*  : per-dimension product (signal variance folded into dim 1)
%     LIN       : sum over dims -> D terms (linear in dim j, const elsewhere)
%     PER       : per-dimension product of 1-D periodic kernels. This is
%                 EXACTLY the selection-stage {covPER,'iso',{covSEiso}}: the
%                 SE of the summed sin/cos embedding factorizes across dims.
%     RQ        : EXACT per-dimension product (covRQiso per dim, shared
%                 alpha) -> K = 1 term (revision plan B1-2). The old
%                 Gauss-Laguerre SE-mixture approximation remains available
%                 via hyp.RQ_legacy_mixture = true (J = hyp.RQ_J, default 7).

    if ischar(expr), expr = parse_expr_str(expr); end
    terms = expand_node(expr, D, hyp);
end

% ------------------------------------------------------------------------
function terms = expand_node(expr, D, hyp)
% recursive expansion; expr.terms entries may be tokens or nested structs
    switch expr.op
        case 'base'
            terms = prim_expand(expr.terms{1}, D, hyp);
        case 'sum'
            terms = [expand_node(as_struct(expr.terms{1}), D, hyp), ...
                     expand_node(as_struct(expr.terms{2}), D, hyp)];
        case 'prod'
            ta = expand_node(as_struct(expr.terms{1}), D, hyp);
            tb = expand_node(as_struct(expr.terms{2}), D, hyp);
            terms = repmat(empty_term(D), 1, numel(ta)*numel(tb));
            k = 0;
            for i = 1:numel(ta)
                for j = 1:numel(tb)
                    k = k + 1;
                    for d = 1:D
                        terms(k).f{d} = {@covProd, {ta(i).f{d}, tb(j).f{d}}};
                        terms(k).h{d} = [ta(i).h{d}; tb(j).h{d}];
                    end
                end
            end
        otherwise
            error('expand_expr_terms:badOp', 'Unknown op "%s".', expr.op);
    end
end

% ------------------------------------------------------------------------
function e = as_struct(t)
    if isstruct(t), e = t;
    else, e = struct('name', t, 'op', 'base', 'terms', {{t}}); end
end

% ------------------------------------------------------------------------
function terms = prim_expand(tok, D, hyp)
    tok = upper(tok);
    t = empty_term(D);
    switch tok
        case {'SE','MA1','MA3','MA5'}
            hv = hyp.(tok); hv = hv(:);
            assert(numel(hv) == D+1, '%s hyp must be [log l_1..l_D, log sf]', tok);
            for d = 1:D
                if tok(1) == 'S'
                    t.f{d} = {@covSEiso};
                else
                    nu = str2double(tok(3));
                    t.f{d} = {@covMaterniso, nu};
                end
                % signal variance on dim 1 only, unit elsewhere
                sf_d = 0; if d == 1, sf_d = hv(end); end
                t.h{d} = [hv(d); sf_d];
            end
            terms = t;

        case 'LIN'
            hv = hyp.LIN; hv = hv(:);
            assert(numel(hv) == D, 'LIN hyp must be [log l_1..l_D]');
            terms = repmat(empty_term(D), 1, D);
            for j = 1:D
                for d = 1:D
                    if d == j
                        terms(j).f{d} = {@covLINiso};
                        terms(j).h{d} = hv(j);
                    else
                        terms(j).f{d} = {@covConst};   % k = sf^2 = 1
                        terms(j).h{d} = 0;
                    end
                end
            end

        case 'PER'
            hv = hyp.PER; hv = hv(:);
            assert(numel(hv) == 3, 'PER hyp must be [log p, log l, log sf]');
            for d = 1:D
                t.f{d} = {@covPER, 'iso', {@covSEiso}};   % 1-D periodic
                sf_d = 0; if d == 1, sf_d = hv(3); end
                t.h{d} = [hv(1); hv(2); sf_d];
            end
            terms = t;

        case 'RQ'
            % kernels_v1: exact direction-product RQ (revision plan B1-2).
            % Per-dim covRQiso, shared alpha, sf on dim 1. The old SE-mixture
            % approximation remains available via hyp.RQ_legacy_mixture = true.
            hv = hyp.RQ; hv = hv(:);
            assert(numel(hv) == D+2, 'RQ hyp must be [log l_1..D, log sf, log alpha]');
            if isfield(hyp, 'RQ_legacy_mixture') && hyp.RQ_legacy_mixture
                logl = hv(1:D); logsf = hv(D+1); alpha = exp(hv(D+2));
                J = 10; if isfield(hyp, 'RQ_J'), J = hyp.RQ_J; end
                [s, w] = se_mixture_fit(alpha, J);
                J = numel(s);                  % zero-weight nodes dropped
                terms = repmat(empty_term(D), 1, J);
                for j = 1:J
                    for d = 1:D
                        terms(j).f{d} = {@covSEiso};
                        sf_d = 0; if d == 1, sf_d = logsf + 0.5*log(w(j)); end
                        terms(j).h{d} = [logl(d) - 0.5*log(s(j)); sf_d];
                    end
                end
            else
                logl = hv(1:D); logsf = hv(D+1); logal = hv(D+2);
                t = empty_term(D);
                for d = 1:D
                    t.f{d} = {@covRQiso};
                    sf_d = 0; if d == 1, sf_d = logsf; end
                    t.h{d} = [logl(d); sf_d; logal];
                end
                terms = t;                       % K = 1 term, EXACT
            end
        otherwise
            error('expand_expr_terms:badTok', 'Unknown primitive "%s".', tok);
    end
end

% ------------------------------------------------------------------------
function [s, w] = se_mixture_fit(alpha, J)
% SE-mixture approximation of the 1-D RQ profile by non-negative least
% squares: (1 + r^2/(2 alpha))^(-alpha) ~= sum_j w_j exp(-s_j r^2 / 2),
% with J inverse-squared-lengthscale nodes s_j log-spaced over a range set
% by the Gamma(alpha,alpha) mixing density's spread (cv = 1/sqrt(alpha)).
% Direct profile fitting is far more accurate than Gauss-Laguerre quadrature
% for small alpha (heavy-tailed mixing). Zero-weight nodes are dropped.
    spread = min(6/sqrt(alpha), 14);                 % half-width in log s
    s = exp(linspace(-spread, spread, J))';
    % r^2 design: from near field out to where the profile decays to ~5e-3,
    % CAPPED at scaled distance rho = 1000 lengthscales: for heavy-tailed
    % small alpha the 5e-3 level lies at astronomical distances no finite
    % domain reaches, and an uncapped range starves the near field.
    r2max = min(2*alpha*(0.005^(-1/alpha) - 1), 1e6);
    r2 = [0; logspace(-6, log10(r2max), 500)'];
    A = exp(-0.5 * r2 * s');
    tgt = (1 + r2/(2*alpha)).^(-alpha);
    rw = ones(size(r2)); rw(1) = 50;                 % pin k(0) (variance)
    w = lsqnonneg(A .* rw, tgt .* rw);
    keep = w > 1e-10;
    s = s(keep); w = w(keep);
end

% ------------------------------------------------------------------------
function t = empty_term(D)
    t = struct('f', {cell(1,D)}, 'h', {cell(1,D)});
end
