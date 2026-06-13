function info = kernel_structure_info(expr, D, opts)
%KERNEL_STRUCTURE_INFO  Kronecker/grid compatibility analysis of a kernel expr.
%
%   info = kernel_structure_info(expr, D)
%   info = kernel_structure_info(expr, D, opts)
%
%   Expands a grammar kernel expression into a sum of dimension-separable
%   terms and decides which SIMULATION route it can take:
%
%     route = 'full-kron' : K = 1 separable term. Single Kronecker product;
%                           fast MVM + eigendecomposition sampling (paper's
%                           current full-speed path).
%     route = 'mvm-iter'  : 2 <= K <= opts.k_max. K_uu = sum of K Kronecker
%                           products; fast MVM still holds (apply each term),
%                           but inverse/eigendecomposition do not -> use an
%                           MVM-only iterative sampler (CG/Lanczos).
%     route = 'exact'     : fallback, K > k_max or undecomposable.
%
%   Separability assumptions (SIMULATION-stage realization of primitives):
%     SE, MA1, MA3, MA5 : per-dimension 1-D kernel product (Kronecker-ready),
%                         K = 1. NOTE: ARD Matern used at SELECTION stage is
%                         not exactly the per-dim product Matern; the grid
%                         simulation realizes Matern as per-dim product
%                         (consistent with the paper's grid code, Eq. K=prod K_d).
%     PER : K = 1, EXACT. {@covPER,'iso',{@covSEiso}} factorizes across
%           dimensions (SE of the summed sin/cos embedding = per-dim product),
%           verified to machine precision in test_reform.m.
%     LIN : sum over dimensions, k(x,x') = sum_d x_d x_d' / l_d^2
%           -> K = D separable terms (linear in dim d, constant elsewhere).
%     RQ  : iso form not separable. Realized as a Gauss-Laguerre scale
%           mixture of opts.n_rq_mix SE components (RQ = gamma mixture of SE)
%           -> K = n_rq_mix terms, flagged needs_reform (controlled approx).
%
%   Operators: K(sum)  = K_a + K_b
%              K(prod) = K_a * K_b
%
%   Inputs:
%     expr : struct from kernel_grammar (fields name/op/terms) or char
%            like 'SE', 'SE+PER', 'SE*LIN'.
%     D    : input dimension.
%     opts : optional struct
%              .k_max    : max #terms for the iterative-MVM route (default 10)
%              .n_rq_mix : #SE components approximating RQ (default 3)
%
%   Output info struct:
%     .name          expression name
%     .K             number of separable terms after expansion
%     .route         'full-kron' | 'mvm-iter' | 'exact'
%     .needs_reform  cellstr of primitives requiring reformed implementation
%     .detail        human-readable expansion note

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'k_max'),    opts.k_max = 32; end
    if ~isfield(opts, 'n_rq_mix'), opts.n_rq_mix = 7; end

    if ischar(expr), expr = parse_simple(expr); end

    switch expr.op
        case 'base'
            [K, reform, note] = prim_terms(expr.terms{1}, D, opts);
        case 'sum'
            [Ka, ra, na] = prim_terms(expr.terms{1}, D, opts);
            [Kb, rb, nb] = prim_terms(expr.terms{2}, D, opts);
            K = Ka + Kb; reform = [ra, rb];
            note = sprintf('%s (+) %s: K = %d + %d', na, nb, Ka, Kb);
        case 'prod'
            [Ka, ra, na] = prim_terms(expr.terms{1}, D, opts);
            [Kb, rb, nb] = prim_terms(expr.terms{2}, D, opts);
            K = Ka * Kb; reform = [ra, rb];
            note = sprintf('%s (x) %s: K = %d x %d', na, nb, Ka, Kb);
        otherwise
            error('kernel_structure_info:badOp', 'Unknown op "%s".', expr.op);
    end

    if K == 1
        route = 'full-kron';
    elseif K <= opts.k_max
        route = 'mvm-iter';
    else
        route = 'exact';
    end

    info = struct('name', expr.name, 'K', K, 'route', route, ...
                  'needs_reform', {unique(reform)}, 'detail', note);
end

% ------------------------------------------------------------------------
function [K, reform, note] = prim_terms(tok, D, opts)
    reform = {};
    switch upper(tok)
        case {'SE','MA1','MA3','MA5'}
            K = 1; note = [tok ':1'];
        case 'PER'
            K = 1;                          % iso form factorizes: exact
            note = 'PER:1(exact)';
        case 'LIN'
            K = D; note = sprintf('LIN:%d(=D)', D);
        case 'RQ'
            K = opts.n_rq_mix; reform = {'RQ'};
            note = sprintf('RQ:%d(SE-mixture)', K);
        otherwise
            error('kernel_structure_info:badTok', 'Unknown primitive "%s".', tok);
    end
end

% ------------------------------------------------------------------------
function e = parse_simple(s)
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
