function [exprs, info] = kernel_grammar(opts)
%KERNEL_GRAMMAR  Bounded compositional grammar of GP covariance expressions.
%
%   [exprs, info] = kernel_grammar()
%   [exprs, info] = kernel_grammar(opts)
%
%   Enumerates all valid kernel expressions of depth <= 2 built from a fixed
%   set of base kernels and the binary operators {+, *} (covSum / covProd).
%   This replaces the fixed 4-kernel library with an open, compositional
%   search space that grows combinatorially, so that exhaustive AIC/BIC
%   evaluation (each candidate requiring multi-restart hyperparameter
%   optimization) becomes genuinely expensive -- establishing the necessity
%   of an intelligent pre-screening stage.
%
%   GRAMMAR (depth <= 2):
%     expr := base | base '+' base | base '*' base
%   where base is one of the primitive tokens below. Operators are
%   commutative, so unordered pairs are enumerated once (canonicalized).
%
%   Each returned expression is a struct with fields:
%     .name   canonical human-readable label, e.g. 'SE', 'SE+PER', 'MA3*LIN'
%     .op     'base' | 'sum' | 'prod'
%     .terms  cell array of primitive tokens, e.g. {'SE'} or {'SE','PER'}
%   T3 expressions '(A*B)+C' (opts.include_t3) have op='sum' with
%   .terms = {prod_struct, C_token} (nested struct; grammar_v1).
%
%   OPTIONS (struct, all optional):
%     opts.primitives   cell array of primitive tokens to use.
%                       default: {'SE','MA1','MA3','MA5','RQ','LIN','PER'}
%     opts.include_sum  enumerate '+' combinations (default true)
%     opts.include_prod enumerate '*' combinations (default true)
%     opts.include_t3   enumerate '(A*B)+C' (grammar_v1 T3; default false).
%                       A*B must be a legal (unpruned) product, C a primitive
%                       distinct from A and B. Option A of grammar_v1:
%                       7 + 21 + 11 + 55 = 94 kernels (376 with 4 means).
%     opts.prune        apply degeneracy pruning rules (default true)
%
%   See also: build_covfunc_from_expr, kernel_primitive_cell, build_covfunc_v1

    if nargin < 1, opts = struct(); end
    prim_order = {'SE','MA1','MA3','MA5','RQ','LIN','PER'};
    if ~isfield(opts,'primitives') || isempty(opts.primitives)
        opts.primitives = prim_order;
    end
    if ~isfield(opts,'include_sum'),  opts.include_sum  = true; end
    if ~isfield(opts,'include_prod'), opts.include_prod = true; end
    if ~isfield(opts,'include_t3'),   opts.include_t3   = false; end
    if ~isfield(opts,'prune'),        opts.prune        = true; end

    P = opts.primitives;
    % Keep primitives in canonical order for stable, deduplicated naming.
    [~, ord] = ismember(P, prim_order);
    [~, si]  = sort(ord);
    P = P(si);
    nP = numel(P);

    exprs = struct('name', {}, 'op', {}, 'terms', {});

    % --- depth 1: single primitives ---
    for i = 1:nP
        exprs(end+1) = make_expr('base', P(i)); %#ok<AGROW>
    end

    % --- depth 2: unordered pairs of distinct primitives ---
    for i = 1:nP
        for j = i+1:nP
            ti = P{i}; tj = P{j};
            if opts.include_sum && ~is_degenerate('sum', ti, tj, opts.prune)
                exprs(end+1) = make_expr('sum', {ti, tj}); %#ok<AGROW>
            end
            if opts.include_prod && ~is_degenerate('prod', ti, tj, opts.prune)
                exprs(end+1) = make_expr('prod', {ti, tj}); %#ok<AGROW>
            end
        end
    end

    % --- grammar_v1 T3: '(A*B)+C' with legal product A*B and distinct C ---
    if opts.include_t3
        for i = 1:nP
            for j = i+1:nP
                ti = P{i}; tj = P{j};
                if ~opts.include_prod || ...
                   is_degenerate('prod', ti, tj, opts.prune)
                    continue
                end
                prod_sub = make_expr('prod', {ti, tj});
                for k = 1:nP
                    tc = P{k};
                    if strcmp(tc, ti) || strcmp(tc, tj), continue; end
                    nm = sprintf('(%s*%s)+%s', ti, tj, tc);
                    exprs(end+1) = struct('name', nm, 'op', 'sum', ...
                        'terms', {{prod_sub, tc}}); %#ok<AGROW>
                end
            end
        end
    end

    % --- summary info ---
    n_base = sum(strcmp({exprs.op}, 'base'));
    n_prod = sum(strcmp({exprs.op}, 'prod'));
    n_t3   = sum(cellfun(@(t) isstruct(t{1}), {exprs.terms}));
    n_sum  = sum(strcmp({exprs.op}, 'sum')) - n_t3;
    info = struct('n_total', numel(exprs), 'n_base', n_base, ...
                  'n_sum', n_sum, 'n_prod', n_prod, 'n_t3', n_t3, ...
                  'primitives', {P});
end

% ------------------------------------------------------------------------
function e = make_expr(op, terms)
    if ischar(terms), terms = {terms}; end
    switch op
        case 'base', nm = terms{1};
        case 'sum',  nm = [terms{1} '+' terms{2}];
        case 'prod', nm = [terms{1} '*' terms{2}];
        otherwise, error('kernel_grammar:badOp', 'Unknown op %s', op);
    end
    e = struct('name', nm, 'op', op, 'terms', {terms});
end

% ------------------------------------------------------------------------
function tf = is_degenerate(op, a, b, do_prune)
%IS_DEGENERATE  Reject combinations that are redundant or ill-posed.
    tf = false;
    if ~do_prune, return; end

    isStat = @(t) any(strcmp(t, {'SE','MA1','MA3','MA5','RQ'}));

    switch op
        case 'prod'
            % Product of two purely stationary, unit-variance-style kernels
            % yields another stationary kernel of similar character while
            % doubling hyperparameters -> redundant. Keep products only when
            % at least one factor adds structure (LIN or PER) or mixes a
            % stationary kernel with LIN/PER (Automatic-Statistician style).
            if isStat(a) && isStat(b)
                tf = true;   % e.g. SE*RQ, MA3*MA5 : prune
            end
            % LIN*LIN handled by distinctness (a~=b); LIN*PER kept.
        case 'sum'
            % Sums are generally meaningful (mixture of structures). No prune.
            tf = false;
    end
end
