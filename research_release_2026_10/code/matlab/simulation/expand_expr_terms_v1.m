function [terms, hypmap] = expand_expr_terms_v1(expr, D, hypvec)
%EXPAND_EXPR_TERMS_V1  Map a grammar_v1 expression + flat v1 hyperparameter
%   vector (build_covfunc_v1 layout, param_v1 redundancy-free
%   parameterization) to sum-of-Kronecker simulation terms.
%
%   [terms, hypmap] = expand_expr_terms_v1(...) also returns the per-token
%   hyp struct (e.g. for kernel_lengthscales_v1 effective-scale reads, P40).
%
%   terms = expand_expr_terms_v1(expr, D, hypvec)
%
%   This is the SIMULATION-side parameter map required by 无冗余参数化方案
%   v1 §5.3 and plan item P40 (no hard-coded slot reading): the flat v1
%   vector is split into per-token fields with the amplitude (sf) placed on
%   exactly the slots that build_covfunc_v1 assigns:
%
%     T0 SE/MA1/MA3/MA5 : [l_1..l_D, sf]          (covScale-wrapped prod)
%     T0 RQ             : [l_1..l_D, sf, alpha]
%     T0 LIN            : [L_1..L_D]              (beta_d = 1/L_d^2)
%     T0 PER            : [p, l, sf]
%     T1 A+B            : A standalone || B standalone
%     T2 stat x LIN     : [stat l_1..l_D (NO sf), LIN L_1..L_D]
%                         (RQ keeps alpha: [l_1..l_D, alpha, LIN L..])
%     T2 stat x PER     : [stat full (with sf), PER p, l]   (PER unit)
%     T2 LIN x PER      : [LIN L_1..L_D, PER p, l]          (PER unit)
%     T3 (A x B)+C      : product part by T2 rules || C standalone
%
%   Canonical factor order inside a product follows build_covfunc_v1:
%   stationary first, then LIN, then PER. In the per-token hyp struct handed
%   to expand_expr_terms, a token whose sf was absorbed gets sf slot = 0
%   (log 1); grammar_v1 uses each primitive at most once per expression, so
%   per-token fields are unambiguous.
%
%   Asserts numel(hypvec) equals the internal count AND (when
%   build_covfunc_v1 is on the path) the selection-side parameter count.
%
%   See also: expand_expr_terms, build_covfunc_v1, parse_expr_str

    if ischar(expr), expr = parse_expr_str(expr); end
    hypvec = hypvec(:);
    [hyp, used] = map_node(expr, D, hypvec);
    assert(used == numel(hypvec), 'expand_expr_terms_v1:hypLen', ...
        'hypvec has %d entries, expression "%s" consumes %d', ...
        numel(hypvec), expr.name, used);
    if exist('build_covfunc_v1', 'file') == 2
        [~, nh] = build_covfunc_v1(expr, D);
        assert(nh == numel(hypvec), 'expand_expr_terms_v1:countMismatch', ...
            'build_covfunc_v1 reports %d params for "%s", got %d', ...
            nh, expr.name, numel(hypvec));
    end
    terms = expand_expr_terms(expr, D, hyp);
    hypmap = hyp;
end

% --------------------------------------------------------------------------
function [hyp, used] = map_node(expr, D, v)
    hyp = struct();
    switch expr.op
        case 'base'
            tok = upper(expr.terms{1});
            n = standalone_count(tok, D);
            hyp.(tok) = v(1:n);
            used = n;
        case 'sum'
            [ha, na] = map_node(as_struct(expr.terms{1}), D, v);
            [hb, nb] = map_node(as_struct(expr.terms{2}), D, v(na+1:end));
            fa = fieldnames(ha); fb = fieldnames(hb);
            assert(isempty(intersect(fa, fb)), 'expand_expr_terms_v1:dupTok', ...
                'primitive repeated in one expression (grammar_v1 forbids)');
            for i = 1:numel(fa), hyp.(fa{i}) = ha.(fa{i}); end
            for i = 1:numel(fb), hyp.(fb{i}) = hb.(fb{i}); end
            used = na + nb;
        case 'prod'
            a = upper(token_of(expr.terms{1}));
            b = upper(token_of(expr.terms{2}));
            stat = {'SE','MA1','MA3','MA5','RQ'};
            isStat = @(t) any(strcmp(t, stat));
            % canonical order: stationary first, then LIN, then PER
            if ~isStat(a) && isStat(b), [a,b] = deal(b,a); end
            if strcmp(a,'PER') && strcmp(b,'LIN'), [a,b] = deal(b,a); end
            if isStat(a) && isStat(b)
                % canonical order for stat x stat (buildable for B2-1
                % pruned-40 re-evaluation; not in the search library)
                pord = {'SE','MA1','MA3','MA5','RQ'};
                if find(strcmp(pord,b)) < find(strcmp(pord,a)), [a,b] = deal(b,a); end
            end
            if isStat(a) && strcmp(b,'LIN')
                % unit stationary (sf absorbed by LIN betas); RQ keeps alpha
                if strcmp(a,'RQ')
                    hyp.(a) = [v(1:D); 0; v(D+1)];      % [l.., sf=0, alpha]
                    ns = D + 1;
                else
                    hyp.(a) = [v(1:D); 0];               % [l.., sf=0]
                    ns = D;
                end
                hyp.LIN = v(ns+1 : ns+D);
                used = ns + D;
            elseif isStat(a) && strcmp(b,'PER')
                ns = standalone_count(a, D);             % stat keeps its sf
                hyp.(a) = v(1:ns);
                hyp.PER = [v(ns+1); v(ns+2); 0];         % unit PER
                used = ns + 2;
            elseif strcmp(a,'LIN') && strcmp(b,'PER')
                hyp.LIN = v(1:D);
                hyp.PER = [v(D+1); v(D+2); 0];
                used = D + 2;
            elseif isStat(a) && isStat(b)
                % stat x stat: unit(a) || standalone(b) (B2-1 pruned-40 only)
                if strcmp(a,'RQ')
                    hyp.(a) = [v(1:D); 0; v(D+1)]; na = D + 1;
                else
                    hyp.(a) = [v(1:D); 0];           na = D;
                end
                nb = standalone_count(b, D);
                hyp.(b) = v(na+1 : na+nb);
                used = na + nb;
            else
                error('expand_expr_terms_v1:badProd', ...
                    'unsupported product %s x %s', a, b);
            end
        otherwise
            error('expand_expr_terms_v1:badOp', 'Unknown op "%s".', expr.op);
    end
end

% --------------------------------------------------------------------------
function n = standalone_count(tok, D)
    switch upper(tok)
        case {'SE','MA1','MA3','MA5'}, n = D + 1;
        case 'RQ',                     n = D + 2;
        case 'LIN',                    n = D;
        case 'PER',                    n = 3;
        otherwise
            error('expand_expr_terms_v1:badTok', 'Unknown primitive "%s".', tok);
    end
end

% --------------------------------------------------------------------------
function tok = token_of(t)
% product children are base tokens in grammar_v1 (nested prod unsupported)
    if isstruct(t)
        assert(strcmp(t.op,'base'), 'expand_expr_terms_v1:badProd', ...
            'product children must be base primitives');
        tok = t.terms{1};
    else
        tok = t;
    end
end

% --------------------------------------------------------------------------
function e = as_struct(t)
    if isstruct(t), e = t;
    else, e = struct('name', t, 'op', 'base', 'terms', {{t}}); end
end
