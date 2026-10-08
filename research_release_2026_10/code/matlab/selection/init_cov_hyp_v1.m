function h = init_cov_hyp_v1(expr, D, logsf, log_rms_x)
%INIT_COV_HYP_V1  Scale-aware initial covariance hyperparameters (log space)
%   for grammar_v1 expressions under the param_v1 redundancy-free layout
%   (无冗余参数化方案 v1; consumed by build_covfunc_v1 / evaluate_expr_aic_v1).
%
%   h = init_cov_hyp_v1(expr, D, logsf, log_rms_x)
%
%   Inputs:
%     expr      grammar_v1 expression (struct or char, T3 '(A*B)+C' ok)
%     D         input dimension
%     logsf     log std of the LS-detrended residual (target signal scale)
%     log_rms_x per-dim log rms(x_d) (for LIN slots: beta_d = 1/L_d^2 starts
%               so the linear kernel's magnitude ~ rms(x_d)^2/L_d^2 matches
%               the target variance)
%
%   Slot layouts (v1):
%     SE/MA*      [l_1..D, sf]        RQ [l_1..D, sf, alpha]
%     LIN         [L_1..D]            PER [p, l, sf]
%     stat x LIN  [l_1..D] (unit stat; RQ keeps alpha) || [L_1..D]
%     stat x PER  [l_1..D, sf] (stat carries sf)        || [p, l] (unit PER)
%     LIN x PER   [L_1..D]                                || [p, l]
%     (A x B)+C   product part as above || C standalone
%
%   Sums split the target variance evenly across additive components
%   (sf -> sf/sqrt(2) per level), matching the old init_cov_hyp convention.
%
%   See also: build_covfunc_v1, expand_expr_terms_v1, evaluate_expr_aic_v1

    if ischar(expr), expr = parse_expr_str(expr); end
    h = init_node(expr, D, logsf, log_rms_x);
end

% --------------------------------------------------------------------------
function h = init_node(expr, D, logsf, lrx)
    switch expr.op
        case 'base'
            h = prim0(expr.terms{1}, D, logsf, lrx);
        case 'sum'
            ha = init_node(as_struct(expr.terms{1}), D, logsf - 0.5*log(2), lrx);
            hb = init_node(as_struct(expr.terms{2}), D, logsf - 0.5*log(2), lrx);
            h = [ha; hb];
        case 'prod'
            a = upper(token_of(expr.terms{1}));
            b = upper(token_of(expr.terms{2}));
            stat = {'SE','MA1','MA3','MA5','RQ'};
            isStat = @(t) any(strcmp(t, stat));
            if ~isStat(a) && isStat(b), [a,b] = deal(b,a); end
            if strcmp(a,'PER') && strcmp(b,'LIN'), [a,b] = deal(b,a); end
            if isStat(a) && isStat(b)
                pord = {'SE','MA1','MA3','MA5','RQ'};
                if find(strcmp(pord,b)) < find(strcmp(pord,a)), [a,b] = deal(b,a); end
            end
            if isStat(a) && strcmp(b,'LIN')
                h = [unit_stat0(a, D); prim0('LIN', D, logsf, lrx)];
            elseif isStat(a) && strcmp(b,'PER')
                h = [prim0(a, D, logsf, lrx); 0; 0];     % unit PER [p; l]
            elseif strcmp(a,'LIN') && strcmp(b,'PER')
                h = [prim0('LIN', D, logsf, lrx); 0; 0];
            elseif isStat(a) && isStat(b)
                % stat x stat (B2-1 pruned-40 only): unit(a) || standalone(b)
                h = [unit_stat0(a, D); prim0(b, D, logsf, lrx)];
            else
                error('init_cov_hyp_v1:badProd', 'unsupported product %s x %s', a, b);
            end
        otherwise
            error('init_cov_hyp_v1:badOp', 'Unknown op "%s".', expr.op);
    end
end

% --------------------------------------------------------------------------
function h = prim0(tok, D, logsf, lrx)
    switch upper(tok)
        case {'SE','MA1','MA3','MA5'}
            h = [zeros(D,1); logsf];
        case 'RQ'
            h = [zeros(D,1); logsf; 0];
        case 'LIN'
            h = lrx(:) - logsf;     % rms(x_d)^2 / L_d^2 ~ sf^2
        case 'PER'
            h = [0; 0; logsf];
        otherwise
            error('init_cov_hyp_v1:badTok', 'Unknown primitive "%s".', tok);
    end
end

% --------------------------------------------------------------------------
function h = unit_stat0(tok, D)
% unit-amplitude stationary slots (no sf): SE/MA* [l..] ; RQ [l.., alpha]
    switch upper(tok)
        case {'SE','MA1','MA3','MA5'}, h = zeros(D,1);
        case 'RQ',                     h = [zeros(D,1); 0];
        otherwise
            error('init_cov_hyp_v1:badTok', 'Unknown primitive "%s".', tok);
    end
end

% --------------------------------------------------------------------------
function tok = token_of(t)
    if isstruct(t)
        assert(strcmp(t.op,'base'), 'init_cov_hyp_v1:badProd', ...
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
