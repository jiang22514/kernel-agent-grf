function [covfunc, n_hyp] = build_covfunc_v1(expr, D)
%BUILD_COVFUNC_V1  Compile a grammar_v1 expression into a GPML covfunc.
%
%   Redundancy-free parameterization (无冗余参数化方案 v1):
%   - sums keep each component's own parameters (identifiable);
%   - products carry ONE overall amplitude per product chain:
%       stationary x LIN : unit stationary  x  LIN(beta)   [beta_d = 1/L_d^2]
%       stationary x PER : stationary(with sf) x PERu(unit)
%       LIN x PER        : LIN(beta) x PERu(unit)
%   - T3 (A x B) + C : product part by the rules above + C with its own params.
%
%   Stationary tokens: SE MA1 MA3 MA5 RQ (unit forms used inside products).
%
%   Returns GPML cell + hyperparameter count (via GPML's no-data convention).

if ischar(expr), expr = parse_expr_v1(expr); end

switch expr.op
    case 'base'
        covfunc = kernel_primitive_cell_v1(expr.terms{1});
    case 'sum'
        covfunc = {@covSum, { build_covfunc_v1(ensure_expr(expr.terms{1}), D), ...
                              build_covfunc_v1(ensure_expr(expr.terms{2}), D) }};
    case 'prod'
        covfunc = prod_cell_v1(expr.terms{1}, expr.terms{2});
    otherwise
        error('build_covfunc_v1:badOp', 'Unknown op: %s', expr.op);
end

cnt_str = feval(covfunc{:});
n_hyp   = eval(cnt_str);
end

% --------------------------------------------------------------------------
function c = prod_cell_v1(a, b)
% product with the one-amplitude rule
if isstruct(a), a = a.terms{1}; end   % tolerate parse_expr_str nested structs
if isstruct(b), b = b.terms{1}; end
a = upper(a); b = upper(b);
stat = {'SE','MA1','MA3','MA5','RQ'};
isStat = @(t) any(strcmp(t, stat));

% canonical order: stationary first, then LIN, then PER (grammar canonicalizes
% pairs already, but stay robust)
if ~isStat(a) && isStat(b), [a,b] = deal(b,a); end
if strcmp(a,'PER') && strcmp(b,'LIN'), [a,b] = deal(b,a); end  % LIN before PER

uStat = @(tok) unit_stat_cell(tok);            % unit-amplitude stationary
sStat = @(tok) kernel_primitive_cell_v1(tok);  % standalone (with sf) stationary

if isStat(a) && strcmp(b,'LIN')
    c = {@covProd, { uStat(a), {@covLINard} }};          % [theta(no sf); beta]
elseif isStat(a) && strcmp(b,'PER')
    c = {@covProd, { sStat(a), {@covPERu} }};            % [theta+sf; p,ell]
elseif strcmp(a,'LIN') && strcmp(b,'PER')
    c = {@covProd, { {@covLINard}, {@covPERu} }};        % [beta; p,ell]
elseif isStat(a) && isStat(b)
    % Pruned from the grammar_v1 SEARCH library, but must remain buildable
    % for the B2-1 pruned-40 re-evaluation (R3#4). One amplitude per chain:
    % unit(a) x standalone(b), canonical primitive order enforced below.
    pord = {'SE','MA1','MA3','MA5','RQ'};
    ia = find(strcmp(pord,a)); ib = find(strcmp(pord,b));
    if ib < ia, [a,b] = deal(b,a); end
    c = {@covProd, { uStat(a), sStat(b) }};              % [a(no sf); b(+sf)]
else
    error('build_covfunc_v1:badProd', 'unsupported product %s x %s', a, b);
end
end

% --------------------------------------------------------------------------
function c = unit_stat_cell(tok)
switch upper(tok)
    case 'SE',  c = {@covSEu};                      % unit ARD SE (kernels_v1)
    case 'MA1', c = {@covMatProdU, 1};
    case 'MA3', c = {@covMatProdU, 3};
    case 'MA5', c = {@covMatProdU, 5};
    case 'RQ',  c = {@covRQprodU};
    otherwise, error('unit_stat_cell:bad','%s', tok);
end
% note: {@covSEu} replaces the earlier {'covSE','ard',[]} (= covSEard), which
% carries an sf slot and reintroduced the sf<->beta redundancy (P03) in
% SE x LIN products; caught and fixed 2026-09-23 (test_expand_v1_map).
c = ensure_handle(c);
end

function c = ensure_handle(c)
if ischar(c{1}), c{1} = str2func(c{1}); end
end

% --------------------------------------------------------------------------
function e = ensure_expr(t)
if isstruct(t), e = t; else, e = struct('name',t,'op','base','terms',{{t}}); end
end

% --------------------------------------------------------------------------
function e = parse_expr_v1(s)
% supports 'SE', 'SE+PER', 'MA1*LIN', '(MA1*LIN)+SE'
s = strrep(strtrim(s), ' ', '');
if s(1) == '('
    close = find(s == ')', 1);
    inner = s(2:close-1);
    rest  = s(close+1:end);
    assert(rest(1) == '+' , 'T3 must be (prod)+prim');
    e = struct('name', s, 'op', 'sum', ...
               'terms', {{ parse_expr_v1(inner), rest(2:end) }});
    return
end
ip = strfind(s, '+'); ix = strfind(s, '*');
if ~isempty(ip)
    e = struct('name', s, 'op', 'sum',  'terms', {{s(1:ip-1), s(ip+1:end)}});
elseif ~isempty(ix)
    e = struct('name', s, 'op', 'prod', 'terms', {{s(1:ix-1), s(ix+1:end)}});
else
    e = struct('name', s, 'op', 'base', 'terms', {{s}});
end
end
