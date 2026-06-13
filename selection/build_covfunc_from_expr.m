function [covfunc, n_hyp, hyp0] = build_covfunc_from_expr(expr, D, init)
%BUILD_COVFUNC_FROM_EXPR  Compile a kernel expression into a GPML covfunc.
%
%   [covfunc, n_hyp]        = build_covfunc_from_expr(expr, D)
%   [covfunc, n_hyp, hyp0]  = build_covfunc_from_expr(expr, D, init)
%
%   Compiles a kernel expression (from kernel_grammar) into a GPML covariance
%   specification cell, using covSum / covProd for the binary operators. The
%   number of covariance hyperparameters is queried automatically through the
%   standard GPML mechanism (calling the covfunc with no data returns a string
%   giving the hyperparameter count as a function of D), so composite kernels
%   need no manual bookkeeping.
%
%   INPUTS
%     expr  struct with fields .op ('base'|'sum'|'prod') and .terms (tokens),
%           or a char primitive token (e.g. 'SE'), or a char expression with a
%           single operator (e.g. 'SE+PER', 'MA3*LIN').
%     D     input dimensionality (number of spatial coordinates).
%     init  (optional) struct controlling the returned initial hyperparameters:
%             init.logell  scalar log length-scale init   (default 0)
%             init.logsf   scalar log signal-std init      (default 0)
%           Any unfilled entries default to 0 (i.e. unit-scale start), which
%           is a neutral starting point for the multi-restart optimizer.
%
%   OUTPUTS
%     covfunc  GPML covariance cell, e.g. {@covSum,{{@covSEard},{@covPERard}}}
%     n_hyp    number of covariance hyperparameters for this expression at D
%     hyp0     n_hyp-by-1 initial hyperparameter vector (log space)
%
%   See also: kernel_grammar, kernel_primitive_cell

    if nargin < 3, init = struct(); end
    if ischar(expr), expr = parse_expr_string(expr); end

    switch expr.op
        case 'base'
            covfunc = kernel_primitive_cell(expr.terms{1});
        case 'sum'
            covfunc = {@covSum,  { kernel_primitive_cell(expr.terms{1}), ...
                                   kernel_primitive_cell(expr.terms{2}) }};
        case 'prod'
            covfunc = {@covProd, { kernel_primitive_cell(expr.terms{1}), ...
                                   kernel_primitive_cell(expr.terms{2}) }};
        otherwise
            error('build_covfunc_from_expr:badOp', ...
                  'Unknown op: %s', expr.op);
    end

    % --- query hyperparameter count via GPML's no-data calling convention ---
    cnt_str = feval(covfunc{:});           % e.g. '(D+1)' or '(D+1)+(D+1)'
    n_hyp   = eval(cnt_str);               %#ok<EVLDIR>  D is in scope here

    if nargout < 3, return; end

    % --- neutral initial hyperparameters (log space) ---
    % All log-hyperparameters default to 0 (length-scales = 1, signal std = 1,
    % period = 1, RQ alpha = 1). The multi-restart optimizer perturbs these.
    hyp0 = zeros(n_hyp, 1);
    if isfield(init,'logell') && ~isempty(init.logell)
        hyp0(:) = init.logell;             % uniform start if requested
    end
    % Signal-std slots are kernel-specific; the uniform start above is a safe
    % default. Callers needing kernel-aware init can post-process hyp0.
end

% ------------------------------------------------------------------------
function e = parse_expr_string(s)
%PARSE_EXPR_STRING  Parse 'SE', 'SE+PER' or 'MA3*LIN' into an expr struct.
    s = strtrim(s);
    if contains(s, '+')
        parts = strsplit(s, '+');
        e = struct('name', s, 'op', 'sum', ...
                   'terms', {{strtrim(parts{1}), strtrim(parts{2})}});
    elseif contains(s, '*')
        parts = strsplit(s, '*');
        e = struct('name', s, 'op', 'prod', ...
                   'terms', {{strtrim(parts{1}), strtrim(parts{2})}});
    else
        e = struct('name', s, 'op', 'base', 'terms', {{s}});
    end
end
