function e = parse_expr_str(s)
%PARSE_EXPR_STR  Parse a grammar_v1 kernel expression string into the
%   nested struct form (fields name/op/terms) shared by the selection and
%   simulation stages.
%
%   Supported (grammar_v1 white list): 'SE', 'SE+PER', 'MA1*LIN',
%   '(MA1*LIN)+SE'. Splits at the first TOP-LEVEL operator (paren depth 0),
%   '+' before '*'. Unparenthesized mixed strings like 'A+B*C' are NOT in
%   grammar_v1 and would be split at '+' first (documented, unsupported).
%
%   Output struct: .name (verbatim string), .op in {'base','sum','prod'},
%   .terms = cell of two nested structs (sum/prod) or one token (base).
%
%   See also: expand_expr_terms, expand_expr_terms_v1, kernel_structure_info,
%             build_covfunc_v1

    s = strrep(strtrim(s), ' ', '');
    [pos, op] = top_level_op(s);
    if ~isempty(pos)
        left  = parse_expr_str(s(1:pos-1));
        right = parse_expr_str(s(pos+1:end));
        if op == '+', opname = 'sum'; else, opname = 'prod'; end
        e = struct('name', s, 'op', opname, 'terms', {{left, right}});
        return
    end
    if numel(s) >= 2 && s(1) == '(' && s(end) == ')'
        e = parse_expr_str(s(2:end-1));     % fully parenthesized: unwrap
        e.name = s;
        return
    end
    e = struct('name', s, 'op', 'base', 'terms', {{s}});
end

% --------------------------------------------------------------------------
function [pos, op] = top_level_op(s)
% first operator at paren depth 0; '+' has lower precedence (split first)
    depth = 0; firstPlus = 0; firstStar = 0;
    for i = 1:numel(s)
        c = s(i);
        if c == '(',     depth = depth + 1;
        elseif c == ')', depth = depth - 1;
        elseif depth == 0 && c == '+' && firstPlus == 0, firstPlus = i;
        elseif depth == 0 && c == '*' && firstStar == 0, firstStar = i;
        end
    end
    if firstPlus > 0,     pos = firstPlus; op = '+';
    elseif firstStar > 0, pos = firstStar; op = '*';
    else,                 pos = [];        op = '';
    end
end
