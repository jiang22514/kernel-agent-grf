function [props, raw] = parse_kernel_proposal(response_text)
%PARSE_KERNEL_PROPOSAL  Parse + validate an LLM proposal message.
%
%   [props, raw] = parse_kernel_proposal(response_text)
%
%   Extracts the first JSON object from the LLM reply (tolerating ```json
%   fences and surrounding prose), then validates every proposal against the
%   bounded grammar (kernel_grammar) and the mean library. Invalid items are
%   kept with .valid = false and a .why message (fed back to the LLM so it
%   can self-correct -- part of the closed loop).
%
%   OUTPUT
%     props : struct array with fields
%             .kernel_raw  string as proposed
%             .name        canonical expression name ('' if invalid)
%             .mean_id     1..4 (NaN if invalid)
%             .mean_name   'Zero'|'Constant'|'Linear'|'Quadratic'
%             .reason      LLM's one-line rationale
%             .valid, .why
%     raw   : decoded JSON struct (with .stop / .final / .analysis when given)
%
%   See also: agent_kernel_search, kernel_grammar

    % ---- canonical name set from the grammar ----
    persistent valid_names prim_order
    if isempty(valid_names)
        ex = kernel_grammar();
        valid_names = {ex.name};
        prim_order = {'SE','MA1','MA3','MA5','RQ','LIN','PER'};
    end

    % ---- extract JSON ----
    raw = struct(); props = struct('kernel_raw',{},'name',{},'mean_id',{}, ...
        'mean_name',{},'reason',{},'valid',{},'why',{});
    s = char(response_text);
    i0 = find(s == '{', 1, 'first'); i1 = find(s == '}', 1, 'last');
    if isempty(i0) || isempty(i1) || i1 <= i0
        raw.parse_error = 'no JSON object found';
        return
    end
    try
        raw = jsondecode(s(i0:i1));
    catch ME
        raw = struct('parse_error', ME.message);
        return
    end
    if ~isfield(raw, 'proposals'), return; end

    plist = raw.proposals;
    if isstruct(plist), plist = num2cell(plist); end   % struct array -> cell
    for i = 1:numel(plist)
        p = plist{i};
        q = struct('kernel_raw','','name','','mean_id',NaN, ...
                   'mean_name','','reason','','valid',false,'why','');
        if isfield(p,'kernel'), q.kernel_raw = strtrim(char(p.kernel)); end
        if isfield(p,'reason'), q.reason = char(p.reason); end

        % --- kernel: tokenize, canonicalize, check against grammar ---
        [nm, why] = canonicalize(q.kernel_raw, prim_order, valid_names);
        q.name = nm; q.why = why;

        % --- mean ---
        mn = ''; if isfield(p,'mean'), mn = lower(strtrim(char(p.mean))); end
        switch mn
            case {'zero'},                    q.mean_id = 1; q.mean_name = 'Zero';
            case {'constant','const'},        q.mean_id = 2; q.mean_name = 'Constant';
            case {'linear','lin'},            q.mean_id = 3; q.mean_name = 'Linear';
            case {'quadratic','quad'},        q.mean_id = 4; q.mean_name = 'Quadratic';
            otherwise
                q.why = strtrim([q.why ' unknown mean "' mn '";']);
        end

        q.valid = ~isempty(q.name) && ~isnan(q.mean_id);
        props(end+1) = q; %#ok<AGROW>
    end
end

% ------------------------------------------------------------------------
function [nm, why] = canonicalize(kstr, prim_order, valid_names)
    nm = ''; why = '';
    if isempty(kstr), why = 'empty kernel;'; return; end
    kstr = upper(strrep(kstr, ' ', ''));
    if contains(kstr,'+') && contains(kstr,'*')
        why = 'depth>2 (mixes + and *);'; return
    end
    if contains(kstr,'+'), op = '+'; elseif contains(kstr,'*'), op = '*';
    else, op = ''; end
    if isempty(op)
        toks = {kstr};
    else
        toks = strsplit(kstr, ['\' op], 'DelimiterType','RegularExpression');
    end
    if numel(toks) > 2, why = 'more than two primitives;'; return; end
    ok = cellfun(@(t) any(strcmp(t, prim_order)), toks);
    if ~all(ok)
        why = sprintf('unknown primitive "%s";', toks{find(~ok,1)}); return
    end
    if numel(toks) == 2
        if strcmp(toks{1}, toks{2})
            why = 'self-combination not allowed;'; return
        end
        [~, oi] = ismember(toks, prim_order);     % canonical ordering
        [~, si] = sort(oi); toks = toks(si);
        nm = [toks{1} op toks{2}];
    else
        nm = toks{1};
    end
    if ~any(strcmp(nm, valid_names))
        why = sprintf('"%s" not in grammar (pruned or invalid);', nm);
        nm = '';
    end
end
