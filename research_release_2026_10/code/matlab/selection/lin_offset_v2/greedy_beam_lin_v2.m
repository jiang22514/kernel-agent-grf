function out = greedy_beam_lin_v2(recs, opts)
% Same layer-updating width-two search as greedy_beam_v3.
% Only the table adapter changes: explicit legal recs, no fitting or fallback.
% Caller verifies the FINAL source; score values are read only on visits.
    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'budget'),     opts.budget = 20;      end
    if ~isfield(opts,'width'),      opts.width  = 2;       end
    if ~isfield(opts,'table_only'), opts.table_only = true; end

    assert(opts.table_only, 'LIN v2 baseline is table-only');
    prims = {'SE','MA1','MA3','MA5','RQ','LIN','PER'};
    scores = containers.Map('KeyType','char','ValueType','double');
    for ki=1:numel(recs)
        if recs(ki).pruned, continue; end
        for mi=1:4
            row=recs(ki).rows{mi}; value=inf;
            if row.ok && isfinite(row.aic_final),value=row.aic_final;end
            scores(sprintf('%s|%d',recs(ki).expr,mi))=value;
        end
    end
    assert(scores.Count==376);

    seq = {}; seqA = []; traj = []; layer_of = [];
    seen = containers.Map('KeyType','char','ValueType','logical');
    n_skipped_offtable = 0;

    % --- L0: primitives x {Zero, Quadratic}（与 v2 一致） ---
    for mi = [1 4]
        for p = prims, do_step(p{1}, mi, 0); end
    end
    beam = topw(seq, seqA, opts.width);
    beam_hist = {beam}; layer = 0;

    % --- 真逐层：每层结束重排全局 top-w ---
    while numel(seq) < opts.budget
        layer = layer + 1;
        cand = {};
        for bi = 1:numel(beam)
            cand = [cand, children_of(beam{bi}, prims)]; %#ok<AGROW>
        end
        cand = unique(cand, 'stable');
        progressed = false;
        for ci2 = 1:numel(cand)
            if numel(seq) >= opts.budget, break; end
            key = cand{ci2};
            if isKey(seen, key), continue; end
            pp = strsplit(key, '|');
            if opts.table_only && ~isKey(scores, key)
                n_skipped_offtable = n_skipped_offtable + 1;
                seen(key) = true;   % 本预算内不再尝试
                continue;
            end
            do_step(pp{1}, str2double(pp{2}), layer);
            progressed = true;
        end
        if ~progressed, break; end
        newbeam = topw(seq, seqA, opts.width);
        if isequal(sort(newbeam), sort(beam)) && layer > 1
            % beam 已稳定且其子代全部评估过：搜索自然终止
            beam = newbeam; beam_hist{end+1} = beam;
            break;
        end
        beam = newbeam; beam_hist{end+1} = beam;
    end

    out = struct('seq', {seq}, 'aic', seqA, 'traj', traj, 'layer', layer_of, ...
                 'evals_used', numel(seq), 'beam_hist', {beam_hist}, ...
                 'n_layers', numel(beam_hist), ...
                 'n_skipped_offtable', n_skipped_offtable, ...
                 'cache_stats', struct('hits', numel(seq), 'misses', 0, ...
                 'gp_time_total', 0));

    function do_step(kernel, mean_id, ly)
        key = sprintf('%s|%d', kernel, mean_id);
        if isKey(seen, key) || numel(seq) >= opts.budget, return; end
        seen(key) = true;
        assert(isKey(scores,key),'Unknown candidate; live fitting is disabled');
        a=scores(key);
        seq{end+1} = key; seqA(end+1) = a;
        traj(end+1) = min(seqA); layer_of(end+1) = ly;
    end
end

%% ================= helpers =================
function beam = topw(seq, seqA, w)
    [~, ord] = sort(seqA);
    beam = seq(ord(1:min(w, numel(ord))));
end

function q = children_of(pairkey, prims)
% 与 v2 相同的四个算子家族（均值换档 / 乘积 / 加法 / 乘积->T3），规范化字典序
    parts = strsplit(pairkey, '|'); kern = parts{1}; pm = str2double(parts{2});
    mem = regexp(strrep(strrep(kern,'(',''),')',''), '[*+]', 'split');
    q = {};
    for mi2 = setdiff(1:4, pm)
        q{end+1} = sprintf('%s|%d', kern, mi2);
    end
    if ~contains(kern,{'+','*'})
        for q2 = {'LIN','PER'}
            if strcmp(q2{1}, kern), continue; end
            [~, o] = ismember({kern, q2{1}}, prims); [~, si] = sort(o);
            t = {kern, q2{1}}; t = t(si);
            q{end+1} = sprintf('%s*%s|%d', t{1}, t{2}, pm);
        end
        for q2 = prims
            if strcmp(q2{1}, kern), continue; end
            [~, o] = ismember({kern, q2{1}}, prims); [~, si] = sort(o);
            t = {kern, q2{1}}; t = t(si);
            q{end+1} = sprintf('%s+%s|%d', t{1}, t{2}, pm);
        end
    end
    if contains(kern,'*') && ~startsWith(kern,'(')
        for q2 = prims
            if any(strcmp(q2{1}, mem)), continue; end
            q{end+1} = sprintf('(%s)+%s|%d', kern, q2{1}, pm);
        end
    end
end
