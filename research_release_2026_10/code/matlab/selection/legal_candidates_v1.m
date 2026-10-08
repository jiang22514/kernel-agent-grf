function [legal_mask, champ] = legal_candidates_v1(recs)
%LEGAL_CANDIDATES_V1  统一合法候选过滤（第七批 B08：94 合法核 x 4 均值 = 376 行）
%   [legal_mask, champ] = legal_candidates_v1(recs)
%
%   legal_mask : NK x 4 逻辑阵——recs(ki) 非 pruned 且 rows{mi}.ok 为 true
%   champ      : 合法冠军 struct('ki','mi','expr','mean_id','aic')；无合法行时为 []
%
%   用法：一切"冠军/排名/命中分母"必须经本函数过滤；416 行完整表仅作诊断。
%   自检：legal_candidates_v1('selftest') —— 构造 pruned 行 AIC 极低的假表，
%   验证合法冠军与排名不受影响（B08 验收用例），失败则 error。
%
%   See also: rescore_dataset_v1（recs 带 .pruned 字段）

    if ischar(recs) && strcmp(recs, 'selftest')
        selftest(); legal_mask = []; champ = []; return
    end

    NK = numel(recs);
    legal_mask = false(NK, 4);
    bestA = inf; champ = [];
    for ki = 1:NK
        pr = isfield(recs(ki),'pruned') && recs(ki).pruned;
        for mi = 1:4
            r = recs(ki).rows{mi};
            if pr || ~r.ok, continue; end
            legal_mask(ki,mi) = true;
            if r.aic_final < bestA
                bestA = r.aic_final;
                champ = struct('ki', ki, 'mi', mi, 'expr', recs(ki).expr, ...
                               'mean_id', mi, 'aic', r.aic_final);
            end
        end
    end
end

function selftest()
% B08 验收：pruned 候选 AIC 极低不得改变合法冠军/合法排名
    mk = @(ok, aic) struct('ok', ok, 'aic_final', aic, 'nlZ_final', aic/2, ...
        'hyp_final', [], 'k', 1, 'mean_name', 'Constant', 'mean_id', 2);
    % rows{mi} 必须是 struct（C01 复审：多包一层 cell 会在 r.ok 处崩溃）；
    % 逐核赋值构造，避免 struct() 对标量 cell 的扩张歧义
    recs = struct('expr', {'SE', 'SE*MA3', 'RQ'}, 'pruned', {false, true, false}, ...
                  'rows', {[], [], []});
    aics = [500 100 510];
    for ki = 1:3
        recs(ki).rows = {mk(false, inf), mk(true, aics(ki)), mk(false, inf), mk(false, inf)};
    end
    % 假表：合法冠军应为 SE+Const (500)；pruned 的 SE*MA3+Const (100) 不得胜出
    [mask, champ] = legal_candidates_v1(recs);
    assert(isstruct(champ) && strcmp(champ.expr, 'SE') && champ.aic == 500, ...
        'B08 selftest 失败：pruned 候选影响了合法冠军');
    assert(nnz(mask) == 2, 'B08 selftest 失败：合法行计数错误');
    % 对照：不过滤时最小 AIC 是 100（证明测试有鉴别力）
    allA = [500 100 510];
    assert(min(allA) == 100, 'selftest 构造错误');
    fprintf('legal_candidates_v1 selftest PASS（pruned AIC=100 被正确排除，合法冠军 SE=500）\n');
end
