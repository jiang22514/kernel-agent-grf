function mg = marginal_check_ski_v2(expr, hyp, mf, xs, y, pack, theta)
%MARGINAL_CHECK_SKI  近似(SKI)采样器 vs 精确后验的全网格边际对照 + 协方差检查
%  C11/D08 检查器（G2：独立文件——检查器修改不影响 b67 采样指纹；
%  trap_weights/expected_gap 为局部稳定副本，勿与 b67 内版本分叉）。
%
%  解析参照 = 精确 GP 条件均值/边际 std（分块线性代数，非抽样量）；
%  判据（预登记，先于结果固定）：RMS 归一均值差 < 0.05；经验/解析 std
%  比中位 ∈ [0.95,1.05]；E[wp] 解析期望差 <= 3xMC_se；协方差结构按
%  归一化比 err/(4x解析MC尺度) <= 1 判定。
%  D08 协方差点位集：中心 5x5 / 近钻孔 25 / 远钻孔 25 / 边界条带 25。
%  诚实边界：本检查比较"样本经验协方差 vs 精确条件协方差"，容差=4x解析
%  MC 尺度；系统性超出部分即近似+离散化误差量级。SKI 近似模型本身的
%  条件协方差可解析计算（见 ski_analytic_check_v1，小子集误差分离）；
%  本函数的全网格经验检查只查各组内部，不宣称覆盖组间交叉协方差或全域
%  联合分布。NS<100 协方差检查降级为记录（冒烟）。
%
%   See also: cov_point_sets_v1, ski_analytic_check_v1, b67_consequence_v3

    mg = struct();
    xq = pack.xq; F = pack.F; fmu = pack.fmu;
    NS = size(F,2);
    covfunc = build_covfunc_v2(parse_expr_str(expr), size(xs,2));
    Kxx = feval(covfunc{:}, hyp.cov, xs);
    mu_tr = feval(mf{:}, hyp.mean, xs);
    sn2 = exp(2*hyp.lik);
    Ky = (Kxx + Kxx')/2 + sn2*eye(size(xs,1));
    L = chol(Ky, 'lower');
    al = L' \ (L \ (y - mu_tr));
    q = size(xq,1); nb = ceil(q/2000);
    mu_ex = zeros(q,1); sd_ex = zeros(q,1);
    for bi = 1:nb
        r1 = (bi-1)*2000+1; r2 = min(bi*2000, q);
        Ksx = feval(covfunc{:}, hyp.cov, xq(r1:r2,:), xs);
        Kss = feval(covfunc{:}, hyp.cov, xq(r1:r2,:), 'diag');
        mu_ex(r1:r2) = feval(mf{:}, hyp.mean, xq(r1:r2,:)) + Ksx * al;
        sd_ex(r1:r2) = sqrt(max(Kss - sum((L \ Ksx').^2, 1)', 1e-16));
    end
    dm = (fmu - mu_ex) ./ max(sd_ex, 1e-12);
    mg.rms_mean = sqrt(mean(dm.^2));
    mg.max_abs_mean = max(abs(dm));
    sr = std(F, 0, 2) ./ max(sd_ex, 1e-12);
    mg.sd_ratio_med = median(sr);
    mg.sd_ratio_q = prctile(sr, [5 95]);   % 分位数展示用（D08：不只中位数）
    mg.n_grid = q;
    % E[wp] 解析期望（解析边际 sd，非同批样本 std，C11）
    W = trap_weights_local_(pack.g1, pack.g2);
    mg.exp_gap = expected_gap_local_(pack.wp, mu_ex, sd_ex, W, pack.g1, pack.g2, theta);
    mg.exp_mc_se = sqrt(var(pack.wp)/numel(pack.wp));
    % ---- D08 多点位协方差结构检查 ----
    mg.cov_done = false; mg.cov_ratio_max = NaN; mg.cov_sets = {};
    if NS >= 100
        sets = cov_point_sets_frozen_v2(pack, xs);
        ratios = []; mg.cov_sets = cell(1, numel(sets));
        for si = 1:numel(sets)
            idx = sets(si).idx;
            Fsub = F(idx, :);
            Kss25 = feval(covfunc{:}, hyp.cov, xq(idx,:));
            Ksx25 = feval(covfunc{:}, hyp.cov, xq(idx,:), xs);
            C25 = Kss25 - Ksx25 * (L' \ (L \ Ksx25'));
            C25 = (C25 + C25')/2;
            Chat = cov(Fsub');
            nrmC = norm(C25, 'fro');
            err = norm(Chat - C25, 'fro') / nrmC;
            mc_sc = sqrt((trace(C25)^2 + nrmC^2) / NS) / nrmC;
            tol = 4 * mc_sc;
            ratios(end+1) = err / tol; %#ok<AGROW>
            mg.cov_sets{si} = struct('name', sets(si).name, 'err', err, ...
                'mc_scale', mc_sc, 'tol', tol, 'ratio', err/tol);
        end
        mg.cov_ratio_max = max(ratios);
        mg.cov_done = true;
        mg.cov_note = sprintf('%d 点位集 max err/tol=%.3f（<=1 过）', numel(sets), mg.cov_ratio_max);
    else
        mg.cov_note = sprintf('NS=%d<100降级为记录', NS);
    end
    pass_cov = ~mg.cov_done || mg.cov_ratio_max <= 1;
    mg.pass = mg.rms_mean < 0.05 && mg.sd_ratio_med >= 0.95 && mg.sd_ratio_med <= 1.05 && ...
              abs(mg.exp_gap) <= 3*mg.exp_mc_se && pass_cov;
    fails = {};
    if mg.rms_mean >= 0.05, fails{end+1} = 'rms_mean'; end
    if mg.sd_ratio_med < 0.95 || mg.sd_ratio_med > 1.05, fails{end+1} = 'sd_ratio'; end
    if abs(mg.exp_gap) > 3*mg.exp_mc_se, fails{end+1} = 'exp_gap'; end
    if ~pass_cov, fails{end+1} = 'covariance'; end
    if isempty(fails), mg.fail_note = ''; else, mg.fail_note = strjoin(fails, ','); end
end

function W = trap_weights_local_(g1, g2)
    wx = [0.5, ones(1, numel(g1)-2), 0.5] * (g1(2)-g1(1));
    wy = [0.5, ones(1, numel(g2)-2), 0.5] * (g2(2)-g2(1));
    W = wy(:) * wx(:).';
end

function gap = expected_gap_local_(wp, mu, sd, W, g1, g2, theta)
% E[wp]=Σ w_i Φ((θ-μ_i)/σ_i) vs 样本均值（应 <3×MC_se）
    MU = reshape(mu, numel(g2), numel(g1));
    SD = reshape(sd, numel(g2), numel(g1));
    Ewp = sum(W .* normcdf((theta - MU) ./ max(SD, 1e-12)), [1 2]) / sum(W(:));
    mc_se = sqrt(var(wp)/numel(wp));
    gap = mean(wp) - Ewp;
    if abs(gap) > 3*mc_se
        fprintf('  *** 解析期望对照 |mean(wp)-E[wp]|=%.4f > 3×MC_se=%.4f（如实记录）\n', abs(gap), 3*mc_se);
    end
end
