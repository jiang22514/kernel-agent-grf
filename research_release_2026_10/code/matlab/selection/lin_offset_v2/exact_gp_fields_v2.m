function [F, ok, info] = exact_gp_fields_v2(expr, hyp, mf, xs, y, xq, NS, seed, opts)
%EXACT_GP_FIELDS_V1  精确稠密 GP 条件模拟（第七批 R08/B09：无诱导网格）
%   [F, ok, info] = exact_gp_fields_v2(expr, hyp, mf, xs, y, xq, NS, seed)
%   [F, ok, info] = exact_gp_fields_v2(..., opts)
%
%   后验: f_q | y ~ N(mu_q, C),  C = Kqq - Kqx Ky^{-1} Kqx'（无观测噪声于
%   查询潜场；均值函数 mf 由调用方给出 cell 形式）。
%
%   B09 修复（Codex：covRQprod 内部产生 3 个 q×q×D 数组，整块调用在
%   24k 点需 ~26GiB 临时）：
%     - Kqq 按行块构建（默认 2000 行/块）：块内三维数组 ~block×q×D×3，
%       峰值 ≈ C + Lc + 块临时 + Kqx/A（见 info.need_gb 估算式）；
%     - 内存门按分块后真实需求估算；memory 读取失败一律拒绝（旧版
%       avail=inf 放行漏洞已封）；
%     - 记录 need_gb / avail_gb / jitter 实际值 / n_blocks / 时长；
%       chol 失败如实返回 ok=false，不硬撑。
%   opts: .block_rows (2000) .mem_budget_gb (48) .jitter_levels
%         （默认 [1e-10 1e-9 1e-8 1e-7 1e-6] x base）
%
%   info: mu_q (qx1), sd_q (qx1 后验边际 std，jitter 前), jitter_used,
%         need_gb, avail_gb, n_blocks, t_kernel, t_chol, t_total, q, n

    if nargin < 9, opts = struct(); end
    if ischar(expr) && strcmp(expr, 'selftest')
        % C16：分块 vs 直接稠密对照（含尾块）+ 同随机数样本一致性 +
        % 内存门拒绝路径——纳入常规验收（b67 每轮正式/冒烟运行前调用）。
        [F, ok, info] = exact_selftest_();
        return
    end
    if ~isfield(opts,'block_rows'),    opts.block_rows = 2000; end
    if ~isfield(opts,'mem_budget_gb'), opts.mem_budget_gb = 48; end
    if ~isfield(opts,'jitter_levels'), opts.jitter_levels = [1e-10 1e-9 1e-8 1e-7 1e-6]; end

    ok = false; F = [];
    info = struct('jitter_used', NaN, 'q', size(xq,1), 'n', size(xs,1));
    t_all = tic;
    q = size(xq,1); n = size(xs,1); D = size(xs,2);

    % ---- 内存门（B09：分块后真实需求；读取失败拒绝） ----
    need_bytes = q*q*8*2.15 ...                    % C + Lc
               + opts.block_rows*q*8*3*D ...       % 块内 covRQprod 三维临时
               + q*n*8*2 ...                       % Kqx + A
               + q*NS*8;                           % F
    info.need_gb = need_bytes / 2^30;
    try
        [~, sysv] = memory;
        avail_bytes = sysv.PhysicalMemory.Available;
    catch
        info.reason = 'memory 查询失败（B09：一律拒绝，不放行）';
        return
    end
    info.avail_gb = avail_bytes / 2^30;
    if avail_bytes < need_bytes * 1.15 || info.need_gb > opts.mem_budget_gb
        info.reason = sprintf('内存门拒绝：需 %.1fGB（预算 %.0fGB），可用 %.1fGB', ...
                              info.need_gb, opts.mem_budget_gb, info.avail_gb);
        return
    end

    root_cov = build_covfunc_v2(parse_expr_str(expr), D);

    % ---- 训练侧 ----
    Kxx = feval(root_cov{:}, hyp.cov, xs);
    mu_tr = feval(mf{:}, hyp.mean, xs);
    sn2 = exp(2*hyp.lik);
    Ky = (Kxx + Kxx')/2 + sn2*eye(n);
    [L, pL] = chol(Ky, 'lower');
    if pL ~= 0
        info.reason = '训练协方差 Ky 的 chol 失败'; return
    end
    Kqx = feval(root_cov{:}, hyp.cov, xq, xs);   % q x n（三维临时 q*n*D 可承受）
    A = L \ Kqx.';                                % n x q
    al = L' \ (L \ (y - mu_tr));
    mu_q = feval(mf{:}, hyp.mean, xq) + Kqx * al;
    clear Kxx Kqx

    % ---- 分块构建 C = Kqq - A'A（B09 核心） ----
    t_k = tic;
    C = zeros(q, q);
    nb = ceil(q / opts.block_rows);
    info.n_blocks = nb;
    for bi = 1:nb
        r1 = (bi-1)*opts.block_rows + 1; r2 = min(bi*opts.block_rows, q);
        Kb = feval(root_cov{:}, hyp.cov, xq(r1:r2,:), xq);   % (block x q)
        C(r1:r2,:) = Kb - (A(:,r1:r2))' * A;
    end
    clear A Kb
    info.t_kernel = toc(t_k);
    C = (C + C')/2;
    sd_q = sqrt(max(diag(C), 0));
    dg0 = diag(C);   % D12：加抖动前对角（jitter 影响的分母必须用它）

    % ---- 抖动递增 chol ----
    t_c = tic;
    base = max(dg0);
    if base <= 0, info.reason = '后验协方差对角非正'; return; end
    Lc = []; p = 1;
    for jit = opts.jitter_levels * base
        C(1:q+1:end) = C(1:q+1:end) + jit;
        [Lc, p] = chol(C, 'lower');
        if p == 0
            info.jitter_used = jit / base;
            break;
        end
        C(1:q+1:end) = C(1:q+1:end) - jit;
    end
    info.t_chol = toc(t_c);
    if isempty(Lc) || p ~= 0
        info.reason = 'chol 在全部抖动档位下失败'; return
    end

    % ---- 采样 ----
    t_s = tic;
    rng(seed);
    F = mu_q + Lc * randn(q, NS);
    info.t_sample = toc(t_s);   % C15：采样段实测（勿用总时差充当采样时间）
    ok = true;
    info.mu_q = mu_q; info.sd_q = sd_q;
    % C16/D12：sd_q 对应加抖动前的条件协方差；样本对应抖动后矩阵。
    % 抖动对边际方差的相对增加 = jitter / 加抖动前方差（分母不得含抖动；
    % 原方差为零/负时记 Inf 并注明，不以"占最终方差比"掩盖）。
    info.sd_note = 'sd_q 为加抖动前条件协方差边际 std；样本对应抖动后矩阵（jitter_used 已记录）';
    if min(dg0) > 0
        info.jitter_var_rel_max = info.jitter_used * base / min(dg0);
    else
        info.jitter_var_rel_max = inf;
        info.jitter_note = '加抖动前最小方差为零/负，相对增加不可用有限比例表示';
    end
    info.t_total = toc(t_all);
end

function [F, ok, info] = exact_selftest_()
% C16 常规验收小例：n=60 训练、q=19 查询、block_rows=7（3 块含尾块）。
% 分块路径 vs 直接稠密公式：均值/协方差/同种子样本逐项对照；另验证
% 内存门拒绝路径与 jitter 记录。
    setup_revision_paths; % relocatable public dependency setup
    ok = false; F = [];
    info = struct('max_diff_mean', NaN, 'max_diff_sd', NaN, 'max_diff_samples', NaN, ...
                  'memgate_rejected', false, 'note', '');
    rng(7);
    xs = rand(60, 2); y = sin(3*xs(:,1)) + xs(:,2) + 0.05*randn(60,1);
    xq = rand(19, 2);
    expr = 'RQ';
    hyp = struct('mean', 0.5, 'cov', log([0.4 0.7 1.2 0.9]), 'lik', log(0.08));
    mf = {@meanConst};
    NS = 4; seed = 4242;
    o = struct('block_rows', 7);
    [Fb, okb, ib] = exact_gp_fields_v2(expr, hyp, mf, xs, y, xq, NS, seed, o);
    if ~okb, info.note = sprintf('分块路径失败: %s', ib.reason); return; end
    % 直接稠密参照
    covfunc = build_covfunc_v2(parse_expr_str(expr), 2);
    Kxx = feval(covfunc{:}, hyp.cov, xs);
    mu_tr = feval(mf{:}, hyp.mean, xs);
    sn2 = exp(2*hyp.lik);
    Ky = (Kxx + Kxx')/2 + sn2*eye(60);
    L = chol(Ky, 'lower');
    Kqx = feval(covfunc{:}, hyp.cov, xq, xs);
    Kqq = feval(covfunc{:}, hyp.cov, xq);
    al = L' \ (L \ (y - mu_tr));
    mu_d = feval(mf{:}, hyp.mean, xq) + Kqx * al;
    C_d = Kqq - Kqx * (Ky \ Kqx');
    C_d = (C_d + C_d')/2;
    jit = ib.jitter_used * max(diag(C_d));
    Cd_j = C_d; Cd_j(1:20:end) = Cd_j(1:20:end) + jit;
    Ld = chol(Cd_j, 'lower');
    rng(seed);
    Fd = mu_d + Ld * randn(19, NS);
    info.max_diff_mean = max(abs(ib.mu_q - mu_d));
    % D12：本字段是"边际标准差差"（旧名 max_diff_cov 名不副实——完整
    % 协方差差异由同随机数样本差间接覆盖，如实命名）
    info.max_diff_sd = max(abs(ib.sd_q - sqrt(max(diag(C_d),0))));
    info.max_diff_samples = max(abs(Fb(:) - Fd(:)));
    % 内存门拒绝路径（预算 0 GB 必须拒绝）
    [~, okm] = exact_gp_fields_v2(expr, hyp, mf, xs, y, xq, 1, 1, struct('mem_budget_gb', 0));
    info.memgate_rejected = ~okm;
    pass = info.max_diff_mean < 1e-9 && info.max_diff_sd < 1e-9 && ...
           info.max_diff_samples < 1e-9 && info.memgate_rejected;
    info.note = sprintf('mean %.2e sd %.2e samples %.2e memgate %d -> %s', ...
        info.max_diff_mean, info.max_diff_sd, info.max_diff_samples, ...
        info.memgate_rejected, tern(pass, 'PASS', 'FAIL'));
    ok = pass;
end

function s = tern(c, a, b), if c, s = a; else, s = b; end, end
