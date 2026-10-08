function [xs, covfunc, mf, n_cov] = eval_geometry_v1(expr, mean_id, x)
%EVAL_GEOMETRY_V1  评估路径"几何"单一来源：坐标缩放 + 核函数 + 均值函数
%   [xs, covfunc, mf] = eval_geometry_v1(expr, mean_id, x)
%   [xs, covfunc, mf, n_cov] = eval_geometry_v1(expr, mean_id, x)
%
%   口径（与 evaluate_expr_aic_v1 / point_nlz_v1 完全一致；D11 复审后
%   evaluate_expr_aic_v1 自身也经本函数取得坐标/核/均值——拟合路径与
%   对照同源，任何一侧被改都会破坏 b65 对齐验收）：
%     - 坐标按各维样本 std 缩放（sx==0 -> 1，不中心化）；
%     - 核由 build_covfunc_v1(parse_expr_str(expr), D) 构造；
%     - 均值族 id 1..4 = Zero/Const/Const+Linear/Const+Poly2。
%
%   See also: point_nlz_v1, evaluate_expr_aic_v1, build_covfunc_v1

    sx = std(x); sx(sx == 0) = 1;
    xs = x ./ sx;
    if ischar(expr), expr = parse_expr_str(expr); end   % 与拟合入口同：接受核名或语法结构
    [covfunc, n_cov] = build_covfunc_v1(expr, size(x, 2));
    switch mean_id
        case 1, mf = {@meanZero};
        case 2, mf = {@meanConst};
        case 3, mf = {@meanSum, {{@meanConst}, {@meanLinear}}};
        case 4, mf = {@meanSum, {{@meanConst}, {@meanPoly, 2}}};
        otherwise
            error('eval_geometry_v1:badMean', 'mean_id 须为 1..4');
    end
end
