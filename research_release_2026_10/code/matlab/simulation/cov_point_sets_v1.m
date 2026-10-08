function sets = cov_point_sets_v1(pack, xs)
%COV_POINT_SETS_V1  协方差检查点位集——中心/近钻孔/远钻孔/边界条带（各 25 点）
%  D08 检查器组件（G2：独立文件，检查器修改不影响采样指纹）。
%
%   See also: marginal_check_ski, ski_analytic_check_v1

    g1 = pack.g1; g2 = pack.g2; xq = pack.xq;
    n1 = numel(g1); n2 = numel(g2);
    % 1) 中心 5x5
    i1c = max(1, min(n1-4, round(n1/2)-2)) + (0:4);
    i2c = max(1, min(n2-4, round(n2/2)-2)) + (0:4);
    [I1, I2] = meshgrid(i1c, i2c);
    idx_center = sub2ind([n2 n1], I2(:), I1(:));
    % 2)/3) 近钻孔 / 远钻孔：按到训练点的最小欧氏距离排序取两端 25 点
    dmin = min(pdist2(xq, xs), [], 2);
    [~, ord] = sort(dmin);
    idx_near = sort(ord(1:25));
    idx_far  = sort(ord(end-24:end));
    % 4) 边界条带：左缘 5x5 角区
    i1b = 1:5; i2b = max(1, min(n2-4, round(n2/2)-2)) + (0:4);
    [J1, J2] = meshgrid(i1b, i2b);
    idx_edge = sub2ind([n2 n1], J2(:), J1(:));
    sets = struct('name', {'center5x5','near_borehole25','far_borehole25','edge_strip25'}, ...
                  'idx', {idx_center, idx_near, idx_far, idx_edge});
end
