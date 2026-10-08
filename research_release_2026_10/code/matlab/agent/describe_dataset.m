function txt = describe_dataset(x, y, case_label)
%DESCRIBE_DATASET  "Perceive" stage of the closed-loop agent: compact,
%  numeric data profile handed to the LLM (no raw data, no leakage of any
%  ground-truth label -- only generic statistics a practitioner could read
%  off the dataset).
%
%   txt = describe_dataset(x, y, case_label)
%
%  Contents: size/dimension, per-dim coordinate ranges, target stats,
%  polynomial-trend fit quality (R2 of linear/quadratic OLS), and an
%  isotropic empirical semivariogram of the linearly detrended residuals
%  (6 lags, normalized) computed in std-scaled coordinates.
%
%   See also: agent_kernel_search

    if nargin < 3, case_label = 'dataset'; end
    [n, D] = size(x);

    L = {};
    L{end+1} = sprintf('Dataset: %s', case_label);
    L{end+1} = sprintf('n = %d observations, D = %d input dimensions.', n, D);
    for d = 1:D
        L{end+1} = sprintf('  x%d: range [%.4g, %.4g], std %.4g', ...
                           d, min(x(:,d)), max(x(:,d)), std(x(:,d))); %#ok<AGROW>
    end
    L{end+1} = sprintf('y: mean %.4g, std %.4g, range [%.4g, %.4g].', ...
                       mean(y), std(y), min(y), max(y));

    % polynomial trend diagnostics (with intercept)
    yc = y - mean(y); sst = sum(yc.^2);
    B1 = [ones(n,1), x];
    r2_lin = 1 - sum((y - B1*(B1\y)).^2) / sst;
    B2 = [B1, x.^2];
    r2_quad = 1 - sum((y - B2*(B2\y)).^2) / sst;
    L{end+1} = sprintf(['Trend fit (OLS R^2): linear %.3f, ', ...
                        'quadratic(diagonal) %.3f.'], r2_lin, r2_quad);

    % isotropic empirical semivariogram of linearly detrended residuals,
    % in std-scaled coordinates (so lags are comparable across dims)
    res = y - B1*(B1\y);
    sx = std(x); sx(sx == 0) = 1;
    xs = x ./ sx;
    m = min(n, 600);                       % cap pair count for speed
    idx = randperm(n, m);
    Dm = pdist2_local(xs(idx,:), xs(idx,:));
    Gv = 0.5 * (res(idx) - res(idx)').^2;
    iu = triu(true(m), 1);
    dv = Dm(iu); gv = Gv(iu);
    edges = quantile(dv, linspace(0.02, 0.7, 7));
    gh = zeros(1,6); hh = zeros(1,6);
    for b = 1:6
        sel = dv >= edges(b) & dv < edges(b+1);
        gh(b) = mean(gv(sel)); hh(b) = mean(dv(sel));
    end
    gh = gh / var(res);                    % normalize by residual variance
    sv = sprintf('(h=%.2f, g=%.2f) ', [hh; gh]);
    L{end+1} = ['Semivariogram of linearly-detrended residuals ', ...
                '(std-scaled coords, normalized; g->1 means decorrelated): ', sv];

    txt = strjoin(L, newline);
end

% ------------------------------------------------------------------------
function Dm = pdist2_local(A, B)
% Euclidean distance matrix without toolbox dependencies.
    Dm = sqrt(max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'), 0));
end
