%% make_borehole_sim_figures.m
%  Borehole random-field simulation figures for the revised paper, with a
%  THREE-way comparison that mirrors the paper's storyline:
%
%    (1) SE + Quadratic      "community default" -- the squared-exponential
%                            kernel used by most geotechnical random-field
%                            studies without any model selection;
%    (2) RQ + Constant       "best single kernel" -- the optimum if model
%                            selection is restricted to single kernels
%                            (overall rank 34/156: everything above it is
%                            composite);
%    (3) MA1*LIN + Constant  "grammar champion" -- the composite found by
%                            exhaustive search and rediscovered by 7/8 LLM
%                            agents at ~12% of the exhaustive cost.
%
%  Each step answers one narrative question: default vs selected (why select
%  at all), single vs composite (why a compositional grammar).
%
%  Outputs (paper_figs/):
%    borehole_fields.png      3x3: posterior mean | posterior std | realization
%    borehole_variogram.png   vertical semivariogram: data vs the three models
%    borehole_sim_results.mat
%
%  Run:  >> make_borehole_sim_figures

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
rng(7);

S  = load('samedata.mat'); x = S.x_same; y = S.y_same;   % physical units
CH = load('champion_borehole.mat');   champion   = CH.champion;
BL = load('baseline_borehole.mat');   baseline   = BL.baseline;
SB = load('singlebest_borehole.mat'); singlebest = SB.singlebest;
sx = champion.sx;
xs_dat = x ./ sx;                       % scaled coords (model frame)
n = size(x,1);

NS      = 200;                          % posterior samples per model
out_dir = fullfile(here, 'paper_figs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---- plotting grid (physical units -> scaled for the model) -------------
ng1 = 240; ng2 = 140;
g1 = linspace(min(x(:,1)), max(x(:,1)), ng1);
g2 = linspace(min(x(:,2)), max(x(:,2)), ng2);
[G1, G2] = meshgrid(g1, g2);
xq = [G1(:), G2(:)] ./ sx;
nq = size(xq,1);

models = struct( ...
    'short',   {'SE+Quad', 'RQ+Const', 'MA1*LIN+Const'}, ...
    'role',    {'community default', 'best single kernel', 'grammar champion'}, ...
    'expr',    {'SE', 'RQ', 'MA1*LIN'}, ...
    'mean_id', {4, 2, 2}, ...
    'hyp',     {baseline.hyp, singlebest.hyp, champion.hyp}, ...
    'aic',     {baseline.aic, singlebest.aic, champion.aic});
n_mod = numel(models);

res = repmat(struct('fmu',[],'fsd',[],'real1',[],'covfunc',[],'sn2',[], ...
                    'info',[]), 1, n_mod);

for mi = 1:n_mod
    M = models(mi);
    fprintf('=== %s (%s) ===\n', M.short, M.role);

    % --- per-primitive hyp struct for the separable-term expansion ---
    hyp_split = struct();
    switch M.expr
        case 'SE'
            hyp_split.SE  = M.hyp.cov(1:3);
            pts_per_l = 5;                      % smooth: l/h>=5 suffices
        case 'RQ'
            hyp_split.RQ  = M.hyp.cov(1:4);
            hyp_split.RQ_J = 15;                % alpha ~ 0.1: heavy-tailed
            pts_per_l = 10;
        case 'MA1*LIN'
            hyp_split.MA1 = M.hyp.cov(1:3);
            hyp_split.LIN = M.hyp.cov(4:5);
            pts_per_l = 10;                     % MA1 kink: denser grid
    end
    terms = expand_expr_terms(M.expr, 2, hyp_split);

    % exact covariance handle (for the variogram + RQ-mixture check)
    [cf, ~] = build_covfunc_from_expr(M.expr, 2);
    res(mi).covfunc = {cf, M.hyp.cov};
    if strcmp(M.expr, 'RQ')                     % verify SE-mixture accuracy
        idx = 1:3:n;
        Kex = feval(cf{:}, M.hyp.cov, xs_dat(idx,:));
        Kap = kdense_terms(terms, xs_dat(idx,:), xs_dat(idx,:));
        err = max(abs(Kap(:)-Kex(:))) / max(abs(Kex(:)));
        fprintf('  RQ SE-mixture (J=%d, K=%d terms): max rel err %.2e\n', ...
                hyp_split.RQ_J, numel(terms), err);
        assert(err < 0.03, 'RQ mixture too inaccurate; raise RQ_J');
    end

    % --- mean values (scaled frame) ---
    switch M.mean_id
        case 2, mf = {@meanConst};
        case 4, mf = {@meanSum, {{@meanConst}, {@meanPoly, 2}}};
    end
    mu_train = feval(mf{:}, M.hyp.mean, xs_dat);
    mu_query = feval(mf{:}, M.hyp.mean, xq);

    % --- inducing grid, density by the l/h rule (capped) ---
    xg = cell(1,2);
    for d = 1:2
        lo_d = min([xq(:,d); xs_dat(:,d)]); hi_d = max([xq(:,d); xs_dat(:,d)]);
        h = exp(M.hyp.cov(d)) / pts_per_l;
        mg = min(max(ceil((hi_d - lo_d)/h) + 1, 60), 1100);
        pad = 4 * (hi_d - lo_d) / (mg - 1);
        xg{d} = linspace(lo_d - pad, hi_d + pad, mg)';
    end
    fprintf('  grid %d x %d (= %d)\n', numel(xg{1}), numel(xg{2}), ...
            numel(xg{1})*numel(xg{2}));

    % --- simulate (dense-chol solver + FITC diagonal correction) ---
    o = struct('sn', exp(M.hyp.lik), 'mu_train', mu_train, ...
               'mu_test', mu_query, 'n_samples', NS, 'solver', 'chol', ...
               'fitc_alpha', 1, 'seed', 100 + mi);
    [fmu_all, F_all, info] = grid_sim_terms(terms, xg, xs_dat, y, xq, o);
    if isfield(info, 'fitc_g_mean')
        fprintf('  FITC diag correction: mean g = %.4g\n', info.fitc_g_mean);
    end
    fprintf('  setup %.1fs, %d samples %.1fs\n', info.t_setup, NS, info.t_samples);

    res(mi).fmu   = reshape(fmu_all, ng2, ng1);
    res(mi).fsd   = reshape(std(F_all, 0, 2), ng2, ng1);
    res(mi).real1 = reshape(F_all(:,1), ng2, ng1);
    res(mi).sn2   = exp(2*M.hyp.lik);
    res(mi).info  = info;
end

%% ---------------- figure 1: fields (3 models x 3 panels) ------------------
fig = figure('Position', [40 40 1500 1120], 'Color', 'w', 'Visible', 'off');
titles = {'Posterior mean', 'Posterior std', 'One realization'};
sd_max = max(arrayfun(@(r) max(r.fsd(:)), res));
for mi = 1:n_mod
    panels = {res(mi).fmu, res(mi).fsd, res(mi).real1};
    for pi = 1:3
        ax = subplot(n_mod, 3, (mi-1)*3 + pi);
        imagesc(g1, g2, panels{pi}); axis xy tight;
        set(ax, 'YDir', 'reverse');
        colormap(ax, 'parula'); colorbar;
        if pi == 2, caxis([0, sd_max]); else, caxis([min(y), max(y)]); end
        if pi == 1
            hold on; scatter(x(:,1), x(:,2), 4, 'k', 'filled', ...
                             'MarkerFaceAlpha', 0.35); hold off;
        end
        title(sprintf('%s (%s, AIC=%.1f) — %s', models(mi).short, ...
              models(mi).role, models(mi).aic, titles{pi}), ...
              'Interpreter', 'none', 'FontSize', 10);
        xlabel('x_1 [m]'); ylabel('depth x_2 [m]');
    end
end
sgtitle(['Borehole CPT: community default  vs  best single kernel  vs  ' ...
         'composite champion'], 'FontWeight', 'bold');
exportgraphics(fig, fullfile(out_dir, 'borehole_fields.png'), 'Resolution', 200);
close(fig);

%% ---------------- figure 2: vertical semivariogram ------------------------
% Data: empirical vertical semivariogram of linearly detrended residuals.
% Models: model-implied gamma(h) from the EXACT kernels (no mixture approx),
% averaged over data locations (the LIN term is nonstationary).
B = [ones(n,1), x];
gv_dat = vert_semivar(x, y - B*(B\y));
hh = linspace(0.05, max(gv_dat.h), 60)';
GV_th = zeros(numel(hh), n_mod);
for mi = 1:n_mod
    GV_th(:, mi) = theo_vert_semivar(res(mi).covfunc, res(mi).sn2, ...
                                     xs_dat, sx, hh);
end

fig = figure('Position', [60 60 780 560], 'Color', 'w', 'Visible', 'off');
hold on;
cols = [0.85 0.33 0.10; 0.93 0.69 0.13; 0.00 0.45 0.74];
for mi = 1:n_mod
    plot(hh, GV_th(:,mi), '-', 'Color', cols(mi,:), 'LineWidth', 2.0);
end
plot(gv_dat.h, gv_dat.g, 'ko', 'LineWidth', 1.4, 'MarkerFaceColor', 'k', ...
     'MarkerSize', 6);
hold off; grid on; box on;
xlabel('vertical lag h [m]'); ylabel('semivariance \gamma(h)');
legend([arrayfun(@(m) sprintf('%s (%s, AIC=%.1f)', m.short, m.role, m.aic), ...
        models, 'UniformOutput', false), {'data (detrended, empirical)'}], ...
       'Location', 'southeast', 'Interpreter', 'none', 'FontSize', 9);
title('Vertical semivariogram: data vs model-implied');
exportgraphics(fig, fullfile(out_dir, 'borehole_variogram.png'), 'Resolution', 220);
close(fig);

save(fullfile(out_dir, 'borehole_sim_results.mat'), 'res', 'models', ...
     'g1', 'g2', 'gv_dat', 'hh', 'GV_th', 'NS', '-v7.3');
fprintf('\nSaved figures + results -> %s\nDONE.\n', out_dir);

%% ---------------- helpers --------------------------------------------------
function g = theo_vert_semivar(covpair, sn2, xs_dat, sx, hh)
% model-implied vertical semivariogram from the exact kernel, averaged over
% data locations: gamma(h) = mean_i[(k(x,x)+k(x',x'))/2 - k(x,x')] + sn2.
    cf = covpair{1}; hyp = covpair{2};
    g = zeros(numel(hh), 1);
    zmax = max(xs_dat(:,2));
    for b = 1:numel(hh)
        dz = hh(b) / sx(2);
        xa = xs_dat(xs_dat(:,2) + dz <= zmax, :);
        xb = xa; xb(:,2) = xb(:,2) + dz;
        kaa = feval(cf{:}, hyp, xa, 'diag');
        kbb = feval(cf{:}, hyp, xb, 'diag');
        kab = diag(feval(cf{:}, hyp, xa, xb));
        g(b) = mean((kaa + kbb)/2 - kab) + sn2;
    end
end

function Kd = kdense_terms(terms, xa, xb)
    D = numel(terms(1).f);
    Kd = 0;
    for k = 1:numel(terms)
        Kk = 1;
        for d = 1:D
            Kk = Kk .* feval(terms(k).f{d}{:}, terms(k).h{d}, xa(:,d), xb(:,d));
        end
        Kd = Kd + Kk;
    end
end

function s = vert_semivar(x, r)
% vertical-direction empirical semivariogram (pairs with |dx1| < 2 m)
    d1 = abs(x(:,1) - x(:,1)');
    d2 = abs(x(:,2) - x(:,2)');
    G  = 0.5 * (r - r').^2;
    iu = triu(true(numel(r)), 1);
    sel = iu & (d1 < 2);
    dv = d2(sel); gv = G(sel);
    edges = linspace(0, quantile(dv, 0.6), 13);
    h = zeros(numel(edges)-1, 1); g = zeros(numel(edges)-1, 1);
    for b = 1:numel(edges)-1
        m = dv >= edges(b) & dv < edges(b+1);
        h(b) = mean(dv(m)); g(b) = mean(gv(m));
    end
    keep = ~isnan(h);
    s = struct('h', h(keep), 'g', g(keep));
end
