%% make_agent_figures.m
%  Paper material from the closed-loop agent study (agent_study_results.mat):
%
%   (1) paper_figs/agent_convergence.png
%       best-so-far regret vs round, per case (8 models), plus an aggregate
%       panel (median over runs, cheap tier vs flagship tier)
%   (2) paper_figs/agent_openloop_vs_closedloop.png
%       open-loop (round-1 only, no feedback) vs closed-loop final regret
%   (3) console + research-log markdown: per-model behavior table
%       (exact hits / top-3 / top-5, evals, rounds, stop behavior, tokens)
%
%  Run:  >> make_agent_figures

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

L = load('agent_study_results.mat');
R = L.R; tags = L.tags; cases = L.cases;
[n_models, n_cases] = size(R);
MAXR = 4;
flagship = ismember(tags, {'gpt-5.5','claude-opus-4.8','gemini-3.1-pro'});
out_dir = fullfile(here, 'paper_figs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---- regret-by-round matrix (runs padded flat after early stop) ----------
REG = NaN(n_models, n_cases, MAXR);
for mi = 1:n_models
    for ci = 1:n_cases
        r = R{mi,ci}; if isempty(r) || ~r.ok, continue; end
        tr = r.traj(:)' - r.gt_aic;
        tr = max(tr, 0);                       % numerical safety
        tr(end+1:MAXR) = tr(end);              % flat after stop
        REG(mi,ci,:) = tr(1:MAXR);
    end
end

% ---- (1) convergence figure ----------------------------------------------
% cheap tier: muted blues/greys; flagship: warm bold
cols = [0.55 0.63 0.72; 0.45 0.55 0.75; 0.60 0.70 0.60; ...
        0.70 0.62 0.50; 0.55 0.50 0.65; ...
        0.85 0.33 0.10; 0.64 0.08 0.18; 0.49 0.18 0.56];
mk = {'o','s','d','^','v','o','s','d'};

fig = figure('Position', [40 40 1480 760], 'Color', 'w', 'Visible', 'off');
for ci = 1:n_cases
    ax = subplot(2, 3, ci); hold(ax, 'on');
    for mi = 1:n_models
        lw = 1.2 + 1.0*flagship(mi);
        plot(ax, 1:MAXR, squeeze(REG(mi,ci,:)), ['-' mk{mi}], ...
             'Color', cols(mi,:), 'LineWidth', lw, 'MarkerSize', 5, ...
             'MarkerFaceColor', cols(mi,:));
    end
    hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on');
    title(ax, cases(ci).name, 'Interpreter', 'none');
    xlabel(ax, 'round'); ylabel(ax, 'regret (\DeltaAIC vs exhaustive optimum)');
    xticks(ax, 1:MAXR); xlim(ax, [0.85 MAXR+0.15]);
end
% aggregate panel: median over the 5 cases, by tier
ax = subplot(2, 3, 6); hold(ax, 'on');
mcheap = squeeze(median(reshape(REG(~flagship,:,:), [], n_cases, MAXR), [1 2], 'omitnan'));
mflag  = squeeze(median(reshape(REG( flagship,:,:), [], n_cases, MAXR), [1 2], 'omitnan'));
plot(ax, 1:MAXR, mcheap, '-o', 'Color', [0.35 0.45 0.60], 'LineWidth', 2.2, ...
     'MarkerFaceColor', [0.35 0.45 0.60]);
plot(ax, 1:MAXR, mflag,  '-s', 'Color', [0.75 0.20 0.15], 'LineWidth', 2.2, ...
     'MarkerFaceColor', [0.75 0.20 0.15]);
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on');
title(ax, 'median over all runs, by tier');
xlabel(ax, 'round'); ylabel(ax, 'median regret');
xticks(ax, 1:MAXR); xlim(ax, [0.85 MAXR+0.15]);
legend(ax, {'5 mid-tier models', '3 flagship models'}, 'Location', 'northeast');
lg = legend(subplot(2,3,1), tags, 'Interpreter', 'none', 'FontSize', 7, ...
            'Location', 'northeast'); %#ok<NASGU>
sgtitle('Closed-loop convergence: best-so-far regret by round', 'FontWeight', 'bold');
exportgraphics(fig, fullfile(out_dir, 'agent_convergence.png'), 'Resolution', 220);
close(fig);

% ---- (2) open-loop vs closed-loop ----------------------------------------
r1  = REG(:,:,1);                              % open-loop (no feedback)
rfin = REG(:,:,MAXR);                          % closed-loop final
fig = figure('Position', [60 60 720 560], 'Color', 'w', 'Visible', 'off');
hold on;
for mi = 1:n_models
    scatter(r1(mi,:), rfin(mi,:), 70, cols(mi,:), mk{mi}, 'filled', ...
            'MarkerFaceAlpha', 0.85);
end
lim = max([r1(:); rfin(:)], [], 'omitnan') * 1.08;
plot([0 lim], [0 lim], 'k--', 'LineWidth', 1);
hold off; grid on; box on; axis([0 lim 0 lim]);
xlabel('open-loop regret (round 1 only, no feedback)');
ylabel('closed-loop final regret');
title('Closed-loop feedback vs open-loop selection (each point = one run)');
legend([tags, {'no-improvement line'}], 'Interpreter', 'none', ...
       'Location', 'northwest', 'FontSize', 8);
exportgraphics(fig, fullfile(out_dir, 'agent_openloop_vs_closedloop.png'), ...
               'Resolution', 220);
close(fig);

n_runs    = sum(~isnan(r1(:)));
n_improve = sum(rfin(:) < r1(:) - 1e-9);
fprintf('Open vs closed loop: %d/%d runs improved after round 1.\n', ...
        n_improve, n_runs);
fprintf('  median regret: open-loop %.2f -> closed-loop %.2f\n', ...
        median(r1(:), 'omitnan'), median(rfin(:), 'omitnan'));

% ---- (3) per-model behavior table -----------------------------------------
fprintf('\n%-18s %5s %5s %5s %8s %6s %7s %9s %7s\n', 'Model', 'hit1', ...
        'top3', 'top5', 'medReg', 'evals', 'rounds', 'llm-stop', 'tokens');
md = cell(n_models,1);
for mi = 1:n_models
    rk = NaN(1,n_cases); rg = NaN(1,n_cases); ev = NaN(1,n_cases);
    rd = NaN(1,n_cases); st = 0; tk = NaN(1,n_cases);
    for ci = 1:n_cases
        r = R{mi,ci}; if isempty(r) || ~r.ok, continue; end
        rk(ci) = r.rank; rg(ci) = r.regret; ev(ci) = r.n_evals;
        rd(ci) = r.n_rounds; tk(ci) = r.tokens;
        st = st + strcmp(r.stopped_by, 'llm-stop');
    end
    fprintf('%-18s %5d %5d %5d %8.2f %6.1f %7.1f %6d/%-2d %7.0f\n', tags{mi}, ...
            sum(rk==1), sum(rk<=3), sum(rk<=5), median(rg,'omitnan'), ...
            mean(ev,'omitnan'), mean(rd,'omitnan'), st, n_cases, ...
            mean(tk,'omitnan'));
    md{mi} = sprintf('| %s | %d/5 | %d/5 | %d/5 | %.2f | %.1f | %.1f | %d/5 | %.0fk |', ...
            tags{mi}, sum(rk==1), sum(rk<=3), sum(rk<=5), median(rg,'omitnan'), ...
            mean(ev,'omitnan'), mean(rd,'omitnan'), st, mean(tk,'omitnan')/1e3);
end

fid = fopen(fullfile(out_dir, 'agent_behavior_table.md'), 'w');
fprintf(fid, ['| 模型 | 精确命中 | top-3 | top-5 | regret中位 | 平均拟合次数 | ' ...
              '平均轮数 | 主动收敛 | 平均tokens |\n|---|---|---|---|---|---|---|---|---|\n']);
fprintf(fid, '%s\n', md{:});
fclose(fid);
fprintf('\nSaved -> %s\n', fullfile(out_dir, 'agent_behavior_table.md'));
fprintf('DONE.\n');
