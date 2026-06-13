%% run_agent_study.m
%  Full closed-loop agent study (A1): every model in the panel x every case.
%
%  For each (model, case): run agent_kernel_search, then score against the
%  exhaustive ground truth:
%    - regret      : AIC(best found) - AIC(ground-truth optimum)
%    - rank        : position of the agent's best among all 156 candidates
%    - final_*     : same metrics for the LLM's own final recommendation
%    - n_evals     : GP fits spent (vs 156 exhaustive)
%    - tokens, wall time, stop reason
%
%  Cases use EXACTLY the same data and eval settings as the ground-truth
%  enumerations (synthetic: enumerate_grammar_aic_results.mat, n=100;
%  borehole: enumerate_borehole_fulln_results.mat, full n=744), so AIC values
%  are directly comparable.
%
%  Results saved incrementally to agent_study_results.mat; transcripts to
%  agent_log_<case>_<model>.txt.
%
%  Run:  >> run_agent_study

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

%% -------- data (identical generators to enumerate_grammar_aic) ----------
rng(1);
sn = 0.1; n_syn = 100;
x1 = rand(n_syn,2); y1 = x1(:,2) + 0.1*x1(:,1) + sn*randn(n_syn,1);
x2 = rand(n_syn,2); y2 = x2(:,2).^2 + x2(:,1).*x2(:,2) + sn*randn(n_syn,1);
x3 = rand(n_syn,2); y3 = sin(x3(:,2)) + x3(:,1) + sn*randn(n_syn,1);
x4 = rand(n_syn,3); y4 = x4(:,1).*x4(:,2) + x4(:,2).^2 + x4(:,3).^2 + sn*randn(n_syn,1);
S = load('samedata.mat'); x5 = S.x_same; y5 = S.y_same;   % full n=744

cases = struct( ...
    'name',  {'EX1','EX2','EX3','EX4','Borehole'}, ...
    'label', {'EX1 (2-D synthetic, unknown generator)', ...
              'EX2 (2-D synthetic, unknown generator)', ...
              'EX3 (2-D synthetic, unknown generator)', ...
              'EX4 (3-D synthetic, unknown generator)', ...
              'Borehole CPT cone resistance over a 2-D vertical cross-section (x1 horizontal distance [m], x2 depth [m])'}, ...
    'x', {x1,x2,x3,x4,x5}, 'y', {y1,y2,y3,y4,y5});
n_cases = numel(cases);

%% -------- ground truth tables --------------------------------------------
Gs = load('enumerate_grammar_aic_results.mat');      % synthetic (cases 1-4)
Gb = load('enumerate_borehole_fulln_results.mat');   % borehole full-n
GT = cell(n_cases,1);
for c = 1:4, GT{c} = squeeze(Gs.AIC(c,:,:)); end
GT{5} = Gb.AIC;                                       % (expr, mean)
expr_names = {Gs.exprs.name};
mean_names = Gs.mean_names;

%% -------- panel -----------------------------------------------------------
cfg = llm_api_config();
models = cfg.models; tags = cfg.tags;
n_models = numel(models);

agent_base = struct('max_rounds', 4, 'n_propose', 5, 'verbose', true);

%% -------- run --------------------------------------------------------------
R = cell(n_models, n_cases);
if exist('agent_study_results.mat', 'file')
    old = load('agent_study_results.mat');
    if isfield(old, 'R')
        % map old runs into the (possibly extended) panel by model tag
        for omi = 1:numel(old.tags)
            nmi = find(strcmp(tags, old.tags{omi}), 1);
            if isempty(nmi), continue; end
            for ci = 1:min(size(old.R,2), n_cases)
                R{nmi, ci} = old.R{omi, ci};
            end
        end
        fprintf('Resuming: %d/%d runs already done.\n', ...
                sum(~cellfun(@isempty, R(:))), numel(R));
    end
end

grand_t0 = tic;
for mi = 1:n_models
    for ci = 1:n_cases
        if ~isempty(R{mi,ci}), continue; end
        fprintf('\n===== %s  x  %s =====\n', tags{mi}, cases(ci).name);
        o = agent_base;
        o.model = models{mi};
        o.case_label = cases(ci).label;
        o.log_file = sprintf('agent_log_%s_%s.txt', cases(ci).name, ...
                             strrep(tags{mi}, '.', '_'));
        t0 = tic;
        try
            out = agent_kernel_search(cases(ci).x, cases(ci).y, o);
        catch ME
            warning('study:run', '%s x %s failed: %s', tags{mi}, ...
                    cases(ci).name, ME.message);
            R{mi,ci} = struct('ok', false, 'error', ME.message);
            save('agent_study_results.mat', 'R', 'models', 'tags', 'cases', ...
                 'expr_names', 'mean_names');
            continue
        end
        wall = toc(t0);

        % ---- score against ground truth ----
        a = GT{ci};
        gt_aic = min(a(:)); n_total = numel(a);
        best_rank = sum(a(:) <= out.best.aic);
        % LLM's own final recommendation (look up in GT table; may be empty)
        [fa, fr] = lookup_final(out.final_llm, a, expr_names, mean_names);

        rec = struct('ok', true, 'model', tags{mi}, 'case', cases(ci).name, ...
            'best_expr', out.best.name, 'best_mean', out.best.mean_name, ...
            'best_aic', out.best.aic, 'gt_aic', gt_aic, ...
            'regret', out.best.aic - gt_aic, 'rank', best_rank, ...
            'n_total', n_total, ...
            'final_expr', out.final_llm.name, 'final_mean', out.final_llm.mean_name, ...
            'final_aic', fa, 'final_rank', fr, ...
            'n_evals', out.n_evals, 'n_rounds', numel(out.rounds), ...
            'tokens', out.tokens, 'stopped_by', out.stopped_by, ...
            'wall_s', wall, 'traj', out.traj, 'detail', out);
        R{mi,ci} = rec;
        fprintf('-> best %s+%s regret %.2f (rank %d/%d), %d evals, %.0fs\n', ...
                rec.best_expr, rec.best_mean, rec.regret, rec.rank, ...
                rec.n_total, rec.n_evals, wall);
        save('agent_study_results.mat', 'R', 'models', 'tags', 'cases', ...
             'expr_names', 'mean_names');
    end
end
fprintf('\nTotal study wall-clock: %.1f min\n', toc(grand_t0)/60);

%% -------- summary table ----------------------------------------------------
fprintf('\n==================== AGENT STUDY SUMMARY ====================\n');
fprintf('%-18s %-9s %-22s %7s %6s %6s %7s\n', 'Model', 'Case', ...
        'Best found', 'regret', 'rank', 'evals', 'tokens');
fprintf('%s\n', repmat('-', 1, 84));
for mi = 1:n_models
    for ci = 1:n_cases
        r = R{mi,ci};
        if isempty(r) || ~r.ok, continue; end
        fprintf('%-18s %-9s %-22s %7.2f %3d/%-3d %6d %7d\n', r.model, r.case, ...
                [r.best_expr '+' r.best_mean], r.regret, r.rank, ...
                r.n_total, r.n_evals, r.tokens);
    end
end
fprintf('\nSaved -> agent_study_results.mat\nDONE.\n');

%% -------- helpers -----------------------------------------------------------
function [fa, fr] = lookup_final(final_llm, a, expr_names, mean_names)
    fa = NaN; fr = NaN;
    if isempty(final_llm.name), return; end
    [p, ~] = parse_kernel_proposal(jsonencode(struct('proposals', ...
        {{struct('kernel', final_llm.name, 'mean', final_llm.mean_name)}})));
    if isempty(p) || ~p(1).valid, return; end
    ei = find(strcmp(expr_names, p(1).name), 1);
    mi = find(strcmp(mean_names, p(1).mean_name), 1);
    if isempty(ei) || isempty(mi), return; end
    fa = a(ei, mi);
    fr = sum(a(:) <= fa);
end
