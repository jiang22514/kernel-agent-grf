%% pilot_agent_search.m
%  Pilot run of the closed-loop agent (A1) on ONE synthetic case with the
%  cheapest model, to debug the protocol before the full multi-model study.
%
%  Steps:
%   (0) offline unit test of the proposal parser (no API cost)
%   (1) agent run on EX3 (trigonometric; PER-relevant, cheap n=100 fits)
%   (2) comparison against the exhaustive ground truth
%
%  Run:  >> pilot_agent_search

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

%% ---------- (0) parser unit test (offline) -------------------------------
mock = ['Sure! Here is my answer:' newline '```json' newline ...
    '{"analysis":"test","proposals":[' ...
    '{"kernel":"LIN*MA1","mean":"constant","reason":"r"},' ...   % needs canonicalization
    '{"kernel":"SE + PER","mean":"Quadratic","reason":"r"},' ... % spaces
    '{"kernel":"SE*RQ","mean":"Zero","reason":"r"},' ...         % pruned product
    '{"kernel":"SE+SE","mean":"Zero","reason":"r"},' ...         % self-combo
    '{"kernel":"MA2","mean":"Linear","reason":"r"},' ...         % unknown primitive
    '{"kernel":"RQ","mean":"banana","reason":"r"}' ...           % bad mean
    '],"stop":false}' newline '```'];
[props, ~] = parse_kernel_proposal(mock);
exp_valid = [true true false false false false];
exp_names = {'MA1*LIN','SE+PER','','','','RQ'};  % #6: kernel ok, mean invalid
ok = numel(props) == 6 && isequal([props.valid], exp_valid) && ...
     isequal({props.name}, exp_names);
fprintf('(0) parser unit test: %s\n', ternary(ok, 'PASS', 'FAIL'));
assert(ok, 'parser unit test failed -- fix before spending API budget');

%% ---------- (1) agent run on EX3 ------------------------------------------
% regenerate EXACTLY the same data as enumerate_grammar_aic.m (rng sequence)
rng(1);
sn = 0.1; n_syn = 100;
x1 = rand(n_syn,2); y1 = x1(:,2) + 0.1*x1(:,1) + sn*randn(n_syn,1); %#ok<NASGU>
x2 = rand(n_syn,2); y2 = x2(:,2).^2 + x2(:,1).*x2(:,2) + sn*randn(n_syn,1); %#ok<NASGU>
x3 = rand(n_syn,2); y3 = sin(x3(:,2)) + x3(:,1) + sn*randn(n_syn,1);

opts = struct('model', 'openai/gpt-4o-mini', ...
              'case_label', 'EX3 (2-D synthetic, unknown generator)', ...
              'max_rounds', 4, 'n_propose', 5, ...
              'log_file', 'agent_pilot_ex3_log.txt', 'verbose', true);
fprintf('\n(1) closed-loop agent on EX3 with %s ...\n', opts.model);
out = agent_kernel_search(x3, y3, opts);

%% ---------- (2) compare with exhaustive ground truth ----------------------
G = load('enumerate_grammar_aic_results.mat');   % AIC(case, expr, mean)
ci = 3;                                          % EX3
a = squeeze(G.AIC(ci,:,:));
[gt_aic, gi] = min(a(:)); [ge, gm] = ind2sub(size(a), gi);
n_total = numel(a);
rank_of = @(v) sum(a(:) <= v);

fprintf('\n================ PILOT SUMMARY ================\n');
fprintf('Ground truth  : %-10s + %-9s  AIC=%.2f  (exhaustive, %d candidates)\n', ...
        G.exprs(ge).name, G.mean_names{gm}, gt_aic, n_total);
fprintf('Agent best    : %-10s + %-9s  AIC=%.2f  (rank %d/%d, regret %.2f)\n', ...
        out.best.name, out.best.mean_name, out.best.aic, ...
        rank_of(out.best.aic), n_total, out.best.aic - gt_aic);
fprintf('LLM final pick: %s + %s\n', out.final_llm.name, out.final_llm.mean_name);
fprintf('Cost          : %d GP fits (%.0f%% of exhaustive), %d rounds, %d tokens, stop=%s\n', ...
        out.n_evals, 100*out.n_evals/n_total, numel(out.rounds), ...
        out.tokens, out.stopped_by);
fprintf('Best-AIC trajectory by round: %s\n', mat2str(out.traj, 6));

save('agent_pilot_ex3.mat', 'out');
fprintf('Saved -> agent_pilot_ex3.mat ; transcript -> agent_pilot_ex3_log.txt\n');
fprintf('DONE.\n');

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
