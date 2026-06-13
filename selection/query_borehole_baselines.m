%% query_borehole_baselines.m  -- best single kernel & key rows from the
%  full-n borehole enumeration (narrative baselines for the simulation figs)
clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
G = load('enumerate_borehole_fulln_results.mat');
a = G.AIC; b = G.BIC; en = {G.exprs.name}; mn = G.mean_names;
base = strcmp({G.exprs.op}, 'base');

fprintf('--- all single-primitive rows (AIC by mean) ---\n');
fprintf('%-6s', 'expr'); fprintf('%12s', mn{:}); fprintf('\n');
for i = find(base)
    fprintf('%-6s', en{i}); fprintf('%12.2f', squeeze(a(i,:))); fprintf('\n');
end

aa = a; aa(~base, :) = inf;
[v, ix] = min(aa(:)); [ei, mi] = ind2sub(size(aa), ix);
fprintf('\nBest single kernel : %s + %s  AIC=%.2f\n', en{ei}, mn{mi}, v);
[v2, ix2] = min(a(:)); [e2, m2] = ind2sub(size(a), ix2);
fprintf('Grammar champion   : %s + %s  AIC=%.2f\n', en{e2}, mn{m2}, v2);
fprintf('SE + Quadratic     : AIC=%.2f\n', a(strcmp(en,'SE'), 4));
fprintf('Overall rank of best single: %d/156\n', sum(a(:) <= v));
fprintf('DONE.\n');
