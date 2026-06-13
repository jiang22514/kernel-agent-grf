%% fix_agent_study.m
%  Clear study cells that terminated on an LLM/network error so that
%  re-running run_agent_study (resume mode) redoes only those runs.
%
%  Run AFTER the main study finishes:  >> fix_agent_study; run_agent_study

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

L = load('agent_study_results.mat');
R = L.R;
n_clear = 0;
for i = 1:numel(R)
    r = R{i};
    if isempty(r), continue; end
    bad = (isfield(r,'ok') && ~r.ok) || ...
          (isfield(r,'stopped_by') && strcmp(r.stopped_by, 'llm-error'));
    if bad
        [mi, ci] = ind2sub(size(R), i);
        fprintf('clearing %s x %s (%s)\n', L.tags{mi}, L.cases(ci).name, ...
                pick(r, 'stopped_by', 'error'));
        R{i} = [];
        n_clear = n_clear + 1;
    end
end
if n_clear > 0
    L.R = R; save('agent_study_results.mat', '-struct', 'L');
    fprintf('%d run(s) cleared. Now execute run_agent_study to redo them.\n', n_clear);
else
    fprintf('No failed runs found.\n');
end

function v = pick(r, f1, f2)
    if isfield(r, f1), v = r.(f1); elseif isfield(r, f2), v = r.(f2);
    else, v = '?'; end
end
