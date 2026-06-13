%% test_kernel_structure.m
%  Routing table: classify all 39 grammar expressions by simulation route
%  (full-kron / mvm-iter / exact) for D = 2 and D = 3.
%
%  Run:  >> test_kernel_structure

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

[exprs, ginfo] = kernel_grammar();
n_expr = numel(exprs);

for D = [2 3]
    fprintf('=============================================================\n');
    fprintf('  SIMULATION ROUTING TABLE  (D = %d, k_max = 10, RQ ~ 3 SE)\n', D);
    fprintf('=============================================================\n');
    fprintf('%-14s %4s  %-10s  %-14s  %s\n', 'expr', 'K', 'route', 'reform', 'expansion');
    cnt = struct('full_kron',0,'mvm_iter',0,'exact',0);
    reform_needed = {};
    for i = 1:n_expr
        s = kernel_structure_info(exprs(i), D);
        rf = strjoin(s.needs_reform, ',');
        if isempty(rf), rf = '-'; end
        fprintf('%-14s %4d  %-10s  %-14s  %s\n', s.name, s.K, s.route, rf, s.detail);
        switch s.route
            case 'full-kron', cnt.full_kron = cnt.full_kron + 1;
            case 'mvm-iter',  cnt.mvm_iter  = cnt.mvm_iter  + 1;
            case 'exact',     cnt.exact     = cnt.exact     + 1;
        end
        reform_needed = [reform_needed, s.needs_reform]; %#ok<AGROW>
    end
    fprintf('-------------------------------------------------------------\n');
    fprintf('full-kron: %d | mvm-iter: %d | exact: %d  (of %d)\n', ...
            cnt.full_kron, cnt.mvm_iter, cnt.exact, n_expr);
    fprintf('primitives needing reformed implementation: %s\n\n', ...
            strjoin(unique(reform_needed), ', '));
end
fprintf('DONE.\n');
