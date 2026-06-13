%% test_kernel_grammar.m
%  Validate the A2 compositional-kernel engine:
%    1. Enumerate the bounded grammar (depth <= 2) and report the space size.
%    2. For D = 2 and D = 3, compile every expression into a GPML covfunc,
%       query its hyperparameter count, and evaluate it on random data to
%       confirm it builds a valid covariance matrix without error.
%
%  Run from the gpml-matlab-v4.2-2018-06-11 folder:  >> test_kernel_grammar

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

fprintf('==============================================================\n');
fprintf('  A2 COMPOSITIONAL KERNEL GRAMMAR -- BUILD & SANITY TEST\n');
fprintf('==============================================================\n\n');

%% 1. Enumerate grammar
[exprs, info] = kernel_grammar();
fprintf('Primitives: %s\n', strjoin(info.primitives, ', '));
fprintf('Grammar size (depth <= 2): %d expressions\n', info.n_total);
fprintf('   base = %d,  sum = %d,  prod = %d\n\n', ...
        info.n_base, info.n_sum, info.n_prod);

fprintf('All kernel expressions:\n');
names = {exprs.name};
for i = 1:numel(names)
    fprintf('  %-10s', names{i});
    if mod(i,6)==0, fprintf('\n'); end
end
fprintf('\n\n');

%% 2. Build + evaluate for D = 2 and D = 3
for D = [2 3]
    fprintf('--------------------------------------------------------------\n');
    fprintf('  D = %d : compile + evaluate every expression\n', D);
    fprintf('--------------------------------------------------------------\n');
    rng(1);
    x = rand(20, D);
    n_ok = 0; n_fail = 0;
    fprintf('%-12s | %-6s | %-8s | status\n', 'expr', 'n_hyp', 'K size');
    fprintf('%s\n', repmat('-', 1, 50));
    for i = 1:numel(exprs)
        e = exprs(i);
        try
            [cov, n_hyp, hyp0] = build_covfunc_from_expr(e, D);
            K = feval(cov{:}, hyp0, x);
            ok = isreal(K) && all(isfinite(K(:))) && isequal(size(K),[20 20]);
            if ok
                n_ok = n_ok + 1; status = 'OK';
            else
                n_fail = n_fail + 1; status = 'BAD-K';
            end
            fprintf('%-12s | %-6d | %-8s | %s\n', e.name, n_hyp, ...
                    sprintf('%dx%d',size(K,1),size(K,2)), status);
        catch ME
            n_fail = n_fail + 1;
            fprintf('%-12s | %-6s | %-8s | ERROR: %s\n', e.name, '-', '-', ME.message);
        end
    end
    fprintf('\n  D=%d summary: %d OK, %d FAIL (of %d)\n\n', D, n_ok, n_fail, numel(exprs));
end

%% 3. Candidate space size vs fixed library (necessity argument)
n_means = 4;   % Zero, Constant, Linear, Quadratic
fprintf('==============================================================\n');
fprintf('  SEARCH SPACE: fixed library vs compositional grammar\n');
fprintf('==============================================================\n');
fprintf('  Fixed (paper):        4 kernels x %d means = %d combinations\n', n_means, 4*n_means);
fprintf('  Compositional (A2):  %d kernels x %d means = %d combinations\n', ...
        info.n_total, n_means, info.n_total*n_means);
fprintf('  -> %.1fx larger; exhaustive multi-restart AIC becomes infeasible.\n', ...
        (info.n_total*n_means)/(4*n_means));

fprintf('\nDONE.\n');
