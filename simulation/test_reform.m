%% test_reform.m
%  Validate the PER/RQ simulation-stage realizations against the
%  selection-stage GPML kernels:
%   (P)  PER: per-dim product == iso {covPER,'iso',{covSEiso}}  (exact)
%   (R)  RQ : J-component SE mixture vs covRQard, alpha in {0.5,1,2,5},
%             J in {5,7,10}  (controlled approximation)
%   (C)  composites containing RQ (SE+RQ, RQ*LIN) vs ARD dense
%   (K)  routing table sanity after the update
%
%  Run:  >> test_reform

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
rng(3);

D = 2;
n = 400;
x = [4*rand(n,1), 3*rand(n,1) + 1];      % arbitrary off-origin points

fail = 0;

%% ---------- (P) PER exactness -------------------------------------------
hyp = struct();
hyp.PER = [log(0.7); log(0.5); log(1.3)];
terms = expand_expr_terms('PER', D, hyp);
Kt = kdense(terms, x, x);
Kref = feval(@covPER, 'iso', {@covSEiso}, hyp.PER, x);
err = max(abs(Kt(:) - Kref(:))) / max(abs(Kref(:)));
fprintf('(P) PER product vs iso form        : rel err = %.2e  %s\n', ...
        err, pf(err < 1e-12));
fail = fail + (err >= 1e-12);

%% ---------- (R) RQ mixture accuracy --------------------------------------
fprintf('(R) RQ mixture vs covRQard (max rel err):\n');
fprintf('    %-8s', 'alpha');
for J = [7 10 15], fprintf('  J=%-7d', J); end; fprintf('\n');
worst_def = 0;                                   % default J = 10
for alpha = [0.1 0.5 1 2 5]
    fprintf('    %-8.1f', alpha);
    hyp.RQ = [log(0.8); log(0.6); log(1.1); log(alpha)];
    Kref = feval(@covRQard, hyp.RQ, x);
    for J = [7 10 15]
        hyp.RQ_J = J;
        terms = expand_expr_terms('RQ', D, hyp);
        Kt = kdense(terms, x, x);
        e = max(abs(Kt(:) - Kref(:))) / max(abs(Kref(:)));
        fprintf('  %.2e', e);
        if J == 10, worst_def = max(worst_def, e); end
    end
    fprintf('\n');
end
fprintf('    worst @J=10 (default): %.2e  %s\n', worst_def, pf(worst_def < 0.02));
fail = fail + (worst_def >= 0.02);
hyp = rmfield(hyp, 'RQ_J');

%% ---------- (C) composites containing RQ ---------------------------------
hyp.SE  = [log(0.9); log(0.7); log(0.8)];
hyp.LIN = [log(3); log(2.5)];
hyp.RQ  = [log(0.8); log(0.6); log(1.1); log(1.5)];

terms = expand_expr_terms('SE+RQ', D, hyp);
Kref = feval(@covSum, {{@covSEard},{@covRQard}}, [hyp.SE; hyp.RQ], x);
e1 = max(abs(kdense(terms,x,x) - Kref), [], 'all') / max(abs(Kref(:)));
fprintf('(C) SE+RQ  (K=%2d) vs ARD dense     : rel err = %.2e  %s\n', ...
        numel(terms), e1, pf(e1 < 0.02));
fail = fail + (e1 >= 0.02);

terms = expand_expr_terms('RQ*LIN', D, hyp);
Kref = feval(@covProd, {{@covRQard},{@covLINard}}, [hyp.RQ; hyp.LIN], x);
e2 = max(abs(kdense(terms,x,x) - Kref), [], 'all') / max(abs(Kref(:)));
fprintf('    RQ*LIN (K=%2d) vs ARD dense     : rel err = %.2e  %s\n', ...
        numel(terms), e2, pf(e2 < 0.02));
fail = fail + (e2 >= 0.02);

%% ---------- (K) routing table after update -------------------------------
[exprs, ~] = kernel_grammar();
cnt = struct('fk',0,'mi',0,'ex',0); reform = {};
for i = 1:numel(exprs)
    s = kernel_structure_info(exprs(i), D);
    switch s.route
        case 'full-kron', cnt.fk = cnt.fk + 1;
        case 'mvm-iter',  cnt.mi = cnt.mi + 1;
        case 'exact',     cnt.ex = cnt.ex + 1;
    end
    reform = [reform, s.needs_reform]; %#ok<AGROW>
end
fprintf('(K) routes (D=%d): full-kron %d | mvm-iter %d | exact %d; reform: %s  %s\n', ...
        D, cnt.fk, cnt.mi, cnt.ex, strjoin(unique(reform), ','), ...
        pf(cnt.ex == 0));
fail = fail + (cnt.ex ~= 0);

%% ---------- verdict -------------------------------------------------------
fprintf('\n%s\n', repmat('=',1,60));
if fail == 0, fprintf('ALL CHECKS PASSED.\n');
else,         fprintf('%d CHECK(S) FAILED.\n', fail); end

%% ---------- helpers -------------------------------------------------------
function Kd = kdense(terms, xa, xb)
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

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
