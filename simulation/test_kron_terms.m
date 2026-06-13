%% test_kron_terms.m
%  Validate the sum-of-Kronecker simulation engine pieces:
%   (A) kron_mvm correctness vs explicit kron()
%   (B) term expansion + Kronecker assembly == GPML ARD dense forms (exact
%       for SE / LIN / SE+LIN / SE*LIN), and == independent covMask-based
%       GPML evaluation for product-form Matern / PER composites
%   (C) prior sampling u ~ N(0, K_uu) by summing K independent per-term
%       Kronecker samples: sample covariance must match K_uu
%   (D) MVM speed: kron_mvm vs dense matrix-vector product
%
%  Run:  >> test_kron_terms

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
rng(7);

D  = 2;
m1 = 14; m2 = 11; m = m1*m2;
x1 = linspace(0, 1, m1)';  x2 = linspace(0, 2, m2)';
xs = {x1, x2};
% full grid point list, dim-2 fastest (matches kron(A1,A2) ordering)
xg = [kron(x1, ones(m2,1)), kron(ones(m1,1), x2)];

% shared hyperparameters (natural log)
hyp = struct();
hyp.SE  = [log(0.3); log(0.5); log(1.2)];
hyp.MA3 = [log(0.4); log(0.6); log(0.9)];
hyp.LIN = [log(0.7); log(1.1)];
hyp.PER = [log(0.5); log(0.4); log(0.8)];

fail = 0;

%% ---------- (A) kron_mvm vs explicit kron ----------
A = randn(m1); A = A*A'; B = randn(m2); B = B*B';
v = randn(m,1);
err = norm(kron_mvm({A,B}, v) - kron(A,B)*v) / norm(kron(A,B)*v);
fprintf('(A) kron_mvm vs kron():            rel err = %.2e  %s\n', ...
        err, pf(err < 1e-10)); fail = fail + (err >= 1e-10);

%% ---------- helper: assemble dense K_uu from terms ----------
assemble = @(terms) assemble_terms(terms, xs);

%% ---------- (B1) exact equality vs GPML ARD forms ----------
checks = {
  'SE',     {@covSEard},  hyp.SE
  'LIN',    {@covLINard}, hyp.LIN
  'SE+LIN', {@covSum,  {{@covSEard},{@covLINard}}}, [hyp.SE; hyp.LIN]
  'SE*LIN', {@covProd, {{@covSEard},{@covLINard}}}, [hyp.SE; hyp.LIN]
};
for i = 1:size(checks,1)
    name = checks{i,1};
    terms = expand_expr_terms(name, D, hyp);
    Kkron = assemble(terms);
    Kref  = feval(checks{i,2}{:}, checks{i,3}, xg);
    err = max(abs(Kkron(:) - Kref(:))) / max(abs(Kref(:)));
    fprintf('(B1) %-7s vs ARD dense (K=%d):    rel err = %.2e  %s\n', ...
            name, numel(terms), err, pf(err < 1e-9));
    fail = fail + (err >= 1e-9);
end

%% ---------- (B2) product-form composites vs covMask reference ----------
for nm = {'MA3*LIN', 'SE*PER', 'MA3+PER'}
    name = nm{1};
    terms = expand_expr_terms(name, D, hyp);
    Kkron = assemble(terms);
    % independent reference: same 1-D factors evaluated through covMask on
    % the full point list (exercises GPML's own indexing, not our kron)
    Kref = zeros(m);
    for k = 1:numel(terms)
        cmask = {@covProd, {{@covMask, {[1 0], terms(k).f{1}}}, ...
                            {@covMask, {[0 1], terms(k).f{2}}}}};
        Kref = Kref + feval(cmask{:}, [terms(k).h{1}; terms(k).h{2}], xg);
    end
    err = max(abs(Kkron(:) - Kref(:))) / max(abs(Kref(:)));
    fprintf('(B2) %-7s vs covMask ref (K=%d):  rel err = %.2e  %s\n', ...
            name, numel(terms), err, pf(err < 1e-9));
    fail = fail + (err >= 1e-9);
end

%% ---------- (C) prior sampling: sum of per-term Kronecker samples ----------
for nm = {'SE*LIN', 'SE+MA3'}
    name = nm{1};
    if strcmp(name, 'SE+MA3')
        hyp.SE2 = hyp.SE; % both primitives present in hyp already
    end
    terms = expand_expr_terms(name, D, hyp);
    K = numel(terms);
    Kuu = assemble(terms);
    % per-term symmetric square roots of the 1-D factor matrices
    S = cell(K, D);
    for k = 1:K
        for d = 1:D
            Kd = feval(terms(k).f{d}{:}, terms(k).h{d}, xs{d});
            Kd = (Kd + Kd')/2;
            [Q, E] = eig(Kd);
            S{k,d} = Q * diag(sqrt(max(diag(E), 0))) * Q';
        end
    end
    N = 40000;
    U = zeros(m, N);
    for k = 1:K
        Xi = randn(m, N);
        for j = 1:N
            U(:,j) = U(:,j) + kron_mvm(S(k,:), Xi(:,j));
        end
    end
    Cs = (U * U') / N;
    relF = norm(Cs - Kuu, 'fro') / norm(Kuu, 'fro');
    fprintf('(C)  %-7s sampling (K=%d, N=%d): rel Frob err = %.3f  %s\n', ...
            name, K, N, relF, pf(relF < 0.06));
    fail = fail + (relF >= 0.06);
end

%% ---------- (D) MVM speed: kron_mvm vs dense ----------
mb1 = 64; mb2 = 64; mb = mb1*mb2;
xb = {linspace(0,1,mb1)', linspace(0,2,mb2)'};
terms = expand_expr_terms('SE*LIN', D, hyp);
K = numel(terms);
Fac = cell(K, D); Kdense = zeros(mb);
for k = 1:K
    for d = 1:D
        Fac{k,d} = feval(terms(k).f{d}{:}, terms(k).h{d}, xb{d});
    end
    Kdense = Kdense + kron(Fac{k,1}, Fac{k,2});
end
vb = randn(mb, 1); reps = 200;
t0 = tic; for r = 1:reps, w1 = Kdense * vb; end; t_dense = toc(t0)/reps;
t0 = tic;
for r = 1:reps
    w2 = zeros(mb,1);
    for k = 1:K, w2 = w2 + kron_mvm(Fac(k,:), vb); end
end
t_kron = toc(t0)/reps;
err = norm(w1 - w2)/norm(w1);
fprintf('(D)  MVM m=%d, K=%d: dense %.3f ms | kron %.3f ms (%.0fx)  agree %.1e  %s\n', ...
        mb, K, t_dense*1e3, t_kron*1e3, t_dense/t_kron, err, pf(err < 1e-10));
fail = fail + (err >= 1e-10);

%% ---------- verdict ----------
fprintf('\n%s\n', repmat('=',1,60));
if fail == 0
    fprintf('ALL CHECKS PASSED.\n');
else
    fprintf('%d CHECK(S) FAILED.\n', fail);
end

%% ---------- local functions ----------
function K = assemble_terms(terms, xs)
    D = numel(xs);
    m = prod(cellfun(@numel, xs));
    K = zeros(m);
    for k = 1:numel(terms)
        Kk = 1;
        for d = 1:D
            Kd = feval(terms(k).f{d}{:}, terms(k).h{d}, xs{d});
            Kk = kron(Kk, Kd);
        end
        K = K + Kk;
    end
end

function s = pf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end
