%% test_grid_sim_terms.m
%  Validate grid_sim_terms (mvm-iter route for K>=2 composite kernels)
%  against the EXACT GP posterior, using the borehole champion structure
%  MA1*LIN + Constant on a synthetic problem of known truth.
%
%  Checks:
%   (1) posterior mean: grid vs exact GP
%   (2) posterior marginal std: sampled vs exact
%   (3) posterior covariance: sample cov vs exact (Frobenius)
%   (4) scalability timing at a large grid/query size
%
%  Run:  >> test_grid_sim_terms

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
rng(42);

%% ---------------- problem setup (mimics scaled borehole frame) ----------
D = 2;
hyp = struct();
hyp.MA1 = [log(0.8); log(0.6); log(0.7)];   % [log l1, log l2, log sf]
hyp.LIN = [log(5);   log(4)];               % LIN lengthscales
terms = expand_expr_terms('MA1*LIN', D, hyp);
K = numel(terms);

c0 = 5.0;                 % constant mean
sn = 0.30;                % noise std
n  = 300;
lo = [3.5 1.0]; hi = [6.5 5.5];
x  = lo + rand(n,D) .* (hi - lo);

% ground-truth latent from the EXACT kernel
kfun = @(xa,xb) kernel_terms_dense(terms, xa, xb);
Kxx = kfun(x,x);
L = chol(Kxx + 1e-10*eye(n), 'lower');
f_true = L * randn(n,1);
y = c0 + f_true + sn*randn(n,1);

% query: 16x16 sub-grid (exact covariance comparison stays cheap)
[q1, q2] = meshgrid(linspace(lo(1)+.1, hi(1)-.1, 16), ...
                    linspace(lo(2)+.1, hi(2)-.1, 16));
xs = [q1(:), q2(:)]; nsq = size(xs,1);

%% ---------------- exact GP posterior ------------------------------------
Ky   = Kxx + sn^2*eye(n);
Ksx  = kfun(x, xs);
Kss  = kfun(xs, xs);
a_e  = Ky \ (y - c0);
mu_e = c0 + Ksx' * a_e;
Sig_e = Kss - Ksx' * (Ky \ Ksx);
std_e = sqrt(max(diag(Sig_e), 0));

%% ---------------- grid route --------------------------------------------
% MA1 (exponential kernel) has a kink at r=0, so SKI interpolation error
% decays only ~linearly with grid spacing: use a dense grid. Empirical rule
% (probe_ski_accuracy.m): lengthscale/spacing >= ~19 gives ~2% mean error
% and ~4-5% std error. Kronecker structure keeps dense grids cheap.
ng = 160;
xg = {linspace(lo(1)-.3, hi(1)+.3, ng)', linspace(lo(2)-.3, hi(2)+.3, ng)'};
NS = 4000;
opts = struct('sn', sn, 'mu_train', c0*ones(n,1), 'mu_test', c0*ones(nsq,1), ...
              'n_samples', NS, 'seed', 1);
[fmu, F, info] = grid_sim_terms(terms, xg, x, y, xs, opts);

%% ---------------- checks ------------------------------------------------
fail = 0;

% (1) posterior mean (error relative to the signal variation around c0)
e_mu = norm(fmu - mu_e) / norm(mu_e - c0);
fprintf('(1) posterior mean rel err          : %.4f  %s\n', e_mu, pf(e_mu < 0.04));
fail = fail + (e_mu >= 0.04);

% sample-based posterior mean must agree with fmu (sanity of sampler)
e_sm = norm(mean(F,2) - fmu) / norm(fmu - c0);
fprintf('    sample-mean vs fmu (stat noise) : %.4f  %s\n', e_sm, pf(e_sm < 0.05));
fail = fail + (e_sm >= 0.05);

% (2) marginal std
std_g = std(F, 0, 2);
e_sd = mean(abs(std_g - std_e)) / mean(std_e);
fprintf('(2) marginal std mean rel err       : %.4f  %s\n', e_sd, pf(e_sd < 0.08));
fail = fail + (e_sd >= 0.08);

% (3) posterior covariance (Frobenius), judged against the statistical
%     noise floor of NS exact posterior samples (self-calibrating)
Cs = cov(F');
e_cv = norm(Cs - Sig_e, 'fro') / norm(Sig_e, 'fro');
Lp = chol(Sig_e + 1e-10*eye(nsq), 'lower');
Fe = mu_e + Lp * randn(nsq, NS);                  % exact posterior samples
floor_cv = norm(cov(Fe') - Sig_e, 'fro') / norm(Sig_e, 'fro');
tol_cv = 1.5*floor_cv + 0.03;
fprintf('(3) posterior cov rel Frob err      : %.4f  (stat floor %.4f, tol %.4f)  %s\n', ...
        e_cv, floor_cv, tol_cv, pf(e_cv < tol_cv));
fail = fail + (e_cv >= tol_cv);

fprintf('    [grid m=%d, K=%d | mean PCG %d iters | %.0f samples, avg PCG %.0f iters]\n', ...
        info.m, K, info.pcg_iter_mean, NS, info.pcg_iter_samples);
fprintf('    [t_setup=%.2fs t_mean=%.2fs t_samples=%.1fs (%.1f ms/sample)]\n', ...
        info.t_setup, info.t_mean, info.t_samples, 1e3*info.t_samples/NS);

%% ---------------- (4) scalability probe ---------------------------------
ng2 = 100;                                   % m = 10,000 inducing points
xg2 = {linspace(lo(1)-.3, hi(1)+.3, ng2)', linspace(lo(2)-.3, hi(2)+.3, ng2)'};
[qq1, qq2] = meshgrid(linspace(lo(1), hi(1), 200), linspace(lo(2), hi(2), 200));
xs2 = [qq1(:), qq2(:)];                      % 40,000 query points
opts2 = struct('sn', sn, 'mu_train', c0*ones(n,1), ...
               'mu_test', c0*ones(size(xs2,1),1), 'n_samples', 5, 'seed', 2);
[~, ~, info2] = grid_sim_terms(terms, xg2, x, y, xs2, opts2);
fprintf('(4) scale probe: m=%d grid, %d query pts -> setup %.1fs, mean %.2fs, %.2f s/sample\n', ...
        info2.m, size(xs2,1), info2.t_setup, info2.t_mean, info2.t_samples/5);

%% ---------------- verdict -----------------------------------------------
fprintf('\n%s\n', repmat('=',1,60));
if fail == 0, fprintf('ALL CHECKS PASSED.\n');
else,         fprintf('%d CHECK(S) FAILED.\n', fail); end

%% ---------------- helpers -----------------------------------------------
function Kd = kernel_terms_dense(terms, xa, xb)
% dense kernel matrix of a sum-of-separable-terms kernel at arbitrary points
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
