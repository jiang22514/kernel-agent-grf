%% probe_ski_accuracy.m
%  Isolate the source of grid_sim_terms vs exact-GP error:
%  kernel smoothness (SE vs MA1) x grid density (40 vs 80 vs 160 per dim).
%
%  Run:  >> probe_ski_accuracy

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
rng(42);

D = 2; c0 = 5.0; sn = 0.30; n = 300;
lo = [3.5 1.0]; hi = [6.5 5.5];
x  = lo + rand(n,D) .* (hi - lo);

hyp = struct();
hyp.SE  = [log(0.8); log(0.6); log(0.7)];
hyp.MA1 = [log(0.8); log(0.6); log(0.7)];
hyp.MA3 = [log(0.8); log(0.6); log(0.7)];
hyp.LIN = [log(5);   log(4)];

[q1, q2] = meshgrid(linspace(lo(1)+.1, hi(1)-.1, 14), ...
                    linspace(lo(2)+.1, hi(2)-.1, 14));
xs = [q1(:), q2(:)]; nsq = size(xs,1);

kernels = {'SE', 'MA3', 'MA1', 'MA1*LIN'};
grids   = [40, 80, 160];

for ki = 1:numel(kernels)
    kn = kernels{ki};
    terms = expand_expr_terms(kn, D, hyp);
    kfun = @(xa,xb) kdense(terms, xa, xb);

    Kxx = kfun(x,x);
    L = chol(Kxx + 1e-10*eye(n), 'lower');
    rng(7); f_true = L*randn(n,1); y = c0 + f_true + sn*randn(n,1);

    Ky = Kxx + sn^2*eye(n);
    Ksx = kfun(x, xs); Kss = kfun(xs, xs);
    mu_e = c0 + Ksx' * (Ky \ (y - c0));
    Sig_e = Kss - Ksx' * (Ky \ Ksx);
    std_e = sqrt(max(diag(Sig_e),0));

    fprintf('--- %s ---\n', kn);
    for ng = grids
        xg = {linspace(lo(1)-.3, hi(1)+.3, ng)', linspace(lo(2)-.3, hi(2)+.3, ng)'};
        opts = struct('sn', sn, 'mu_train', c0*ones(n,1), ...
                      'mu_test', c0*ones(nsq,1), 'n_samples', 3000, 'seed', 1);
        [fmu, F, info] = grid_sim_terms(terms, xg, x, y, xs, opts);
        e_mu = norm(fmu - mu_e)/norm(mu_e - c0);
        e_sd = mean(abs(std(F,0,2) - std_e))/mean(std_e);
        e_cv = norm(cov(F') - Sig_e,'fro')/norm(Sig_e,'fro');
        fprintf('  ng=%3d (h=%.3f, l2/h=%4.1f): mean %.4f | std %.4f | cov %.4f\n', ...
                ng, (hi(2)-lo(2)+.6)/(ng-1), 0.6/((hi(2)-lo(2)+.6)/(ng-1)), ...
                e_mu, e_sd, e_cv);
    end
end
fprintf('DONE.\n');

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
