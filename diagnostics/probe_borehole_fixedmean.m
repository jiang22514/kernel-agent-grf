%% probe_borehole_fixedmean.m
%  After the mean fix (intercept + LS init), verify our exact-GP pipeline
%  reproduces the paper's borehole Table 2 Gaussian column on RAW coordinates,
%  then re-check the key composite competitors.
%
%  Run:  >> probe_borehole_fixedmean

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

S = load('samedata.mat'); x = S.x_same; y = S.y_same;
opts = struct('n_restart', 5, 'n_iter', -100, 'quiet', true);
paper = [1026.57 666.28 653.14 547.20];
mean_names = {'Zero','Constant','Linear','Quadratic'};

fprintf('--- SE, raw coords, fixed means (paper Gaussian col in brackets) ---\n');
for mi = 1:4
    r = evaluate_expr_aic('SE', mi, x, y, opts);
    fprintf('  SE + %-9s : AIC=%8.2f  k=%d  [paper %8.2f]\n', ...
            mean_names{mi}, r.aic, r.k, paper(mi));
    if mi == 4 && r.ok
        fprintf('    l1=%.2f m  l2=%.2f m  [paper 74.43 / 1.30]\n', ...
                exp(r.hyp.cov(1)), exp(r.hyp.cov(2)));
    end
end

%% --- exact-GP nlZ at the PAPER's published SE+Quadratic hyperparameters ---
% Settles whether the paper's 547.20 is reachable under the exact marginal
% likelihood or is a grid/SPEP-approximation artifact.
meanfunc = {@meanSum, {{@meanConst}, {@meanPoly, 2}}};
hypP = struct();
hypP.mean = [0.0092; -0.0553; 0.0032; 0.0001; 0.0377];  % from Train_Simulation_same_Test.m
hypP.cov  = log([74.43; 1.30; 5.32]);
hypP.lik  = log(0.48);
nlZP = gp(hypP, @infGaussLik, meanfunc, {@covSEard}, {@likGauss}, x, y);
kP = 9;
fprintf('\nExact nlZ at paper hyps (SE+Quad): nlZ=%.2f -> AIC=%.2f [paper grid AIC 547.20 => nlZ %.2f]\n', ...
        nlZP, 2*kP + 2*nlZP, (547.20 - 2*kP)/2);
% and refine from there
[~, hypP2] = evalc(['minimize(hypP, @gp, -200, @infGaussLik, ' ...
                    'meanfunc, {@covSEard}, {@likGauss}, x, y)']);
nlZP2 = gp(hypP2, @infGaussLik, meanfunc, {@covSEard}, {@likGauss}, x, y);
fprintf('  ...after exact-GP refinement from paper hyps: nlZ=%.2f -> AIC=%.2f (l1=%.2f, l2=%.2f m)\n', ...
        nlZP2, 2*kP + 2*nlZP2, exp(hypP2.cov(1)), exp(hypP2.cov(2)));

fprintf('\n--- key competitors, raw coords, fixed means ---\n');
combos = {'MA1',2; 'RQ',2; 'SE+MA5',2; 'MA1*LIN',2; 'RQ*LIN',2; 'MA1*LIN',4; 'SE*LIN',4};
for i = 1:size(combos,1)
    r = evaluate_expr_aic(combos{i,1}, combos{i,2}, x, y, opts);
    fprintf('  %-8s + %-9s : AIC=%8.2f BIC=%8.2f\n', ...
            combos{i,1}, mean_names{combos{i,2}}, r.aic, r.bic);
end
fprintf('DONE.\n');
