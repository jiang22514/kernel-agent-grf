%% test_cubic_mean.m
%  Decision experiment (SELF-CONTAINED): does adding a CUBIC (degree-3
%  polynomial) mean ever improve AIC over the 4-mean library
%  {Zero,Constant,Linear,Quadratic}?
%
%  To avoid any cross-run inconsistency, this script recomputes BOTH the
%  <=Quadratic baseline AND the Cubic candidates in the SAME run, on the SAME
%  data, with the SAME settings. For each case it enumerates all 39 kernel
%  expressions x means {Zero..Cubic} and reports whether the global-best AIC
%  uses the Cubic mean.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

N_RESTART = 3;        % screening-grade restarts (yes/no question; ranking is stable)
N_SUB     = 150;      % borehole subsample for tractable exact GP
eval_opts = struct('n_restart', N_RESTART, 'n_iter', -100, 'quiet', true);
mean_names = {'Zero','Constant','Linear','Quadratic','Cubic'};

[exprs, ~] = kernel_grammar(); n_expr = numel(exprs);

% datasets (same generators as the main enumeration)
rng(1); sn = 0.1; n_syn = 100;
x1 = rand(n_syn,2); y1 = x1(:,2) + 0.1*x1(:,1) + sn*randn(n_syn,1);
x2 = rand(n_syn,2); y2 = x2(:,2).^2 + x2(:,1).*x2(:,2) + sn*randn(n_syn,1);
x3 = rand(n_syn,2); y3 = sin(x3(:,2)) + x3(:,1) + sn*randn(n_syn,1);
x4 = rand(n_syn,3); y4 = x4(:,1).*x4(:,2) + x4(:,2).^2 + x4(:,3).^2 + sn*randn(n_syn,1);
S = load('samedata.mat'); x5 = S.x_same; y5 = S.y_same;
Xs = {x1,x2,x3,x4,x5}; Ys = {y1,y2,y3,y4,y5};
case_names = {'EX1 (linear)','EX2 (polynomial)','EX3 (trigonometric)', ...
              'EX4 (3D nonlinear)','Borehole (CPT)'};
n_cases = 5;

fprintf('==============================================================\n');
fprintf('  CUBIC-MEAN VERIFICATION (self-contained, %d restarts)\n', N_RESTART);
fprintf('  39 kernels x means {Zero..Cubic}, same data for both\n');
fprintf('==============================================================\n');
fprintf('%-22s | %-22s | %-22s | Cubic better?\n', ...
        'Case', 'best (<=Quadratic)', 'best (Cubic)');
fprintf('%s\n', repmat('-',1,92));

any_win = false;
for c = 1:n_cases
    x = Xs{c}; y = Ys{c}; n_orig = size(x,1);
    if n_orig > N_SUB, rng(12345); sel = randperm(n_orig, N_SUB); x = x(sel,:); y = y(sel); end

    best_qd = inf; kq = '';     % best over means 1..4
    best_cb = inf; kc = '';     % best with mean 5 (Cubic)
    for ei = 1:n_expr
        for mi = 1:5
            r = evaluate_expr_aic(exprs(ei), mi, x, y, eval_opts);
            if ~r.ok, continue; end
            if mi <= 4 && r.aic < best_qd, best_qd = r.aic; kq = exprs(ei).name; end
            if mi == 5 && r.aic < best_cb, best_cb = r.aic; kc = exprs(ei).name; end
        end
    end
    wins = best_cb < best_qd - 1e-6;
    any_win = any_win || wins;
    fprintf('%-22s | %-13s %-8.2f | %-13s %-8.2f | %s (d=%.2f)\n', ...
        case_names{c}, [kq '+Quad?'], best_qd, [kc '+Cubic'], best_cb, ...
        tern(wins,'YES','no'), best_cb - best_qd);
end

fprintf('%s\n', repmat('-',1,92));
if any_win
    fprintf('\n>>> Cubic improves AIC in >=1 case -> consider adding Cubic to the library.\n');
else
    fprintf('\n>>> Cubic NEVER beats the <=Quadratic best -> KEEP 4 means.\n');
    fprintf('    Reportable: extending the polynomial trend to cubic gives no AIC benefit;\n');
    fprintf('    the trend hierarchy {Zero,Constant,Linear,Quadratic} is sufficient.\n');
end
fprintf('DONE.\n');

function s = tern(c,a,b), if c, s=a; else, s=b; end, end
