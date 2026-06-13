function [fmu, F, info] = grid_sim_terms(terms, xg, x, y, xs, opts)
%GRID_SIM_TERMS  Grid-based GP posterior mean + sampling for sum-of-Kronecker
%                (K >= 1 separable terms) covariances. The 'mvm-iter' route.
%
%   [fmu, F, info] = grid_sim_terms(terms, xg, x, y, xs, opts)
%
%   Model (SKI/grid inducing-point):  y = m(X) + W u + eps,
%     u ~ N(0, K_uu),  K_uu = sum_k kron_d K^{(k,d)},  eps ~ N(0, sn^2 I),
%   where W is the sparse cubic-interpolation matrix from the inducing grid
%   to the training points (apxGrid), and queries use f* = m(X*) + W* u.
%
%   Posterior mean:
%     fmu = m(X*) + W* K_uu W' (W K_uu W' + sn^2 I)^{-1} (y - m(X))
%   solved by (P)CG -- only MVMs with K_uu are needed, and an MVM with a sum
%   of Kronecker products is just the sum of per-term Kronecker MVMs.
%
%   Posterior sampling (Matheron / pathwise; EXACT in distribution, no
%   eigendecomposition of the SUM is ever required):
%     u0   ~ N(0, K_uu)   = sum_k (independent per-term Kronecker samples)
%     eps0 ~ N(0, sn^2 I)
%     u|y  = u0 + K_uu W' (W K_uu W' + sn^2 I)^{-1} (ytil - W u0 - eps0)
%     f*   = m(X*) + W* (u|y)
%
%   INPUTS
%     terms : struct array from expand_expr_terms (.f{d}, .h{d})
%     xg    : 1xD cell of equispaced grid coordinate vectors (must cover the
%             ranges of x and xs with some margin)
%     x, y  : training inputs (n x D) and targets (n x 1)
%     xs    : query points (ns x D)
%     opts  : .sn        noise std (required)
%             .mu_train  mean at x   (n x 1, default 0)
%             .mu_test   mean at xs  (ns x 1, default 0)
%             .n_samples number of posterior samples (default 0)
%             .pcg_tol / .pcg_maxit  (default 1e-6 / 1000)
%             .solver    'pcg' (default) or 'chol': form the n x n system
%                        A = W K_uu W' + sn^2 I densely (n MVMs, one-off)
%                        and Cholesky-factorize. Exact, immune to the bad
%                        conditioning of near-interpolation fits (tiny sn),
%                        and fast for n up to a few thousand.
%             .fitc_alpha  ('chol' only, default 0) FITC/SPEP diagonal
%                        correction: Lambda = sn^2 I + alpha*diag(g) with
%                        g = diag(K_ff) - diag(W K_uu W') >= 0, the variance
%                        the interpolation underestimates. Regularizes
%                        near-interpolation fits exactly where the SKI
%                        approximation is weakest (paper's scoeff analogue).
%             .deg       interpolation degree (default 3 = cubic)
%             .seed      rng seed for sampling (default: untouched)
%
%   OUTPUTS
%     fmu   : ns x 1 posterior mean
%     F     : ns x n_samples posterior samples (empty if n_samples = 0)
%     info  : struct with timings, pcg iteration counts, sizes
%
%   See also: expand_expr_terms, kron_mvm, kernel_structure_info

    if ~isfield(opts,'sn'), error('grid_sim_terms:sn','opts.sn required'); end
    n  = size(x,1); ns = size(xs,1);
    D  = numel(xg);
    K  = numel(terms);
    if ~isfield(opts,'mu_train'),  opts.mu_train  = zeros(n,1);  end
    if ~isfield(opts,'mu_test'),   opts.mu_test   = zeros(ns,1); end
    if ~isfield(opts,'n_samples'), opts.n_samples = 0;           end
    if ~isfield(opts,'pcg_tol'),   opts.pcg_tol   = 1e-6;        end
    if ~isfield(opts,'pcg_maxit'), opts.pcg_maxit = 1000;        end
    if ~isfield(opts,'solver'),    opts.solver    = 'pcg';       end
    if ~isfield(opts,'fitc_alpha'),opts.fitc_alpha = 0;          end
    if ~isfield(opts,'deg'),       opts.deg       = 3;           end
    if isfield(opts,'seed') && ~isempty(opts.seed), rng(opts.seed); end
    sn2 = opts.sn^2;

    t0 = tic;
    % --- per-term per-dimension grid factor matrices + symmetric sqrts ---
    Fac = cell(K, D); Sq = cell(K, D);
    for k = 1:K
        for d = 1:D
            Kd = feval(terms(k).f{d}{:}, terms(k).h{d}, xg{d}(:));
            Kd = (Kd + Kd')/2;
            Fac{k,d} = Kd;
            [Q, E] = eig(Kd);
            Sq{k,d} = Q * diag(sqrt(max(diag(E), 0))) * Q';
        end
    end
    m = prod(cellfun(@numel, xg));

    % --- interpolation matrices (sparse) ---
    W  = apxGrid('interp', xg(:)', x,  opts.deg);
    Ws = apxGrid('interp', xg(:)', xs, opts.deg);

    % --- Kronecker ordering of apxGrid's grid expansion ---
    % kron_mvm({A1..AD}) assumes the LAST dim varies fastest; apxGrid may
    % expand with the FIRST dim fastest. Detect and reverse factors if so.
    xe2 = apxGrid('expand', xg(:)');
    if size(xe2,1) > 1 && xe2(1,1) ~= xe2(2,1)
        ord = D:-1:1;          % first dim fastest -> reverse factor order
    else
        ord = 1:D;
    end

    mvmK = @(v) mvm_terms(Fac, ord, v);
    info = struct('m', m, 'K', K, 'n', n, 'ns', ns);

    if strcmp(opts.solver, 'chol')
        % dense n x n system: n Kronecker MVMs once, then exact triangular
        % solves per RHS. Robust to sn -> 0 (near-interpolation fits).
        WKWt = zeros(n, n);
        Wt = W';
        for j = 1:n
            WKWt(:,j) = W * mvmK(Wt(:,j));
        end
        WKWt = (WKWt + WKWt')/2;
        Lam = sn2 * ones(n,1);
        if opts.fitc_alpha > 0
            kff = zeros(n,1);            % exact prior variances at train pts
            for k = 1:K
                pk = ones(n,1);
                for d = 1:D
                    pk = pk .* feval(terms(k).f{d}{:}, terms(k).h{d}, ...
                                     x(:,d), 'diag');
                end
                kff = kff + pk;
            end
            g = max(kff - diag(WKWt), 0);
            Lam = Lam + opts.fitc_alpha * g;
            info.fitc_g_mean = mean(g);
        end
        A = WKWt + diag(Lam);
        [LA, pflag] = chol(A, 'lower');
        if pflag > 0   % tiny sn + interpolation error can leave A indefinite
            jit = 1e-10 * trace(A)/n;
            while pflag > 0
                jit = jit * 10;
                [LA, pflag] = chol(A + jit*eye(n), 'lower');
            end
            warning('grid_sim_terms:jitter', 'chol needed jitter %.1e', jit);
        end
        solveA = @(b) LA' \ (LA \ b);
        sLam = sqrt(Lam);                 % per-point noise std for Matheron
    else
        Afun = @(v) W*(mvmK(W'*v)) + sn2*v;
        solveA = @(b) pcg_solve(Afun, b, opts);
        sLam = opts.sn * ones(n,1);
    end
    info.t_setup = toc(t0);

    % --- posterior mean ---
    t0 = tic;
    ytil = y - opts.mu_train;
    alpha = solveA(ytil);
    fmu = opts.mu_test + Ws * mvmK(W' * alpha);
    info.t_mean = toc(t0); info.pcg_iter_mean = NaN;

    % --- posterior samples (Matheron) ---
    F = zeros(ns, opts.n_samples);
    t0 = tic;
    for j = 1:opts.n_samples
        u0 = zeros(m,1);
        for k = 1:K                              % exact prior sample
            u0 = u0 + kron_mvm(Sq(k, ord), randn(m,1));
        end
        eps0 = sLam .* randn(n,1);
        rhs  = ytil - W*u0 - eps0;
        rr = solveA(rhs);
        u_post = u0 + mvmK(W' * rr);
        F(:,j) = opts.mu_test + Ws * u_post;
    end
    info.t_samples = toc(t0);
    info.pcg_iter_samples = NaN;
end

% --------------------------------------------------------------------------
function w = mvm_terms(Fac, ord, v)
    [K, ~] = size(Fac);
    w = zeros(size(v));
    for k = 1:K
        w = w + kron_mvm(Fac(k, ord), v);
    end
end

% --------------------------------------------------------------------------
function x = pcg_solve(Afun, b, opts)
    [x, flag, relres, ~] = pcg(Afun, b, opts.pcg_tol, opts.pcg_maxit);
    if flag ~= 0
        warning('grid_sim_terms:pcg', 'PCG flag=%d relres=%.2e', flag, relres);
    end
end
