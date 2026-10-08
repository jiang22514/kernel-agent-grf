function [K, dK] = covMatProdU(nu2, hyp, x, z)
%COVMATPRODU  Unit-amplitude direction-product (separable) Matern kernel.
%             kernels_v1 — GPML v4.2 closure convention.
%
%   k(x,z) = prod_{d=1..D} k_nu(|x_d - z_d| / ell_d),   nu = nu2/2, nu2 in {1,3,5}
%
%   Hyperparameters: hyp = [ log(ell_1); ...; log(ell_D) ]   (unit amplitude)
%   Use with covScale for the signal variance:
%     MA1 = {@covScale, {@covMatProdU, 1}}   % hyp = [log ell_1..D; log sf]
%
%   Replaces radial covMaternard so the SELECTION stage uses exactly the
%   per-dimension product form that the sum-of-Kronecker SIMULATION engine
%   realizes (revision plan B1-1, kernels_v1).
%
%   Conventions (GPML v4.2):
%     covMatProdU(nu2)              -> 'D'      (parameter count)
%     K = covMatProdU(nu2,hyp,x)              -> covariance matrix
%     k = covMatProdU(nu2,hyp,x,'diag')       -> diagonal
%     K = covMatProdU(nu2,hyp,x,z)            -> cross-covariance
%     [K,dK] = ...                  -> dK(Q) returns d tr(Q'K)/d hyp (column)

if nargin < 3, K = 'D'; return; end              % report number of parameters
if nargin < 4, z = []; end
dg = ischar(z) && strcmp(z,'diag');
if dg, z = []; end

D   = size(x,2);
ell = exp(hyp(1:D));

if dg                                            % diagonal: k(x,x) = 1
    n = size(x,1); K = ones(n,1);
    if nargout > 1, dK = @(Q) zeros(D,1); end    % no ell-dependence on diagonal
    return
end

if isempty(z), z = x; end
n = size(x,1); m = size(z,1);

A = zeros(n,m,D);                                % scaled per-dim distances
for d = 1:D
    A(:,:,d) = abs(bsxfun(@minus, x(:,d), z(:,d)')) / ell(d);
end

KF = zeros(n,m,D);                               % per-dim kernel factors
for d = 1:D
    KF(:,:,d) = mat1d(nu2, A(:,:,d));
end
K = prod(KF, 3);
K = K(:,:,1);                                    % n x m

if nargout > 1
    dK = @(Q) dirder(Q, nu2, A, KF, D);
end
end

% --------------------------------------------------------------------------
function [dhyp, dx] = dirder(Q, nu2, A, KF, D)
% dhyp(i) = d tr(Q' K)/ d log ell_i = sum(Q .* dK/d log ell_i)
%   dK/d log ell_i = (-a_i * dk_i/da_i) * prod_{e~=i} k_e
    dhyp = zeros(D,1);
    for i = 1:D
        [~, dk] = mat1d(nu2, A(:,:,i));
        dKi = -A(:,:,i) .* dk;                   % d k_i / d log ell_i
        for e = 1:D
            if e ~= i, dKi = dKi .* KF(:,:,e); end
        end
        dhyp(i) = sum(sum(Q .* dKi));
    end
    dx = [];                                     % not needed on the exact path
end

% --------------------------------------------------------------------------
function [k, dk] = mat1d(nu2, a)
% 1-D unit-amplitude Matern factor k_nu(a) and derivative dk/da.
switch nu2
    case 1
        t  = a;
        k  = exp(-t);
        dk = -k;
    case 3
        t  = sqrt(3)*a;
        e  = exp(-t);
        k  = (1+t).*e;
        dk = -sqrt(3)*t.*e;
    case 5
        t  = sqrt(5)*a;
        e  = exp(-t);
        k  = (1+t+t.^2/3).*e;
        dk = -sqrt(5)*(t+t.^2)/3.*e;
    otherwise
        error('mat1d:badNu','nu2 must be 1, 3 or 5');
end
end
