function [K, dK] = covSEu(hyp, x, z)
%COVSEU  Unit-amplitude ARD squared-exponential kernel (kernels_v1).
%         k(x,z) = prod_d exp(-a_d^2/2),  a_d = |x_d - z_d|/ell_d.
%
%   Hyperparameters: hyp = [ log(ell_1); ...; log(ell_D) ]  (NO sf).
%   Used INSIDE products under the one-amplitude-per-chain rule
%   (param_v1, 无冗余参数化方案): SE x LIN = covSEu x covLINard.
%   (GPML ships covSEisoU but no ARD unit variant; covSEard carries sf and
%   would reintroduce the sf<->beta redundancy of revision plan B1-3/P03.)
%
%   Conventions (GPML v4.2 closure, same as covMatProdU/covRQprodU):
%     covSEu()                      -> 'D'      (parameter count)
%     K = covSEu(hyp,x) / covSEu(hyp,x,z) / covSEu(hyp,x,'diag')
%     [K,dK] = ...                  -> dK(Q) returns d tr(Q'K)/d hyp (column)

if nargin < 2, K = 'D'; return; end              % report number of parameters
if nargin < 3, z = []; end
dg = ischar(z) && strcmp(z,'diag');
if dg, z = []; end

D   = size(x,2);
ell = exp(hyp(1:D));

if dg                                            % diagonal: k(x,x) = 1
    n = size(x,1); K = ones(n,1);
    if nargout > 1, dK = @(Q) zeros(D,1); end
    return
end

if isempty(z), z = x; end
n = size(x,1); m = size(z,1);

A2 = zeros(n,m,D);                               % squared scaled distances
for d = 1:D
    ad = abs(bsxfun(@minus, x(:,d), z(:,d)')) / ell(d);
    A2(:,:,d) = ad.^2;
end
KF = exp(-0.5*A2);                               % per-dim kernel factors
K  = prod(KF, 3);
K  = K(:,:,1);                                   % n x m

if nargout > 1
    dK = @(Q) dirder(Q, A2, K, D);
end
end

% --------------------------------------------------------------------------
function [dhyp, dx] = dirder(Q, A2, K, D)
% dK/d log ell_i = K .* a_i^2
%   (d exp(-a^2/2)/d log ell = a^2 * exp(-a^2/2); other factors cancel)
    dhyp = zeros(D,1);
    for i = 1:D
        dKi = K .* A2(:,:,i);
        dhyp(i) = sum(sum(Q .* dKi));
    end
    dx = [];                                     % not needed on the exact path
end
