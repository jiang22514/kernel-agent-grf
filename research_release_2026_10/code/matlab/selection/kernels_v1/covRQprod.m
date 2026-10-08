function [K, dK] = covRQprod(hyp, x, z)
%COVRQPROD  Direction-product (separable) rational quadratic kernel.
%           kernels_v1 — GPML v4.2 closure convention.
%
%   k(x,z) = sf^2 * prod_{d=1..D} ( 1 + a_d^2/(2*alpha) )^(-alpha),
%   a_d = |x_d - z_d| / ell_d.
%
%   Hyperparameters: hyp = [ log(ell_1); ...; log(ell_D); log(sf); log(alpha) ]
%   —— 与 covRQard 的参数顺序一致（[ell..D, sf, alpha]），便于参数映射复用。
%
%   Replaces radial covRQard AND the SE-mixture approximation used at the
%   simulation stage: exact and separable, one definition for both stages
%   (revision plan B1-2, kernels_v1).
%
%   Conventions (GPML v4.2):
%     covRQprod()                    -> '(D+2)'   (parameter count)
%     K = covRQprod(hyp,x) / covRQprod(hyp,x,z) / (hyp,x,'diag')
%     [K,dK] = ...                   -> dK(Q) returns d tr(Q'K)/d hyp (column)

if nargin < 2, K = '(D+2)'; return; end          % report number of parameters
if nargin < 3, z = []; end
dg = ischar(z) && strcmp(z,'diag');
if dg, z = []; end

D   = size(x,2);
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));
al  = exp(hyp(D+2));

if dg
    n = size(x,1); K = sf2*ones(n,1);
    if nargout > 1
        dK = @(Q) [zeros(D,1); 2*sum(Q(:)); 0];
    end
    return
end

if isempty(z), z = x; end
n = size(x,1); m = size(z,1);

A2 = zeros(n,m,D);
for d = 1:D
    ad = abs(bsxfun(@minus, x(:,d), z(:,d)')) / ell(d);
    A2(:,:,d) = ad.^2;
end
F  = 1 + A2/(2*al);
KF = F.^(-al);
K  = sf2 * prod(KF, 3);
K  = K(:,:,1);

if nargout > 1
    dK = @(Q) dirder(Q, A2, F, KF, K, sf2, al, D);
end
end

% --------------------------------------------------------------------------
function [dhyp, dx] = dirder(Q, A2, F, KF, K, sf2, al, D)
% dK/d log ell_i = sf2 * a_i^2 * f_i^(-al-1) * prod_{e~=i} f_e^(-al)
% dK/d log sf    = 2K
% dK/d log alpha = K * ( -al * sum_d ln f_d + sum_d a_d^2/(2 f_d) )
    dhyp = zeros(D+2,1);
    for i = 1:D
        dKi = sf2 * (A2(:,:,i) .* F(:,:,i).^(-al-1));
        for e = 1:D
            if e ~= i, dKi = dKi .* KF(:,:,e); end
        end
        dhyp(i) = sum(sum(Q .* dKi));
    end
    dhyp(D+1) = 2*sum(sum(Q .* K));
    S = -al * sum(log(F),3) + sum(A2 ./ (2*F),3);
    dhyp(D+2) = sum(sum(Q .* (K .* S(:,:,1))));
    dx = [];
end
