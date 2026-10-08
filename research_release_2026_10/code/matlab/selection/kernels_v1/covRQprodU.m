function [K, dK] = covRQprodU(hyp, x, z)
%COVRQPRODU  Unit-amplitude direction-product rational quadratic (kernels_v1).
%   Companion of covRQprod for use INSIDE products (product rule: one single
%   overall amplitude per product chain — the other factor carries it).
%   hyp = [ log(ell_1); ...; log(ell_D); log(alpha) ]
%   k(x,z) = prod_d ( 1 + a_d^2/(2 alpha) )^(-alpha),  a_d = |x_d-z_d|/ell_d.
%   Conventions identical to covMatProdU (GPML v4.2 closure).

if nargin < 2, K = 'D+1'; return; end
if nargin < 3, z = []; end
dg = ischar(z) && strcmp(z,'diag');
if dg, z = []; end

D   = size(x,2);
ell = exp(hyp(1:D));
al  = exp(hyp(D+1));

if dg
    n = size(x,1); K = ones(n,1);
    if nargout > 1, dK = @(Q) zeros(D+1,1); end
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
K  = prod(KF, 3);
K  = K(:,:,1);

if nargout > 1
    dK = @(Q) dirder(Q, A2, F, KF, K, al, D);
end
end

% --------------------------------------------------------------------------
function [dhyp, dx] = dirder(Q, A2, F, KF, K, al, D)
    dhyp = zeros(D+1,1);
    for i = 1:D
        dKi = A2(:,:,i) .* F(:,:,i).^(-al-1);
        for e = 1:D
            if e ~= i, dKi = dKi .* KF(:,:,e); end
        end
        dhyp(i) = sum(sum(Q .* dKi));
    end
    S = -al * sum(log(F),3) + sum(A2 ./ (2*F),3);
    dhyp(D+1) = sum(sum(Q .* (K .* S(:,:,1))));
    dx = [];
end
