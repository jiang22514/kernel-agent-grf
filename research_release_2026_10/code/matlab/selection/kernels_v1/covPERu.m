function [K, dK] = covPERu(hyp, x, z)
%COVPERU  Unit-amplitude isotropic periodic kernel, per-dimension factorized.
%         kernels_v1 — GPML v4.2 closure convention.
%
%   k(x,z) = prod_d exp( -2 sin^2(pi |x_d - z_d| / p) / ell^2 )
%
%   hyp = [ log(p); log(ell) ]   (period p shared across dims, defined on the
%   standardized coordinates — see revision plan P35)
%
%   Unit-amplitude companion of {@covPER,'iso',{@covSEiso}} for use INSIDE
%   products (product rule: one overall amplitude per product chain).
%   Identical functional form minus sf: covPER-iso embeds x ->
%   [sin(pi x/p); cos(pi x/p)] per dim and applies covSEiso; the squared
%   embedded distance is sum_d 2 sin^2(pi d_d/p), which factorizes per dim.

if nargin < 2, K = '2'; return; end              % parameter count: p, ell
if nargin < 3, z = []; end
dg = ischar(z) && strcmp(z,'diag');
if dg, z = []; end

p   = exp(hyp(1));
ell = exp(hyp(2));

if dg
    n = size(x,1); K = ones(n,1);
    if nargout > 1, dK = @(Q) zeros(2,1); end
    return
end

if isempty(z), z = x; end
D = size(x,2);
n = size(x,1); m = size(z,1);

K = ones(n,m);
SS = zeros(n,m,D);   % sin^2(pi d / p) per dim (for log-ell derivative)
TP = zeros(n,m,D);   % d * sin(2 pi d / p) per dim (for log-p derivative)
for d = 1:D
    Dd = bsxfun(@minus, x(:,d), z(:,d)');        % signed difference
    sd = sin(pi*Dd/p);
    SS(:,:,d) = sd.^2;
    K  = K .* exp(-2 * SS(:,:,d) / ell^2);
    TP(:,:,d) = Dd .* sin(2*pi*Dd/p);
end

if nargout > 1
    dK = @(Q) dirder(Q, K, SS, TP, p, ell, D);
end
end

% --------------------------------------------------------------------------
function [dhyp, dx] = dirder(Q, K, SS, TP, p, ell, D)
% ln k = -(2/ell^2) sum_d sin^2(pi d_d/p)
%   d/d log ell = (4/ell^2) sum_d sin^2
%   d/d log p   = (2 pi/(ell^2 p)) sum_d d_d sin(2 pi d_d/p)
    dhyp = zeros(2,1);
    dhyp(1) = sum(sum(Q .* (K .* (2*pi/(ell^2*p)) .* sum(TP,3))));
    dhyp(2) = sum(sum(Q .* (K .* (4/ell^2)      .* sum(SS,3))));
    dx = [];
end
