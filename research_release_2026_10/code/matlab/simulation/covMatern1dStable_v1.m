function [K,dK] = covMatern1dStable_v1(nu2,hyp,x,z)
%COVMATERN1DSTABLE_V1 Stable 1-D Matern leaf with the GPML two-hyper layout.
%   hyp=[log(ell);log(sf)], nu2=1/3/5. Parameter-count call returns '2'.
%   Delegates to the SAME covMatProdU direct abs(x-z)/ell implementation
%   used by selection, with the standard covScale amplitude wrapper. It
%   does not evaluate squared norms followed by subtracting and square root.
%   Hyperparameter directional derivatives follow the existing GPML closure
%   convention. Coordinate derivatives are not provided by covMatProdU and
%   are not used by this sampling-only leaf.
assert(isnumeric(nu2) && isscalar(nu2) && any(nu2==[1,3,5]), ...
    'union:maternOrder','Supported doubled Matern orders are 1, 3, and 5');
if nargin<3, K='2'; return; end
if nargin<4, z=[]; end
assert(isa(hyp,'double') && isreal(hyp) && numel(hyp)==2 && all(isfinite(hyp(:))), ...
    'union:maternHyp','The one-dimensional Matern leaf requires two finite real hyperparameters');
assert(isa(x,'double') && isreal(x) && ismatrix(x) && size(x,2)==1 && all(isfinite(x(:))), ...
    'union:maternDimension','The stable Matern leaf requires finite one-dimensional inputs');
if ~isempty(z)
    assert((ischar(z) && strcmp(z,'diag')) || ...
        (isa(z,'double') && isreal(z) && ismatrix(z) && size(z,2)==1 && all(isfinite(z(:)))), ...
        'union:maternDimension','z must be a finite one-dimensional input or diag');
end
if nargout>1
    [K,dK]=covScale({@covMatProdU,nu2},hyp,x,z);
else
    K=covScale({@covMatProdU,nu2},hyp,x,z);
end
assert(all(isfinite(K(:))),'union:maternNonfinite','Stable Matern evaluation produced a nonfinite value');
end
