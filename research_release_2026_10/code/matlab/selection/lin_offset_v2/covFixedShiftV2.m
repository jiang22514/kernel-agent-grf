function [K,dK]=covFixedShiftV2(base,shift,hyp,x,z)
%COVFIXEDSHIFTV2 Fixed shift for a one-dimensional simulation factor.
if nargin<4, K=feval(base{:}); return; end
if nargin<5, z=[]; end
x=x-shift;
if ~isempty(z)&&~ischar(z), z=z-shift; end
if nargout>1, [K,dK]=feval(base{:},hyp,x,z);
else, K=feval(base{:},hyp,x,z); end
end
