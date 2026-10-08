function [wp,W] = b67_union_weakprop_v1(F,g1,g2,theta)
%B67_UNION_WEAKPROP_V1 Physical-grid trapezoidal fraction of F < theta.
% F uses MATLAB meshgrid order (x2 fastest), one complete field per column.
% This deterministic helper does not draw, fit, save, or alter random state.
assert(isa(F,'double') && isreal(F) && ismatrix(F) && ~isempty(F), ...
    'b67union:functionalInput','F must be a nonempty real double matrix');
assert(isa(theta,'double') && isreal(theta) && isscalar(theta) && isfinite(theta), ...
    'b67union:functionalInput','theta must be one finite real double');
check_axis_(g1); check_axis_(g2);
n1=numel(g1); n2=numel(g2);
assert(size(F,1)==n1*n2,'b67union:functionalShape','F rows must match the physical grid');
wx=[0.5,ones(1,n1-2),0.5]*(g1(2)-g1(1));
wy=[0.5,ones(1,n2-2),0.5]*(g2(2)-g2(1));
W=wy(:)*wx(:).'; total=sum(W(:));
assert(isfinite(total) && total>0,'b67union:functionalWeights','Invalid physical area');
wp=zeros(size(F,2),1); w=W(:);
% Per-field evaluation bounds temporary storage even for 500 fine fields.
for j=1:size(F,2)
    v=F(:,j);
    assert(all(isfinite(v)),'b67union:nonfiniteField', ...
        'Field contains NaN/Inf; comparison must not silently count it as strong');
    wp(j)=sum(w.*(v<theta))/total;
end
end

function check_axis_(g)
assert(isa(g,'double') && isreal(g) && isvector(g) && numel(g)>=2 && all(isfinite(g(:))), ...
    'b67union:functionalGrid','Grid axes must contain at least two finite doubles');
d=diff(g(:));
assert(all(d>0) && max(abs(d-d(1)))<=128*eps(max(1,max(abs(g(:))))), ...
    'b67union:functionalGrid','Trapezoidal helper requires a strictly increasing uniform axis');
end
