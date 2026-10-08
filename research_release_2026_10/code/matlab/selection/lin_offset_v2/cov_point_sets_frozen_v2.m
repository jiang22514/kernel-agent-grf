function sets=cov_point_sets_frozen_v2(pack,xs)
%COV_POINT_SETS_FROZEN_V2 Keep v1 physical locations under the v2 geometry.
% Only the selection of the diagnostic indices uses the original uncentered
% standardized coordinates; all covariance calculations use the new kernel
% and centered coordinates. Numerical acceptance thresholds are unchanged.
assert(isfield(pack,'physical_train')&&isfield(pack,'geometry'), ...
    'v2check:reference','Physical training coordinates and fitted geometry are required');
x=pack.physical_train;g=pack.geometry;s=std(x);s(s==0)=1;
[a,b]=meshgrid(pack.g1,pack.g2);q=[a(:),b(:)];
assert(isequal(xs,(x-g.offset)./g.scale)&&isequal(pack.xq,(q-g.offset)./g.scale), ...
    'v2check:geometry','Pack/training coordinates do not use the recorded physical geometry');
ref=struct('g1',pack.g1,'g2',pack.g2,'xq',q./s);
sets=cov_point_sets_v1(ref,x./s);
if isfield(pack,'cov_check_sets')
    assert(isequaln(sets,pack.cov_check_sets),'v2check:indices','Frozen physical diagnostic indices changed');
end
end
