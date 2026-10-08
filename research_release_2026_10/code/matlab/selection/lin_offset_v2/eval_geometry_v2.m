function [xs,cf,mf,n_cov,geometry]=eval_geometry_v2(expr,mean_id,x)
%EVAL_GEOMETRY_V2 Training-only horizontal centering and per-axis scaling.
% Reuse geometry.offset/scale for every query, never re-estimate on test data.
sx=std(x); sx(sx==0)=1;
offset=zeros(1,size(x,2)); offset(1)=mean(x(:,1));
xs=(x-offset)./sx;
[cf,n_cov]=build_covfunc_v2(expr,size(x,2));
switch mean_id
    case 1, mf={@meanZero};
    case 2, mf={@meanConst};
    case 3, mf={@meanSum,{{@meanConst},{@meanLinear}}};
    case 4, mf={@meanSum,{{@meanConst},{@meanPoly,2}}};
    otherwise, error('linOffset:mean','Mean id must be 1..4');
end
geometry=struct('offset',offset,'scale',sx,'definition','horizontal_training_mean_and_axis_std_v2');
end
