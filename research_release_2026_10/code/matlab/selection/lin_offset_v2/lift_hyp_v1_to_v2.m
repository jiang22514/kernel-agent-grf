function h=lift_hyp_v1_to_v2(expr,mean_id,h_old,x)
%LIFT_HYP_V1_TO_V2 Exact embedding; does not optimize or replace fit records.
% v1 uses x/std(x); v2 uses (x-[mean(x_h),0,...])/std(x).
D=size(x,2); sx=std(x); sx(sx==0)=1; shift=mean(x(:,1))/sx(1);
h=h_old; h.cov=h.cov(:); h.mean=h.mean(:);
[~,~,slot]=build_covfunc_v2(expr,D);
if slot>0, h.cov=[h.cov(1:slot-1);-shift;h.cov(slot:end)]; end
if mean_id>=3
    h.mean(1)=h_old.mean(1)+h_old.mean(2)*shift;
end
if mean_id==4
    % meanPoly order is [x_1,...,x_D,x_1^2,...,x_D^2].
    h.mean(1)=h.mean(1)+h_old.mean(D+2)*shift^2;
    h.mean(2)=h_old.mean(2)+2*h_old.mean(D+2)*shift;
end
assert(mean_id>=1&&mean_id<=4,'linOffset:mean','Only current four means are supported');
end
