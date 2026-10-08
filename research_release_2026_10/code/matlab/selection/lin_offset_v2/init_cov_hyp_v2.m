function h=init_cov_hyp_v2(expr,D,logsf,log_rms_x)
%INIT_COV_HYP_V2 Uses horizontally centered coordinates; offset starts at 0.
h=init_cov_hyp_v1(expr,D,logsf,log_rms_x);
[~,~,slot]=build_covfunc_v2(expr,D);
if slot>0, h=[h(1:slot-1);0;h(slot:end)]; end
end
