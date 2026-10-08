function report=check_consequence_fixture_v2()
% Small compatibility fixture for the copied diagnostics, not a model test.
% One fixed ensemble is checked in its old and exactly mapped new coordinates.
% No formal field pack, formal grid, acceptance threshold, or seed is changed.
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'kernel-agent-grf'));setup_revision_paths;addpath(here);maxNumCompThreads(2);
x=[441,5.8;478,13.7;618,8.3;752,18.2;687,10.4];y=[3.2;4.8;2.5;5.9;4.1];theta=4.14;
g1=linspace(min(x(:,1)),max(x(:,1)),7);g2=linspace(min(x(:,2)),max(x(:,2)),8);
[a,b]=meshgrid(g1,g2);q=[a(:),b(:)];sx=std(x);xo=x./sx;qo=q./sx;
expr='(MA1*LIN)+PER';mi=2;[~,~,mf,nc]=eval_geometry_v1(expr,mi,x);
ho=struct('cov',zeros(nc,1),'mean',3.8,'lik',log(.3));
[mu,fields,info]=union_gp_fields_v1(expr,ho,mf,xo,y,qo,struct('n_samples',500,'seed',410071,'check_indices',(1:56)'));
po=struct('g1',g1,'g2',g2,'xq',qo,'F',fields,'fmu',mu,'fsd',info.sd_exact);
po.wp=b67_union_weakprop_v1(fields,g1,g2,theta);
old=marginal_check_ski(expr,ho,mf,xo,y,po,theta);
hn=lift_hyp_v1_to_v2(expr,mi,ho,x);[xn,~,mfn,~,gn]=eval_geometry_v2(expr,mi,x);
pn=po;pn.xq=(q-gn.offset)./gn.scale;pn.geometry=gn;pn.physical_train=x;
pn.cov_check_sets=cov_point_sets_frozen_v2(pn,xn);
assert(isequaln(pn.cov_check_sets,cov_point_sets_v1(po,xo)));
new=marginal_check_ski_v2(expr,hn,mfn,xn,y,pn,theta);
fields_to_compare={'rms_mean','max_abs_mean','sd_ratio_med','sd_ratio_q','exp_gap','exp_mc_se','cov_ratio_max'};
error_max=0;
for j=1:numel(fields_to_compare)
    key=fields_to_compare{j};error_max=max(error_max,max(abs(old.(key)(:)-new.(key)(:))));
end
cov_error=0;
for j=1:4
    assert(strcmp(old.cov_sets{j}.name,new.cov_sets{j}.name));
    for key={'err','mc_scale','tol','ratio'}
        k=key{1};cov_error=max(cov_error,abs(old.cov_sets{j}.(k)-new.cov_sets{j}.(k)));
    end
end
assert(error_max<1e-9&&cov_error<1e-9&&old.pass==new.pass, ...
    'fixture:checkerMismatch','Mapped v2 diagnostic differs from frozen v1 diagnostic');
% Exact source comparison proves the formal wrapper retains every old gate.
old_source=fileread(fullfile(root,'kernel-agent-grf','simulation','b67_union_check_pack_v1.m'));
expected=strrep(strrep(strrep(old_source,'b67_union_check_pack_v1','b67_union_check_pack_v2'), ...
    'B67_UNION_CHECK_PACK_V1','B67_UNION_CHECK_PACK_V2'),'marginal_check_ski(','marginal_check_ski_v2(');
actual=fileread(fullfile(here,'b67_union_check_pack_v2.m'));
assert(strcmp(normalize_(actual),normalize_(expected)),'fixture:gateChanged','Formal checker changed beyond versioned call routing');
old_marginal=fileread(fullfile(root,'kernel-agent-grf','simulation','marginal_check_ski.m'));
expected=strrep(strrep(strrep(old_marginal,'function mg = marginal_check_ski(','function mg = marginal_check_ski_v2('), ...
    'build_covfunc_v1(','build_covfunc_v2('),'sets = cov_point_sets_v1(pack, xs);','sets = cov_point_sets_frozen_v2(pack, xs);');
assert(strcmp(normalize_(fileread(fullfile(here,'marginal_check_ski_v2.m'))),normalize_(expected)), ...
    'fixture:gateChanged','Marginal checker changed beyond kernel/index routing');
report=struct('passed',true,'n_query',56,'n_fields',500,'seed',410071, ...
    'max_metric_difference',error_max,'max_covariance_metric_difference',cov_error, ...
    'old_ensemble_pass',old.pass,'new_ensemble_pass',new.pass,'formal_gates_unchanged',true, ...
    'scope','Diagnostic compatibility only; one small fixture, no formal grid simulation');
disp(report);
end
function s=normalize_(s)
s=strrep(s,sprintf('\r\n'),sprintf('\n'));
end
