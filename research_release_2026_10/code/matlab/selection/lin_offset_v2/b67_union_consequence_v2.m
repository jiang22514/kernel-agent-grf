function result=b67_union_consequence_v2(options)
%B67_UNION_CONSEQUENCE_V2 FINAL-champion fields with unchanged old baselines.
% Modes: plan (default), preflight, generate, recheck, benchmark.
% Required for every active mode: score_file and final_manifest (JSON).
% Original baselines are never drawn again. New fields are saved before checks.
if nargin<1,options=struct();end
if ~isfield(options,'mode'),options.mode='plan';end
assert(ismember(options.mode,{'plan','preflight','generate','recheck','benchmark'}));
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'kernel-agent-grf'));setup_revision_paths;addpath(here);
if ~isfield(options,'threads'),options.threads=6;end
if ~isfield(options,'out_dir'),options.out_dir=fullfile(root,'output','revision','review_fixes_2026-10-06','lin_v2','consequence');end
out=canonical_(options.out_dir);assert(startsWith(out,[canonical_(root) filesep]));
assert(isscalar(options.threads)&&options.threads>=1&&mod(options.threads,1)==0);
old_dir=fullfile(root,'output','revision','07_整改_B7','consequence_union_v1');
base_names={'SE_main','SE_fine','RQ_main'};base_expr={'SE','SE','RQ'};base_mid=[4,4,2];
grids={[201,121],[401,241]};seeds=[103,153];units={'champion_main','champion_fine'};
gates=struct('cov_relative',1e-6,'cov_standardized_max',1e-3,'mean_gp_standardized_rms',1e-4, ...
    'mean_gp_standardized_max',1e-3,'variance_gp_relative_max',1e-3,'conditioning_jitter',0);
settings=struct('n_samples',500,'batch_size',16,'node_cap',2000000,'threads',options.threads, ...
    'theta_quantile',.1,'event_threshold',.1,'main_grid',grids{1},'fine_grid',grids{2}, ...
    'seeds',seeds,'minimum_available_gib',18,'baseline_policy','Reuse original SE_main, SE_fine, RQ_main fields after exact posterior mapping checks');
result=struct('status','plan','protocol','b67_union_lin_offset_v2','output_directory',out, ...
    'settings',settings,'gates',gates,'required_inputs',{{'FINAL score_file','FINAL manifest'}}, ...
    'benchmark_policy','Separate explicit mode; requires training_idle=true and runs both routes serially');
if strcmp(options.mode,'plan'),return;end
assert(isfield(options,'score_file')&&isfield(options,'final_manifest'));
F=jsondecode(fileread(options.final_manifest));table_hash=sha_(options.score_file);
assert(strcmp(F.status,'FINAL')&&strcmp(F.case_name,'Borehole_free')&&strcmp(F.table_sha256,table_hash), ...
    'b67v2:final','Require independently finalized CPT free-noise table and matching hash');
D=load(fullfile(root,'samedata.mat'),'x_same','y_same');x=D.x_same;y=D.y_same(:);
assert(size(x,2)==2&&strcmp(F.data_sha256,array_hash_(x,y)));
T=load(options.score_file,'recs','metadata','fingerprint','run_identity_sha256','source_manifest');
assert(strcmp(F.run_identity_sha256,T.run_identity_sha256)&&strcmp(T.fingerprint.data_sha256,F.data_sha256));
assert(strcmp(T.metadata.kernel_version,'lin_horizontal_offset_v2'));
verify_(T.source_manifest);
[legal,champ]=legal_candidates_v1(T.recs);assert(nnz(legal)==376&&isequaln(champ,T.metadata.champion));
row=T.recs(champ.ki).rows{champ.mi};hyp=row.hyp_final;
[xs,~,mf,~,geometry]=eval_geometry_v2(champ.expr,champ.mi,x);
theta=quantile(y,.1);assert(abs(theta-4.14)<1e-12,'b67v2:threshold','The physical threshold changed');
maxNumCompThreads(options.threads);
if ~isfolder(out),mkdir(out);end
sample_sources=sources_(T.source_manifest,[{options.score_file,options.final_manifest,fullfile(root,'samedata.mat')}, ...
    names_to_paths_({'b67_union_consequence_v2','union_gp_fields_v2','expand_expr_terms_v2', ...
    'covFixedShiftV2','checked_psd_sqrt_union','stable_union_matern_terms_v1','covMatern1dStable_v1', ...
    'kron_mvm','expand_expr_terms_v1','expand_expr_terms','predict_gp_v2','exact_gp_fields_v2', ...
    'covLINiso','covLIN','covConst','covRQiso','covRQ','covMaterniso','covMatern'})]);
checker_sources=sources_([],names_to_paths_({'b67_union_check_pack_v2','marginal_check_ski_v2', ...
    'cov_point_sets_frozen_v2','cov_point_sets_v1','b67_union_weakprop_v1'}));
sample_fp=struct('schema_version','lin_offset_v2','kernel_definition','lin_first_coordinate_offset_v2', ...
    'sources',sample_sources,'settings',settings,'champion',champ,'hyp',hyp,'geometry',geometry, ...
    'data_sha256',F.data_sha256,'table_sha256',table_hash,'run_identity_sha256',T.run_identity_sha256,'matlab_version',version);
checker_fp=struct('sources',checker_sources,'gates',gates,'matlab_version',version);
sample_id=text_sha_(jsonencode(sample_fp));checker_id=text_sha_(jsonencode(checker_fp));
if strcmp(options.mode,'benchmark')
    assert(isfield(options,'training_idle')&&isequal(options.training_idle,true), ...
        'b67v2:trainingActive','Benchmark requires explicit acknowledgement that fitting queues are finished');
    result=benchmark_(out,champ,hyp,mf,geometry,xs,x,y,settings,sample_fp,sample_id);return;
end
plan=struct('sample_fp',sample_fp,'sample_identity_sha256',sample_id,'units',{units},'baseline_units',{base_names});
save_or_match_(fullfile(out,'run_plan.mat'),struct('plan',plan),'plan');
% Prove baseline posteriors match; reuse their immutable raw fields and their
% original passed Monte Carlo receipts rather than repeating those checks.
proofs=cell(1,3);base_results=cell(1,3);
for j=1:3
    [proofs{j},base_results{j}]=reuse_baseline_(old_dir,base_names{j},base_expr{j},base_mid(j),T.recs,x,y,theta,out,sample_id);
end
% Both new grids must pass zero-field checks before either receives fields.
preflight=cell(1,2);
for j=1:2
    pf=fullfile(out,[units{j} '_zero_field_' checker_id(1:16) '.mat']);
    if isfile(pf)
        P=load(pf,'receipt');receipt=P.receipt;
        assert(strcmp(receipt.sample_identity_sha256,sample_id)&&strcmp(receipt.checker_identity_sha256,checker_id));
        assert(receipt.check.pass,'b67v2:zero','Saved zero-field check failed; no sampling');
    else
        resource_gate_(18);
        p=grid_pack_(x,geometry,grids{j});sets=cov_point_sets_frozen_v2(p,xs);
        opt=struct('n_samples',0,'batch_size',16,'node_cap',2000000,'check_indices',unique([sets.idx]));
        [mu,~,info]=union_gp_fields_v2(champ.expr,hyp,mf,xs,y,p.xq,opt);
        check=zero_check_(champ.expr,hyp,mf,xs,y,p.xq,mu,info,gates);
        receipt=struct('unit',units{j},'sample_identity_sha256',sample_id,'checker_identity_sha256',checker_id, ...
            'sample_fp',sample_fp,'checker_fp',checker_fp,'check',check,'info',info,'grid',p,'generated_fields',0);
        save_new_(pf,struct('receipt',receipt));sidecar_(pf);
        assert(check.pass,'b67v2:zero','Zero-field check failed; saved diagnostics, no sampling');
    end
    preflight{j}=struct('unit',units{j},'file',pf,'sha256',sha_(pf),'check',receipt.check);
end
write_once_(fullfile(out,['preflight_' checker_id(1:16) '.json']),struct('status','PASS','preflight',{preflight}, ...
    'baseline_proofs',{proofs},'generated_fields',0,'sample_identity_sha256',sample_id,'checker_identity_sha256',checker_id));
if strcmp(options.mode,'preflight'),result=struct('status','PREFLIGHT_PASS','preflight',{preflight},'baseline_proofs',{proofs});return;end
new_results=cell(1,2);
for j=1:2
    unit=units{j};sfile=fullfile(out,[unit '_samples.mat']);
    if isfile(sfile)
        check_sidecar_(sfile);S=load(sfile,'pack','sample_fp','sample_identity_sha256');
        assert(strcmp(S.sample_identity_sha256,sample_id)&&isequaln(S.sample_fp,sample_fp),'b67v2:sampleIdentity','Saved fields use different inputs; refusing replacement');
        pack=S.pack;clear S;
        assert(size(pack.F,2)==500&&pack.seed==seeds(j)&&strcmp(pack.expr,champ.expr)&&pack.mean_id==champ.mi);
        fprintf('REUSE_SAMPLES_V2 %s\n',unit);
    else
        assert(~strcmp(options.mode,'recheck'),'b67v2:missing','Recheck cannot draw missing fields');
        resource_gate_(18);pack=grid_pack_(x,geometry,grids{j});pack.cov_check_sets=cov_point_sets_frozen_v2(pack,xs);
        opt=struct('n_samples',500,'batch_size',16,'node_cap',2000000,'seed',seeds(j), ...
            'check_indices',unique([pack.cov_check_sets.idx]),'progress_callback',@(n)progress_(out,unit,n));
        fprintf('START_SAMPLES_V2 %s %s mean%d seed%d\n',unit,champ.expr,champ.mi,seeds(j));
        [mu,fields,info]=union_gp_fields_v2(champ.expr,hyp,mf,xs,y,pack.xq,opt);
        pack.F=fields;clear fields;pack.fmu=mu;pack.fsd=info.sd_exact;pack.info=info;
        pack.expr=champ.expr;pack.hyp=hyp;pack.mf=mf;pack.mean_id=champ.mi;pack.kind='union';
        pack.seed=seeds(j);pack.theta=theta;pack.unit=unit;pack.kernel_version='lin_horizontal_offset_v2';
        pack.source_table_sha256=table_hash;pack.source_data_sha256=F.data_sha256;pack.source_files=sample_sources;
        pack.cov_indices_source='Original physical query grid / original axis standard deviations, identical v1 index selection';
        sample_identity_sha256=sample_id;
        % Preserve all 500 fields before any statistical acceptance decision.
        save_new_(sfile,struct('pack',pack,'sample_fp',sample_fp,'sample_identity_sha256',sample_identity_sha256));sidecar_(sfile);
    end
    [r,acceptance]=b67_union_check_pack_v2(pack,xs,y,theta);
    r.unit=unit;r.sample_file=sfile;r.sample_sha256=sha_(sfile);
    r.sample_identity_sha256=sample_id;r.checker_identity_sha256=checker_id;
    rr=struct('result',r,'acceptance',acceptance,'checker_fp',checker_fp);
    write_once_(fullfile(out,[unit '_result_' checker_id(1:16) '.json']),rr);
    new_results{j}=rr;
    assert(acceptance.pass,'b67v2:statisticalCheck','Saved fields failed a frozen statistical gate; retain them, do not redraw or change seed');
    clear pack;
end
verify_(sample_sources);verify_(checker_sources);
result=struct('status','PASS','protocol','b67_union_lin_offset_v2','theta',theta,'champion',champ, ...
    'sample_identity_sha256',sample_id,'checker_identity_sha256',checker_id,'preflight',{preflight}, ...
    'baseline_proofs',{proofs},'results',{{new_results{1},new_results{2},base_results{1},base_results{2},base_results{3}}}, ...
    'scope','Two new champion packs plus three unchanged original baseline packs; no baseline resampling');
write_once_(fullfile(out,['summary_' checker_id(1:16) '.json']),result);
end

function [proof,result]=reuse_baseline_(old_dir,unit,expr,mid,recs,x,y,theta,out,sample_id)
sfile=fullfile(old_dir,[unit '_samples.mat']);rfile=fullfile(old_dir,[unit '_result.json']);
check_sidecar_(sfile);source_hash=sha_(sfile);old_result=jsondecode(fileread(rfile));
assert(old_result.acceptance.pass&&strcmp(old_result.result.sample_sha256,source_hash),'b67v2:baselineReceipt','Old baseline receipt is not valid for the saved fields');
pf=fullfile(out,[unit '_reuse_proof.mat']);
if isfile(pf)
    Q=load(pf,'proof');proof=Q.proof;
    assert(strcmp(proof.sample_identity_sha256,sample_id)&&strcmp(proof.old_sample_sha256,source_hash) ...
        &&strcmp(proof.old_result_sha256,sha_(rfile))&&proof.pass);
else
    S=load(sfile,'pack');p=S.pack;clear S;
    assert(strcmp(p.expr,expr)&&p.mean_id==mid&&p.theta==theta&&size(p.F,2)==500);
    [a,b]=meshgrid(p.g1,p.g2);q=[a(:),b(:)];sx=std(x);
    assert(isequal(p.xq,q./sx),'b67v2:baselineGrid','Old baseline physical query domain differs');
    ki=find(strcmp({recs.expr},expr));assert(isscalar(ki));h=recs(ki).rows{mid}.hyp_final;
    mapped=lift_hyp_v1_to_v2(expr,mid,p.hyp,x);
    assert(norm(h.cov(:)-mapped.cov(:),inf)<=1e-12&&norm(h.mean(:)-mapped.mean(:),inf)<=1e-12 ...
        &&abs(h.lik-mapped.lik)<=1e-12,'b67v2:baselineParameters','Final non-LIN parameters no longer equal the exact old-model mapping');
    [xs,~,~,~,g]=eval_geometry_v2(expr,mid,x);
    new=predict_gp_v2(expr,mid,struct('hyp',h,'geometry',g),x,y,q);
    mean_error=max(abs(new.mean-p.fmu));sd_error=max(abs(sqrt(new.latent_variance)-p.fsd));
    standardized_mean_error=max(abs(new.mean-p.fmu)./p.fsd);
    relative_variance_error=max(abs(new.latent_variance-p.fsd.^2)./(p.fsd.^2));
    assert(standardized_mean_error<1e-6&&relative_variance_error<1e-6,'b67v2:baselineChanged','Baseline posterior changed; reuse not justified');
    adapted=grid_pack_(x,g,[numel(p.g1),numel(p.g2)]);
    new_sets=cov_point_sets_frozen_v2(adapted,xs);old_sets=cov_point_sets_v1(p,x./sx);
    assert(isequaln(new_sets,old_sets),'b67v2:baselineIndices','Physical covariance check indices changed');
    proof=struct('pass',true,'unit',unit,'expr',expr,'mean_id',mid,'sample_identity_sha256',sample_id, ...
        'old_sample_file',sfile,'old_sample_sha256',source_hash,'old_result_file',rfile,'old_result_sha256',sha_(rfile), ...
        'max_absolute_mean_error',mean_error,'max_absolute_sd_error',sd_error, ...
        'max_standardized_mean_error',standardized_mean_error,'max_relative_variance_error',relative_variance_error, ...
        'exact_parameter_mapping_checked',true, ...
        'cov_check_sets',new_sets,'geometry',g,'hyp',h,'fields_reused',500,'new_fields_drawn',0, ...
        'policy','Full-grid v2 posterior agrees with the old moments; use unchanged fields and their original passed Monte Carlo checks');
    save_new_(pf,struct('proof',proof));clear p new;
end
result=old_result;result.reuse=proof;
end

function p=grid_pack_(x,g,ng)
g1=linspace(min(x(:,1)),max(x(:,1)),ng(1));g2=linspace(min(x(:,2)),max(x(:,2)),ng(2));
[a,b]=meshgrid(g1,g2);p=struct('g1',g1,'g2',g2,'xq',([a(:),b(:)]-g.offset)./g.scale, ...
    'physical_train',x,'geometry',g);
end
function r=zero_check_(expr,h,mf,xs,y,xq,mu,info,gates)
idx=info.check_indices;cf=build_covfunc_v2(parse_expr_str(expr),2);
[~,~,mg,vg]=gp(h,@infGaussLik,mf,cf,@likGauss,xs,y,xq(idx,:));
ce=info.check_cov_exact;ca=info.check_cov_actual;dv=diag(ce);assert(all(dv>0));
sd=sqrt(dv);dc=(ca-ce)./(sd*sd');dm=(mu(idx)-mg)./sd;
r=struct('cov_relative',info.analytic_cov_relative_error,'cov_standardized_max',max(abs(dc(:))), ...
    'mean_gp_standardized_rms',sqrt(mean(dm.^2)),'mean_gp_standardized_max',max(abs(dm)), ...
    'variance_gp_relative_max',max(abs(vg-dv)./dv),'conditioning_jitter',info.conditioning_jitter);
r.pass=r.cov_relative<gates.cov_relative&&r.cov_standardized_max<gates.cov_standardized_max ...
    &&r.mean_gp_standardized_rms<gates.mean_gp_standardized_rms&&r.mean_gp_standardized_max<gates.mean_gp_standardized_max ...
    &&r.variance_gp_relative_max<gates.variance_gp_relative_max&&r.conditioning_jitter==0;
end
function r=benchmark_(out,champ,h,mf,g,xs,x,y,settings,sample_fp,sample_id)
bo=fullfile(out,'same_model_benchmark');if ~isfolder(bo),mkdir(bo);end
file=fullfile(bo,'paired_timing.mat');
if isfile(file),Q=load(file,'result');assert(strcmp(Q.result.sample_identity_sha256,sample_id));r=Q.result;return;end
resource_gate_(18);p=grid_pack_(x,g,settings.main_grid);seed=410006;
opt=struct('n_samples',500,'batch_size',16,'node_cap',2000000,'seed',seed,'check_indices',[1,4000,12000,24321]);
uf=fullfile(bo,'union_stage.mat');
if isfile(uf)
    Q=load(uf,'stage');us=Q.stage;assert(strcmp(us.sample_identity_sha256,sample_id));
else
    timer=tic;[mu,U,info]=union_gp_fields_v2(champ.expr,h,mf,xs,y,p.xq,opt);elapsed=toc(timer);
    assert(all(isfinite(U(:))));clear U;
    us=struct('sample_identity_sha256',sample_id,'mean',mu,'info',info,'wall_seconds',elapsed);
    save_new_(uf,struct('stage',us));
end
resource_gate_(18);df=fullfile(bo,'dense_stage.mat');
if isfile(df)
    Q=load(df,'stage');ds=Q.stage;assert(strcmp(ds.sample_identity_sha256,sample_id));
else
    timer=tic;[V,ok,info]=exact_gp_fields_v2(champ.expr,h,mf,xs,y,p.xq,500,seed);elapsed=toc(timer);
    if ok,finite=all(isfinite(V(:)));else,finite=false;end
    clear V;
    ds=struct('sample_identity_sha256',sample_id,'info',info,'wall_seconds',elapsed,'ok',ok,'finite',finite);
    save_new_(df,struct('stage',ds));
end
assert(ds.ok&&ds.finite,'b67v2:benchmarkFailed','Dense benchmark failed; receipt preserved, no automatic retry');
mean_standardized=max(abs(us.mean-ds.info.mu_q)./us.info.sd_exact);
variance_relative=max(abs(us.info.sd_exact.^2-ds.info.sd_q.^2)./(us.info.sd_exact.^2));
assert(mean_standardized<1e-6&&variance_relative<1e-6&&ds.info.jitter_used<=1e-8, ...
    'b67v2:benchmarkAgreement','Benchmark moment agreement or numerical jitter gate failed; both stage receipts retained');
r=struct('status','PASS','sample_identity_sha256',sample_id,'champion',champ, ...
    'n_query',size(p.xq,1),'n_train',size(x,1),'n_samples',500,'threads',settings.threads,'seed',seed, ...
    'union_total_seconds',us.wall_seconds,'union_setup_seconds',us.info.t_setup_s,'union_sample_seconds',us.info.t_samples_s, ...
    'dense_total_seconds',ds.wall_seconds,'dense_kernel_seconds',ds.info.t_kernel, ...
    'dense_cholesky_seconds',ds.info.t_chol,'dense_sample_seconds',ds.info.t_sample, ...
    'dense_to_union_ratio',ds.wall_seconds/us.wall_seconds, ...
    'mean_max_absolute_difference',max(abs(us.mean-ds.info.mu_q)), ...
    'variance_max_absolute_difference',max(abs(us.info.sd_exact.^2-ds.info.sd_q.^2)), ...
    'mean_standardized_max_difference',mean_standardized,'variance_relative_max_difference',variance_relative, ...
    'dense_jitter_relative_to_max_variance',ds.info.jitter_used, ...
    'dense_jitter_relative_to_min_variance',ds.info.jitter_var_rel_max, ...
    'dense_peak_estimate_gib',ds.info.need_gb,'note','One serial same-model pair; includes setup and 500 fields in each route. Dense draws use the recorded numerical jitter. Not a weak-zone ensemble.');
save_new_(file,struct('result',r,'sample_fp',sample_fp));write_once_(fullfile(bo,'paired_timing.json'),r);
end
function files=names_to_paths_(names)
files=cell(size(names));for j=1:numel(names),files{j}=which(names{j});assert(~isempty(files{j}));end
end
function S=sources_(prior,files)
S=struct('path',{},'sha256',{});if ~isempty(prior),S=prior;end
for j=1:numel(files),S(end+1)=struct('path',files{j},'sha256',sha_(files{j}));end %#ok<AGROW>
[~,idx]=unique(string({S.path}),'stable');S=S(idx);
end
function verify_(S)
for j=1:numel(S),assert(strcmp(sha_(S(j).path),S(j).sha256),'b67v2:sourceChanged','Source changed: %s',S(j).path);end
end
function resource_gate_(n)
[~,s]=memory;assert(s.PhysicalMemory.Available/2^30>=n,'b67v2:memory','Available physical memory below 18 GiB');
end
function progress_(out,unit,n)
p=fullfile(out,'progress.json');fid=fopen(p,'w');assert(fid>=0);fwrite(fid,unicode2native(jsonencode(struct('unit',unit,'completed',n,'total',500)),'UTF-8'),'uint8');fclose(fid);
end
function p=canonical_(p)
p=char(java.io.File(p).getCanonicalPath());
end
function h=sha_(p)
fid=fopen(p,'rb');assert(fid>=0);c=onCleanup(@()fclose(fid)); %#ok<NASGU>
md=java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid),b=fread(fid,1048576,'*uint8');if ~isempty(b),md.update(typecast(b,'int8'));end;end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
function h=array_hash_(x,y)
h=bytes_sha_([typecast(double([size(x),size(y)]),'uint8')';typecast(x(:),'uint8');typecast(y(:),'uint8')]);
end
function h=text_sha_(s)
h=bytes_sha_(unicode2native(s,'UTF-8'));
end
function h=bytes_sha_(b)
md=java.security.MessageDigest.getInstance('SHA-256');md.update(typecast(uint8(b(:)),'int8'));h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
function check_sidecar_(p)
assert(isfile(p)&&isfile([p '.sha256'])&&strcmp(sha_(p),strtrim(fileread([p '.sha256']))),'b67v2:integrity','Saved sample integrity failure');
end
function sidecar_(p)
s=[p '.sha256'];assert(~isfile(s));fid=fopen(s,'w');assert(fid>=0);fprintf(fid,'%s',sha_(p));fclose(fid);
end
function save_new_(p,v)
assert(~isfile(p),'b67v2:immutable','Refusing overwrite: %s',p);t=[tempname(fileparts(p)) '.partial.mat'];
save(t,'-struct','v','-v7.3');assert(~isfile(p));[ok,msg]=movefile(t,p);assert(ok,msg);
end
function save_or_match_(p,v,key)
if isfile(p),Q=load(p,key);assert(isequaln(Q.(key),v.(key)),'b67v2:identity','Existing identity differs');else,save_new_(p,v);end
end
function write_once_(p,v)
if isfile(p),assert(isequaln(jsondecode(fileread(p)),jsondecode(jsonencode(v))));return;end
t=[tempname(fileparts(p)) '.partial.json'];fid=fopen(t,'w');assert(fid>=0);
fwrite(fid,unicode2native(jsonencode(v,'PrettyPrint',true),'UTF-8'),'uint8');fclose(fid);
assert(~isfile(p));[ok,msg]=movefile(t,p);assert(ok,msg);
end
