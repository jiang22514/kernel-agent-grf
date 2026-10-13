function result=matlab_union_delft(model_id,n_samples,seed,x_query,cell_area_m2)
%MATLAB_UNION_DELFT Saved-fit latent conditional fields without refitting.
% Default: 0 samples (moments only) on the frozen 201 x 115 grid.
% Explicitly request 500 samples and seed 111026 for the study field ensemble.
if nargin<2,n_samples=0;end
if nargin<3,seed=111026;end
[m,d,index]=matlab_load_delft(model_id);
if nargin<4
    root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
    assert(strcmp(index.query_file,'data/fields_201x115.mat'));
    file=fullfile(root,index.query_file);assert(strcmp(hash_(file),index.query_sha256));
    q=load(file);x_query=q.x_query;cell_area_m2=q.cell_area_m2;
elseif nargin<5
    cell_area_m2=[];
end
[xs,~,mf,~,g]=eval_geometry_v2(m.kernel,m.mean_id,d.x);
assert(isequaln(g,m.geometry));qs=(x_query-g.offset)./g.scale;
checks=unique(round(linspace(1,size(qs,1),min(128,size(qs,1)))))';
options=struct('n_samples',n_samples,'seed',seed,'batch_size',16, ...
    'node_cap',2000000,'check_indices',checks);
[mu,F,info]=union_gp_fields_v2(m.kernel,m.hyp,mf,xs,d.y_log,qs,options);
assert(info.conditioning_jitter==0&&~info.query_observation_noise);
v=info.variance_exact;
result=struct('model_id',m.model_id,'x_query',x_query,'log_mean',mu, ...
    'latent_log_variance',v,'log_fields',F,'qc_fields_MPa',exp(F), ...
    'latent_mean_qc_MPa',exp(mu+v/2), ...
    'latent_sd_qc_MPa',sqrt(expm1(v).*exp(2*mu+v)), ...
    'union_info',info,'seed',seed,'cell_area_m2',cell_area_m2);
if ~isempty(cell_area_m2)
    w=cell_area_m2(:);assert(numel(w)==numel(mu)&&all(isfinite(w))&&all(w>0));
    area=sum(w);thresholds=[0.2,0.1,0.3,0.5];
    result.spatial_mean_qc_MPa=(w'/area*result.qc_fields_MPa)';
    result.spatial_mean_expectation_qc_MPa=w'/area*result.latent_mean_qc_MPa;
    result.thresholds_MPa=thresholds;
    result.low_resistance_area_fraction=zeros(n_samples,numel(thresholds));
    result.expected_low_resistance_area_fraction=zeros(1,numel(thresholds));
    for k=1:numel(thresholds)
        result.low_resistance_area_fraction(:,k)=(w'/area*(F<log(thresholds(k))))';
        probability=0.5*erfc(-(log(thresholds(k))-mu)./sqrt(2*v));
        result.expected_low_resistance_area_fraction(k)=w'/area*probability;
    end
    result.low_resistance_area_m2=result.low_resistance_area_fraction*area;
end
end
function h=hash_(p)
f=fopen(p,'rb');assert(f>=0);c=onCleanup(@()fclose(f));b=fread(f,inf,'*uint8');
m=java.security.MessageDigest.getInstance('SHA-256');m.update(typecast(b(:),'int8'));
h=lower(reshape(dec2hex(typecast(m.digest(),'uint8'),2)',1,[]));
end
