function result=matlab_refit_delft(model_id,options)
%MATLAB_REFIT_DELFT Optional selected-structure refit, never the full library.
% Default mode='plan'. mode='run' performs 15 starts x 200 evaluations + 500.
% selection: main_12, means 1..target, neutral plus polished parent seed.
% fixed_structure: target mean only, neutral starts, for a supplied public case.
if nargin<2,options=struct();end
[model,main,index]=matlab_load_delft(model_id);
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
defaults=struct('mode','plan','fit_role','selection', ...
    'case_file',fullfile(root,index.data_file),'output_dir',fullfile(root,'results','refits'));
for names=fieldnames(defaults)',name=names{1};if ~isfield(options,name),options.(name)=defaults.(name);end,end
assert(ismember(options.mode,{'plan','run'})&&ismember(options.fit_role,{'selection','fixed_structure'}));
D=load(options.case_file,'x','y','y_log');x=double(D.x);y=double(D.y_log(:));
assert(size(x,2)==2&&size(x,1)==numel(y)&&all(isfinite(x(:)))&&all(isfinite(y)));
assert(all(D.y(:)>0)&&max(abs(log(double(D.y(:)))-y))<1e-12&&size(unique(x,'rows'),1)==size(x,1));
if strcmp(options.fit_role,'selection')
    assert(isequaln(x,main.x)&&isequaln(y,main.y_log(:)), ...
        'delftRelease:selection','Selection inheritance is defined for the frozen main_12 case');
    means=1:model.mean_id;
else
    means=model.mean_id;
end
library=kernel_grammar(struct('include_t3',true));assert(numel(library)==94);
ki=find(strcmp({library.name},model.kernel));assert(isscalar(ki));
assert(strcmp(model_id,sprintf('k%03d_m%d',ki,model.mean_id)));
identity=struct('schema','delft_portable_refit_v1','kernel',model.kernel, ...
    'kernel_index',ki,'target_mean',model.mean_id,'fit_role',options.fit_role, ...
    'data_sha256',hash_(options.case_file),'release_sha256',hash_(fullfile(root,'matlab_manifest.json')), ...
    'n_restart',15,'n_iter',-200,'polish_max_evaluations',500, ...
    'jitter_sd',0.5,'seed0',20261011,'threads',3,'free_noise',true, ...
    'response_scale','log','matlab_version',version);
out=fullfile(options.output_dir,[options.fit_role '_' identity.data_sha256(1:12)],model_id);
result=struct('status','PLAN','model_id',model_id,'required_mean_ids',means, ...
    'maximum_new_fits',numel(means),'output_directory',out,'identity',identity);
if strcmp(options.mode,'plan'),return;end
if ~isfolder(out),mkdir(out);end
maxNumCompThreads(3);parent=[];
for mi=means
    tag=sprintf('k%03d_m%d',ki,mi);file=fullfile(out,[tag '.mat']);
    raw_file=fullfile(out,[tag '_raw.mat']);
    ev=struct('n_restart',15,'n_iter',-200,'jitter_sd',0.5,'quiet',true, ...
        'standardize',true,'seed',20261011+100*ki+mi,'seed_hyps',{{}});
    if ~isempty(parent)
        hp=parent.hyp;counts=[0,1,3,5];hp.mean=[hp.mean(:);zeros(counts(mi)-numel(hp.mean),1)];
        [xs,cf,mf]=eval_geometry_v2(model.kernel,mi,x);
        assert(abs(gp(hp,@infGaussLik,mf,cf,@likGauss,xs,y)-parent.nlZ)<1e-6);
        ev.seed_hyps={hp};
    end
    if isfile(file)
        C=load(file,'record');r=C.record;
        assert(isequaln(r.identity,identity)&&r.mean_id==mi&&isequaln(r.options,ev));
        fit=r.fit;
    else
        if isfile(raw_file)
            C=load(raw_file,'record');r=C.record;
            assert(isequaln(r.identity,identity)&&r.mean_id==mi&&isequaln(r.options,ev));raw=r.fit;
        else
            fprintf('DELFT_PORTABLE_REFIT %s (%d observations)\n',tag,size(x,1));
            raw=evaluate_expr_aic_v2(model.kernel,mi,x,y,ev);
            record=struct('identity',identity,'mean_id',mi,'options',ev,'fit',raw);save_once_(raw_file,record);
        end
        assert(raw.n_success>0,'delftRelease:fit','All restarts failed; diagnostics retained');
        check_(raw,model.kernel,mi,x,y,parent);
        fit=polish_fit_v2(model.kernel,mi,raw,x,y,500);
        check_(fit,model.kernel,mi,x,y,parent);
        record=struct('identity',identity,'mean_id',mi,'options',ev,'fit',fit);save_once_(file,record);
    end
    check_(fit,model.kernel,mi,x,y,parent);parent=fit;
end
result.status='COMPLETE';result.fit=fit;result.fit_file=file;
end
function check_(f,expr,mi,x,y,parent)
[xs,cf,mf,nc,g]=eval_geometry_v2(expr,mi,x);counts=[0,1,3,5];k=nc+counts(mi)+1;
assert(f.ok&&f.free_noise&&isequaln(f.geometry,g)&&f.k==k&&isfinite(f.nlZ));
assert(abs(f.aic-2*k-2*f.nlZ)<1e-6&&abs(gp(f.hyp,@infGaussLik,mf,cf,@likGauss,xs,y)-f.nlZ)<1e-6);
if ~isempty(parent),assert(f.nlZ<=parent.nlZ+1e-6);end
end
function save_once_(file,record)
assert(~isfile(file));temp=[tempname(fileparts(file)) '.partial.mat'];
save(temp,'record','-v7');assert(~isfile(file));[ok,msg]=movefile(temp,file);assert(ok,msg);
end
function h=hash_(p)
f=fopen(p,'rb');assert(f>=0);c=onCleanup(@()fclose(f));b=fread(f,inf,'*uint8');
m=java.security.MessageDigest.getInstance('SHA-256');m.update(typecast(b(:),'int8'));
h=lower(reshape(dec2hex(typecast(m.digest(),'uint8'),2)',1,[]));
end
