function result = run_holdout_case(case_name, options)
%RUN_HOLDOUT_CASE Fresh, uniformly fitted 94 x 4 LIN-v2 test-score table.
% options.kernel_indices controls scheduling only; all scientific settings
% and per-candidate seeds remain identical for preflight and full/resume.
if nargin < 2, options=struct(); end
if ~isfield(options,'kernel_indices'), options.kernel_indices=1:94; end
if ~isfield(options,'max_new_fits'), options.max_new_fits=inf; end
if ~isfield(options,'threads'), options.threads=3; end
assert(options.threads==3,'holdout:threads','The frozen protocol uses three threads per process');
assert(ischar(case_name)&&~isempty(regexp(case_name,'^H0[1-6]$','once')));
assert(all(ismember(options.kernel_indices,1:94))&&numel(unique(options.kernel_indices))==numel(options.kernel_indices));
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(fileparts(here)));
base=fullfile(root,'recomputed','holdout');
addpath(fullfile(root,'code','matlab')); setup_revision_paths;
v2=fullfile(root,'code','matlab','selection','lin_offset_v2'); addpath(v2);
assert(strcmp(which('evaluate_expr_aic_v2'),fullfile(v2,'evaluate_expr_aic_v2.m')));
maxNumCompThreads(options.threads);
data_file=fullfile(root,'data','holdout',[case_name '.mat']);
S=load(data_file,'x_raw','y'); x=double(S.x_raw); y=double(S.y(:));
assert(isequal(size(x),[96,2])&&numel(y)==96&&all(isfinite(x(:)))&&all(isfinite(y)));
assert(all(std(x)>0)&&std(y)>0);
lib=kernel_grammar(struct('include_t3',true)); assert(numel(lib)==94);
assets_file=fullfile(root,'data','holdout','assets.json');
A=jsondecode(fileread(assets_file)); assert(isequal({lib.name}',A.kernels(:)),'holdout:catalog','Kernel order differs from development catalog');
out=fullfile(base,'scores',case_name); units=fullfile(out,'candidates');
if ~isfolder(units),mkdir(units);end
source_files=source_manifest_(here,assets_file);
case_id=str2double(case_name(2:end));
identity=struct('schema','heldout_fresh_lin_v2_20261008','case_name',case_name, ...
    'n',96,'D',2,'data_file',data_file,'data_file_sha256',file_hash_(data_file), ...
    'data_sha256',array_hash_(x,y),'sources',source_files,'matlab_version',version, ...
    'kernel_version','lin_horizontal_offset_v2','kernel_count',94,'mean_count',4, ...
    'n_restart',15,'n_iter',-200,'polish_max_evaluations',500,'jitter_sd',0.5, ...
    'noise','free','seed0',108860,'seed_rule','108860+100000*case_number+100*kernel_id+mean_id', ...
    'initialization','neutral plus same-expression fitted parent mean; no other case or generating parameters', ...
    'mean_order',{{'Zero','Constant','Linear','Quadratic'}},'threads',3);
identity_sha256=bytes_hash_(unicode2native(jsonencode(identity),'UTF-8'));
plan=struct('identity',identity,'identity_sha256',identity_sha256,'case_directory',out);
plan_file=fullfile(out,'run_plan.mat');
if isfile(plan_file)
    P=load(plan_file,'plan');assert(strcmp(P.plan.identity_sha256,identity_sha256),'holdout:resume','Data, source or protocol changed; refusing mixed resume');
else
    save_once_(plan_file,struct('plan',plan));
end
write_once_(fullfile(out,'run_plan.json'),plan);
profile_rng=rng; rng(108860+case_id); profile=describe_dataset(x,y,case_name); rng(profile_rng);
new_fits=0; completed=0;
for ki=sort(options.kernel_indices(:)')
    parent=[]; parent_hash='';
    for mi=1:4
        tag=sprintf('k%03d_m%d',ki,mi); file=fullfile(units,[tag '.mat']);
        if isfile(file)
            U=load(file,'unit');unit=U.unit;
            assert(strcmp(unit.identity_sha256,identity_sha256)&&unit.ki==ki&&unit.mi==mi&&strcmp(unit.expr,lib(ki).name));
            assert(strcmp(unit.parent_checkpoint_sha256,parent_hash),'holdout:parent','Parent changed on resume');
            check_fit_(unit.fit,lib(ki).name,mi,x,y,parent);
            fprintf('REUSE %s %s AIC %.9f\n',case_name,tag,unit.fit.aic);
        else
            if new_fits>=options.max_new_fits||isfile(fullfile(out,'STOP_AFTER_CANDIDATE'))
                result=struct('status','PARTIAL','case_name',case_name,'new_fits',new_fits,'completed_this_call',completed);return;
            end
            verify_sources_(source_files); assert(strcmp(file_hash_(data_file),identity.data_file_sha256));
            ev=struct('n_restart',15,'n_iter',-200,'jitter_sd',0.5,'quiet',true, ...
                'seed',108860+100000*case_id+100*ki+mi,'seed_hyps',{{}});
            if ~isempty(parent)
                hp=parent.hyp; nm=mean_count_(mi,2); hp.mean=[hp.mean(:);zeros(nm-numel(hp.mean),1)];
                [xs,cf,mf]=eval_geometry_v2(lib(ki).name,mi,x);
                z=gp(hp,@infGaussLik,mf,cf,@likGauss,xs,y);
                assert(abs(z-parent.nlZ)<1e-6,'holdout:embedding','Nested-mean embedding changes likelihood');
                ev.seed_hyps={hp};
            end
            fprintf('START %s %s %s mean%d\n',case_name,tag,lib(ki).name,mi);
            timer=tic; raw=evaluate_expr_aic_v2(lib(ki).name,mi,x,y,ev);
            assert(raw.n_success>0,'holdout:fit','All 15 optimization starts failed');
            check_fit_(raw,lib(ki).name,mi,x,y,parent);
            [fit,polish]=polish_fit_v2(lib(ki).name,mi,raw,x,y,500);
            elapsed=toc(timer);check_fit_(fit,lib(ki).name,mi,x,y,parent);
            verify_sources_(source_files);assert(strcmp(file_hash_(data_file),identity.data_file_sha256));
            unit=struct('identity_sha256',identity_sha256,'ki',ki,'mi',mi,'expr',lib(ki).name, ...
                'fit',fit,'raw_fit',raw,'polish',polish,'options',ev, ...
                'elapsed_seconds',elapsed,'parent_checkpoint_sha256',parent_hash);
            save_once_(file,struct('unit',unit));new_fits=new_fits+1;
            fprintf('DONE %s %s AIC %.9f seconds %.2f\n',case_name,tag,fit.aic,elapsed);
        end
        h=file_hash_(file);
        summary=struct('case_name',case_name,'ki',ki,'mi',mi,'kernel',lib(ki).name, ...
            'mean',unit.fit.mean_name,'aic',unit.fit.aic,'nlZ',unit.fit.nlZ,'k',unit.fit.k, ...
            'ok',unit.fit.ok,'elapsed_seconds',unit.elapsed_seconds,'identity_sha256',identity_sha256,'checkpoint_sha256',h);
        write_once_(fullfile(units,[tag '.json']),summary);
        parent=unit.fit;parent_hash=h;completed=completed+1;
    end
end
% Only publish when every legal candidate has an independently saved fit.
files=dir(fullfile(units,'k???_m?.mat'));
if numel(files)~=376
    result=struct('status','PARTIAL','case_name',case_name,'completed_total',numel(files),'new_fits',new_fits);return;
end
recs=struct('expr',{},'pruned',{},'rows',{},'done4',{}); rows=struct('kernel',{},'mean_id',{},'mean',{},'aic',{},'k',{},'ok',{});
hashes=cell(94,4); total_seconds=0;
for ki=1:94
    recs(ki)=struct('expr',lib(ki).name,'pruned',false,'rows',{cell(1,4)},'done4',true);
    parent=[];parent_hash='';
    for mi=1:4
        file=fullfile(units,sprintf('k%03d_m%d.mat',ki,mi));U=load(file,'unit');u=U.unit;
        assert(strcmp(u.identity_sha256,identity_sha256)&&u.ki==ki&&u.mi==mi&&strcmp(u.expr,lib(ki).name));
        assert(strcmp(u.parent_checkpoint_sha256,parent_hash)); check_fit_(u.fit,u.expr,mi,x,y,parent);
        f=u.fit;r=struct('expr_name',u.expr,'mean_id',mi,'mean_name',f.mean_name, ...
            'hyp',f.hyp,'hyp_final',f.hyp,'nlZ',f.nlZ,'nlZ_final',f.nlZ, ...
            'aic',f.aic,'aic_final',f.aic,'bic',f.bic,'bic_final',f.bic,'k',f.k,'ok',f.ok, ...
            'n_success',f.n_success,'protocol','fresh_lin_v2_15x200_plus500','time',u.elapsed_seconds);
        recs(ki).rows{mi}=r;
        rows(end+1)=struct('kernel',u.expr,'mean_id',mi,'mean',f.mean_name,'aic',f.aic,'k',f.k,'ok',f.ok); %#ok<AGROW>
        parent=f;parent_hash=file_hash_(file);hashes{ki,mi}=parent_hash;total_seconds=total_seconds+u.elapsed_seconds;
    end
end
[mask,champ]=legal_candidates_v1(recs);assert(nnz(mask)==376);
verify_sources_(source_files);assert(strcmp(file_hash_(data_file),identity.data_file_sha256));
metadata=struct('status','COMPLETE','complete',true,'case_name',case_name,'candidate_count',376, ...
    'identity_sha256',identity_sha256,'data_sha256',identity.data_sha256,'champion',champ, ...
    'fit_seconds',total_seconds,'checkpoint_sha256',{hashes},'kernel_version','lin_horizontal_offset_v2');
table_file=fullfile(out,'scores.mat');
if ~isfile(table_file)
    save_once_(table_file,struct('recs',recs,'metadata',metadata,'identity',identity,'profile',profile));
else
    T=load(table_file,'metadata');assert(isequaln(T.metadata,metadata));
end
result=struct('status','COMPLETE','case_name',case_name,'kernels',{{lib.name}},'profile',profile, ...
    'rows',rows,'champion',champ,'candidate_count',376,'fit_seconds',total_seconds, ...
    'identity_sha256',identity_sha256,'data_sha256',identity.data_sha256,'table_sha256',file_hash_(table_file));
write_once_(fullfile(out,'scores.json'),result);
fprintf('HOLDOUT_COMPLETE %s candidates=376 seconds=%.2f\n',case_name,total_seconds);
end

function check_fit_(f,expr,mi,x,y,parent)
assert(f.ok&&isfinite(f.nlZ)&&isfinite(f.aic)&&isfinite(f.bic)&&isreal(f.nlZ));
[xs,cf,mf,nc]=eval_geometry_v2(expr,mi,x); k=nc+mean_count_(mi,size(x,2))+1;
assert(f.k==k&&f.free_noise&&isfield(f.hyp,'lik')&&isfinite(f.hyp.lik));
assert(abs(f.aic-(2*k+2*f.nlZ))<=1e-6&&abs(f.bic-(k*log(size(x,1))+2*f.nlZ))<=1e-6);
z=gp(f.hyp,@infGaussLik,mf,cf,@likGauss,xs,y);assert(abs(z-f.nlZ)<=1e-6);
if ~isempty(parent),assert(f.nlZ<=parent.nlZ+1e-6,'holdout:nesting','Nested mean lost parent feasible point');end
end
function n=mean_count_(mi,d)
v=[0,1,1+d,1+2*d];n=v(mi);
end
function sources=source_manifest_(here,assets)
names={'evaluate_expr_aic_v2','polish_fit_v2','eval_geometry_v2','build_covfunc_v2','init_cov_hyp_v2', ...
    'covLINhOffset','covRQprodV2','build_covfunc_v1','init_cov_hyp_v1','parse_expr_str','kernel_grammar', ...
    'legal_candidates_v1','describe_dataset','setup_revision_paths','gp','minimize','infGaussLik','apx', ...
    'solve_chol','sq_dist','likGauss','meanZero','meanConst','meanLinear','meanPoly','meanSum', ...
    'covSum','covProd','covLINard','covScale','covPER','covSEard','covSEiso','covSE','covMaha', ...
    'kernel_primitive_cell_v1','covMatProdU','covRQprod','covRQprodU','covSEu','covPERu'};
files={fullfile(here,'run_holdout_case.m'),assets};
for i=1:numel(names),p=which(names{i});assert(~isempty(p));files{end+1}=p;end %#ok<AGROW>
files=sort(unique(files));sources=struct('path',{},'sha256',{});
for i=1:numel(files),sources(i)=struct('path',files{i},'sha256',file_hash_(files{i}));end
end
function verify_sources_(s)
for i=1:numel(s),assert(strcmp(file_hash_(s(i).path),s(i).sha256),'holdout:source','Frozen source changed');end
end
function h=file_hash_(p)
f=fopen(p,'rb');assert(f>=0);c=onCleanup(@()fclose(f)); %#ok<NASGU>
h=bytes_hash_(fread(f,inf,'*uint8'));
end
function h=array_hash_(x,y)
h=bytes_hash_([typecast(double([size(x),size(y)]),'uint8')';typecast(x(:),'uint8');typecast(y(:),'uint8')]);
end
function h=bytes_hash_(b)
m=java.security.MessageDigest.getInstance('SHA-256');m.update(typecast(uint8(b(:)),'int8'));
h=lower(reshape(dec2hex(typecast(m.digest(),'uint8'),2)',1,[]));
end
function save_once_(p,value)
assert(~isfile(p),'holdout:immutable','Refusing to overwrite a checkpoint');
t=[tempname(fileparts(p)) '.partial.mat'];save(t,'-struct','value','-v7.3');
assert(~isfile(p));[ok,msg]=movefile(t,p);assert(ok,msg);
end
function write_once_(p,value)
if isfile(p),assert(isequaln(jsondecode(fileread(p)),jsondecode(jsonencode(value))),'holdout:json','Existing output differs');return;end
t=[tempname(fileparts(p)) '.partial.json'];f=fopen(t,'w');assert(f>=0);
fwrite(f,unicode2native(jsonencode(value,'PrettyPrint',true),'UTF-8'),'uint8');fclose(f);
assert(~isfile(p));[ok,msg]=movefile(t,p);assert(ok,msg);
end
