function root=matlab_setup_delft(gpml_root)
%MATLAB_SETUP_DELFT Relative package paths plus external, hash-identified GPML.
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(here));
if nargin<1,gpml_root=fullfile(root,'third_party','gpml');end
gpml_root=char(java.io.File(char(gpml_root)).getCanonicalPath());
assert(isfile(fullfile(gpml_root,'Copyright'))&&isfolder(fullfile(gpml_root,'cov')), ...
    'delftRelease:gpml','Place compatible GPML in third_party/gpml or pass its directory');
dependencies=jsondecode(fileread(fullfile(root,'third_party','GPML_dependencies.json')));
for j=1:numel(dependencies.files)
    r=dependencies.files(j);file=fullfile(gpml_root,r.path);
    assert(isfile(file)&&strcmp(hash_(file),r.sha256),'delftRelease:gpmlVersion', ...
        'GPML source differs from the recorded dependency: %s',r.path);
end
addpath(gpml_root);
for d={'cov','mean','util','inf','lik'},addpath(fullfile(gpml_root,d{1}));end
for d={'simulation','selection','selection/kernels_v1','selection/lin_offset_v2'}
    addpath(fullfile(here,d{1}));
end
names={'build_covfunc_v1','build_covfunc_v2','eval_geometry_v2','predict_gp_v2', ...
    'union_gp_fields_v2','expand_expr_terms','checked_psd_sqrt_union', ...
    'evaluate_expr_aic_v2','polish_fit_v2','init_cov_hyp_v1','init_cov_hyp_v2','kernel_grammar'};
for j=1:numel(names)
    actual=which(names{j});assert(startsWith(actual,[here filesep]), ...
        'delftRelease:shadow','Project function is shadowed: %s',names{j});
end
for j=1:numel(dependencies.files)
    r=dependencies.files(j);[~,name]=fileparts(r.path);actual=which(name);
    expected=char(java.io.File(fullfile(gpml_root,r.path)).getCanonicalPath());
    assert(strcmp(actual,expected),'delftRelease:shadow','GPML function is shadowed: %s',name);
end
end
function h=hash_(p)
f=fopen(p,'rb');assert(f>=0);c=onCleanup(@()fclose(f));b=fread(f,inf,'*uint8');
m=java.security.MessageDigest.getInstance('SHA-256');m.update(typecast(b(:),'int8'));
h=lower(reshape(dec2hex(typecast(m.digest(),'uint8'),2)',1,[]));
end
