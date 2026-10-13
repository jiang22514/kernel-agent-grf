function [model,data,index]=matlab_load_delft(model_id)
%MATLAB_LOAD_DELFT Load a whitelisted saved fit and public training observations.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
index=jsondecode(fileread(fullfile(root,'models','matlab_models.json')));
assert(strcmp(index.schema_version,'delft_portable_models_v1')&&strcmp(index.status,'COMPLETE'));
assert(ischar(model_id)||isstring(model_id));model_id=char(model_id);
assert(~isempty(regexp(model_id,'^k[0-9]{3}_m[1-4]$','once')));
j=find(strcmp({index.models.model_id},model_id));assert(numel(j)==1,'delftRelease:model','Unknown model id');
R=index.models(j);expected=['models/matlab_model_' model_id '.mat'];
assert(strcmp(R.model_file,expected)&&strcmp(index.data_file,'data/main_12.mat'));
file=fullfile(root,expected);data_file=fullfile(root,index.data_file);
assert(strcmp(hash_(file),R.sha256)&&strcmp(hash_(data_file),index.data_sha256));
C=load(file,'model');model=C.model;data=load(data_file);
assert(strcmp(model.schema_version,'delft_portable_fit_v1')&&strcmp(model.model_id,model_id));
assert(strcmp(model.response_scale,'log')&&strcmp(model.kernel,R.kernel)&&model.mean_id==R.mean_id);
assert(all(data.y(:)>0)&&max(abs(log(data.y(:))-data.y_log(:)))<1e-12);
[~,~,~,~,g]=eval_geometry_v2(model.kernel,model.mean_id,data.x);
assert(isequaln(g,model.geometry)&&size(data.x,1)==model.n_train);
end
function h=hash_(p)
f=fopen(p,'rb');assert(f>=0);c=onCleanup(@()fclose(f));b=fread(f,inf,'*uint8');
m=java.security.MessageDigest.getInstance('SHA-256');m.update(typecast(b(:),'int8'));
h=lower(reshape(dec2hex(typecast(m.digest(),'uint8'),2)',1,[]));
end
