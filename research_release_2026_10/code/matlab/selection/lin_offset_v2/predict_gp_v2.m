function p=predict_gp_v2(expr,mean_id,fit,x,y,xq)
%PREDICT_GP_V2 Exact latent and observation predictions with saved geometry.
[xs,cf,mf,~,g]=eval_geometry_v2(expr,mean_id,x);
assert(isequaln(g,fit.geometry),'linOffset:geometry','Fit and prediction geometries differ');
q=(xq-g.offset)./g.scale; h=fit.hyp; y=y(:);
K=feval(cf{:},h.cov,xs); sn2=exp(2*h.lik);
L=chol((K+K')/2+sn2*eye(numel(y)),'lower');
alpha=L'\(L\(y-feval(mf{:},h.mean,xs)));
mu=zeros(size(q,1),1); var_latent=mu;
for start=1:2000:size(q,1)
    ii=start:min(start+1999,size(q,1));
    Q=feval(cf{:},h.cov,q(ii,:),xs); A=L\Q';
    mu(ii)=feval(mf{:},h.mean,q(ii,:))+Q*alpha;
    var_latent(ii)=feval(cf{:},h.cov,q(ii,:),'diag')-sum(A.^2,1)';
end
assert(all(isfinite(mu))&&all(isfinite(var_latent))&&all(var_latent>0), ...
    'linOffset:variance','Exact latent variance must be positive; no clipping is applied');
p=struct('mean',mu,'latent_variance',var_latent,'observation_variance',var_latent+sn2, ...
    'noise_variance',sn2,'geometry',g,'kernel_version','lin_horizontal_offset_v2');
end
