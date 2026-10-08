function prediction=noise_log_expectation_v2(unit,x,y_MPa,grid_size)
%NOISE_LOG_EXPECTATION_V2 Exact marginal expectation, no field simulation.
% grid_size=[horizontal nodes, vertical nodes], main grid=[201,121].
if nargin<4, grid_size=[201,121]; end
assert(numel(grid_size)==2 && all(grid_size>=2) && all(mod(grid_size,1)==0));
y=y_MPa(:); if unit.is_log_response, y=log(y); end
[xs,cf,mf,~,geometry]=eval_geometry_v2(unit.expr,unit.mean_id,x);
assert(isequaln(geometry,unit.fit.geometry) && isequaln(geometry,unit.geometry));
g1=linspace(min(x(:,1)),max(x(:,1)),grid_size(1));
g2=linspace(min(x(:,2)),max(x(:,2)),grid_size(2));
[a,b]=meshgrid(g1,g2); xq=[a(:),b(:)]; q=(xq-geometry.offset)./geometry.scale;
h=unit.fit.hyp; sn2=exp(2*h.lik);
K=feval(cf{:},h.cov,xs); L=chol((K+K')/2+sn2*eye(numel(y)),'lower');
alpha=L'\(L\(y-feval(mf{:},h.mean,xs)));
mu=zeros(size(q,1),1); raw_variance=mu; prior_diagonal=mu;
for first=1:1500:size(q,1)
    ii=first:min(first+1499,size(q,1));
    Q=feval(cf{:},h.cov,q(ii,:),xs); R=L\Q';
    mu(ii)=feval(mf{:},h.mean,q(ii,:))+Q*alpha;
    prior_diagonal(ii)=feval(cf{:},h.cov,q(ii,:),'diag');
    raw_variance(ii)=prior_diagonal(ii)-sum(R.^2,1)';
end
tol=64*eps*(numel(y)+1)*max(1,max(abs(prior_diagonal)));
assert(all(isfinite(mu)) && all(isfinite(raw_variance)) && all(raw_variance>=-tol), ...
    'stressV2:variance','Nonfinite or materially negative latent variance');
variance=max(raw_variance,0); theta=4.14;
if unit.is_log_response, theta=log(theta); end
probability=0.5*erfc((mu-theta)./sqrt(2*variance));
probability(variance==0)=double(mu(variance==0)<theta);
assert(all(isfinite(probability)) && all(probability>=0 & probability<=1));
w1=ones(1,grid_size(1)); w1([1,end])=0.5;
w2=ones(grid_size(2),1); w2([1,end])=0.5;
W=w2*w1; weights=W(:)/sum(W(:));
fraction=weights'*probability;
domain_area=(g1(end)-g1(1))*(g2(end)-g2(1));
prediction=struct('query_coordinates_m',xq,'grid_size',grid_size,'geometry',geometry, ...
    'mean_fitted_scale',mu,'raw_latent_variance_fitted_scale',raw_variance, ...
    'latent_variance_fitted_scale',variance,'observation_variance_fitted_scale',variance+sn2, ...
    'mean_unit',unit.response_unit,'variance_unit',unit.variance_unit, ...
    'noise_sd',unit.noise_sd,'noise_sd_unit',unit.noise_sd_unit, ...
    'latent_probability_below_4_14_MPa',probability,'trapezoid_normalized_weights',weights, ...
    'mean_weak_fraction',fraction,'mean_weak_area_m2',domain_area*fraction,'domain_area_m2',domain_area, ...
    'threshold_MPa',4.14,'threshold_on_fitted_scale',theta, ...
    'roundoff_variance_tolerance',tol,'roundoff_clipped_count',sum(raw_variance<0), ...
    'conditioning_jitter',0,'is_log_response',unit.is_log_response, ...
    'method','Trapezoidal expectation of latent weak-zone area on the fixed grid; no independent observation noise or field Monte Carlo');
end
