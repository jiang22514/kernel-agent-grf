function [mu,F,info] = union_gp_fields_v2(expr,hyp,mf,x,y,xq,opts)
%UNION_GP_FIELDS_V1 Coordinate-union sampling with original-GP conditioning.
%   [mu,F,info] = union_gp_fields_v2(expr,hyp,mf,x,y,xq,opts)
%   A two-dimensional Cartesian union contains every train/query coordinate
%   exactly. Independent separable terms form one shared prior realization;
%   the original Kxx/Kqx and actual observation noise give the Matheron map.
%   There is no interpolation, FITC correction, added query noise, or jitter.
%   Actual factor roundoff is retained in the reported joint covariance.
%
%   opts: n_samples (0), batch_size (16), node_cap (2000000), optional seed,
%   check_indices (at most 512), progress_callback (empty). The callback is
%   called once after each completed batch with its cumulative sample count.
%   It receives no samples, performs no internal checkpoint/hash work, and
%   cannot advance this sampler's RNG state; callback errors still propagate.
%   info.sd_exact is the original GP latent-field marginal standard deviation
%   at every query. Any nonpositive variance is rejected, never floored.

if nargin<7, opts=struct(); end
assert(isstruct(opts) && isscalar(opts),'union:options','opts must be a scalar struct');
assert(ischar(expr) && isrow(expr) && ~isempty(expr), ...
    'union:expression','expr must be a nonempty character row');
assert(isstruct(hyp) && isscalar(hyp) && all(isfield(hyp,{'cov','mean','lik'})), ...
    'union:hyperparameters','hyp must contain cov, mean, and lik');
finite_double_(hyp.cov,'hyp.cov'); finite_double_(hyp.mean,'hyp.mean');
finite_double_(hyp.lik,'hyp.lik');
assert(isvector(hyp.cov) && ~isempty(hyp.cov) && ...
    (isvector(hyp.mean) || isempty(hyp.mean)) && isscalar(hyp.lik), ...
    'union:hyperparameters','cov/mean must be vectors and lik a scalar');
assert(iscell(mf) && ~isempty(mf),'union:meanFunction','mf must be a nonempty GPML function cell');
finite_double_(x,'x'); finite_double_(xq,'xq'); finite_double_(y,'y');
assert(ismatrix(x) && ismatrix(xq) && size(x,2)==2 && size(xq,2)==2 && ...
    ~isempty(x) && ~isempty(xq),'union:D','Nonempty train/query matrices must have two columns');
assert(isvector(y) && size(x,1)==numel(y),'union:observations','y must have one value per training row');
n=size(x,1); nq=size(xq,1); y=y(:);
if ~isfield(opts,'n_samples'), opts.n_samples=0; end
if ~isfield(opts,'batch_size'), opts.batch_size=16; end
if ~isfield(opts,'node_cap'), opts.node_cap=2000000; end
if ~isfield(opts,'progress_callback'), opts.progress_callback=[]; end
if ~isfield(opts,'check_indices')
    opts.check_indices=unique(round(linspace(1,nq,min(25,nq))));
end
integer_scalar_(opts.n_samples,0,'n_samples');
integer_scalar_(opts.batch_size,1,'batch_size');
integer_scalar_(opts.node_cap,1,'node_cap');
assert(isempty(opts.progress_callback) || isa(opts.progress_callback,'function_handle'), ...
    'union:callback','progress_callback must be empty or a function handle');
ci=opts.check_indices;
assert(isnumeric(ci) && isreal(ci) && isvector(ci) && ~isempty(ci) && ...
    all(isfinite(ci(:))) && all(ci(:)==floor(ci(:))) && all(ci(:)>=1 & ci(:)<=nq), ...
    'union:checkIndices','check_indices must contain finite valid integer query indices');
idx=unique(ci(:));
assert(numel(idx)<=512,'union:checkCapacity','Explicit joint check subset exceeds 512 points');
sn=exp(hyp.lik); sn2=sn^2;
assert(isfinite(sn) && isfinite(sn2) && sn>0 && sn2>0, ...
    'union:noise','Observation noise variance must be positive and finite without underflow');
if isfield(opts,'seed')
    integer_scalar_(opts.seed,0,'seed');
    assert(opts.seed<=2^32-1,'union:seed','seed must fit the MATLAB twister seed range');
    rng(opts.seed,'twister');
end
t0=tic;
xg=cell(1,2); ix=zeros(n,2); iq=zeros(nq,2);
for d=1:2
    [xg{d},~,map]=unique([x(:,d);xq(:,d)]);
    ix(:,d)=map(1:n); iq(:,d)=map(n+1:end);
    assert(isequal(xg{d}(ix(:,d)),x(:,d)) && isequal(xg{d}(iq(:,d)),xq(:,d)), ...
        'union:selector','Coordinate union did not exactly reconstruct all input coordinates');
end
ms=cellfun(@numel,xg); m=prod(ms);
assert(isfinite(m) && m<=opts.node_cap,'union:capacity','Union grid exceeds the explicit node cap');
ind_x=(ix(:,1)-1)*ms(2)+ix(:,2);
ind_q=(iq(:,1)-1)*ms(2)+iq(:,2);
terms=expand_expr_terms_v2(expr,2,hyp.cov);
[terms,term_eval_report]=stable_union_matern_terms_v1(terms); nk=numel(terms);
Sq=cell(nk,2); K0=cell(nk,2);
factor_report=struct('min_eigenvalue',{},'negative_roundoff_bound',{}, ...
    'negative_mass',{},'relative_reconstruction_error',{});
for k=1:nk
    for d=1:2
        A=feval(terms(k).f{d}{:},terms(k).h{d},xg{d});
        [Sq{k,d},K0{k,d},factor_report(k,d)]=checked_psd_sqrt_union(A);
    end
end
cf=build_covfunc_v2(parse_expr_str(expr),2);
Kxx=feval(cf{:},hyp.cov,x); finite_double_(Kxx,'Kxx');
assert(isequal(size(Kxx),[n,n]),'union:kernelShape','Unexpected training covariance shape');
Kxx=(Kxx+Kxx')/2; Ky=Kxx+sn2*eye(n); finite_double_(Ky,'Ky');
[L,p]=chol(Ky,'lower');
assert(p==0,'union:conditioning','Original GP conditioning matrix is not positive definite; no jitter applied');
Kqx=feval(cf{:},hyp.cov,xq,x); finite_double_(Kqx,'Kqx');
assert(isequal(size(Kqx),[nq,n]),'union:kernelShape','Unexpected cross covariance shape');
mx=feval(mf{:},hyp.mean,x); mq=feval(mf{:},hyp.mean,xq);
finite_double_(mx,'training mean'); finite_double_(mq,'query mean');
assert(isequal(size(mx),[n,1]) && isequal(size(mq),[nq,1]), ...
    'union:meanShape','Mean functions must return one column per point');
alpha=L'\(L\(y-mx)); mu=mq+Kqx*alpha; finite_double_(mu,'posterior mean');

% Original-GP latent variance, with bounded solve workspace (n by <=2000).
% The tolerance only classifies failures; it never substitutes a variance.
var_exact=zeros(nq,1); variance_tolerance=zeros(nq,1); block_size=2000;
for first=1:block_size:nq
    last=min(nq,first+block_size-1); rows=first:last;
    kd=feval(cf{:},hyp.cov,xq(rows,:),'diag'); finite_double_(kd,'query prior variance');
    assert(isequal(size(kd),[numel(rows),1]),'union:kernelShape','Unexpected prior variance shape');
    v=L\Kqx(rows,:)'; removed=sum(v.^2,1)';
    raw=kd-removed;
    tol=128*eps*n*max(abs(kd)+abs(removed),realmin);
    finite_double_(raw,'posterior variance'); finite_double_(tol,'variance roundoff tolerance');
    if any(raw<=0)
        j=find(raw<=0,1);
        if raw(j)<-tol(j), id='union:negativeVariance'; else, id='union:nonpositiveVariance'; end
        error(id,['Original-GP latent variance is nonpositive at query %d: ' ...
            'variance=%.17g, roundoff tolerance=%.17g; no floor or query noise applied'], ...
            rows(j),raw(j),tol(j));
    end
    var_exact(rows)=raw; variance_tolerance(rows)=tol;
end
sd_exact=sqrt(var_exact);

K0xx=selected_cov_(K0,ix,ix);
K0sx=selected_cov_(K0,iq(idx,:),ix);
K0ss=selected_cov_(K0,iq(idx,:),iq(idx,:));
T=(L'\(L\Kqx(idx,:)'))';
Kss=feval(cf{:},hyp.cov,xq(idx,:)); finite_double_(Kss,'check prior covariance');
Cex=Kss-Kqx(idx,:)*(L'\(L\Kqx(idx,:)'));
Cactual=K0ss-T*K0sx'-K0sx*T'+T*(K0xx+sn2*eye(n))*T';
Cex=(Cex+Cex')/2; Cactual=(Cactual+Cactual')/2;
finite_double_(Cex,'exact check posterior covariance'); finite_double_(Cactual,'actual check posterior covariance');
info=struct('method','exact_coordinate_union_matheron_lin_offset_v2','expr',expr, ...
    'n',n,'n_query',nq,'n_samples',opts.n_samples,'solver','chol', ...
    'conditioning_model','original_gp','conditioning_jitter',0, ...
    'fitc_applied',false,'fitc_alpha',0,'query_observation_noise',false, ...
    'training_noise_sd',sn,'training_noise_variance',sn2, ...
    'union_axis_sizes',ms,'union_nodes',m,'node_cap',opts.node_cap, ...
    'factor_report',factor_report,'term_evaluation',term_eval_report, ...
    'train_selector',ind_x,'query_selector',ind_q, ...
    'check_indices',idx,'check_mu_exact',mu(idx), ...
    'check_cov_exact',Cex,'check_cov_actual',Cactual, ...
    'analytic_cov_relative_error',norm(Cactual-Cex,'fro')/max(norm(Cex,'fro'),realmin), ...
    'prior_train_relative_error',norm(K0xx-Kxx,'fro')/max(norm(Kxx,'fro'),realmin), ...
    'analytic_mean_standardized_rms',0,'sd_exact',sd_exact,'variance_exact',var_exact, ...
    'variance_roundoff_tolerance',variance_tolerance, ...
    'variance_diagnostics',struct('block_size',block_size,'minimum',min(var_exact), ...
        'nonpositive_count',0,'positive_within_roundoff_count',sum(var_exact<=variance_tolerance), ...
        'floor_applied',false,'query_noise_added',false, ...
        'roundoff_tolerance_rule','128*eps*n*(abs(prior variance)+abs(removed variance))', ...
        'nonpositive_policy','reject; classify material negative versus within-roundoff nonpositive'), ...
    't_setup_s',toc(t0));
% The zero mean error is algebraic, not independent evidence of correctness.
F=zeros(nq,opts.n_samples); info.rng_before_sampling=rng; t1=tic;
for first=1:opts.batch_size:opts.n_samples
    last=min(opts.n_samples,first+opts.batch_size-1); nb=last-first+1;
    q0=zeros(nq,nb); rhs=zeros(n,nb);
    for j=1:nb
        u=zeros(m,1);
        for k=1:nk
            u=u+kron_mvm(Sq(k,:),randn(m,1));
        end
        e=sn*randn(n,1);
        q0(:,j)=u(ind_q); rhs(:,j)=y-mx-u(ind_x)-e;
    end
    batch=mq+q0+Kqx*(L'\(L\rhs)); finite_double_(batch,'conditional samples');
    F(:,first:last)=batch;
    if ~isempty(opts.progress_callback)
        notify_progress_(opts.progress_callback,last);
    end
end
info.t_samples_s=toc(t1); info.total_elapsed_s=toc(t0); info.rng_after_sampling=rng;
info.batch_size=opts.batch_size;
info.progress_callback_enabled=~isempty(opts.progress_callback);
info.progress_callback_rng_policy='restore sampler RNG state after callback, also on callback error';
info.scope='Joint prior on the coordinate union, original-GP Matheron map; factor roundoff retained in analytic covariance check';
end

function C=selected_cov_(Fac,ia,ib)
C=zeros(size(ia,1),size(ib,1));
for k=1:size(Fac,1)
    A=Fac{k,1}(ia(:,1),ib(:,1)); B=Fac{k,2}(ia(:,2),ib(:,2));
    C=C+A.*B;
end
end

function finite_double_(v,name)
assert(isa(v,'double') && isreal(v) && all(isfinite(v(:))), ...
    'union:nonfiniteInput','%s must contain only real finite doubles',name);
end

function integer_scalar_(v,lower,name)
assert(isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v) && ...
    v==floor(v) && v>=lower && v<=flintmax, ...
    'union:integerOption','%s must be a finite integer scalar in the supported range',name);
end

function notify_progress_(callback,completed)
state=rng; restore=onCleanup(@()rng(state)); %#ok<NASGU>
callback(completed);
end
