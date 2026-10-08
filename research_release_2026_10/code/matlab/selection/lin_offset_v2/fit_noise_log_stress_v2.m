function unit=fit_noise_log_stress_v2(expr,mean_id,x,y_MPa,scenario,options)
%FIT_NOISE_LOG_STRESS_V2 Independent fit under one explicit stress scenario.
% Production runner fixes 15x200 plus one best-point continuation of 500
% objective evaluations. Smaller budgets are used only by the synthetic test.
% This helper does not change frozen core functions or archived experiments.
if nargin<6, options=struct(); end
if ~isfield(options,'n_restart'), options.n_restart=15; end
if ~isfield(options,'n_iter'), options.n_iter=-200; end
if ~isfield(options,'polish_iter'), options.polish_iter=-500; end
if ~isfield(options,'seed'), options.seed=107600; end
assert(options.n_iter<0 && options.polish_iter<0 && options.n_restart>=1);
assert(ismember(scenario,{'sigma010','sigma020','log_free'}));
y_MPa=y_MPa(:); assert(all(isfinite(y_MPa)) && all(y_MPa>0));
islog=strcmp(scenario,'log_free'); response=y_MPa;
ev=struct('n_restart',options.n_restart,'n_iter',options.n_iter, ...
    'jitter_sd',0.5,'quiet',true,'seed',options.seed);
fixed_sn=[];
if islog
    response=log(y_MPa); % numerical y is q_c / MPa
    assert(~isfield(options,'raw_seed'),'stressV2:logSeed','Do not use raw-scale seed in a log-response fit');
elseif strcmp(scenario,'sigma010')
    fixed_sn=0.1;
else
    fixed_sn=0.2;
end
if ~islog
    ev.sn_fixed=fixed_sn;
    if isfield(options,'raw_seed'), ev.seed_hyps={options.raw_seed}; end
end
started=tic;
stage1=evaluate_expr_aic_v2(expr,mean_id,x,response,ev);
stage1_seconds=toc(started);
assert(stage1.ok && stage1.n_success>0,'stressV2:failed','No successful first-stage fit');
[xs,cf,mf,ncov,geometry]=eval_geometry_v2(expr,mean_id,x);
assert(isequaln(stage1.geometry,geometry));
expected_k=ncov+numel(stage1.hyp.mean)+double(islog);
assert(stage1.k==expected_k,'stressV2:count','Fixed noise must be excluded from k');
best=stage1; h0=stage1.hyp;
if ~islog, h0=rmfield(h0,'lik'); end
continuation=struct('initial_hyp',h0,'hyp',h0,'ok',false,'accepted',false, ...
    'nlZ',NaN,'iterations',NaN,'objective_trace',[],'error','');
started=tic;
try
    if islog
        objective=@gp; args={@infGaussLik,mf,cf,@likGauss,xs,response};
    else
        objective=@(h,varargin)fixed_objective_(h,log(fixed_sn),mf,cf,xs,response); args={};
    end
    [~]=evalc('[h,fX,it]=minimize(h0,objective,options.polish_iter,args{:});');
    hz=h; if ~islog, hz.lik=log(fixed_sn); end
    z=gp(hz,@infGaussLik,mf,cf,@likGauss,xs,response);
    continuation.hyp=hz; continuation.nlZ=z; continuation.iterations=it; continuation.objective_trace=fX;
    continuation.ok=isfinite(z) && isreal(z);
    if continuation.ok && z<stage1.nlZ
        best.hyp=hz; best.nlZ=z; best.aic=2*best.k+2*z;
        best.bic=best.k*log(numel(response))+2*z;
        best.best_source='best_point_continuation'; continuation.accepted=true;
    end
catch ME
    continuation.error=ME.message;
end
continuation_seconds=toc(started);
assert(best.nlZ<=stage1.nlZ,'stressV2:best','Continuation discarded the valid better point');
direct=gp(best.hyp,@infGaussLik,mf,cf,@likGauss,xs,response);
assert(abs(direct-best.nlZ)<1e-6,'stressV2:objective','Final objective cannot be reproduced');
if ~islog
    assert(abs(exp(best.hyp.lik)-fixed_sn)<1e-12);
    response_unit='MPa'; variance_unit='MPa^2'; noise_unit='MPa';
    jacobian_nlZ=0;
else
    response_unit='log(q_c / MPa)'; variance_unit='log(q_c / MPa)^2';
    noise_unit='dimensionless log standard deviation'; jacobian_nlZ=sum(log(y_MPa));
end
unit=struct('expr',expr,'mean_id',mean_id,'scenario',scenario,'is_log_response',islog, ...
    'kernel_version','lin_horizontal_offset_v2','options',ev,'polish_iter',options.polish_iter, ...
    'stage1',stage1,'continuation',continuation,'fit',best,'geometry',geometry, ...
    'aic_fitted_scale',best.aic,'jacobian_nlZ',jacobian_nlZ, ...
    'aic_raw_MPa_scale',best.aic+2*jacobian_nlZ, ...
    'response_unit',response_unit,'variance_unit',variance_unit,'noise_sd_unit',noise_unit, ...
    'noise_sd',exp(best.hyp.lik),'fixed_noise_sd_MPa',fixed_sn, ...
    'stage1_seconds',stage1_seconds,'continuation_seconds',continuation_seconds, ...
    'scope','Assumption stress test; fixed sigma values are not instrument calibration');
end

function [z,dz]=fixed_objective_(h,logsn,mf,cf,x,y)
h.lik=logsn;
if nargout>1
    [z,dz]=gp(h,@infGaussLik,mf,cf,@likGauss,x,y); dz=rmfield(dz,'lik');
else
    z=gp(h,@infGaussLik,mf,cf,@likGauss,x,y);
end
end
