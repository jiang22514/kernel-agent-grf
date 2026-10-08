function [fit,polish]=polish_fit_v2(expr,mi,raw_fit,x,y,max_evals)
%POLISH_FIT_V2 Uniform continuation from the best verified fitted point.
% Keeps the previous feasible solution if continuation fails or is worse.
% Fixed noise is neither optimized nor included in the gradient norm.
fit=raw_fit;
[xs,cf,mf,~,g]=eval_geometry_v2(expr,mi,x);
assert(isequaln(g,fit.geometry)&&fit.ok,'linOffset:polish','Incompatible or invalid fit');
sn=[]; h=fit.hyp;
if ~fit.free_noise, sn=h.lik; h=rmfield(h,'lik'); end
obj=@(hh) objective(hh,sn,mf,cf,xs,y(:));
[before,db]=obj(h);
assert(abs(before-fit.nlZ)<1e-6,'linOffset:polishScore','Best-point likelihood mismatch');
polish=struct('initial_hyp',h,'hyp',h,'before_nlZ',before,'after_nlZ',before, ...
    'initial_gradient_norm',gradient_norm(db),'final_gradient_norm',gradient_norm(db), ...
    'max_evaluations',max_evals,'evaluations',0,'accepted',false,'ok',false,'error','','elapsed_seconds',0);
timer=tic;
try
    evalc('[hh,fx,it]=minimize(h,obj,-max_evals);');
    [after,da]=obj(hh); polish.hyp=hh; polish.after_nlZ=after;
    polish.final_gradient_norm=gradient_norm(da); polish.evaluations=it;
    polish.ok=isfinite(after)&&isreal(after)&&isfinite(polish.final_gradient_norm);
    if polish.ok&&after<before
        fit.hyp=hh; if ~fit.free_noise, fit.hyp.lik=sn; end
        fit.nlZ=after; fit.aic=2*fit.k+2*after; fit.bic=fit.k*log(fit.n)+2*after;
        fit.best_source='best_point_continuation'; fit.seed_used=false; fit.floor_used=false;
        fit.status='optimized'; polish.accepted=true;
    end
catch ME
    polish.error=ME.message;
end
polish.elapsed_seconds=toc(timer);
[checked,dchecked]=obj(strip_fixed(fit.hyp,fit.free_noise));
assert(abs(checked-fit.nlZ)<1e-6&&fit.nlZ<=raw_fit.nlZ+1e-8,'linOffset:polishScore','Continuation lost a valid better point');
fit.final_gradient_norm=gradient_norm(dchecked);
fit.polish=polish;
fit.protocol='scoring_lin_offset_v2_15x200_plus_best500';
end
function h=strip_fixed(h,free)
if ~free, h=rmfield(h,'lik'); end
end
function [nlz,d]=objective(h,sn,mf,cf,x,y)
if ~isempty(sn), h.lik=sn; end
[nlz,d]=gp(h,@infGaussLik,mf,cf,@likGauss,x,y);
if ~isempty(sn), d=rmfield(d,'lik'); end
end
function n=gradient_norm(d)
v=[d.mean(:);d.cov(:)];
if isfield(d,'lik'), v=[v;d.lik(:)]; end
n=norm(v);
end
