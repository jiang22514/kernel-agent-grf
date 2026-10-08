function report=test_noise_log_stress_v2()
%TEST_NOISE_LOG_STRESS_V2 Tiny synthetic test; does not load CPT or score tables.
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(fileparts(here)));
addpath(genpath(fullfile(root,'gpml','gpml-matlab-v4.2-2018-06-11')));
addpath(genpath(fullfile(root,'kernel-agent-grf','simulation')));
addpath(genpath(fullfile(root,'kernel-agent-grf','selection'))); addpath(here);
maxNumCompThreads(1);
[h,v]=ndgrid([12,27,49],[-7,-3,1,4,8,13]); x=[h(:),v(:)];
y=4.6+0.28*sin(x(:,1)/11)+0.10*cos(x(:,2)/3)+0.025*sin((1:size(x,1))');
models={'SE',4;'RQ',2;'(MA1*LIN)+PER',2}; scenarios={'sigma010','sigma020','log_free'};
max_mu_error=0; max_variance_error=0; count=0;
for m=1:3
    for s=1:3
        opts=struct('n_restart',1,'n_iter',-15,'polish_iter',-20,'seed',8180+10*m+s);
        u=fit_noise_log_stress_v2(models{m,1},models{m,2},x,y,scenarios{s},opts);
        p=noise_log_expectation_v2(u,x,y,[7,5]);
        assert(size(p.query_coordinates_m,1)==35 && abs(sum(p.trapezoid_normalized_weights)-1)<1e-14);
        assert(p.mean_weak_fraction>=0 && p.mean_weak_fraction<=1);
        assert(u.fit.nlZ<=u.stage1.nlZ && all(p.observation_variance_fitted_scale>=p.latent_variance_fitted_scale));
        assert(max(abs(p.observation_variance_fitted_scale-p.latent_variance_fitted_scale-exp(2*u.fit.hyp.lik)))<1e-12);
        [xs,cf,mf,ncov,g]=eval_geometry_v2(u.expr,u.mean_id,x);
        q=(p.query_coordinates_m-g.offset)./g.scale; response=y;
        if s==3, response=log(y); end
        [~,~,mu,variance]=gp(u.fit.hyp,@infGaussLik,mf,cf,@likGauss,xs,response,q);
        max_mu_error=max(max_mu_error,max(abs(mu-p.mean_fitted_scale)));
        max_variance_error=max(max_variance_error,max(abs(variance-p.latent_variance_fitted_scale)));
        assert(u.fit.k==ncov+numel(u.fit.hyp.mean)+double(s==3));
        if s==3
            assert(abs(u.aic_raw_MPa_scale-u.fit.aic-2*sum(log(y)))<1e-12);
            assert(p.threshold_on_fitted_scale==log(4.14));
        else
            assert(abs(u.noise_sd-[0.1,0.2]*(s==[1,2])')<1e-12);
            assert(p.threshold_on_fitted_scale==4.14 && u.jacobian_nlZ==0);
        end
        count=count+1;
    end
end
% Independent change-of-variable check for one positive observation.
q=3.7; mu_log=1.3; v_log=0.08;
nlz_log=0.5*log(2*pi*v_log)+0.5*(log(q)-mu_log)^2/v_log;
lognormal_density=exp(-(log(q)-mu_log)^2/(2*v_log))/(q*sqrt(2*pi*v_log));
assert(abs(-log(lognormal_density)-(nlz_log+log(q)))<1e-14);
assert(max_mu_error<1e-8 && max_variance_error<1e-8);
report=struct('passed',true,'synthetic_observations',numel(y),'models',3,'scenarios',3, ...
    'completed_units',count,'grid_nodes',35,'max_mean_error_vs_gpml',max_mu_error, ...
    'max_latent_variance_error_vs_gpml',max_variance_error, ...
    'production_data_loaded',false,'production_experiment_started',false, ...
    'test_budget','1x15 + best20 per synthetic unit, not the production protocol');
out=fullfile(root,'output','revision','review_fixes_2026-10-06','lin_v2','sensitivity');
if ~isfolder(out), mkdir(out); end
fid=fopen(fullfile(out,'synthetic_preflight.json'),'w'); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(report,'PrettyPrint',true));
disp(report);
end
