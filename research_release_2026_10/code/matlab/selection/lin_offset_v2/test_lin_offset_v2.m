function report=test_lin_offset_v2(out)
% Fast numerical preflight covering every legal expression, not full fitting.
if nargin<1, out=pwd; end
rng(1062026); D=2;
x=[-1.7,0.6;0.2,1.4;1.8,2.3;0.8,0.9]; z=[-0.8,1.1;2.1,1.9;0.4,3.2];
expressions=kernel_grammar(struct('include_t3',true)); assert(numel(expressions)==94);
maperr=0; graderr=0; embeddederr=0; nlin=0;
for i=1:numel(expressions)
    e=expressions(i); [cf,nh,slot]=build_covfunc_v2(e,D);
    [old,nold]=build_covfunc_v1(e,D); v=0.2*randn(nh,1);
    assert(nh==nold+contains(e.name,'LIN')); nlin=nlin+(slot>0);
    for mode=1:3
        switch mode
            case 1, zz=[];
            case 2, zz=z;
            case 3, zz='diag';
        end
        [K,dk]=feval(cf{:},v,x,zz); Q=randn(size(K)); analytic=dk(Q);
        numerical=zeros(nh,1); dt=1e-5;
        for j=1:nh
            v1=v; v2=v; v1(j)=v1(j)+dt; v2(j)=v2(j)-dt;
            delta=(feval(cf{:},v1,x,zz)-feval(cf{:},v2,x,zz))/(2*dt);
            numerical(j)=sum(Q(:).*delta(:));
        end
        ge=norm(analytic-numerical)/max(1,norm(numerical));
        if ge>1e-7, fprintf('GRADIENT %s mode%d error=%.6g\n',e.name,mode,ge); disp([analytic,numerical]); end
        graderr=max(graderr,ge);
        terms=expand_expr_terms_v2(e,D,v); [terms,~]=stable_union_matern_terms_v1(terms);
        T=zeros(size(K));
        for j=1:numel(terms)
            P=ones(size(K));
            for d=1:D
                if isempty(zz)||ischar(zz), zd=zz; else, zd=zz(:,d); end
                P=P.*feval(terms(j).f{d}{:},terms(j).h{d},x(:,d),zd);
            end
            T=T+P;
        end
        maperr=max(maperr,norm(K-T,'fro')/max(1,norm(K,'fro')));
    end
    if slot>0
        v(slot)=0; v0=v; v0(slot)=[];
        embeddederr=max(embeddederr,norm(feval(cf{:},v,x)-feval(old{:},v0,x),'fro'));
    end
end
fprintf('ALL EXPRESSIONS lin=%d gradient=%.4g mapping=%.4g embedding=%.4g\n',nlin,graderr,maperr,embeddederr);
assert(nlin==48 && graderr<1e-7 && maperr<1e-11 && embeddederr<1e-11);
% Old coordinate geometry and all four mean families embed exactly.
physical=[441.2,5.8;477.8,13.7;618.4,8.3;751.9,18.2;687.3,10.4];
meanerr=0; nlzerr=0;
for mi=1:4
    expr='(MA1*LIN)+PER';
    [xo,cfo,mf,nh]=eval_geometry_v1(expr,mi,physical);
    h=struct('cov',zeros(nh,1),'mean',0.13*randn(eval(feval(mf{:})),1),'lik',log(0.3));
    h2=lift_hyp_v1_to_v2(expr,mi,h,physical);
    [xn,cfn,mfn]=eval_geometry_v2(expr,mi,physical);
    meanerr=max(meanerr,max(abs(feval(mf{:},h.mean,xo)-feval(mfn{:},h2.mean,xn))));
    yy=[3.2;4.8;2.5;5.9;4.1];
    a=gp(h,@infGaussLik,mf,cfo,@likGauss,xo,yy);
    b=gp(h2,@infGaussLik,mfn,cfn,@likGauss,xn,yy);
    nlzerr=max(nlzerr,abs(a-b));
end
assert(meanerr<1e-11&&nlzerr<1e-9);
% Fitting invariance with train-derived initialization and identical seeds.
t=linspace(0,1,16)'; xx=[440+320*t,5+13*mod(7*t,1)]; yy=3.7+0.4*sin(8*t)+0.13*cos(13*t);
o=struct('n_restart',3,'n_iter',-50,'seed',1066);
a=evaluate_expr_aic_v2('LIN',2,xx,yy,o);
b=evaluate_expr_aic_v2('LIN',2,xx+[3100,0],yy,o);
fit_delta=abs(a.aic-b.aic); assert(a.ok&&b.ok&&fit_delta<1e-5);
o.sn_fixed=0.2; f=evaluate_expr_aic_v2('LIN',2,xx,yy,o);
assert(f.k==a.k-1 && abs(exp(f.hyp.lik)-0.2)<1e-14);
% Exact conditional covariance, including new LIN offset, on an irregular grid.
[xs,cf,mf]=eval_geometry_v2('(MA1*LIN)+PER',2,physical);
h=struct('cov',[0;0;0.2;-0.1;0.8;0;0;log(0.4)],'mean',3.8,'lik',log(0.2));
[g1,g2]=ndgrid(linspace(-1.5,1.5,7),linspace(1.1,4.5,8)); xq=[g1(:),g2(:)];
[mu,F,info]=union_gp_fields_v2('(MA1*LIN)+PER',h,mf,xs,yy(1:5),xq,struct('n_samples',5,'check_indices',(1:56)','seed',1922));
[~,~,mu_ref,var_ref]=gp(h,@infGaussLik,mf,cf,@likGauss,xs,yy(1:5),xq);
assert(max(abs(mu-mu_ref))<1e-10 && max(abs(info.sd_exact.^2-var_ref))<1e-10);
assert(info.analytic_cov_relative_error<1e-10 && all(isfinite(F(:))));
report=struct('passed',true,'expressions',94,'lin_expressions',nlin,'gradient_relative_error',graderr, ...
    'term_mapping_relative_error',maperr,'old_embedding_error',embeddederr,'mean_embedding_error',meanerr, ...
    'nlz_embedding_error',nlzerr,'fit_translation_aic_difference',fit_delta,'union_covariance_relative_error',info.analytic_cov_relative_error);
if ~isfolder(out), mkdir(out); end
fid=fopen(fullfile(out,'mathematical_preflight.json'),'w'); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
disp(report);
end
