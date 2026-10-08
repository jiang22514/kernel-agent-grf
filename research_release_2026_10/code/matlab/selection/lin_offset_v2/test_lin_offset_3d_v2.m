function report=test_lin_offset_3d_v2(out)
% Three-dimensional synthetic-case coverage, plus reusable non-LIN geometry.
rng(1062027); D=3;
x=[-1.4,0.8,2.1;0.1,1.9,-0.3;1.7,2.2,0.5;0.6,1.3,1.2]; z=[-0.2,1.5,0.7;2.4,0.2,1.9];
exprs=kernel_grammar(struct('include_t3',true)); maperr=0; graderr=0; meanerr=0;
for i=1:numel(exprs)
    [cf,nh]=build_covfunc_v2(exprs(i),D); v=0.15*randn(nh,1);
    [K,dk]=feval(cf{:},v,x,z); Q=randn(size(K)); ga=dk(Q); gn=zeros(nh,1); step=1e-6;
    for j=1:nh
        a=v;b=v;a(j)=a(j)+step;b(j)=b(j)-step;
        d=(feval(cf{:},a,x,z)-feval(cf{:},b,x,z))/(2*step); gn(j)=sum(Q(:).*d(:));
    end
    graderr=max(graderr,norm(ga-gn)/max(1,norm(gn)));
    terms=expand_expr_terms_v2(exprs(i),D,v); [terms,~]=stable_union_matern_terms_v1(terms); T=zeros(size(K));
    for j=1:numel(terms)
        P=ones(size(K));
        for d=1:D, P=P.*feval(terms(j).f{d}{:},terms(j).h{d},x(:,d),z(:,d)); end
        T=T+P;
    end
    maperr=max(maperr,norm(K-T,'fro')/max(1,norm(K,'fro')));
    for mi=1:4
        xr=x+[11,2,3]; [xo,old,mfo,no]=eval_geometry_v1(exprs(i),mi,xr);
        h=struct('cov',zeros(no,1),'mean',0.3*randn(eval(feval(mfo{:})),1),'lik',log(0.2));
        h2=lift_hyp_v1_to_v2(exprs(i),mi,h,xr); [xn,new,mfn]=eval_geometry_v2(exprs(i),mi,xr);
        meanerr=max(meanerr,max(abs(feval(mfo{:},h.mean,xo)-feval(mfn{:},h2.mean,xn))));
        ko=feval(old{:},h.cov,xo); kn=feval(new{:},h2.cov,xn);
        assert(norm(ko-kn,'fro')/max(1,norm(ko,'fro'))<1e-10);
    end
end
assert(maperr<1e-11&&graderr<1e-7&&meanerr<1e-10);
report=struct('passed',true,'dimension',3,'expressions',94,'gradient_error',graderr,'mapping_error',maperr,'mean_embedding_error',meanerr);
fid=fopen(fullfile(out,'three_dimensional_preflight.json'),'w'); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid); disp(report);
end
