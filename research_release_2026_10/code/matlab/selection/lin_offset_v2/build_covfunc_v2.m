function [cf,nh,offset_slot]=build_covfunc_v2(expr,D)
%BUILD_COVFUNC_V2 v1 grammar and amplitudes, LIN gains one real offset.
cf=build_covfunc_v1(expr,D);
[cf,nh,offset_slot]=replace(cf,D);
assert(nh==eval(feval(cf{:})),'linOffset:count','Compiler count mismatch');
end
function [out,n,slot]=replace(in,D)
out=in; name=func2str(in{1}); slot=0;
if strcmp(name,'covLINard')
    out={@covLINhOffset}; n=D+1; slot=n;
elseif strcmp(name,'covRQprod')
    out={@covRQprodV2}; n=D+2;
elseif any(strcmp(name,{'covSum','covProd'}))
    n=0;
    for j=1:numel(in{2})
        [out{2}{j},nj,sj]=replace(in{2}{j},D);
        if sj>0
            assert(slot==0,'linOffset:duplicate','Grammar allows LIN only once');
            slot=n+sj;
        end
        n=n+nj;
    end
else
    n=eval(feval(in{:}));
end
end
