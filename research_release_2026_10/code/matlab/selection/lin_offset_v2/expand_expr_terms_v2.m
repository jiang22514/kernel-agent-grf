function [terms,hypmap]=expand_expr_terms_v2(expr,D,hypvec)
%EXPAND_EXPR_TERMS_V2 Preserve v1 terms, shift only horizontal LIN leaves.
[~,nh,slot]=build_covfunc_v2(expr,D);
hypvec=hypvec(:); assert(numel(hypvec)==nh,'linOffset:count','Wrong hyperparameter count');
v=hypvec; shift=0;
if slot>0, shift=v(slot); v(slot)=[]; end
[terms,hypmap]=expand_expr_terms_v1(expr,D,v);
if slot>0
    for j=1:numel(terms), terms(j).f{1}=replace(terms(j).f{1},shift); end
    hypmap.LIN=[hypmap.LIN;shift];
end
end
function out=replace(in,c)
out=in; name=func2str(in{1});
if strcmp(name,'covLINiso'), out={@covFixedShiftV2,in,c};
elseif any(strcmp(name,{'covSum','covProd'}))
    for j=1:numel(in{2}), out{2}{j}=replace(in{2}{j},c); end
end
end
