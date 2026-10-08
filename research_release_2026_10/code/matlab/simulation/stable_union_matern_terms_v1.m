function [out,report] = stable_union_matern_terms_v1(terms)
%STABLE_UNION_MATERN_TERMS_V1 Replace only covMaterniso leaves in term cells.
%   Preserves term order, all nesting/composition, and every hyperparameter.
%   The stable replacement has the SAME two-hyperparameter count. covProd
%   continues to perform its original slicing; no values are remapped here.
assert(isstruct(terms) && all(isfield(terms,{'f','h'})), ...
    'union:termShape','Expected the existing separable-term struct array');
out=terms; counts=[];
for k=1:numel(terms)
    assert(iscell(terms(k).f) && iscell(terms(k).h) && ...
        numel(terms(k).f)==numel(terms(k).h),'union:termShape','Factor/hyperparameter axes must agree');
    for d=1:numel(terms(k).f)
        [out(k).f{d},counts(k,d)]=replace_cell_(terms(k).f{d}); %#ok<AGROW>
        assert(isequaln(out(k).h{d},terms(k).h{d}), ...
            'union:parameterMutation','Leaf replacement must not change hyperparameters');
    end
end
report=struct('replaced_leaf_count',sum(counts(:)), ...
    'replaced_per_term_axis',counts,'hyperparameters_unchanged',true, ...
    'replacement','covMaterniso -> covMatern1dStable_v1', ...
    'distance_implementation','selection-side covMatProdU direct abs(x-z)/ell', ...
    'parameter_count_per_leaf',2);
end

function [out,count]=replace_cell_(in)
assert(iscell(in) && ~isempty(in),'union:factorCell','Each covariance factor must be a nonempty cell');
out=in; count=0; name='';
if isa(in{1},'function_handle'), name=func2str(in{1});
elseif ischar(in{1}), name=in{1}; end
if strcmp(name,'covMaterniso')
    assert(numel(in)==2 && isnumeric(in{2}) && isscalar(in{2}) && any(in{2}==[1,3,5]), ...
        'union:maternLeaf','Unsupported covMaterniso constructor; refusing to change its meaning');
    out{1}=@covMatern1dStable_v1; count=1; return;
end
for j=1:numel(in)
    if iscell(in{j})
        [out{j},c]=replace_cell_(in{j}); count=count+c;
    end
end
end
