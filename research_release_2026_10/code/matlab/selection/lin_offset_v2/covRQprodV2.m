function [K,dK]=covRQprodV2(hyp,x,z)
%COVRQPRODV2 Same RQ covariance; correct the legacy diagonal sf derivative.
% Exact-GP training uses the unchanged full-matrix derivative.
if nargin<2, K='(D+2)'; return; end
if nargin<3, z=[]; end
if nargout>1
    [K,dK]=covRQprod(hyp,x,z);
    if ischar(z)&&strcmp(z,'diag')
        D=size(x,2); sf2=exp(2*hyp(D+1));
        dK=@(Q) [zeros(D,1);2*sf2*sum(Q(:));0];
    end
else
    K=covRQprod(hyp,x,z);
end
end
