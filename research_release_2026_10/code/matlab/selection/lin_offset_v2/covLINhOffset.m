function [K,dK] = covLINhOffset(hyp,x,z)
%COVLINHOFFSET Linear covariance with a fitted horizontal origin.
% hyp = [log(L_1); ...; log(L_D); c_h]. c_h is REAL, not logarithmic.
% k = (x_1-c_h)(z_1-c_h)/L_1^2 + sum_{d>1} x_d z_d/L_d^2.
if nargin<2, K='D+1'; return; end
if nargin<3, z=[]; end
D=size(x,2); hyp=hyp(:);
assert(numel(hyp)==D+1,'linOffset:count','Expected D log scales and one real offset');
dg=ischar(z)&&strcmp(z,'diag'); same=isempty(z);
x(:,1)=x(:,1)-hyp(end);
if dg || same, z=x; else, z(:,1)=z(:,1)-hyp(end); end
b=exp(-2*hyp(1:D));
if dg, K=sum((x.*z).*b',2); else, K=(x.*b')*z'; end
if nargout>1, dK=@(Q) deriv(Q,x,z,b,dg,same); end
end
function [dh,dx]=deriv(Q,x,z,b,dg,same)
D=size(x,2); dh=zeros(D+1,1);
if dg
    for j=1:D, dh(j)=-2*b(j)*sum(Q.*x(:,j).*z(:,j)); end
    dh(end)=-2*b(1)*sum(Q.*x(:,1));
    dx=2*Q.*x.*b';
else
    for j=1:D, dh(j)=-2*b(j)*sum(x(:,j).*(Q*z(:,j))); end
    dh(end)=-b(1)*(sum(Q,2)'*x(:,1)+sum(Q,1)*z(:,1));
    dx=(Q*z).*b';
    if same, dx=dx+(Q'*x).*b'; end
end
end
