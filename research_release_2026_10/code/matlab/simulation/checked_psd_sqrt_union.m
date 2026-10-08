function [S,K0,report] = checked_psd_sqrt_union(A)
%CHECKED_PSD_SQRT_UNION Pure checked factorization for the union-grid prototype.
%   [S,K0,report] = checked_psd_sqrt_union(A) returns K0=S*S'.
%   Significant negative spectrum is rejected; only dimension-scaled
%   roundoff negativity is clipped. No jitter, RNG, fitting, or I/O occurs.
%   The spectral bound and four report fields are unchanged from the
%   original prototype. An input symmetry gate prevents symmetrization
%   from silently concealing a non-covariance matrix.

assert(isa(A,'double') && isreal(A),'union:factorType', ...
    'The covariance factor must be a real double matrix');
assert(ismatrix(A) && ~isempty(A) && size(A,1)==size(A,2), ...
    'union:factorShape','The covariance factor must be a nonempty square matrix');
assert(all(isfinite(A(:))),'union:factorNonfinite', ...
    'The covariance factor contains a nonfinite entry');
n=size(A,1);
symmetry_bound=128*eps*n*max(norm(A,'fro'),realmin);
assert(norm(A-A','fro')<=symmetry_bound,'union:factorAsymmetry', ...
    'Asymmetry exceeds a dimension-scaled roundoff bound');
A=(A+A')/2;
assert(all(isfinite(A(:))),'union:factorNonfinite', ...
    'Symmetrization produced a nonfinite entry');
[Q,E]=eig(A); ev=diag(E); spectral_scale=max(abs(ev));
assert(all(isfinite(ev)) && all(isfinite(Q(:))), ...
    'union:factorNonfinite','Eigendecomposition produced a nonfinite result');
bound=128*eps*n*max(spectral_scale,realmin);
assert(min(ev)>=-bound,'union:negativeSpectrum', ...
    'Negative eigenvalue exceeds a dimension-scaled roundoff bound');
S=Q.*sqrt(max(ev,0)).';
K0=S*S';
report=struct('min_eigenvalue',min(ev), ...
    'negative_roundoff_bound',bound, ...
    'negative_mass',sum(max(-ev,0)), ...
    'relative_reconstruction_error',norm(K0-A,'fro')/max(norm(A,'fro'),realmin));
end
