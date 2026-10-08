function w = kron_mvm(As, v)
%KRON_MVM  Matrix-vector product with a Kronecker product, without forming it.
%
%   w = kron_mvm({A1,...,AD}, v)  computes  w = (A1 kron A2 kron ... kron AD) v
%
%   Ordering convention: the LAST dimension varies fastest in v, i.e. for
%   D = 2 the linear index is i = (i1-1)*m2 + i2, matching kron(A1,A2) and a
%   point list built as [kron(x1,ones(m2,1)), kron(ones(m1,1),x2)].
%
%   Cost: O(m * sum_d m_d) instead of O(m^2). Standard algorithm (Saatchi 2011).

    D = numel(As);
    ms = zeros(1, D);
    for d = 1:D, ms(d) = size(As{d}, 1); end
    w = v(:);
    for d = D:-1:1
        W = reshape(w, ms(d), []);
        w = reshape((As{d} * W)', [], 1);
    end
end
