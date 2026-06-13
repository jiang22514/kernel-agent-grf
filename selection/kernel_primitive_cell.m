function [c, label] = kernel_primitive_cell(tok)
%KERNEL_PRIMITIVE_CELL  Map a primitive token to a GPML covariance cell.
%
%   [c, label] = kernel_primitive_cell(tok)
%
%   Returns the GPML covariance specification cell `c` (ARD form, so it works
%   for arbitrary input dimensionality D) and a human-readable `label`.
%
%   Supported tokens:
%     'SE'  squared-exponential / Gaussian (covSEard)         nu -> inf
%     'MA1' Matern 1/2 (Markovian / exponential)              (covMaternard,1)
%     'MA3' Matern 3/2                                          (covMaternard,3)
%     'MA5' Matern 5/2                                          (covMaternard,5)
%     'RQ'  rational quadratic (covRQard)
%     'LIN' linear (covLINard)
%     'PER' periodic: SE base under isotropic periodic embedding,
%           {@covPER,'iso',{@covSEiso}} -- the textbook periodic kernel,
%           valid for any input dimensionality (3 hyperparameters).
%
%   See also: kernel_grammar, build_covfunc_from_expr

    switch upper(tok)
        case 'SE',  c = {@covSEard};        label = 'Gaussian (SE)';
        case 'MA1', c = {@covMaternard, 1}; label = 'Matern(1/2)';
        case 'MA3', c = {@covMaternard, 3}; label = 'Matern(3/2)';
        case 'MA5', c = {@covMaternard, 5}; label = 'Matern(5/2)';
        case 'RQ',  c = {@covRQard};        label = 'Rational Quadratic';
        case 'LIN', c = {@covLINard};       label = 'Linear';
        case 'PER', c = {@covPER, 'iso', {@covSEiso}}; label = 'Periodic';
        otherwise
            error('kernel_primitive_cell:unknown', ...
                  'Unknown primitive token: %s', tok);
    end
end
