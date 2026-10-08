function [c, label] = kernel_primitive_cell_v1(tok)
%KERNEL_PRIMITIVE_CELL_V1  kernels_v1 primitive map (revision plan B1-1/B1-2).
%
%   Direction-product definitions shared with the simulation engine:
%     SE   {@covSEard}                        (already per-dim product)
%     MA1/MA3/MA5  {@covScale,{@covMatProdU,nu2}}   (NEW: replaces covMaternard)
%     RQ   {@covRQprod}                       (NEW: replaces covRQard; [l,sf,al])
%     LIN  {@covLINard}                       (sum over dims; scale = amplitude)
%     PER  {@covPER,'iso',{@covSEiso}}        (unchanged; defined on scaled coords)
%
%   See also: build_covfunc_v1, kernel_grammar

    switch upper(tok)
        case 'SE',  c = {@covSEard};                        label = 'Gaussian (SE)';
        case 'MA1', c = {@covScale,{@covMatProdU,1}};       label = 'Matern(1/2) prod';
        case 'MA3', c = {@covScale,{@covMatProdU,3}};       label = 'Matern(3/2) prod';
        case 'MA5', c = {@covScale,{@covMatProdU,5}};       label = 'Matern(5/2) prod';
        case 'RQ',  c = {@covRQprod};                       label = 'Rational Quadratic prod';
        case 'LIN', c = {@covLINard};                       label = 'Linear';
        case 'PER', c = {@covPER, 'iso', {@covSEiso}};      label = 'Periodic';
        otherwise
            error('kernel_primitive_cell_v1:unknown', 'Unknown primitive token: %s', tok);
    end
end
