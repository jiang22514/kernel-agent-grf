# GPML dependency

The numerical experiments used the GPML MATLAB toolbox **version 4.2, 2018-06-11**. Obtain the official distribution from the GPML project and place its contents under `third_party/gpml/`. The expected main entry is `third_party/gpml/gp.m`.

Project homepage recorded by that distribution: http://gaussianprocess.org/gpml/code

`GPML_Copyright.txt` reproduces the toolbox authors' FreeBSD license. The toolbox itself is not bundled here. Avoid adding project scripts to the toolbox directory because that can change MATLAB path precedence. `setup_revision_paths` adds GPML first and the project's current scientific functions afterwards.

The fitter requires `gp`, `minimize`, `infGaussLik`, `apx`, `solve_chol`, `sq_dist`, `likGauss`, `meanZero`, `meanConst`, `meanLinear`, `meanPoly`, `meanSum`, `covSum`, `covProd`, `covLINard`, `covScale`, `covPER`, `covSEard`, `covSEiso`, `covSE`, `covMaha` and their toolbox dependencies. Install the complete official GPML toolbox rather than manually copying this list.

Offline score/search verification uses only Python's standard library. Synthetic regeneration needs NumPy and SciPy; plotting also needs Matplotlib. None of these steps needs GPU packages or a paid model API.
