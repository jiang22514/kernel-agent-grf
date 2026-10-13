# Public CPT saved-model and selected-refit reproduction

This package contains public Leendert de Boerspolder CPT observations, selected saved Gaussian-process parameters, and the MATLAB source needed to predict, generate conditional latent fields, or refit a selected structure. It does not automatically evaluate the 376-model library. The source dataset is near Leiden, the Netherlands; `Delft` in function names identifies the source institution.

The study used MATLAB R2025b Update 5 with three computational threads. The portable adapters require modern MATLAB with `jsondecode`, implicit array expansion and Java support. No MATLAB license, MATLAB Engine, Python-to-MATLAB bridge or compiled MathWorks component is included. No Octave compatibility claim is made. No additional MathWorks toolbox is required by the documented exact-GP path. A legacy, unused RQ approximation branch in an unchanged source file mentions `lsqnonneg`; these adapters never enable it.

## Dependencies and setup

Run `python matlab_verify_package.py` from this directory to verify file hashes using Python 3.10+ and its standard library. This verifies bytes, not numerical agreement.

GPML dependency identity is recorded in `third_party/GPML_dependencies.json`. The authors' local folder and Copyright label say GPML 4.2, but `gp.m` and `infGaussLik.m` have 2022-04-04 source headers. Therefore a folder name or a generic 4.2 download is insufficient to establish exact dependency identity. Setup checks the recorded SHA-256 of each required `.m` file. By default the build includes only the 29 explicitly listed GPML runtime files and their Copyright in `third_party/gpml`. If the builder explicitly omitted that directory, provide a matching GPML tree yourself; the package does not download or install it. A different GPML version is a separate numerical configuration and should be validated before comparison.

From the package root in MATLAB:

```matlab
addpath(fullfile(pwd,'code','matlab'));
root = matlab_setup_delft();                 % bundled or locally placed GPML
% Alternatively: root = matlab_setup_delft('/your/local/gpml');
index = jsondecode(fileread(fullfile(root,'models','matlab_models.json')));
id = index.roles.astra_r1.model_id;           % prespecified first Astra search
```

The relative package tree can be moved to another drive. All data and model paths in the public index are relative. The fit exporter used an explicit field list; original local checkpoints, absolute author paths, restart logs and application orchestration are not included.

## Predict without refitting

```matlab
p = matlab_predict_delft(id);                % 638 points in 11 held-out CPTs
[model,data] = matlab_load_delft(id);
```

`data.x` has columns `[local_X_m, elevation_NAP_m]`. `data.y` remains cone resistance in MPa; `data.y_log` is the frozen natural logarithm `ln(qc / 1 MPa)`, without an epsilon. Model fitting uses only the 696 training observations from 12 CPTs. The 11 held-out CPTs do not enter fitting or model selection. The original measurement source, sampling choices and data-rebuild instructions are provided in the companion public data package.

`p.log_prediction` is the exact original-GP prediction in log units. For log mean `mu` and variance `v`, the physical mean is `exp(mu + v/2)` and physical standard deviation is `sqrt(expm1(v) * exp(2*mu + v))`. `exp(mu)` is the median. `latent_mean_qc_MPa` uses latent variance; `observation_mean_qc_MPa` and the 90% observation interval also include fitted observation-noise variance. The fitted horizontal LIN offset is a real parameter, not a logarithm. Saved coordinate geometry is reused unchanged at query points.

## Conditional latent fields and low-resistance areas

```matlab
preview = matlab_union_delft(id);            % analytic moments, zero samples
fields = matlab_union_delft(id,500,111026);   % explicit full study ensemble
```

The default grid has 201 horizontal locations by 115 elevations (23,115 query points), over approximately 50 m by 11.4 m. This is a substantial computation; the zero-sample preview still constructs GP conditioning matrices. Query rows follow MATLAB `meshgrid(x_axis,z_axis)` flattened columnwise: 115 elevation rows per horizontal location. Reshape a query vector with `reshape(vector,115,201)`.

`log_fields` and `qc_fields_MPa` have query points in rows and samples in columns. The coordinate-union sampler conditions the original covariance and the fitted observation noise. It adds neither query observation noise nor conditioning jitter. Its `union_info` reports the independent analytic joint-covariance comparison and factor-roundoff diagnostics. The public entry uses the same 16-sample batches, seed, 128-point check selection and two-million-node cap as the study field call. Floating-point libraries and eigenspace bases can affect individual draws across machines even when their Gaussian distribution agrees.

Each log field is exponentiated once to obtain physical qc. Area weights use a tensor product of trapezoidal integration weights over the same domain, including half weights at edges. `low_resistance_area_fraction` has one row per sample and four columns in threshold order `[0.2,0.1,0.3,0.5]` MPa. The equivalent log comparison is `log_field < log(threshold)`. These are descriptive low-resistance areas, not engineering failure probabilities.

Custom query locations and optional positive area weights can be supplied as the fourth and fifth arguments. Use weights defined for that same domain; the adapter does not infer cell areas from an irregular query list.

## Optional refit of the selected structure

```matlab
plan = matlab_refit_delft(id);               % reports work, no fitting
r = matlab_refit_delft(id,struct('mode','run'));
```

This reuses the unchanged `evaluate_expr_aic_v2` and `polish_fit_v2`: 15 restarts, at most 200 objective evaluations per restart, and at most 500 further evaluations from the best verified point. Gaussian observation noise is free. The base seed is `20261011 + 100*kernel_catalog_index + mean_id`; the evaluator adds restart index. The kernel catalog order is the same 94-kernel grammar used by the study. Three computational threads are requested.

For the main-case selection result, the adapter fits the same kernel with means `1..target_mean_id` in order (Zero, Constant, Linear, Quadratic). Every mean retains its neutral initialization; after the first, the polished parent parameters are padded with zero new mean coefficients and supplied as a feasible seed. This matches the formal main-table protocol and can require up to four fits for one selected model. It does not initialize from the released target model's optimized hyperparameters. Raw and polished checkpoints under `results/refits` allow restart after interruption, without replacing any released fit.

For a different public density or resolution case, use the original fixed-structure protocol: target mean only with neutral initialization. Obtain the corresponding MAT from the companion data package, then supply its path explicitly:

```matlab
o = struct('mode','run','fit_role','fixed_structure', ...
    'case_file','/your/public/data/density_06.mat');
r = matlab_refit_delft(id,o);
d = load(o.case_file);
p_new = predict_gp_v2(model.kernel,model.mean_id,r.fit,d.x,d.y_log,d.x_test);
```

These optional refits can take minutes or longer per structure. Differences in MATLAB, BLAS or GPML can change optimizer trajectories; compare both resulting likelihood and predictions rather than promising bit-identical optimization on every machine. A refit result remains separate from the released saved-parameter result.

## Scope, validation and licensing

`matlab_manifest.json` records source hashes and the actual status of relocated MATLAB validation. A COMPLETE build means the declared package was assembled; it does not certify an unperformed cross-machine numerical run. The original scientific functions are copied byte-for-byte; the portable setup, loading and optional refit adapters are separate files. The small independent dense-conditioning reference and matched timing experiment are outside this minimal package; the original sampler's analytic joint-covariance checks remain available in `union_info`.

Project code is under the included MIT `LICENSE`. GPML is separately covered by the retained FreeBSD license in `third_party/GPML_Copyright.txt`; any bundled GPML files retain their original source notices. The public source dataset is by Tom de Gast, Michael A. Hicks and Philip Vardon (2020), [DOI 10.4121/uuid:0a9b77bc-3f6d-40e2-846c-ddb992dd78a6](https://doi.org/10.4121/uuid:0a9b77bc-3f6d-40e2-846c-ddb992dd78a6), under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). `data/source_license.json` preserves its citation and license record. The software license does not replace the dataset or GPML license.
