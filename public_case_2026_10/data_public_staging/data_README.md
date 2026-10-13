# Reproduce the public Leendert de Boerspolder CPT cases

This package rebuilds four fixed two-dimensional CPT cases from the original public `data.zip`. No GP code, model fitting, API calls, or private Wuhan data are included. It works from any directory and does not require the authors' drive or project layout. Python 3.10 or later, NumPy, and SciPy are required; `data_requirements.txt` lists minimum versions. Use an isolated Python environment if installation is necessary.

## Run

From this package directory:

```text
python data_rebuild.py --archive source/data.zip --output rebuilt
python data_verify.py --data rebuilt
```

The output directory must be new or empty; existing research outputs are never overwritten. `--archive` and `--output` can be absolute paths. The frozen adjacent `data_preparation_config.json` is used by default. Retain that configuration to reproduce this study. The bundled archive is the unchanged official release (32,925,310 bytes; approximately 32.9 MB).

Alternatively, obtain **data.zip** from the dataset [DOI landing page](https://doi.org/10.4121/uuid:0a9b77bc-3f6d-40e2-846c-ddb992dd78a6), and pass its local path to `--archive`. The originally archived file URL is [Figshare file 24153797](https://ndownloader.figshare.com/files/24153797). The scripts do not download data. The required SHA-256 is:

```text
432d0b747d68c2141249d76895157ed0b61f4b1b0545770d9a209c1576216865
```

The dataset is by Tom de Gast, Michael A. Hicks and Philip Vardon (2020), published by 4TU.ResearchData under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). `source/data_source_license.json` preserves the existing DOI, author, file and license metadata. The official report is retained inside the unmodified ZIP. The site is near Leiden, the Netherlands; “Delft” in internal case naming identifies the source institution, not the site city. CC0 here describes the source dataset; this note does not assign a new license to software.

## Fixed data definition

- Crest Zone 1, Line 7: CPT **34–44 and 69–86**, 29 complete sounding identities. Official raw member `Raw_data/608312_DKP(200+CPT_ID).GEF` contains each sounding.
- Use original GEF column 2 for cone resistance `qc` in MPa. Local coordinates are the official `XID_storage.mat` and `YID_storage.mat` values. Elevation is **GEF surface NAP elevation minus original penetration length**; MAT column 3 is already elevation and is checked against GEF. No added inclination correction is applied.
- Physical window: local X **0 to 49.993907628797416 m**, NAP elevation **−12.0 to −0.6 m**. Select the nearest original measurement to each 0.2 m or 0.1 m NAP grid target. The chosen points coincide with targets up to floating-point roundoff; there is no interpolation, interval averaging, smoothing, imputation, or response-based sample filter.
- Original `−9999` values are retained in raw fields and flagged/represented by NaN in cleaned fields. Nonpositive raw values remain flagged in source/common-domain exports. All points on both fixed study grids are positive. The four nonpositive qc values in the original common domain fall outside the declared grid window; they were not removed by a fitted-model rule.
- The frozen modeled response is **`ln(qc / 1 MPa)`**, chosen before fitting because qc is positive and right-skewed. `y` and `y_test` stay in MPa; fit with `y_log` and `y_test_log`. No offset or epsilon is added. `data_response_decision.json` records this decision; older raw-field metadata does not override it.

The X-sorted CPT IDs are:

```text
44,78,43,77,42,76,41,86,75,85,40,84,74,83,39,82,73,81,38,80,72,79,37,71,36,70,35,69,34
```

The fixed test CPT IDs, in X order, are:

```text
78,77,41,75,84,83,73,38,79,71,35
```

Their 1-based X ranks are `2,4,7,9,12,14,17,19,22,24,27`. The remaining 18 training CPTs are nested using coordinates alone: leftmost, rightmost, then the point maximizing distance to its nearest selected training CPT; ties use smaller X. The selection order is:

```text
44,34,39,76,36,72,85,43,81,70,74,37,42,69,86,40,80,82
```

| Case | Training CPTs | Grid spacing | Train/test points |
|---|---:|---:|---:|
| `main_12` | First 12 | 0.2 m | 696 / 638 |
| `density_06` | First 6 | 0.2 m | 348 / 638 |
| `density_18` | All 18 | 0.2 m | 1044 / 638 |
| `fine_12` | First 12 | 0.1 m | 1380 / 1265 |

Test identities never overlap training identities. Test qc must not enter fitting, model ranking, LLM search context, or candidate revision. The 0.2 m density cases use the same 11 test CPTs; the 0.1 m case uses the same identities and physical window at the finer spacing.

## Outputs and validation

Each case is written as MAT, NPZ and JSON. Arrays include `x`/`x_raw = [local_X_m, z_NAP_m]`, `y` (MPa), `y_log`, `sounding_id`/`train_id`, and corresponding `x_test`, `y_test`, `y_test_log`, `test_sounding_id`/`test_id`. Source GEF data-row indices are `source_row_1based` and `source_row_1based_test`; they count data rows after `#EOH`, including the initial placeholder row. Rows are ordered by increasing local X, then increasing NAP elevation. `x_query` contains geometry for all 29 registered positions and no responses.

`source_all_rows.csv`, `clean_common_long.csv`, `sampled_0p20m_long.csv`, and `sampled_0p10m_long.csv` preserve the source member, CPT ID, raw row, original value, status flags, target height and actual offset. Grid MAT/NPZ files and the data manifest are also generated. The source common domain is larger than the declared analysis window.

`data_verify.py` checks **every numerical array in both rebuilt MAT and NPZ files** against the unchanged four public reference NPZs, and writes `rebuilt/rebuild_verification.json`. The default requires exact array equality. MAT container bytes can differ because SciPy writes a creation timestamp; a changed file hash alone is not evidence of changed numerical data. If a different platform's logarithm implementation differs at the last bit, explicitly request `--log-atol 1e-14`; only `y_log`/`y_test_log` then receive that tolerance, and the report still states whether each array is exact. Raw observations, coordinates and source-row indices remain exact.

`data_package_manifest.json` lists checksums for package inputs. `original_sources/` preserves the original preparation, reader/diagnostic, and coordinate-selection scripts unchanged for provenance. Run the portable `data_rebuild.py` entry point above; its path handling and deferred plotting imports are the only operational adaptations. The optional original diagnostic plotting path requires Matplotlib; data reconstruction does not.
