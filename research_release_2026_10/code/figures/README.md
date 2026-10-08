# Manuscript figure source code

`figure_source_map.json` maps all 15 manuscript figure filenames to the actual drawing scripts and required source types. The scripts and style helpers are supplied; no CPT pointwise figure data, conditional-mean grids or field arrays are included.

These are figure-source snapshots rather than a single automatic build of private inputs. Their original provenance checks intentionally require the full source schema. The public safe tables remove account paths, provider content and private inputs, so they cannot masquerade as the original unredacted input object. Use the public tools for directly executable synthetic-data regeneration and search-count/trajectory plots:

```text
python -B tools/reproduce_synthetic.py
python -B tools/plot_search_results.py
```

For the original figure layout, use its plotting functions with the corresponding public data or with source data obtained from the authors. Scripts with command-line `--input`, `--summary`, `--directory` or `--data-dir` accept the data location explicitly. `FIGURE_INPUT_ROOT` replaces the author's former project root; it must point to your source-data tree when a script expects the historical directory layout. Some archival draw scripts also check source manifests; update those paths/hashes to the files you actually received. This changes file location metadata, not scientific values or plotting functions.

Synthetic prediction MATs retain observations, true values, query coordinates, predicted means and displayed indices; private source manifests were removed. For `draw_synthetic_v2.py`, the mapping/format checks in its original loader describe the original metadata, and the drawing layout itself is in `synthetic_style.py`. The safe MAT fields can be mapped into that layout directly. H01–H06 data are synthetic and fully public. The private CPT figure scripts cannot run until the corresponding observations or derived pointwise inputs have been obtained.

Additional optional figure dependencies are `h5py`, `Pillow` and `PyMuPDF` for MATLAB v7.3 reads and PDF preview routines. The release contains drawing source only, not historical QA images or execution logs. The source scripts may create their own QA outputs when explicitly run locally.
