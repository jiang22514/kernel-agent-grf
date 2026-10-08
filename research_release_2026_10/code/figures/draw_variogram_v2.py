"""Plot FINAL v2 observation-model semivariances, without fitting or sampling."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from common import *
import variogram_style as style


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    sources=Sources();sources.add(__file__);sources.add(style.__file__)
    side=Path(str(a.input)+'.sha256');sources.add(side);sources.add(a.input,side.read_text().strip())
    data=loadmat(a.input,simplify_cells=True);meta=data['meta']
    need(meta['kernel_version']=='lin_horizontal_offset_v2' and meta['kernel_definition']=='lin_first_coordinate_offset_v2','Old model inputs refused')
    need(meta['protocol']=='projected_observation_semivariance_v1' and meta['GP_fits']==meta['new_fields']==0,'Diagnostic definition changed')
    for r in np.atleast_1d(data['source_manifest']):sources.add(r['path'],r['sha256'])
    labels=[str(v) for v in np.atleast_1d(data['labels'])]
    rows,report=style.compute(data['x'],data['y'],data['mu'],data['C'],data['edges'],labels)
    out=output_dir(a.output);csv_write(out/'borehole_variogram_source.csv',rows)
    if not a.validate_only:style.render(rows,labels,out)
    report.update(status='VALIDATED' if a.validate_only else 'RENDERED_PENDING_VISUAL_QA',figures_generated=not a.validate_only,
        kernel_version=meta['kernel_version'],table_sha256=meta['table_sha256'],model_names=labels,
        diagnostic_arithmetic_recomputed=True,covariance_or_mean_refitted=False,
        selection=f"All {len(data['y'])} observations; unique i<j pairs with abs(dx1)<2m; 12 left-closed/right-open bins up to MATLAB 60th lag percentile.")
    (out/'caption.txt').write_text('CPT projected-observation semivariances using the FINAL LIN-v2 SE, RQ and selected models. Observations and model expectations use the identical global linear residual projection, near-vertical point pairs and bins. Model expectations include squared projected fitted-mean differences plus projected observation covariance; observation noise is included once, before projection. No jitter is added. This is a fixed fitted-model diagnostic at observed locations; correlated point pairs are not treated as independent replicates and no confidence interval is inferred.',encoding='utf-8')
    sources.finish(out,report)


if __name__=='__main__':main()
