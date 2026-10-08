"""Render new exact-champion predictions and unchanged archived BCS means."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from common import *
import synthetic_style as style


def load_cases(folder,sources):
    data=[]
    for case in CASES[:4]:
        path=folder/(case+'_display.mat');side=Path(str(path)+'.sha256')
        sources.add(side);sources.add(path,side.read_text().strip())
        raw=loadmat(path,simplify_cells=True);meta=raw['meta'];ident=raw['identity']
        need(meta['case_name']==case and meta['kernel_version']=='lin_horizontal_offset_v2','Wrong prediction version')
        need(ident['protocol']=='lin_v2_exact_synthetic_means_reuse_frozen_BCS','Wrong prediction protocol')
        need(not meta['GP_refitted'] and not meta['bcs_retrained'] and not meta['SKI_computed'],'Unexpected prediction route')
        records=raw['source_manifest'];records=[records] if isinstance(records,dict) else list(records)
        for r in records:sources.add(r['path'],r['sha256'])
        dim,ng=(3,20) if case=='EX4' else (2,60)
        truth=np.asarray(raw['truth']).ravel();exact=np.asarray(raw['mu_exact']).ravel();bcs=np.asarray(raw['mu_bcs']).ravel()
        xq=np.asarray(raw['xq']);g=np.asarray(raw['g']).ravel()
        expected=np.column_stack([a.ravel(order='F') for a in np.meshgrid(*([g]*dim),indexing='xy')])
        need(len(g)==ng and np.allclose(xq,expected,rtol=0,atol=1e-14),'Wrong query grid/order')
        need(truth.shape==exact.shape==bcs.shape==(ng**dim,) and np.isfinite([truth,exact,bcs]).all(),'Invalid predictions')
        ix=np.asarray(raw['display_index']).ravel().astype(int)-1
        wanted=np.arange(ng**dim) if dim==2 else np.arange(9*ng*ng,10*ng*ng)
        need(np.array_equal(ix,wanted),'Wrong display slice')
        metrics={}
        for key,value in [('exact',exact),('bcs',bcs)]:
            rmse=float(np.sqrt(np.mean((value-truth)**2)))
            need(abs(rmse-float(raw['rmse_'+key]))<1e-10,'Whole-grid RMSE mismatch')
            metrics['rmse_'+key]=rmse
        data.append(dict(case=case,dim=dim,ng=ng,g=g,truth=truth,exact=exact,bcs=bcs,index=ix,metrics=metrics,meta=meta))
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    sources=Sources();sources.add(__file__);sources.add(Path(style.__file__))
    cases=load_cases(a.data_dir,sources);out=output_dir(a.output)
    csv_write(out/'metric_source.csv',[{'case':d['case'],'n_query':len(d['truth']),'n_display':len(d['index']),
        'kernel':d['meta']['expr'],'mean_id':d['meta']['mean_id'],**d['metrics']} for d in cases])
    if not a.validate_only:
        plt=style.configure_matplotlib();fig,_=style.make_figure(cases,plt)
        save_figure(fig,out,'synthetic_prediction_b7');plt.close(fig)
    (out/'caption.md').write_text('Synthetic point predictions: EX1–EX4 rows; known truth, exact GP mean, archived BCS mean, and their prediction errors. Within a row, value panels share one scale and error panels share a symmetric scale. EX4 displays x3=9/19; RMSE uses its full three-dimensional query grid. No interpolation comparator is carried over from the old kernel definition.\n',encoding='utf-8')
    sources.finish(out,{'status':'VALIDATED' if a.validate_only else 'RENDERED_PENDING_VISUAL_QA','figures_generated':not a.validate_only,
        'excluded_queries_from_metrics':0,'display_rule':'Full 2D grids; predeclared EX4 slice only','case_count':4})


if __name__=='__main__':main()
