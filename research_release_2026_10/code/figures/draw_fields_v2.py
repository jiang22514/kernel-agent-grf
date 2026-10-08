"""Map the v2 checked-field summary to the existing three field figure layouts."""
import argparse
from pathlib import Path
import numpy as np
from common import *
import field_style as style

UNITS=['champion_main','champion_fine','SE_main','SE_fine','RQ_main']


def load_batch(path,sources):
    sources.add(path);summary=read(path)
    need(summary.get('status')=='PASS' and summary.get('protocol')=='b67_union_lin_offset_v2','Need completed v2 field checks')
    documents={d['result']['unit']:d for d in summary['results']}
    need(set(documents)==set(UNITS) and len(summary['results'])==5,'Need exactly five checked model/grid units')
    runs=[];perfield=[];maps={}
    for unit in UNITS:
        doc=documents[unit];r=doc['result'];need(doc['acceptance']['pass'] is True,'A field check failed')
        path=Path(r['sample_file']);sources.add(path,r['sample_sha256'])
        if unit.startswith('champion'):
            need(r['sample_identity_sha256']==summary['sample_identity_sha256'] and
                 r['checker_identity_sha256']==summary['checker_identity_sha256'],'New champion identity mismatch')
            need(r['expr']==summary['champion']['expr'],'Wrong selected covariance')
        else:
            reuse=doc['reuse'];need(reuse['pass'] is True and reuse['new_fields_drawn']==0 and reuse['fields_reused']==500,'Baseline reuse is not verified')
            need(reuse['sample_identity_sha256']==summary['sample_identity_sha256'] and reuse['old_sample_sha256']==r['sample_sha256'],'Reuse identity mismatch')
            sources.add(reuse['old_result_file'],reuse['old_result_sha256'])
        wp=np.asarray(r['wp_perfield'],dtype=float).ravel();need(wp.shape==(500,) and np.isfinite(wp).all(),'Require every saved field')
        need(np.all((wp>=0)&(wp<=1)) and r['n_samples']==500 and r['theta']==summary['theta'] and r['event_threshold']==.1,'Wrong field functional')
        events=int(np.sum(wp>.10));need(r['k_of_N']==[events,500] and abs(r['P']-events/500)<1e-12,'Event summary mismatch')
        delta=np.asarray(r['paired_delta_perfield'],dtype=float).ravel()
        if r['nested_done']:
            nested=np.asarray(r['wp_nested_perfield'],dtype=float).ravel()
            need(delta.shape==nested.shape==(500,) and np.allclose(delta,wp-nested,rtol=0,atol=1e-12),'Wrong same-field pairing')
        row={'unit':unit,'event_probability':r['P'],'event_count':events,'wilson95_low':r['wilson'][0],'wilson95_high':r['wilson'][1],
             'wp_mean':float(wp.mean()),'paired_delta_median':float(np.median(delta)) if delta.size else '',
             'paired_delta_rms':float(np.sqrt(np.mean(delta**2))) if delta.size else ''}
        runs.append(row)
        for i,value in enumerate(wp):perfield.append({'unit':unit,'sample_index_matlab':i+1,'weak_area_fraction':float(value),
            'same_field_finer_minus_coarser':float(delta[i]) if delta.size else ''})
        # Only the fixed first saved realization and analytic maps are read;
        # per-field spatial summaries already come from the passed checker.
        if unit.endswith('main'):
            with style.Mat73(path) as raw:
                mu=style.vector(raw.field('pack','fmu'));sd=style.vector(raw.field('pack','fsd'))
                g1=style.vector(raw.field('pack','g1'));g2=style.vector(raw.field('pack','g2'));ds=raw.samples()
                need(ds.shape==(500,len(mu)) and len(mu)==len(g1)*len(g2) and mu.shape==sd.shape,'Invalid map dimensions')
                need(len(g1)==201 and len(g2)==121,'Layout caption expects the formal 201 × 121 main query grid')
                need(np.isfinite(mu).all() and np.isfinite(sd).all() and np.all(sd>0),'Invalid latent maps')
                need(raw.field('pack','expr')==r['expr'],'Map covariance differs from result')
                first=np.asarray(ds[0,:],dtype=float);need(np.isfinite(first).all(),'Invalid first stored field')
                maps[unit]={'g1':g1,'g2':g2,'mu':mu,'sd':sd,'realization1':first}
    return {'runs':runs,'perfield':perfield,'fields':maps,'summary':summary}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    sources=Sources();sources.add(__file__);sources.add(style.__file__);batch=load_batch(a.summary,sources);out=output_dir(a.output)
    csv_write(out/'scalar_source.csv',batch['runs']);csv_write(out/'perfield_source.csv',batch['perfield'])
    arrays={u+'_'+k:v for u,field in batch['fields'].items() for k,v in field.items()};np.savez_compressed(out/'map_source.npz',**arrays)
    qa={} if a.validate_only else style.draw_figures(batch,out)
    qa.update(status='VALIDATED' if a.validate_only else 'RENDERED_PENDING_VISUAL_QA',figures_generated=not a.validate_only,
        field_count=2500,representative_field='First stored realization, no selection',
        old_baselines_used_only_with_new_reuse_proofs=True,raw_fields_resampled=False,
        data_scope='Saved per-field metrics from accepted checker; map arrays from immutable packs; no GP recomputation.')
    sources.finish(out,qa)


if __name__=='__main__':main()
