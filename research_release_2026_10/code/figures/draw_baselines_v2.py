"""Plot only the saved uniform-random and greedy records from FINAL v2 scores."""
import argparse
from pathlib import Path
import numpy as np
from common import *


def load_data(path,sources):
    sources.add(path);data=read(path)
    need(data['status']=='complete' and data['GP_fits']==data['API_calls']==0,'Need completed table-only v2 baseline report')
    identity=data['identity'];need(identity['cases']==CASES and identity['random_repetitions']==2000,'Unexpected baseline design')
    need(identity['random_budgets']==[5,10,15,20] and identity['greedy_budgets']==[20,40,80,160,500],'Budget design changed')
    records={r['case']:r for r in data['results']};need(set(records)==set(CASES),'Need all five cases')
    for case,r in records.items():
        need(r['identity_sha256']==data['identity_sha256'] and r['library_size']==376 and r['raw_random_orders_saved'],'Baseline identity mismatch')
        sources.add(r['source_table'],r['source_table_sha256'])
        need(len(r['random'])==4 and len(r['greedy'])==5,'Incomplete baseline arrays')
        for b,s in zip([5,10,15,20],r['random']):
            need(s['budget']==b and s['runs']==2000,'Random repetition count mismatch')
            need(abs(s['hit_count']/2000-s['hit_rate'])<1e-12 and abs(s['near_count']/2000-s['near_rate'])<1e-12,'Invalid random rates')
            need(s['regret_quartiles'][0]<=s['regret_median']<=s['regret_quartiles'][2],'Invalid quartile order')
        for cap,g in zip([20,40,80,160,500],r['greedy']):
            a=np.atleast_1d(g['aic']);traj=np.atleast_1d(g['traj']);seq=g['seq']
            need(g['cap']==cap and len(seq)==len(a)==len(traj)==g['evals_used']<=cap,'Invalid greedy visit count')
            need(len(set(seq))==len(seq) and np.allclose(traj,np.minimum.accumulate(a),atol=1e-9,rtol=0),'Invalid greedy trajectory')
            need(abs(g['regret']-(traj[-1]-r['champion_aic']))<1e-8 and g['cache_stats']['misses']==0,'Wrong table-only regret')
    return data,records


def source_rows(records):
    rows=[]
    for case in CASES:
        r=records[case];g=r['greedy'][0];traj=np.atleast_1d(g['traj'])-r['champion_aic']
        for s in r['random']:
            n=min(s['budget'],len(traj));gap=float(traj[n-1]);rows.append({'case':display(case),'budget':s['budget'],
                'random_trials':s['runs'],'random_regret_median':s['regret_median'],'random_regret_q25':s['regret_quartiles'][0],
                'random_regret_q75':s['regret_quartiles'][2],'random_hit_rate':s['hit_rate'],
                'random_hit_ci_low':s['hit_wilson95'][0],'random_hit_ci_high':s['hit_wilson95'][1],
                'random_near_rate':s['near_rate'],'greedy_actual_visits':n,'greedy_prefix_regret':gap,'greedy_prefix_hit':gap<=1e-9})
    return rows


def render(records,out):
    plt=plot_setup();fig,axs=plt.subplots(3,5,figsize=(183/25.4,170/25.4),squeeze=False)
    fig.subplots_adjust(left=.10,right=.99,top=.95,bottom=.13,wspace=.43,hspace=.47)
    rows=source_rows(records);blue='#0072B2';orange='#D55E00'
    for ci,case in enumerate(CASES):
        r=records[case];s=[x for x in rows if x['case']==display(case)];x=[v['budget'] for v in s]
        ax=axs[0,ci];ax.set_title(display(case));ax.fill_between(x,[v['random_regret_q25'] for v in s],[v['random_regret_q75'] for v in s],color=blue,alpha=.18)
        ax.plot(x,[v['random_regret_median'] for v in s],'o-',color=blue,label='Uniform random');ax.plot(x,[v['greedy_prefix_regret'] for v in s],'s--',color=orange,label='Greedy')
        ax.set_yscale('symlog',linthresh=2);ax.axhline(2,color='0.5',ls=':',lw=.7);ax.set_ylim(bottom=0);ax.set_xticks(x)
        ax=axs[1,ci];rates=np.array([v['random_hit_rate'] for v in s]);lo=np.array([v['random_hit_ci_low'] for v in s]);hi=np.array([v['random_hit_ci_high'] for v in s])
        ax.errorbar(x,rates,yerr=[rates-lo,hi-rates],color=blue,marker='o',capsize=2,lw=.8)
        ax.plot(x,[float(v['greedy_prefix_hit']) for v in s],'s--',color=orange);ax.set_ylim(-.03,1.07);ax.set_xticks(x);ax.set_xlabel('Candidate budget')
        ax=axs[2,ci];g=r['greedy'];caps=[v['cap'] for v in g];gaps=[v['regret'] for v in g]
        ax.plot(range(5),gaps,'s-',color=orange);ax.set_xticks(range(5),[str(v) for v in caps])
        upper=max(2.4,1.3*max(gaps))
        if max(gaps)<=2:
            ax.set_yscale('linear');ticks=[0,1,2]
        else:
            ax.set_yscale('symlog',linthresh=2);ticks=[v for v in [0,1,2,5,10,20,50,100,200,500] if v<=upper]
        ax.set_yticks(ticks,[str(v) for v in ticks]);ax.set_ylim(0,upper)
        ax.axhline(2,color='0.5',ls=':',lw=.7);ax.set_xlabel('Greedy cap')
        for k,v in enumerate(g):ax.annotate(str(v['evals_used']),(k,v['regret']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6)
    for ri,label in enumerate(['Best ΔAIC\n(lower is better)','Champion hit rate','Greedy best ΔAIC\n(labels: actual visits)']):axs[ri,0].set_ylabel(label)
    for ri,l in enumerate('abc'):axs[ri,0].text(-.45,1.06,l,transform=axs[ri,0].transAxes,fontweight='bold',fontsize=9)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.018))
    save_figure(fig,out,'free_baseline_budgets_b7');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    sources=Sources();sources.add(__file__);data,records=load_data(a.summary,sources);out=output_dir(a.output)
    csv_write(out/'baseline_source.csv',source_rows(records));csv_write(out/'greedy_caps_source.csv',[{'case':display(c),'cap':g['cap'],'actual_visits':g['evals_used'],'regret':g['regret'],'hit':g['hit'],'near':g['near']} for c in CASES for g in records[c]['greedy']])
    if not a.validate_only:render(records,out)
    (out/'caption.txt').write_text('Uniform-random search (2,000 sampled candidate orders per case) and deterministic greedy search on FINAL LIN-v2 tables. Random centers and bands are medians and interquartile ranges; champion hit intervals are 95% Wilson intervals. Greedy prefixes at budgets 5–20 come from its saved cap-20 trajectory; actual visits are in the source table. Larger-cap greedy trajectories are separate saved runs; labels show actual candidate visits. ΔAIC is relative to the exhaustively evaluated library minimum; the dotted line is ΔAIC=2. No model was fitted while drawing this figure.',encoding='utf-8')
    sources.finish(out,{'status':'VALIDATED' if a.validate_only else 'RENDERED_PENDING_VISUAL_QA','figures_generated':not a.validate_only,'baseline_identity':data['identity_sha256'],'random_trials_per_case':2000,'greedy_repetitions':1})


if __name__=='__main__':main()
