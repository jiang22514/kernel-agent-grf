"""Read-only LIN-v2 A50, A+B20 and C20 figures; never issue API requests."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from common import *

LABELS={'astra':'GPT-6 Astra','gpt55':'GPT-5.5','fable51':'Claude Fable 5.1','gemini31pro':'Gemini 3.1 Pro',
        'deepseek4pro':'DeepSeek V4 Pro','qwen38max':'Qwen 3.8 Max','kimi3':'Kimi K3','grok47':'Grok 4.7',
        'gpt4omini':'GPT-4o mini','gemini38flash':'Gemini 3.8 Flash'}
FLAGSHIPS=['astra','fable51']


def load_data(directory,sources):
    directory=directory.resolve();need(not (directory/'RUNNING.lock').exists(),'Wait until active batch finishes before making a consistent figure snapshot')
    adapter=module('figure_frozen_v2_adapter',directory/'adapter.py')
    report=adapter.status(directory,persist=False);plan,assets=adapter.check(directory)
    plan={**plan,'model_order':read(directory/'design.json')['model_order']}
    need(plan['kernel_definition']=='lin_first_coordinate_offset_v2','Only the new LIN definition is accepted')
    for name,digest in plan['input_hashes'].items():sources.add(directory/name,digest)
    sources.add(directory/'plan.json');sources.add(directory/'plan.sha256');sources.add(directory/'assets.json',plan['assets_sha256'])
    ledger=[]
    if (directory/'ledger.json').exists():sources.add(directory/'ledger.json');ledger=read(directory/'ledger.json')
    rows=[];trajectories={}
    for r in report['runs']:
        r=dict(r);r['completed']=r['state']=='complete' and r.get('terminal_present',False)
        r['full_budget_completed']=r['completed'] and r['n_evals']==20
        r['completed_hit']=r['completed'] and r['hit'];r['completed_near']=r['completed'] and r['near_optimal']
        entries=[e for e in ledger if e['run_id']==r['run_id']]
        r['known_cost_usd']=sum(e['charged_or_reserved_usd'] for e in entries if not e['billing_unknown'])
        r['unsettled_requests']=sum(e['billing_unknown'] for e in entries)
        path=directory/'runs'/r['run_id']/'result.json'
        if path.exists():
            sources.add(path);result=read(path);visits=result.get('visited',[])
            optimum=min(s['aic'] for s in assets['cases'][r['case']]['rows'])
            trajectories[r['run_id']]=(np.minimum.accumulate([v['aic'] for v in visits])-optimum).tolist() if visits else []
            if r['stage']=='C' and visits:
                actual=[{k:v[k] for k in ('kernel','mean_id')} for v in visits[:5]]
                need(len(visits)>=5 and actual==r['initial_pairs'],'Same-start trajectory did not use the frozen five pairs')
            if (path.parent/'terminal.json').exists():
                sources.add(path.parent/'terminal.json');terminal=read(path.parent/'terminal.json')
                need(terminal['state']==r['state'],'Terminal and saved result disagree')
        else:trajectories[r['run_id']]=[]
        rows.append(r)
    need(Counter(r['stage'] for r in rows)=={'A':50,'B':10,'C':20},'Expected the full 80-search planned matrix')
    for c in CASES:
        for m in FLAGSHIPS:
            ab=[r for r in rows if r['stage'] in ('A','B') and r['case']==c and r['model_slug']==m]
            need(sorted(r['replicate'] for r in ab)==[1,2],'Flagships must have exactly two planned repeats per case')
            paired=[r for r in rows if r['stage']=='C' and r['case']==c and r['model_slug']==m]
            need(len(paired)==2 and {r['method'] for r in paired}=={'budget_complete','structured'},'Expected two same-start methods')
            need(paired[0]['initial_pairs']==paired[1]['initial_pairs'] and len(paired[0]['initial_pairs'])==5,'Same-start plans differ')
    return plan,report,rows,trajectories


def shortgap(value):
    if value is None:return '—'
    if value<=1e-9:return '0'
    if value<0.1:return f'{value:.2f}'
    return f'{value:.1f}' if value<100 else f'{value:.0f}'


def annotate_status(fig,rows):
    complete=sum(r['completed'] for r in rows);planned=len(rows)
    text=f'{complete}/{planned} completed searches'
    if complete<planned:text+='; ! / open markers: incomplete; —: no visits'
    fig.text(.99,.008,text,ha='right',va='bottom',fontsize=6,color='0.3')


def panel_a(plan,rows,out):
    plt=plot_setup();from matplotlib.patches import Rectangle
    models=plan['model_order'];selected=[r for r in rows if r['stage']=='A'];lookup={(r['model_slug'],r['case']):r for r in selected}
    gaps=np.full((len(models),5),np.nan)
    for i,m in enumerate(models):
        for j,c in enumerate(CASES):
            v=lookup[m,c].get('regret');gaps[i,j]=np.nan if v is None else max(v,0)
    colors=np.log10(1+gaps);cmap=plt.get_cmap('cividis_r').copy();cmap.set_bad('#ededed')
    fig,(ax,side)=plt.subplots(1,2,figsize=(183/25.4,128/25.4),gridspec_kw={'width_ratios':[5,2.25]})
    fig.subplots_adjust(left=.19,right=.98,bottom=.14,top=.88,wspace=.08)
    im=ax.imshow(colors,cmap=cmap,vmin=0,vmax=max(1,float(np.nanmax(colors)) if np.isfinite(colors).any() else 1),aspect='auto')
    ax.set_xticks(range(5),[display(c) for c in CASES]);ax.set_yticks(range(len(models)),[LABELS[m] for m in models]);ax.tick_params(length=0)
    for i,m in enumerate(models):
        for j,c in enumerate(CASES):
            r=lookup[m,c];label=shortgap(r.get('regret'))+('!' if r['n_evals'] and not r['completed'] else '')
            if r['completed_hit']:label+='*'
            ax.text(j,i,label,ha='center',va='center',fontsize=7,color='white' if np.isfinite(colors[i,j]) and colors[i,j]>.55*im.norm.vmax else 'black')
            if r['completed_near']:ax.add_patch(Rectangle((j-.47,i-.47),.94,.94,fill=False,ec='#D55E00',lw=1))
    ax.set_title('a  First structured search: ten models × five cases',loc='left',fontweight='bold',pad=28)
    side.set_xlim(0,3);side.set_ylim(len(models)-.5,-.5);side.axis('off')
    for x,t in zip([.45,1.45,2.45],['Near /\nplanned','Complete /\nplanned','Known\ncost ($)']):side.text(x,-.8,t,ha='center',fontsize=7)
    for i,m in enumerate(models):
        rr=[r for r in selected if r['model_slug']==m]
        vals=[f"{sum(r['completed_near'] for r in rr)}/5",f"{sum(r['completed'] for r in rr)}/5",f"{sum(r['known_cost_usd'] for r in rr):.2f}"+('+' if any(r['unsettled_requests'] for r in rr) else '')]
        for x,t in zip([.45,1.45,2.45],vals):side.text(x,i,t,ha='center',va='center',fontsize=7)
    cb=fig.colorbar(im,ax=[ax,side],orientation='horizontal',fraction=.045,pad=.12,shrink=.55);cb.set_label(r'$\log_{10}(1+\Delta\mathrm{AIC})$; cell text shows $\Delta$AIC (lower is better)')
    annotate_status(fig,selected);save_figure(fig,out,'flagship_first_panel');plt.close(fig)


def repeats(rows,trajectories,out):
    plt=plot_setup();fig,axs=plt.subplots(2,5,figsize=(183/25.4,113/25.4));fig.subplots_adjust(left=.09,right=.985,bottom=.13,top=.88,wspace=.32,hspace=.48)
    selected=[r for r in rows if r['stage'] in ('A','B') and r['model_slug'] in FLAGSHIPS]
    for mi,m in enumerate(FLAGSHIPS):
        for ci,c in enumerate(CASES):
            ax=axs[mi,ci];rr=sorted([r for r in selected if r['model_slug']==m and r['case']==c],key=lambda r:r['replicate'])
            for r,col,ls in zip(rr,['#0072B2','#D55E00'],['-','--']):
                vals=trajectories[r['run_id']]
                if vals:
                    ax.plot(range(1,len(vals)+1),vals,color=col,ls=ls,lw=.9,label='Repeat '+str(r['replicate']))
                    ax.plot(len(vals),vals[-1],marker='o',ms=4,mec=col,mfc=col if r['completed'] else 'white')
            ax.axhline(2,color='0.55',ls=':',lw=.7);ax.set_yscale('symlog',linthresh=2);ax.set_ylim(bottom=0);ax.set_xlim(1,20);ax.set_xticks([5,10,15,20])
            ax.set_title(display(c)+f" ({sum(r['completed'] for r in rr)}/2 complete)",fontsize=7)
            if ci==0:ax.set_ylabel(LABELS[m]+'\nBest ΔAIC')
            if mi==1:ax.set_xlabel('Candidate visits')
    fig.suptitle('b  Flagship search trajectories: two planned runs per case',fontsize=9,x=.09,ha='left',fontweight='bold')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color='#0072B2',label='Repeat 1 (A)'),Line2D([0],[0],color='#D55E00',ls='--',label='Repeat 2 (B)')],loc='lower left',bbox_to_anchor=(.09,.0),ncol=2,fontsize=7)
    annotate_status(fig,selected);save_figure(fig,out,'flagship_repeat_stability');plt.close(fig)


def paired(rows,out):
    plt=plot_setup();fig,axs=plt.subplots(1,5,sharey=True,figsize=(183/25.4,87/25.4));fig.subplots_adjust(left=.09,right=.985,bottom=.28,top=.82,wspace=.38)
    selected=[r for r in rows if r['stage']=='C']
    upper=max(3.0, 1.2*max((r['regret'] for r in selected if r.get('regret') is not None), default=0))
    ticks=[v for v in (0,1,2,5,10,20,50,100) if v<=upper]
    for ci,c in enumerate(CASES):
        ax=axs[ci]
        for m,col,offset in zip(FLAGSHIPS,['#0072B2','#D55E00'],[-.035,.035]):
            rr=[next(r for r in selected if r['case']==c and r['model_slug']==m and r['method']==method) for method in ('budget_complete','structured')]
            xx=np.array([0,1])+offset;vv=[r.get('regret') for r in rr]
            if all(v is not None for v in vv):ax.plot(xx,vv,color=col,lw=.85)
            for x,v,r in zip(xx,vv,rr):
                if v is not None:ax.plot(x,v,'o',ms=4,mec=col,mfc=col if r['completed'] else 'white')
        ax.set_title(display(c));ax.set_xticks([0,1],['Budget-complete\nbaseline','Structured\nagent'],fontsize=6);ax.set_xlim(-.2,1.2)
        ax.set_yscale('symlog',linthresh=2);ax.set_ylim(0,upper);ax.set_yticks(ticks,[str(v) for v in ticks]);ax.axhline(2,color='0.55',ls=':',lw=.7)
    axs[0].set_ylabel('Final visited best ΔAIC\n(lower is better)')
    fig.suptitle('c  Same-start search: ten paired comparisons',fontsize=9,x=.09,ha='left',fontweight='bold')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=col,marker='o',label=LABELS[m]) for m,col in zip(FLAGSHIPS,['#0072B2','#D55E00'])],loc='lower left',bbox_to_anchor=(.09,.08),ncol=2)
    annotate_status(fig,selected);save_figure(fig,out,'flagship_same_start_v2');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true',help='Clearly labelled snapshot; never fill absent runs from older experiments')
    p.add_argument('--require-terminal',action='store_true',help='Final report: all 80 attempts must have actual adapter terminal records; failed/stopped/interrupted attempts remain unsuccessful')
    p.add_argument('--validate-only',action='store_true');a=p.parse_args();sources=Sources();sources.add(__file__)
    plan,report,rows,trajectories=load_data(a.directory,sources)
    complete=all(r['completed'] for r in rows)
    terminated=report['status']=='ALL_PLANNED_ATTEMPTS_TERMINATED' and all(r.get('terminal_present') for r in rows)
    if a.require_terminal:
        need(terminated,'Some planned searches lack terminal records; root must decide how to report missing attempts. Do not rerun paid searches to satisfy a figure gate.')
    else:
        need(complete or a.allow_incomplete,'Not all 80 searches completed; use --allow-incomplete only for an explicitly labelled snapshot')
    display_status='COMPLETE' if complete else ('TERMINATED_WITH_UNSUCCESSFUL_ATTEMPTS' if a.require_terminal else 'INTERIM_INCOMPLETE')
    out=output_dir(a.output)
    columns=['run_id','stage','case','model_slug','method','replicate','state','n_evals','completed','full_budget_completed','regret','hit','near_optimal','completed_hit','completed_near','known_cost_usd','unsettled_requests']
    csv_write(out/'agent_planned_source.csv',[{k:display(r[k]) if k=='case' else r.get(k,'') for k in columns} for r in rows])
    csv_write(out/'agent_trajectories.csv',[{'run_id':run,'candidate_visit':i+1,'best_delta_aic':v} for run,vals in trajectories.items() for i,v in enumerate(vals)])
    write(out/'agent_status_source.json',report)
    if not a.validate_only:panel_a(plan,rows,out);repeats(rows,trajectories,out);paired(rows,out)
    (out/'caption.txt').write_text('The entire design is A50 + C20 + B10 = 80 planned searches. A: one structured-agent search per model and case (50 planned); cells show ΔAIC against the exhaustive FINAL LIN-v2 minimum, * denotes a completed champion hit and orange outlines completed near-optimal results (ΔAIC≤2). + by a known cost marks unsettled billing. A+B: two planned runs per flagship and case (20 total), comprising 10 flagship searches already included in A and 10 additional B searches. These panels overlap and their counts must not be summed to 90. Repeats are shown individually without a confidence band; curves end at actual candidate visits. C: the independent same-start experiment has 2 flagship models × 5 cases × 2 methods = 20 planned searches, each method given the identical five initial candidate/mean pairs. No C result is pooled with A or B. Open endpoint markers / ! indicate searches that did not complete; missing results are absent or dashed, never successful. Completed counts and planned denominators are always retained. No hypothesis-test or significance claim is inferred from these displays.',encoding='utf-8')
    sources.finish(out,{'status':display_status+('_VALIDATED' if a.validate_only else '_RENDERED_PENDING_VISUAL_QA'),
        'figures_generated':not a.validate_only,'planned_searches':80,'completed_searches':sum(r['completed'] for r in rows),
        'terminal_searches':sum(bool(r.get('terminal_present')) for r in rows),'states':dict(Counter(r['state'] for r in rows)),
        'first_panel_planned':50,'flagship_AB_planned':20,'same_start_C_planned':20,'old_results_used':False})


if __name__=='__main__':main()
