"""Redraw existing fixed-model posterior-mean diagnostics; no scientific simulation.

All 30 source requests are exported. Exactly repeated (model,n1,n2) settings
are represented once in the plot, only after their original errors match.
"""
from pathlib import Path
from collections import OrderedDict
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter
import fitz
from PIL import Image

HERE=Path(__file__).resolve().parent
import os
ROOT=Path(os.environ.get('FIGURE_INPUT_ROOT', str(Path(__file__).resolve().parents[2]))).resolve()
SOURCE=ROOT/'output/revision/06_整改_B6/density_v2'
INPUTS={
 'measurements':SOURCE/'b66_density.json',
 'dimensions_log':SOURCE/'b66_density_log.txt',
 'original_script':ROOT/'kernel-agent-grf/simulation/b66_density_study_v2.m',
 'original_figure':SOURCE/'b66_density_curves.png',
}
EXPECTED={
 'measurements':'4619e8441a43fe6d179651d6f6bc750ba90ecf0ea70e91e6840d05eeee6e0a22',
 'dimensions_log':'4e600c6633aed831691cca57644ab9a0fb7896e44cff2fbb65569a5535d6d2e6',
 'original_script':'40c67e19c06a4d13c74be60770d2b9195f2becc70b9cd4538cd5f69d39fab368',
 'original_figure':'ba1bbca5bf164cb32cff105d0443943636c561818805a0952783617c87912146',
}
# Display labels only; covariance semantics are not reimplemented here.
MODEL=OrderedDict([
 ('MA1',('Matérn 1/2','#0072B2','o','-')),
 ('RQ',('Rational quadratic','#D55E00','s','--')),
 ('MA1*LIN',('Matérn 1/2 × linear','#56A6CB','^','-.')),
 ('(MA1*LIN)+PER',('(Matérn 1/2 × linear) + periodic','#946A9F','D',':')),
 ('LIN*PER',('Linear × periodic','#009E73','v','--')),
 ('(RQ*PER)+MA1',('(Rational quadratic × periodic) + Matérn 1/2','#4D4D4D','P','-')),
])
WIDTH_MM=183.0
HEIGHT_MM=148.0
DPI=600
plt.rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'legend.fontsize':8.5,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,
 'legend.frameon':False,'savefig.facecolor':'white',
})

def write_csv(name,rows):
    assert rows
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_existing():
    hashes={}
    for kind,p in INPUTS.items():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h!=EXPECTED[kind]:
            raise ValueError(f'Input fingerprint changed for {kind}; review before redraw')
        hashes[kind]={'path_relative_to_project':p.relative_to(ROOT).as_posix(),'sha256':h}
    j=json.loads(INPUTS['measurements'].read_text(encoding='utf-8-sig'))
    log=INPUTS['dimensions_log'].read_text(encoding='utf-8-sig')
    script=INPUTS['original_script'].read_text(encoding='utf-8-sig')
    for token in ("'fitc_alpha', 0","'n_samples', 0","'solver', 'chol'",'CAP = 260','n = 80; nsq = 200; sn = 0.1'):
        assert token in script,token
    vals={v['expr']:v for v in j.values() if 'ppl_list' in v}
    assert list(vals)==list(MODEL)
    logrows={}
    current=None
    for line in log.splitlines():
        if line.startswith('== '):
            current=next((k for k in MODEL if line.startswith('== '+k+'（')),None)
        elif current:
            hit=re.search(r'ppl=\s*(\d+)\s+网格\s+(\d+)x(\d+)\s+均值相对误差\s+([\d.]+)',line)
            if hit: logrows[(current,int(hit[1]))]=(int(hit[2]),int(hit[3]),float(hit[4]))
    assert len(logrows)==30
    raw=[];unique=OrderedDict(); conflicts=[]
    for expr,v in vals.items():
        assert v['ppl_list']==[5,10,20,40,60]
        ell=v['ell'];span=[3+ell[0],2+ell[1]]
        for req,err in zip(v['ppl_list'],v['emean']):
            assert math.isfinite(err) and err>0
            reqdims=[math.ceil(span[d]*req/ell[d])+1 for d in range(2)]
            dims=[min(260,n) for n in reqdims]
            logged=logrows[(expr,req)]
            assert tuple(dims)==logged[:2],(expr,req,dims,logged)
            assert abs(err-logged[2])<=.00005000001,(expr,req,err,logged[2])
            actual=[ell[d]*(dims[d]-1)/span[d] for d in range(2)]
            row={'source_row':len(raw)+1,'expression':expr,'model_label':MODEL[expr][0],
                 'requested_ppl':req,'raw_n1':reqdims[0],'raw_n2':reqdims[1],
                 'n1':dims[0],'n2':dims[1],'m':dims[0]*dims[1],
                 'actual_ppl1':actual[0],'actual_ppl2':actual[1],
                 'capped':any(a>b for a,b in zip(reqdims,dims)),
                 'relative_L2_discrepancy':err,'discrepancy_percent':100*err}
            raw.append(row)
            key=(expr,*dims)
            if key in unique:
                group=unique[key]
                if group['relative_L2_discrepancy']!=err:
                    conflicts.append({'key':key,'original':group['relative_L2_discrepancy'],'next':err})
                group['source_rows'].append(row['source_row'])
                group['requested_ppl_values'].append(req)
            else:
                unique[key]={k:row[k] for k in ('expression','model_label','n1','n2','m','actual_ppl1','actual_ppl2','capped','relative_L2_discrepancy','discrepancy_percent')}
                unique[key].update(source_rows=[row['source_row']],requested_ppl_values=[req])
    if conflicts:
        (HERE/'duplicate_conflicts.json').write_text(json.dumps(conflicts,indent=2),encoding='utf-8')
        raise ValueError('Identical model/grid settings have unequal original errors; no figure generated')
    assert len(raw)==30 and len(unique)==24
    merged=list(unique.values())
    for row in merged:
        row['source_rows']=';'.join(map(str,row['source_rows']))
        row['requested_ppl_values']=';'.join(map(str,row['requested_ppl_values']))
    write_csv('source_requests.csv',raw)
    write_csv('plotted_unique_grids.csv',merged)
    # Keep fixed positive parameter values in their original compiler order.
    block=script.split('structs_ = {',1)[1].split('};',1)[0]
    paramrows=[]
    for expr,vec in re.findall(r"'([^']+)'\s*,\s*\[([^]]+)\]",block):
        pv=re.findall(r'log\(([^)]+)\)',vec)
        assert expr in MODEL and pv
        paramrows.append({'expression':expr,'original_positive_parameter_values':';'.join(pv),
                          'effective_scale1':vals[expr]['ell'][0],
                          'effective_scale2':vals[expr]['ell'][1]})
    assert len(paramrows)==6
    write_csv('fixed_parameters.csv',paramrows)
    hashes['source_counts']={'request_rows':30,'plotted_unique_model_grid_rows':24,
                             'exact_duplicate_request_rows_merged':6,'model_count':6,
                             'duplicate_conflicts':0}
    hashes['measure_definition']='norm(mu_grid-mu_exact)/max(norm(mu_exact-mean(mu_exact)),1e-12)'
    hashes['settings']={'training_n':80,'query_n':200,'domain':[[0,3],[0,2]],
                        'response_realizations_per_model':1,'noise_sd':.1,'fitc_alpha':0,
                        'posterior_draws':0,'solver':'chol','cap_per_direction':260,
                        'seed_in_original_study':43}
    hashes['counterexample_retained_not_plotted']=j['audit_counterexample']
    (HERE/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    return raw,merged


def draw(rows):
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),dpi=300)
    ax=fig.add_axes([23/WIDTH_MM,51/HEIGHT_MM,151/WIDTH_MM,83/HEIGHT_MM])
    handles=[]
    for expr,(label,color,marker,ls) in MODEL.items():
        rr=sorted([r for r in rows if r['expression']==expr],key=lambda r:r['m'])
        xx=[r['m'] for r in rr]; yy=[r['discrepancy_percent'] for r in rr]
        assert all(math.isfinite(v) and v > 0 for v in xx + yy), 'Log axes require strictly positive data'
        marker_size=6.0 if expr=='MA1' else 4.6
        ax.plot(xx,yy,color=color,linestyle=ls,linewidth=1.15,zorder=2)
        for r in rr:
            ax.plot(r['m'],r['discrepancy_percent'],marker=marker,markersize=marker_size,
                    markeredgecolor=color,markeredgewidth=.9,
                    markerfacecolor='white' if r['capped'] else color,linestyle='none',zorder=4)
        handles.append(Line2D([],[],color=color,linestyle=ls,linewidth=1.15,
                              marker=marker,markersize=marker_size,label=label))
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlim(500,105000);ax.set_ylim(.00015,70)
    ax.set_xticks([1000,10000,100000],['1,000','10,000','100,000'])
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax.tick_params(which='both',direction='out',width=.6)
    ax.grid(axis='y',which='major',color='#E2E5E8',linewidth=.5,zorder=0)
    ax.set_xlabel('Total inducing points, m',labelpad=5)
    ax.set_ylabel('Posterior-mean relative L2 discrepancy (%)',labelpad=6)
    ax.axhline(2,color='#6B6B6B',linewidth=.9,linestyle=(0,(5,3)),zorder=1)
    ax.text(102000,2.55,'2% reference',ha='right',va='bottom',fontsize=8.5,color='#555555')
    fig.text(.5,142/HEIGHT_MM,'Grid refinement of six fixed covariance models',
             ha='center',va='center',fontsize=10,fontweight='bold',color='#223344')
    leg=fig.legend(handles=handles,loc='center',bbox_to_anchor=(.52,28/HEIGHT_MM),
                   ncol=2,columnspacing=2.0,handlelength=2.3,handletextpad=.7,labelspacing=.9,
                   fontsize=8.5)
    fig.text(.5,11.5/HEIGHT_MM,'Filled: requested grid attained. Open: at least one direction capped at 260 points.',
             ha='center',va='center',fontsize=8.5,color='#4D4D4D')
    fig.text(.5,5.3/HEIGHT_MM,'One response per model · 80 training / 200 query points · α = 0 · no posterior draws',
             ha='center',va='center',fontsize=8.5,color='#4D4D4D')
    fig.canvas.draw()
    rend=fig.canvas.get_renderer()
    for t in fig.texts+leg.get_texts()+[ax.xaxis.label,ax.yaxis.label]:
        bb=t.get_window_extent(rend).transformed(fig.transFigure.inverted())
        assert bb.x0>0 and bb.x1<1 and bb.y0>0 and bb.y1<1,('text outside page',t.get_text(),bb)
    legendbb=leg.get_window_extent(rend).transformed(fig.transFigure.inverted())
    xlabelbb=ax.xaxis.label.get_window_extent(rend).transformed(fig.transFigure.inverted())
    assert legendbb.y1<xlabelbb.y0
    for ext in ('pdf','svg','png'):
        fig.savefig(HERE/f'b66_density_actual_grid.{ext}',dpi=DPI)
    plt.close(fig)
    return {'legend_bounds_fraction':list(legendbb.bounds),'text_bounds':'PASS'}


def exports_qa(layout):
    pdf=HERE/'b66_density_actual_grid.pdf'
    with fitz.open(pdf) as doc:
        page=doc[0]
        assert len(doc)==1
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for ln in b['lines'] for s in ln['spans']]
        assert min(s['size'] for s in spans)>=8.49
        assert all(f[1]!='n/a' for f in page.get_fonts(full=True))
        size=[page.rect.width*25.4/72,page.rect.height*25.4/72]
        assert abs(size[0]-WIDTH_MM)<.01 and abs(size[1]-HEIGHT_MM)<.01
        page.get_pixmap(dpi=300,alpha=False).save(HERE/'b66_density_actual_grid_pdf_preview.png')
        layout.update(width_mm=size[0],height_mm=size[1],pdf_min_font_pt=min(s['size'] for s in spans),
                      fonts=[f[3] for f in page.get_fonts(full=True)])
    svg=ET.parse(HERE/'b66_density_actual_grid.svg')
    texts=svg.findall('.//{http://www.w3.org/2000/svg}text')
    assert texts
    with Image.open(HERE/'b66_density_actual_grid.png') as im:
        assert min(im.info['dpi'])>599
        layout.update(png_size=im.size,png_dpi=im.info['dpi'],svg_text_nodes=len(texts))
    layout['output_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in HERE.iterdir() if p.suffix in {'.pdf','.svg','.png','.csv'}}
    (HERE/'export_qa.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
    return layout


if __name__=='__main__':
    raw,merged=read_existing()
    qa=exports_qa(draw(merged))
    print(json.dumps({'requests':len(raw),'unique_model_grids':len(merged),
                      'duplicate_errors_equal':True,'width_mm':qa['width_mm'],
                      'height_mm':qa['height_mm'],'min_font_pt':qa['pdf_min_font_pt']},indent=2))