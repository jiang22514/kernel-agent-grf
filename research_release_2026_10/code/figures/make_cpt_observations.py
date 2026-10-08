"""Observed CPT data only: all 744 rows, no simulation, fitting or interpolation."""
from pathlib import Path
import csv, hashlib, json, platform
import numpy as np
from scipy.io import loadmat
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import font_manager
import fitz
from PIL import Image

import os
ROOT=Path(os.environ.get('FIGURE_INPUT_ROOT', str(Path(__file__).resolve().parents[2]))).resolve()
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'samedata.mat'
COPY=ROOT/'gpml/gpml-matlab-v4.2-2018-06-11/samedata.mat'
WIDTH_MM=183
HEIGHT_MM=132
DPI=600

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

s,copy=loadmat(SOURCE),loadmat(COPY)
x=np.asarray(s['x_same'],dtype=float);q=np.asarray(s['y_same'],dtype=float).ravel()
require(x.shape==(744,2) and q.shape==(744,),'Unexpected source shape')
require(np.isfinite(x).all() and np.isfinite(q).all(),'Nonfinite observations; do not silently exclude')
require(np.array_equal(s['x_same'],copy['x_same']) and np.array_equal(s['y_same'],copy['y_same']),'Root and GPML arrays differ')
require(digest(SOURCE)==digest(COPY),'Root and GPML file hashes differ')
positions=np.unique(x[:,0]);require(len(positions)==6,'Expected six observed horizontal coordinates')
ids=np.searchsorted(positions,x[:,0])+1
rows=[{'source_row':i+1,'borehole_id':f'B{ids[i]}','horizontal_coordinate_m':float(x[i,0]),
       'vertical_coordinate_m':float(x[i,1]),'cone_resistance_MPa':float(q[i])} for i in range(744)]
write_csv(OUT/'cpt_observations_744.csv',rows)
summary=[]
for j,pos in enumerate(positions,1):
    mask=x[:,0]==pos;z=np.sort(x[mask,1]);require(np.allclose(np.diff(z),.1,rtol=0,atol=5e-14),'Vertical spacing is not .1 m')
    summary.append({'borehole_id':f'B{j}','horizontal_coordinate_m':float(pos),'n_observations':int(mask.sum()),
                    'vertical_min_m':float(z.min()),'vertical_max_m':float(z.max()),
                    'qc_min_MPa':float(q[mask].min()),'qc_max_MPa':float(q[mask].max()),'vertical_step_m':.1})
require([r['n_observations'] for r in summary]==[100,122,133,123,128,138],'Unexpected borehole counts')
write_csv(OUT/'borehole_summary.csv',summary)
with (OUT/'cpt_observations_744.csv').open(encoding='utf-8-sig',newline='') as f:restored=list(csv.DictReader(f))
roundtrip=np.array([[float(r[k]) for k in ('horizontal_coordinate_m','vertical_coordinate_m','cone_resistance_MPa')] for r in restored])
require(np.array_equal(roundtrip,np.column_stack([x,q])),'CSV changed source values')

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'axes.titlesize':8.5,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
 'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
 'text.color':'#20252A','axes.labelcolor':'#20252A','axes.edgecolor':'#48525B',
 'xtick.color':'#48525B','ytick.color':'#48525B'})
fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),facecolor='white')
fig.text(.04,.96,'a',fontweight='bold',fontsize=10)
fig.text(.10,.96,'Observed locations and cone resistance',fontsize=9.5)
fig.text(.897,.96,'744 observations',ha='right',fontsize=8.5,color='#48525B')
ax=fig.add_axes([.10,.67,.795,.235])
norm=Normalize(vmin=float(q.min()),vmax=float(q.max()),clip=False)
points=ax.scatter(x[:,0],x[:,1],c=q,cmap='viridis',norm=norm,s=13,marker='_',linewidths=.75)
ax.set_xlim(426,777);ax.set_ylim(4,21.7)
ax.set_xticks(positions,labels=[f'{v:.2f}' for v in positions])
ax.set_yticks([4,8,12,16,20]);ax.set_xlabel('Horizontal coordinate (m)',labelpad=5)
ax.set_ylabel('Vertical coordinate (m)',labelpad=5)
ax.grid(axis='y',color='#DDE2E6',linewidth=.4,zorder=0);ax.set_axisbelow(True)
for j,pos in enumerate(positions,1):ax.text(pos,20.65,f'B{j}',ha='center',va='center',fontsize=8.5)
cax=fig.add_axes([.915,.67,.015,.235]);cb=fig.colorbar(points,cax=cax,ticks=[4,6,8]);cb.outline.set_linewidth(.5)
cb.set_label('Cone resistance, $q_c$ (MPa)',fontsize=8.5,labelpad=6)

fig.text(.04,.555,'b',fontweight='bold',fontsize=10)
fig.text(.10,.555,'Individual borehole profiles',fontsize=9.5)
fig.text(.93,.555,'Shared axis ranges',ha='right',fontsize=8.5,color='#48525B')
left,right,gap=.10,.93,.025;width=(right-left-5*gap)/6
profile_axes=[]
for j,pos in enumerate(positions):
    ap=fig.add_axes([left+j*(width+gap),.12,width,.335],sharey=profile_axes[0] if profile_axes else None)
    mask=x[:,0]==pos;ind=np.flatnonzero(mask);ind=ind[np.argsort(x[ind,1],kind='stable')]
    ap.plot(q[ind],x[ind,1],color='#285D78',linewidth=.7,marker='o',markersize=1.1,markeredgewidth=0)
    ap.set_xlim(3,9.4);ap.set_ylim(4,21.7);ap.set_xticks([4,6,8]);ap.set_yticks([4,8,12,16,20])
    ap.set_title(f'B{j+1} · n = {len(ind)}\n{pos:.2f} m',pad=5,linespacing=1.4)
    ap.grid(axis='y',color='#DDE2E6',linewidth=.4);ap.set_axisbelow(True)
    if j==0:ap.set_ylabel('Vertical coordinate (m)',labelpad=5)
    else:ap.tick_params(labelleft=False)
    profile_axes.append(ap)
fig.text(.515,.042,'Cone resistance, $q_c$ (MPa)',ha='center',fontsize=9)
fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.canvas.get_width_height()
overflow=[]
for text in fig.findobj(matplotlib.text.Text):
    if not text.get_visible() or not text.get_text():continue
    b=text.get_window_extent(renderer)
    # Matplotlib keeps unused axis labels beyond the view; only rendered in-view centers count.
    cx,cy=(b.x0+b.x1)/2,(b.y0+b.y1)/2
    if 0<=cx<=fw and 0<=cy<=fh and (b.x0<-.5 or b.y0<-.5 or b.x1>fw+.5 or b.y1>fh+.5):overflow.append(text.get_text())
require(not overflow,'Visible text extends outside figure: '+str(overflow))
base=OUT/'cpt_observations_b7'
fig.savefig(str(base)+'.pdf',metadata={'Title':'Observed CPT data: six boreholes, 744 observations','Author':''})
fig.savefig(str(base)+'.svg')
fig.savefig(str(base)+'.png',dpi=DPI)
plt.close(fig)

doc=fitz.open(str(base)+'.pdf');page=doc[0]
require(len(doc)==1,'Unexpected PDF page count')
page.get_pixmap(dpi=300,alpha=False).save(OUT/'cpt_observations_b7_pdf_preview.png')
page.get_pixmap(dpi=300,alpha=False).save(OUT/'cpt_observations_b7_final_size_preview.png')
text=page.get_text();require('744 observations' in text and all(f'B{j}' in text for j in range(1,7)),'Expected text missing in PDF')
fonts=page.get_fonts(full=True)
font_info=[{'name':f[3],'type':f[2],'embedded_bytes':len(doc.extract_font(f[0])[3])} for f in fonts]
require(all(f['embedded_bytes']>0 for f in font_info),'PDF font not embedded')
mm=np.array([page.rect.width,page.rect.height])*25.4/72
require(np.allclose(mm,[WIDTH_MM,HEIGHT_MM],atol=.02),'Export changed physical dimensions')
image=Image.open(str(base)+'.png')
qa={'observations_source':744,'observations_panel_a':len(points.get_offsets()),'observations_panel_b':sum(r['n_observations'] for r in summary),
    'excluded_rows':0,'csv_roundtrip_exact':True,'root_gpml_arrays_exactly_equal':True,'root_gpml_bytes_identical':True,
    'borehole_counts':[r['n_observations'] for r in summary],'color_limits_MPa':[float(q.min()),float(q.max())],
    'coordinate_limits':[x.min(axis=0).tolist(),x.max(axis=0).tolist()],'vertical_axis_increases_upward':True,
    'axis_length_scales_differ':True,'profile_smoothing':False,'interpolated_field':False,
    'pdf_dimensions_mm':mm.tolist(),'png_pixels':list(image.size),'png_dpi':image.info.get('dpi'),
    'pdf_fonts':font_info,'svg_editable_text':'<text' in Path(str(base)+'.svg').read_text(encoding='utf-8'),
    'min_requested_font_pt':8.5,'text_overflow':overflow,
    'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
    'visual_QA':'Pending direct inspection of the generated PDF previews; this entry is not a visual pass.'}
(OUT/'export_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
inputs=[SOURCE,COPY,ROOT/'submission_CG/figures/image197.pdf',ROOT/'AI_figure/make_cpt_schematic.py',Path(__file__)]
(OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in inputs},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(qa,ensure_ascii=False,indent=2))