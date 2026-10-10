"""Closed-loop covariance discovery and covariance-preserving field simulation.

Conceptual scientific schematic only. No fitting, sampling, API calls, or
experimental values are generated. Exports are confined to this revision.
Python/Matplotlib is the sole drawing/export/preview backend.
"""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as PlotPath
from matplotlib.transforms import Bbox
import fitz
from PIL import Image

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
WIDTH_MM=183.0
HEIGHT_MM=145.0
DPI=600
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':9.0,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
 'text.usetex':False,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
C={'ink':'#243747','blue':'#315D83','blue_fill':'#EDF3F8','orange':'#A46834',
   'orange_fill':'#FBF0E6','gray':'#697984','gray_fill':'#F2F4F5','line':'#637A8D'}
fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),dpi=300)
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH_MM),ylim=(0,HEIGHT_MM));ax.axis('off')
texts=[];zones=[]
def text(x,y,s,size=9.0,bold=False,ha='center',color=None,zone=None):
 t=ax.text(x,y,s,ha=ha,va='center',fontsize=size,fontweight='bold' if bold else 'normal',
           linespacing=1.17,color=color or C['ink'],zorder=5)
 texts.append((t,zone));return t

def box(x,y,w,h,title,body='',family='blue',title_size=9.0,body_size=9.0):
 col=C[family];fill=C[family+'_fill']
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.25',
   facecolor=fill,edgecolor=col,linewidth=.85,zorder=3))
 z=(x,y,w,h,title);zones.append(z)
 if body:
  text(x+w/2,y+h-4.25,title,size=title_size,bold=True,color=col,zone=z)
  text(x+w/2,y+(h-8.0)/2-.1,body,size=body_size,zone=z)
 else:text(x+w/2,y+h/2,title,size=title_size,bold=True,color=col,zone=z)

def arrow(pts,color=None,dashed=False,end=True,width=.9):
 p=PlotPath(pts,[PlotPath.MOVETO]+[PlotPath.LINETO]*(len(pts)-1))
 ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>' if end else '-',mutation_scale=8,
   linewidth=width,color=color or C['line'],linestyle=(0,(3,2)) if dashed else '-',zorder=2))

text(5,140.2,'a  Agent decision–tool–feedback loop',size=10.0,bold=True,ha='left',color=C['blue'])
box(5,122,173,12,'Observation summaries  +  legal, unranked kernel–mean catalogue',title_size=9.1)
text(91.5,115.1,'20 scored candidates: structured 4 × 5  |  competing-structure 5 × 4',size=9.0)
box(5,88,39,22,'LLM: propose','Kernel–mean pairs\n+2 reserves\nShort rationale',family='orange')
box(50,88,34,22,'Validate + dedup','First b valid, unseen\nBudget cap: 20',body_size=9.0)
box(91,88,46,22,'Numerical scorer','Fit GP parameters → AIC\nBenchmark: cached scores')
box(141,88,37,22,'Update history','AIC, Δ to visited best\nk + fit status\nBest-visited neighbors',family='orange',body_size=9.0)
arrow([(24.5,122),(24.5,110)])
arrow([(44,99),(50,99)])
arrow([(84,99),(91,99)])
arrow([(137,99),(141,99)])
# One branch returns only visited-model feedback while budget remains.
arrow([(161,88),(161,79),(24.5,79),(24.5,88)],color=C['orange'],width=1.1)
text(89,83.3,'History + neighborhoods → next proposal batch if budget remains',size=9.0,color=C['orange'])
# The terminal branch retains the best successful candidate, never an unseen score.
arrow([(161,79),(161,70)],color=C['orange'],width=1.1)
text(146,74.4,'At budget end',size=9.0,color=C['orange'])
box(5,60,173,10,'Select the successful visited candidate with the lowest AIC',family='orange',title_size=9.0)

text(5,54.8,'b  Covariance-preserving conditional simulation',size=10.0,bold=True,ha='left',color=C['blue'])
arrow([(161,60),(161,50.5),(24.5,50.5),(24.5,47)])
box(5,27,39,20,'Compile covariance','Separable components',title_size=9.0)
box(50,27,40,20,'Independent priors','Coordinate union\nSum components',title_size=9.0,body_size=9.0)
box(96,27,38,20,'Joint Matheron','Condition the prior sum\nFitted mean + noise',title_size=9.0,body_size=9.0)
box(140,27,38,20,'Conditional fields','Weak-zone area\n+ exceedance',title_size=9.0)
arrow([(44,37),(50,37)])
arrow([(90,37),(96,37)])
arrow([(134,37),(140,37)])
# Independent reference branch: all structures are selected using full data.
# The borehole evaluation refits their parameters, not a nested agent search.
box(5,4,83,16,'Full-data references','SE · RQ · composite',family='gray',title_size=9.0)
box(96,4,82,16,'Whole-borehole prediction','Fix structures; refit each training fold',family='gray',title_size=9.0,body_size=9.0)
arrow([(88,12),(96,12)],color=C['gray'],dashed=True)
arrow([(46.5,20),(46.5,23),(159,23),(159,27)],color=C['gray'],dashed=True)

# Layout checks use rendered text, not estimated character widths.
fig.canvas.draw();renderer=fig.canvas.get_renderer();inv=ax.transData.inverted();ext=[]
for t,z in texts:
 b=t.get_window_extent(renderer).transformed(inv)
 assert b.x0>=.8 and b.x1<=WIDTH_MM-.8 and b.y0>=.8 and b.y1<=HEIGHT_MM-.8,('crop',t.get_text(),b)
 if z:
  x,y,w,h,title=z
  assert b.x0>=x+.7 and b.x1<=x+w-.7 and b.y0>=y+.6 and b.y1<=y+h-.6,('node boundary',title,t.get_text(),b)
 ext.append((t,b))
overlap=[]
for i,(ta,ba) in enumerate(ext):
 for tb,bb in ext[i+1:]:
  v=Bbox.intersection(ba,bb)
  if v is not None and v.width>.15 and v.height>.15:overlap.append((ta.get_text(),tb.get_text()))
assert not overlap,overlap
OUT.mkdir(parents=True,exist_ok=True);base=OUT/'overall_framework_b7'
for typ in ['pdf','svg','png']:fig.savefig(base.with_suffix('.'+typ),dpi=DPI,facecolor='white')
plt.close(fig)
qa={'dimensions_mm':[WIDTH_MM,HEIGHT_MM],'layout_text_checks':'PASS','text_overlap':'PASS',
    'minimum_requested_font_pt':min(t.get_fontsize() for t,_ in texts),'data_generation':'none',
    'figure_type':'conceptual workflow; not a quantitative evidence panel'}
with fitz.open(base.with_suffix('.pdf')) as d:
 assert len(d)==1
 p=d[0];sp=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for ln in b['lines'] for s in ln['spans']]
 qa['pdf_min_font_pt']=min(s['size'] for s in sp)
 qa['pdf_dimensions_mm']=[p.rect.width*25.4/72,p.rect.height*25.4/72]
 qa['pdf_fonts']=[f[3] for f in p.get_fonts(full=True)]
 qa['pdf_selectable_text']=bool(sp)
 txt=p.get_text()
 for req in ['LLM: propose','Numerical scorer','Update history','20 scored candidates','Joint Matheron','Full-data references','Whole-borehole prediction']:
  assert req in txt,req
 p.get_pixmap(dpi=300,alpha=False).save(OUT/'overall_framework_b7_pdf_preview.png')
 (HERE/'figure_text.txt').write_text(txt,encoding='utf8')
svg=ET.parse(base.with_suffix('.svg'));qa['svg_text_nodes']=len(svg.findall('.//{http://www.w3.org/2000/svg}text'));assert qa['svg_text_nodes']>0
with Image.open(base.with_suffix('.png')) as im:qa['png_pixels']=im.size;qa['png_dpi']=im.info.get('dpi');assert min(qa['png_dpi'])>599
qa['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base.with_suffix('.pdf'),base.with_suffix('.svg'),base.with_suffix('.png')]}
(HERE/'figure_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf8')
print(json.dumps(qa,indent=2))
