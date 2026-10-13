"""Rebuild the two manuscript workflow schematics with Python/matplotlib.

Conceptual diagrams only: no observations, scores, fitting or sampling are
created. The content contract and scientific-source map accompany this file.
All drawing, export and PDF-preview rendering use the Python backend.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
import fitz
from PIL import Image

HERE = Path(__file__).resolve().parent
WIDTH_MM = 183.0
DPI = 600
BODY_PT = 8.5
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False,
    'savefig.facecolor': 'white',
})
C = {'ink': '#171717', 'line': '#333333', 'blue': '#171717',
     'blue_fill': '#FFFFFF', 'neutral': '#F7F7F7', 'gray': '#333333',
     'accent': '#171717', 'accent_fill': '#F0F0F0', 'white': '#FFFFFF'}

class Diagram:
    def __init__(self, name, height_mm):
        self.name, self.height = name, height_mm
        self.fig = plt.figure(figsize=(WIDTH_MM/25.4, height_mm/25.4), dpi=300)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH_MM), ylim=(0, height_mm))
        self.ax.axis('off')
        self.texts = []
        self.boxes = []

    def text(self, x, y, s, *, size=BODY_PT, bold=False, color=None,
             ha='center', va='center', rotation=0, zone=None):
        t = self.ax.text(x, y, s, fontsize=size, fontweight='bold' if bold else 'normal',
                         color=color or C['ink'], ha=ha, va=va, linespacing=1.22,
                         rotation=rotation, zorder=5)
        self.texts.append((t, zone))
        return t

    def box(self, x, y, w, h, title, body='', *, family='blue', size=BODY_PT,
            title_size=9.5):
        color = C['blue'] if family=='blue' else C['gray'] if family=='neutral' else C['accent']
        fill = C['blue_fill'] if family=='blue' else C['neutral'] if family=='neutral' else C['accent_fill']
        patch = FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0',
                              facecolor=fill,edgecolor=color,linewidth=.8,zorder=3)
        self.ax.add_patch(patch)
        zone=(x,y,w,h,title)
        self.boxes.append(zone)
        if body:
            self.text(x+w/2,y+h-3.8,title,size=title_size,bold=True,color=color,zone=zone)
            self.text(x+w/2,y+(h-8)/2+0.15,body,size=size,zone=zone)
        else:
            self.text(x+w/2,y+h/2,title,size=title_size,bold=True,color=color,zone=zone)
        return zone

    def route(self, pts, *, dashed=False, color=None, arrow=True):
        path=MplPath(pts,[MplPath.MOVETO]+[MplPath.LINETO]*(len(pts)-1))
        a=FancyArrowPatch(path=path,arrowstyle='-|>' if arrow else '-',
                          mutation_scale=8,linewidth=.85,
                          linestyle=(0,(3,2)) if dashed else '-',
                          color=color or C['line'],zorder=1)
        self.ax.add_patch(a)

    def validate_layout(self):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        inv=self.ax.transData.inverted()
        extents=[]
        for t,zone in self.texts:
            bb=t.get_window_extent(renderer).transformed(inv)
            assert bb.x0>=0.8 and bb.x1<=WIDTH_MM-.8, (self.name,'horizontal crop',t.get_text(),bb)
            assert bb.y0>=.8 and bb.y1<=self.height-.8, (self.name,'vertical crop',t.get_text(),bb)
            if zone:
                x,y,w,h,title=zone
                assert bb.x0>=x+1 and bb.x1<=x+w-1 and bb.y0>=y+1 and bb.y1<=y+h-1, (
                    self.name,'text outside node',title,t.get_text(),bb)
            extents.append((t,bb))
        overlaps=[]
        for i,(a,ba) in enumerate(extents):
            for b,bb in extents[i+1:]:
                overlap=Bbox.intersection(ba,bb)
                if overlap is not None and overlap.width>.2 and overlap.height>.2:
                    overlaps.append((a.get_text(),b.get_text()))
        assert not overlaps,(self.name,'text overlap',overlaps)
        return {'text_count':len(extents),'min_font_pt':min(t.get_fontsize() for t,_ in self.texts),
                'text_bounds':'PASS','text_overlaps':'PASS'}

    def export(self):
        qa=self.validate_layout()
        for ext in ('pdf','svg','png'):
            self.fig.savefig(HERE/f'{self.name}.{ext}',dpi=DPI,
                             facecolor='white',edgecolor='none')
        plt.close(self.fig)
        pdf=HERE/f'{self.name}.pdf'
        with fitz.open(pdf) as doc:
            assert len(doc)==1
            page=doc[0]; w,h=page.rect.width*25.4/72,page.rect.height*25.4/72
            assert abs(w-WIDTH_MM)<.01 and abs(h-self.height)<.01,(w,h)
            spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b
                   for ln in b['lines'] for s in ln['spans']]
            assert spans and min(s['size'] for s in spans)>=8.49
            fonts=page.get_fonts(full=True)
            assert fonts and all(f[1]!='n/a' for f in fonts)
            text=page.get_text()
            for bad in ('GPT-5.5','507.53','523.60','20 fits','MVM-only','pre-registered','nonconservative'):
                assert bad not in text, (self.name,'retired content',bad)
            page.get_pixmap(dpi=300,alpha=False).save(HERE/f'{self.name}_pdf_preview.png')
            qa.update({'width_mm':w,'height_mm':h,'pdf_text_spans':len(spans),
                       'pdf_min_font_pt':min(s['size'] for s in spans),
                       'pdf_fonts':[f[3] for f in fonts],'pdf_selectable_text':'PASS'})
        svg=ET.parse(HERE/f'{self.name}.svg')
        texts=svg.findall('.//{http://www.w3.org/2000/svg}text')
        assert texts,'SVG lacks editable text'
        with Image.open(HERE/f'{self.name}.png') as im:
            dpi=im.info.get('dpi'); size=im.size
            assert dpi and min(dpi)>599
            assert abs(size[0]-round(WIDTH_MM/25.4*DPI))<=1
        qa.update({'svg_text_nodes':len(texts),'png_size':size,'png_dpi':dpi})
        qa['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in HERE.glob(self.name+'.*') if p.suffix in {'.pdf','.svg','.png'}}
        return qa


def overall():
    d=Diagram('overall_framework_b7',174)
    d.box(6,145,45,23,'Spatial observations',
          'Coordinates and responses\nNumerical data profile\nTraining data define the fit')
    d.box(62,145,53,23,'Bounded model library',
          '94 kernels × 4 means\n376 kernel–mean candidates\nShared covariance definitions')
    d.box(126,145,51,23,'Reference scores',
          'Exact marginal likelihood\nFit parameters; compute AIC\nFrozen for controlled search')
    d.route([(51,156.5),(62,156.5)])
    d.route([(115,156.5),(126,156.5)])
    d.box(6,105,103,27,'Search within a candidate budget',
          'Profile → propose → validate → access scores\nIndependent runs may pool accessed candidates\nEach run can stop or return no valid model')
    d.box(120,105,57,27,'Fitted simulation models',
          'Formal comparison:\nSE baseline · best single RQ\nReference champion')
    d.route([(151.5,145),(151.5,138),(57.5,138),(57.5,132)])
    d.text(92,141.1,'Requested entries only',size=8.5)
    d.route([(151.5,138),(148.5,138),(148.5,132)])
    d.box(6,69,83,23,'Grid conditional fields',
          'Sum of separable covariance terms\nn × n Cholesky + pathwise sampling\nAccuracy checked for the fitted model')
    d.box(99,69,78,23,'RQ reference: dense GP',
          'Original fitted covariance definition\nConditional query covariance\nRecorded numerical sampling shift',family='neutral')
    d.route([(148.5,105),(148.5,98),(47.5,98),(47.5,92)])
    d.route([(148.5,98),(138,98),(138,92)])
    d.box(6,29,52,25,'Numerical accuracy',
          'Means + joint covariances\nApproximation and sampling\nchecked separately')
    d.box(65.5,29,52,25,'Held-out prediction',
          'Train-fold profile + selection\nRefit within each training fold\nPredict excluded borehole',family='neutral')
    d.box(125,29,52,25,'Spatial functionals',
          'Weak-zone area fraction\nModel-conditional event\nprobability',family='accent')
    d.route([(47.5,69),(47.5,61),(138,61),(138,69)],arrow=False)
    d.route([(47.5,61),(32,61),(32,54)])
    d.route([(138,61),(151,61),(151,54)])
    # Fold-specific prediction branches from observations, never from the full-data winner.
    d.route([(6,156.5),(2,156.5),(2,22),(91.5,22),(91.5,29)],dashed=True,color=C['gray'])
    d.text(6,12,'Solid: full-data workflow. Dashed: separate leave-one-borehole-out evaluation.',
           ha='left',size=8.5)
    d.text(6,6.5,'Each simulation uses its model\'s fitted kernel, mean and parameters.',
           ha='left',size=8.5,color=C['gray'])
    return d.export()


def agent():
    d=Diagram('framework_diagram_b7',188)
    d.box(6,158,171,23,'Shared inputs and candidate-access caps',
          'Training-data profile + bounded grammar: 94 kernels × 4 means\nThe same profile, per-round cap and total cap apply to both arms',family='neutral',size=8.7)
    d.text(13,150,'a  Feedback search',ha='left',size=10,bold=True,color=C['blue'])
    d.text(108,150,'b  No-feedback control',ha='left',size=10,bold=True,color=C['gray'])
    d.route([(91.5,158),(91.5,145.5),(46,145.5),(46,143)])
    d.route([(91.5,145.5),(141,145.5),(141,143)])
    for x,family in [(13,'blue'),(108,'neutral')]:
        d.box(x,130,66,13,'LLM proposes candidates','Kernel + mean; structured reply',family=family)
        d.box(x,109,66,13,'Validate proposals','Grammar, duplicates and caps',family=family)
        d.box(x,88,66,13,'Access candidate scores','New valid kernel–mean pairs only',family=family)
        d.route([(x+33,130),(x+33,122)])
        d.route([(x+33,109),(x+33,101)])
        d.route([(x+33,88),(x+33,80)])
        if x==13:
            d.box(x,67,66,13,'Return ranked results','AIC + fit status for visited models',family=family)
        else:
            d.box(x,67,66,13,'Return evaluated names','Scores and fit status withheld',family=family)
        d.route([(x,73.5),(x-4,73.5),(x-4,136.5),(x,136.5)])
        d.text(x-7.2,105,'next round',size=8.5,rotation=90,color=C['gray'])
        d.route([(x+33,67),(x+33,55)])
        d.text(x+39,61,'stop',size=8.5,color=C['gray'],ha='left')
        d.box(x,37,66,18,'Select per run',
              'Lowest AIC among successful\naccessed candidates; else failure',family=family)
    d.text(91.5,30,'Common caps and stalled/error exits; only the feedback arm accepts a model stop.',size=8.5)
    d.box(6,4,81,21,'Controlled search study',
          'Requested entries from the frozen\nexact-likelihood score table',family='accent',title_size=9.3)
    d.box(96,4,81,21,'Within each LOBO training fold',
          'Profile, select and fit using\nretained boreholes only',family='accent',title_size=9.3)
    return d.export()


if __name__=='__main__':
    qa={'overall':overall(),'agent':agent()}
    (HERE/'export_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ('width_mm','height_mm','pdf_min_font_pt','png_size')}
                      for k,v in qa.items()},indent=2))
