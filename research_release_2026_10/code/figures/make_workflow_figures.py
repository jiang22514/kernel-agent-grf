"""Draw the final manuscript framework with Python/matplotlib.

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
C = {'ink': '#223344', 'line': '#596C7A', 'blue': '#315D83',
     'blue_fill': '#EDF3F8', 'neutral': '#F3F4F5', 'gray': '#65717A',
     'accent': '#82673C', 'accent_fill': '#FAF5E9', 'white': '#FFFFFF'}

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
        patch = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.4',
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
    d.box(6,105,103,27,'AI-agent-guided covariance search',
          'Profile → propose → validate → access AIC scores\nStructured search: four rounds, up to 20 accesses\nSelect the lowest-AIC accessed candidate')
    d.box(120,105,57,27,'Fitted simulation models',
          'Formal comparison:\nSE baseline · best single RQ\nLibrary-best composite')
    d.route([(109,118.5),(120,118.5)])
    d.route([(151.5,145),(151.5,138),(57.5,138),(57.5,132)])
    d.text(92,141.1,'Requested entries only',size=8.5)
    d.route([(151.5,138),(148.5,138),(148.5,132)])
    d.box(6,69,83,23,'SE / composite: exact simulation',
          'Exact coordinate union; separable terms\nOriginal-GP Matheron conditioning\nNumerical fidelity checked for the fit')
    d.box(99,69,78,23,'RQ reference: dense GP',
          'Original fitted covariance definition\nConditional query covariance\nRecorded numerical sampling shift',family='neutral')
    d.route([(148.5,105),(148.5,98),(47.5,98),(47.5,92)])
    d.route([(148.5,98),(138,98),(138,92)])
    d.box(6,29,52,25,'Numerical accuracy',
          'Means + joint covariances\nNumerics and sampling\nchecked separately')
    d.box(65.5,29,52,25,'Fixed-model prediction',
          'Full-data structures fixed\nRefit on training boreholes\nPredict omitted borehole',family='neutral')
    d.box(125,29,52,25,'Spatial functionals',
          'Weak-zone area fraction\nModel-conditional event\nprobability',family='accent')
    d.route([(47.5,69),(47.5,61),(138,61),(138,69)],arrow=False)
    d.route([(47.5,61),(32,61),(32,54)])
    d.route([(138,61),(151,61),(151,54)])
    # Predictive checks refit parameters with held-out observations omitted,
    # while retaining the structures selected with all data.
    d.route([(6,156.5),(2,156.5),(2,22),(91.5,22),(91.5,29)],dashed=True,color=C['gray'])
    d.route([(177,118.5),(181,118.5),(181,22),(91.5,22)],dashed=True,color=C['gray'],arrow=False)
    d.text(6,12,'Solid: search and simulation. Dashed: fixed-structure borehole prediction.',
           ha='left',size=8.5)
    d.text(6,6.5,'Each simulation uses its model\'s fitted kernel, mean and parameters.',
           ha='left',size=8.5,color=C['gray'])
    return d.export()


if __name__=='__main__':
    qa={'overall':overall()}
    (HERE/'export_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ('width_mm','height_mm','pdf_min_font_pt','png_size')}
                      for k,v in qa.items()},indent=2))
