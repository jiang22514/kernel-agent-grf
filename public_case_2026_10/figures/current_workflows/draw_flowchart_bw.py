"""Conceptual diagrams of the accepted coordinate-union method; no data draws."""
from pathlib import Path
import importlib.util,json,hashlib
T=Path(__file__).resolve().parent
OUT=T/'flowchart_bw'; OUT.mkdir(exist_ok=True)
source=T/'workflow_layout_bw.py'
spec=importlib.util.spec_from_file_location('workflow_layout',source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'font.size':8.5, 'svg.fonttype':'none', 'pdf.fonttype':42})
mod.HERE=OUT
D=mod.Diagram
def export_with_math(self):
    qa=self.validate_layout()
    self.fig.savefig(OUT/f'{self.name}.pdf',dpi=600,facecolor='white',edgecolor='none')
    self.fig.savefig(OUT/f'{self.name}.svg',dpi=600,facecolor='white',edgecolor='none')
    self.fig.savefig(OUT/f'{self.name}.png',dpi=600,facecolor='white',edgecolor='none')
    mod.plt.close(self.fig)
    with mod.fitz.open(OUT/f'{self.name}.pdf') as doc:
        assert len(doc)==1
        page=doc[0]
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for ln in b['lines'] for s in ln['spans']]
        # Mathematical subscripts are intentionally smaller than the >=8.5pt
        # body text, whose actual artist sizes are checked above.
        assert spans and min(s['size'] for s in spans)>=5.5
        fonts=page.get_fonts(full=True);assert fonts and all(f[1]!='n/a' for f in fonts)
        page.get_pixmap(dpi=300,alpha=False).save(OUT/f'{self.name}_pdf_preview.png')
        qa.update(pdf_min_glyph_pt=min(s['size'] for s in spans),body_min_font_pt=qa['min_font_pt'],
                  mathematical_subscripts_smaller=True,pdf_fonts=[f[3] for f in fonts])
    svg=mod.ET.parse(OUT/f'{self.name}.svg')
    assert svg.findall('.//{http://www.w3.org/2000/svg}text')
    qa['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in OUT.glob(self.name+'.*') if p.suffix in {'.pdf','.svg','.png'}}
    return qa
D.export=export_with_math

d=D('grid_conditioning_union_b7',164)
d.text(91.5,160,'Conditional simulation of the original fitted GP',size=10,bold=True)
d.box(6,130,171,22,'Fitted model and exact coordinate union',
      'Same kernel, mean, parameters and noise as model selection\n'+
      r'$U_d=\mathrm{unique}(X_{:,d}\cup X_{*,:,d})$;  $U=U_1\times\cdots\times U_D$')
d.box(6,90,81,30,'Separable covariance terms',
      r'$K_{uu}=\sum_{r=1}^{R}\bigotimes_{d=1}^{D}K^{(r,d)}$'+'\n'+
      r'$B_{r,d}B_{r,d}^{T}=K^{(r,d)}$'+'\nNonuniform Cartesian axes are allowed')
d.box(96,90,81,30,'Original GP conditioning matrix',
      r'$A_{\rm ex}=K_{XX}+\sigma_n^2 I_n=LL^{T}$'+'\n'+
      r'$P_X,\ P_*$: exact node selectors'+'\nNo inducing interpolation',family='accent')
d.box(6,49,81,31,'Independent prior contributions',
      r'$u_0=\sum_r(\bigotimes_d B_{r,d})\,\xi_r$'+'\n'+
      r'Independent $\xi_r\sim\mathcal{N}(0,I_m)$'+'\n'+
      r'Independent $\varepsilon_0\sim\mathcal{N}(0,\sigma_n^2I_n)$')
d.box(96,49,81,31,'Reusable training-system solve',
      r'$r_y=y-\mu_X-P_Xu_0-\varepsilon_0$'+'\n'+
      r'$LL^{T}a=r_y$'+'\n'+r'One $n\times n$ factor reused per draw',family='accent')
d.box(6,16,171,23,'Condition the joint draw at the query nodes',
      r'$f_*=\mu_*+P_*u_0+K_{*X}a$'+'\n'+
      r'No added query observation noise; no $q\times q$ covariance factorization',family='neutral')
for a,b in [((46.5,130),(46.5,120)),((136.5,130),(136.5,120)),
            ((46.5,90),(46.5,80)),((136.5,90),(136.5,80)),
            ((87,64.5),(96,64.5)),((46.5,49),(46.5,39)),((136.5,49),(136.5,39))]:
    d.route([a,b])
d.text(91.5,9,r'$n$: observations; $m$: union nodes; $q$: query nodes; $R$: separable terms',size=8.5)
d.text(91.5,3.5,'Original-GP identity; factor roundoff and Monte Carlo agreement are checked separately.',size=8.5)
qa={'conditioning':d.export()}


(OUT/'diagram_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','output':str(OUT)}))
