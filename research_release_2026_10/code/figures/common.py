"""Shared plotting I/O. Never fit, draw new fields, call models, or copy to submission."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import numpy as np

HERE=Path(__file__).resolve().parent
import os
ROOT=Path(os.environ.get('FIGURE_INPUT_ROOT', str(Path(__file__).resolve().parents[2]))).resolve()
CASES=['EX1','EX2','EX3','EX4','Borehole']


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=clean),encoding='utf-8')


def csv_write(path,rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def output_dir(path):
    out=Path(path).resolve()
    need(out.is_relative_to(HERE) and out!=HERE,'Output must be a new child folder of this figures directory')
    need(not out.exists(),'Use a new output folder; existing figures/evidence are preserved')
    out.mkdir(parents=True)
    return out


def display(case):return 'CPT' if case=='Borehole' else case


def plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,'legend.fontsize':7,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'axes.linewidth':.65,'figure.facecolor':'white','savefig.facecolor':'white'})
    return plt


def save_figure(fig,out,name):
    fig.canvas.draw()
    fig.savefig(out/(name+'.pdf'),facecolor='white')
    fig.savefig(out/(name+'.svg'),facecolor='white')
    fig.savefig(out/(name+'.png'),dpi=600,facecolor='white')


class Sources:
    def __init__(self):self.files={}
    def add(self,path,expected=None):
        path=Path(path).resolve();need(path.is_file(),'Missing source: '+str(path))
        stat=path.stat();sig=[stat.st_size,stat.st_mtime_ns]
        if str(path) not in self.files:self.files[str(path)]={'sha256':sha(path),'signature':sig}
        else:need(self.files[str(path)]['signature']==sig,'Source changed during read')
        digest=self.files[str(path)]['sha256']
        if expected is not None:need(digest==str(expected).strip().lower(),'Source hash mismatch: '+path.name)
        return digest
    def finish(self,out,detail):
        for p,v in self.files.items():
            st=Path(p).stat();need([st.st_size,st.st_mtime_ns]==v['signature'],'Source changed while plotting')
        detail.update(sources=self.files,rendered_visual_qa='PENDING actual rendered-file inspection',
            science_recomputed=False,api_calls=0,submission_files_modified=False)
        write(out/'figure_qa.json',detail)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
