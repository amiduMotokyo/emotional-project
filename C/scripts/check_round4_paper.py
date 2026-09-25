"""Read-only Word pagination check and preview export for the paper fragment."""
import hashlib
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from C.scripts.export_round4_paper import SOURCE, ROOT
import win32com.client

def main():
    assets=SOURCE.parent/'assets/round4'
    markdown=SOURCE.read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',markdown):
        if not link.startswith('http'):assert (SOURCE.parent/link).exists(),link
    from PIL import Image
    for p in assets.glob('fig*.png'):
        with Image.open(p) as im:assert min(im.info['dpi'])>299
    app=win32com.client.DispatchEx('Word.Application')
    app.Visible=False;app.DisplayAlerts=0;app.AutomationSecurity=3
    d=None
    try:
        d=app.Documents.Open(str(SOURCE.with_suffix('.docx')),ReadOnly=True,AddToRecentFiles=False)
        d.Repaginate()
        pics=[d.InlineShapes(i).Range.Information(3) for i in range(1,d.InlineShapes.Count+1)]
        captions=[p.Range.Information(3) for p in d.Paragraphs if re.match('图C4-[0-9] ',p.Range.Text)]
        tables=[]
        for i in range(1,d.Tables.Count+1):
            r=d.Tables(i).Range.Duplicate
            start=d.Range(r.Start,r.Start).Information(3)
            end=d.Range(r.End-1,r.End-1).Information(3)
            assert start==end,(i,start,end)
            tables.append(start)
        assert pics==captions and len(pics)==5
        assert d.OMaths.Count==3 and d.Tables.Count==2
        preview=ROOT/'C/outputs/q2_round4_paper_preview';preview.mkdir(exist_ok=True)
        d.ExportAsFixedFormat(str(preview/'preview.pdf'),17)
        result=dict(pages=d.ComputeStatistics(2),figures=len(pics),tables=len(tables),equations=d.OMaths.Count,
                    figure_pages=pics,caption_pages=captions,table_pages=tables,
                    docx_sha256=hashlib.sha256(SOURCE.with_suffix('.docx').read_bytes()).hexdigest())
        (assets/'word_layout_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(result,flush=True)
    finally:
        if d is not None:d.Close(0)
        app.Quit(0)

if __name__=='__main__':main()
