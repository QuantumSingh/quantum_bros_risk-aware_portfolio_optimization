from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/project_pipeline_explained.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', fontName='Helvetica-Bold', fontSize=25, leading=29, textColor=colors.HexColor('#16334A'), spaceAfter=14))
styles.add(ParagraphStyle(name='SectionCustom', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=colors.HexColor('#16334A'), spaceBefore=18, spaceAfter=9, keepWithNext=True))
styles.add(ParagraphStyle(name='SubCustom', fontName='Helvetica-Bold', fontSize=11, leading=15, spaceBefore=10, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=10.5, leading=14.8, spaceAfter=7, splitLongWords=True, allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle(name='BulletCustom', parent=styles['BodyCustom'], leftIndent=12, firstLineIndent=-8, spaceAfter=5))
styles.add(ParagraphStyle(name='SourceCustom', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#596671'), spaceBefore=2, spaceAfter=9, splitLongWords=True))
styles.add(ParagraphStyle(name='FormulaCustom', parent=styles['BodyCustom'], fontName='Helvetica-Bold', leftIndent=12, borderColor=colors.HexColor('#D4DEE5'), borderWidth=0.5, borderPadding=9, spaceBefore=5, spaceAfter=14))

class Pipeline(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width=500
        self.height=146
    def draw(self):
        c=self.canv
        labels=[('01','Market data'),('02','Daily returns'),('03','Train / test split'),('04','History windows'),('05','Train models'),('06','Backtest'),('07','Risk and returns'),('08','Checks and reports')]
        for i,(num,label) in enumerate(labels):
            row,col=divmod(i,4)
            x=col*126; y=83-row*68
            c.setFillColor(colors.HexColor('#F1F5F7')); c.setStrokeColor(colors.HexColor('#CDD8DF'))
            c.roundRect(x,y,116,53,5,fill=1,stroke=1)
            c.setFillColor(colors.HexColor('#466878')); c.setFont('Helvetica-Bold',9); c.drawString(x+10,y+35,num)
            c.setFillColor(colors.HexColor('#16334A')); c.setFont('Helvetica',9); c.drawString(x+10,y+16,label)
            if col<3:
                c.setStrokeColor(colors.HexColor('#728896')); c.line(x+117,y+26,x+125,y+26)
                c.line(x+122,y+29,x+125,y+26); c.line(x+122,y+23,x+125,y+26)
        c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#596671')); c.drawString(0,0,'Read left to right across each row. The second row continues the pipeline.')

def footer(c,doc):
    w,h=doc.pagesize
    c.saveState(); c.setStrokeColor(colors.HexColor('#D4DEE5')); c.line(54,43,w-54,43)
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#596671'))
    c.drawString(54,30,'QUANTUM BROS  /  PROJECT PIPELINE GUIDE')
    c.drawRightString(w-54,30,str(doc.page))
    c.restoreState()

story=[]
for line in (ROOT/'tmp/pdfs/project_explanation.txt').read_text(encoding='utf-8').splitlines():
    if not line.strip(): continue
    if line=='@diagram':
        story.extend([Spacer(1,7),Pipeline(),Spacer(1,5)]); continue
    if line.startswith('# '): style='TitleCustom'; line=line[2:]
    elif line.startswith('## '): style='SectionCustom'; line=line[3:]
    elif line.startswith('### '): style='SubCustom'; line=line[4:]
    elif line.startswith('- '): style='BulletCustom'; line='- '+line[2:]
    elif line.startswith('> '): style='FormulaCustom'; line=line[2:]
    elif line.startswith('Source'): style='SourceCustom'
    else: style='BodyCustom'
    story.append(Paragraph(escape(line),styles[style]))
doc=SimpleDocTemplate(str(OUT),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=49,bottomMargin=58,title='Risk-aware portfolio optimization: Project pipeline explained',author='Project walkthrough',subject='Plain-language explanation of data, training, quantum models, evaluation, and research outputs')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
pdf=pymupdf.open(OUT)
render=ROOT/'tmp/pdfs/render'; render.mkdir(parents=True,exist_ok=True)
for i,page in enumerate(pdf):
    page.get_pixmap(matrix=pymupdf.Matrix(1.15,1.15)).save(render/f'page-{i+1:02d}.png')
    text=page.get_text()
    assert text.strip(),f'Blank page {i+1}'
    for block in page.get_text('blocks'):
        assert block[0]>=40 and block[2]<=570 and block[1]>=30 and block[3]<=770,(i+1,block)
print(f'Created {OUT}; {len(pdf)} pages; {sum(len(p.get_text().split()) for p in pdf)} extracted words')
from PIL import Image,ImageOps,ImageDraw
thumbs=[]
for file in [render/f'page-{i+1:02d}.png' for i in range(len(pdf))]:
    im=Image.open(file).convert('RGB'); im.thumbnail((306,396))
    tile=Image.new('RGB',(326,426),'#dce1e5'); tile.paste(im,((326-im.width)//2,10))
    ImageDraw.Draw(tile).text((12,408),file.stem,fill='black'); thumbs.append(tile)
sheet=Image.new('RGB',(326*3,426*((len(thumbs)+2)//3)),'white')
for i,im in enumerate(thumbs): sheet.paste(im,((i%3)*326,(i//3)*426))
sheet.save(render/'contact-sheet.png')
