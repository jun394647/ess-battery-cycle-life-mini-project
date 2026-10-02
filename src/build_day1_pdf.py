"""Render the Day 1 or Day 2 Markdown report as a Korean PDF."""
import argparse
from pathlib import Path
import html
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--day',type=int,choices=(1,2),default=1)
DAY=parser.parse_args().day
SOURCE=ROOT/f'DAY{DAY}-REPORT.md'
DEST=ROOT/(f'DS-MINI-Design-울산_1반-박준형-DAY2.pdf' if DAY==2 else 'DS-MINI-Design-울산_1반-박준형.pdf')
system_font=Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf')
body_font=system_font if system_font.exists() else ROOT/'assets/fonts/NanumGothic-Regular.ttf'
pdfmetrics.registerFont(TTFont('AppleGothic',str(body_font)))
pdfmetrics.registerFontFamily('AppleGothic',normal='AppleGothic',bold='AppleGothic',italic='AppleGothic',boldItalic='AppleGothic')
pdfmetrics.registerFont(TTFont('NanumGothicBold',str(ROOT/'assets/fonts/NanumGothic-Bold.ttf')))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='KTitle',fontName='NanumGothicBold',fontSize=19,leading=28,alignment=TA_CENTER,textColor=colors.HexColor('#18324b'),spaceAfter=14))
styles.add(ParagraphStyle(name='KSubtitle',fontName='AppleGothic',fontSize=9.2,leading=15,alignment=TA_RIGHT,spaceAfter=10))
styles.add(ParagraphStyle(name='KHeading',fontName='NanumGothicBold',fontSize=12.5,leading=18,textColor=colors.HexColor('#18324b'),spaceBefore=12,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='KSubheading',fontName='NanumGothicBold',fontSize=10.5,leading=16,textColor=colors.HexColor('#29485c'),spaceBefore=9,spaceAfter=4,keepWithNext=True))
styles.add(ParagraphStyle(name='KBody',fontName='AppleGothic',fontSize=9.2,leading=15,spaceAfter=7,wordWrap='CJK'))
styles.add(ParagraphStyle(name='KSmall',fontName='AppleGothic',fontSize=7.8,leading=12,wordWrap='CJK'))
styles.add(ParagraphStyle(name='KCaption',fontName='AppleGothic',fontSize=8,leading=12,alignment=TA_CENTER,textColor=colors.HexColor('#536b7a'),spaceAfter=10))
styles.add(ParagraphStyle(name='KPanel',fontName='AppleGothic',fontSize=8.6,leading=14,wordWrap='CJK'))
styles.add(ParagraphStyle(name='KTocNumber',fontName='AppleGothic',fontSize=9,leading=13,textColor=colors.HexColor('#15689a')))
styles.add(ParagraphStyle(name='KTocBody',fontName='AppleGothic',fontSize=8.7,leading=13,wordWrap='CJK'))
if DAY==2:
    styles['KBody'].spaceAfter=5
    styles['KHeading'].spaceBefore=9
    styles['KSubheading'].spaceBefore=7

def inline(text):
    text=html.escape(text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<link href="\2" color="#15689a">\1</link>',text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font color="#345c74">\1</font>',text)
    return text

def page_number(canvas,doc):
    canvas.saveState(); canvas.setFont('AppleGothic',8)
    canvas.setFillColor(colors.HexColor('#6f7e88'))
    canvas.drawString(43,27,f'ESS battery cycle life · Day {DAY}')
    canvas.drawRightString(A4[0]-43,27,str(doc.page))
    canvas.restoreState()

story=[]
lines=SOURCE.read_text().splitlines()
i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:
        i+=1;continue
    if line == '<!-- PAGE BREAK -->':
        story.append(PageBreak())
        i+=1;continue
    if line in ('## 요약', '## 목차'):
        heading=line[3:]
        story.append(Paragraph(heading,styles['KHeading']))
        i+=1
        while i<len(lines) and not lines[i].strip(): i+=1
        items=[]
        while i<len(lines) and lines[i].strip().startswith('- '):
            raw=lines[i]
            item=raw.strip()[2:]
            if heading=='목차' and raw.startswith('  ') and items:
                items[-1]['subitems'].append(item)
            elif heading=='목차':
                items.append({'title':item,'subitems':[]})
            else:
                items.append(item)
            i+=1
        if heading=='요약':
            rows=[[Paragraph(inline(item),styles['KPanel'])] for item in items]
            panel=Table(rows,colWidths=[A4[0]-86],hAlign='LEFT')
            panel_style=[('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#edf4f8')),
                         ('BOX',(0,0),(-1,-1),.7,colors.HexColor('#c6dce8')),
                         ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
                         ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]
            for row in range(len(rows)-1):
                panel_style.append(('LINEBELOW',(0,row),(-1,row),.35,colors.HexColor('#d5e5ed')))
        else:
            rows=[[Paragraph(f'{n:02d}',styles['KTocNumber']),
                   Paragraph(inline(item['title'])+'<br/><font color="#667986" size="7.7">'+
                             ('<br/>'.join if len(item['subitems'])>2 else ' · '.join)
                             (inline(sub) for sub in item['subitems'])+'</font>',styles['KTocBody'])]
                  for n,item in enumerate(items,1)]
            panel=Table(rows,colWidths=[45,A4[0]-131],hAlign='LEFT')
            panel_style=[('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f7f9fb')),
                         ('BOX',(0,0),(-1,-1),.7,colors.HexColor('#d8e2e9')),
                         ('VALIGN',(0,0),(-1,-1),'TOP'),
                         ('LEFTPADDING',(0,0),(0,-1),12),('RIGHTPADDING',(0,0),(0,-1),2),
                         ('LEFTPADDING',(1,0),(1,-1),3),('RIGHTPADDING',(1,0),(1,-1),10),
                         ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]
        panel.setStyle(TableStyle(panel_style))
        story.extend([panel,(PageBreak() if DAY==1 else Spacer(1,18)) if heading=='목차' else Spacer(1,12)])
        continue
    if line.startswith('![어') or line.startswith('!['):
        m=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if m:
            path=ROOT/m.group(2)
            if not path.exists(): raise FileNotFoundError(path)
            iw,ih=ImageReader(str(path)).getSize()
            width=min(A4[0]-86,445);height=width*ih/iw
            if height>340:height=340;width=height*iw/ih
            story.append(KeepTogether([Image(str(path),width=width,height=height),
                                       Paragraph(inline(m.group(1)),styles['KCaption'])]))
        i+=1;continue
    if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['KTitle']));i+=1;continue
    if line.startswith('**DS-MINI-Design'):
        story.append(Paragraph(inline(line),styles['KSubtitle']));i+=1;continue
    if line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['KSubheading']));i+=1;continue
    if line.startswith('## '):
        story.append(Paragraph(inline(line[3:]),styles['KHeading']));i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            items=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(set(x)<=set(':- ') for x in items): rows.append(items)
            i+=1
        if rows:
            col_count=max(map(len,rows))
            widths=[(A4[0]-86)/col_count]*col_count
            data=[[Paragraph(inline(r[j]) if j<len(r) else '',styles['KSmall']) for j in range(col_count)] for r in rows]
            tbl=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f4')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafb')]),
                ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d8e0e5')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            table_block=[tbl,Spacer(1,9)]
            if DAY==2 and len(rows)<=10:
                story.append(KeepTogether(table_block))
            else:
                story.extend(table_block)
        continue
    if line.startswith('- '):
        story.append(Paragraph('• '+inline(line[2:]),styles['KBody']));i+=1;continue
    para=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
        para.append(lines[i].strip());i+=1
    story.append(Paragraph(inline(' '.join(para)),styles['KBody']))

doc=SimpleDocTemplate(str(DEST),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=42,bottomMargin=45,
                      title=f'DS, Mini Project : ESS 배터리 수명 예측 (DAY{DAY})',author='박준형(U015)')
doc.build(story,onFirstPage=page_number,onLaterPages=page_number)
print(DEST)
