"""
📝 족보닷컴 통합 시험지 PDF 생성기
- 배포그룹별 PDF에서 QR코드 자동 추출
- 시험지 QR코드 자동 제거
- 정답/해설 자동 감지 & 분리/합본 출력
- 맨뒷장 템플릿에 그룹별 QR코드 배치

플래닛 과학학원 전용
"""

import streamlit as st
from pypdf import PdfReader, PdfWriter
from PIL import Image, ImageDraw
from pyzbar.pyzbar import decode as pyzbar_decode
from pdf2image import convert_from_bytes
from reportlab.pdfgen import canvas as rl_canvas
import io
import tempfile
import os
import re
import numpy as np
from datetime import datetime

Image.MAX_IMAGE_PIXELS = 200_000_000

MAX_GROUPS = 6
RENDER_DPI = 200
DEFAULT_TEMPLATE = "맨뒷장_템플릿.pdf"
ANSWER_KEYWORDS = ['정답 및 해설', '정답및해설', '정답']


# ──────────────────────────────────────────
# 파일명 생성
# ──────────────────────────────────────────

def clean_exam_name(filename):
    """족보닷컴 파일명에서 배포그룹번호와 날짜를 제거"""
    name = filename
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    # (그룹번호)_날짜
    name = re.sub(r'\(\d+\)_\d{4}-\d{2}-\d{2}$', '', name)
    # __그룹번호__날짜
    name = re.sub(r'__\d+__\d{4}-\d{2}-\d{2}$', '', name)
    # _그룹번호_날짜
    name = re.sub(r'_\(\d+\)_\d{4}-\d{2}-\d{2}$', '', name)
    return name.rstrip('_ ').strip()


def make_download_name(original_filename, label):
    """
    다운로드 파일명 생성
    label: '문제', '정답', '문제+정답'
    결과: YYMMDDHHMM_[label]_정리된파일명.pdf
    """
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    base = clean_exam_name(original_filename)
    return f"{timestamp}_[{label}]_{base}.pdf"


# ──────────────────────────────────────────
# 정답 페이지 감지
# ──────────────────────────────────────────

def find_answer_start_page(pdf_bytes):
    """정답/해설 시작 페이지 인덱스 반환 (0-based), 못찾으면 None"""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").replace(" ", "")
        for kw in ANSWER_KEYWORDS:
            if kw.replace(" ", "") in text:
                return i
    return None


# ──────────────────────────────────────────
# 템플릿 박스 위치 자동 감지
# ──────────────────────────────────────────

def detect_box_positions(template_img):
    arr = np.array(template_img)
    h, w = arr.shape[:2]

    vborders = _find_dark_lines(np.mean(arr[int(0.70*h), :, :], axis=1))
    vborders = [b for b in vborders if b[0] > w*0.1 and b[1] < w*0.95]

    hborders = _find_dark_lines(np.mean(arr[:, int(0.25*w), :], axis=1))
    hborders = [b for b in hborders if b[0] > h*0.60]

    cols = _borders_to_inner(vborders)
    rows = _borders_to_inner(hborders)

    if len(cols) < 3:
        cols = [(int(w*r), int(w*(r+0.14))) for r in [0.17, 0.43, 0.69]]
    if len(rows) < 2:
        rows = [(int(h*0.658), int(h*0.750)), (int(h*0.824), int(h*0.916))]

    cols = sorted(sorted(cols, key=lambda x: x[1]-x[0], reverse=True)[:3])
    rows = sorted(sorted(rows, key=lambda x: x[1]-x[0], reverse=True)[:2])

    return [(c1, r1, c2, r2) for r1, r2 in rows for c1, c2 in cols]


def _find_dark_lines(gray, max_w=60, thresh=100):
    idx = np.where(gray < thresh)[0]
    if len(idx) == 0: return []
    groups, s = [], idx[0]
    for i in range(1, len(idx)):
        if idx[i] - idx[i-1] > 3:
            if idx[i-1] - s + 1 <= max_w: groups.append((s, idx[i-1]))
            s = idx[i]
    if idx[-1] - s + 1 <= max_w: groups.append((s, idx[-1]))
    return groups


def _borders_to_inner(borders):
    return [(b1[1]+1, b2[0]-1) for b1, b2 in zip(borders, borders[1:]) if b2[0]-b1[1] > 100]


# ──────────────────────────────────────────
# QR 코드 추출
# ──────────────────────────────────────────

def extract_qr_from_pdf(pdf_bytes):
    qr, url = _qr_embedded(pdf_bytes)
    return (qr, url) if qr else _qr_render(pdf_bytes)


def _qr_embedded(pdf_bytes):
    try:
        for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
            if '/Resources' not in page or '/XObject' not in page['/Resources']: continue
            for k in page['/Resources']['/XObject'].get_object():
                try:
                    obj = page['/Resources']['/XObject'].get_object()[k].get_object()
                    if obj.get('/Subtype') != '/Image': continue
                    w, h, data = int(obj['/Width']), int(obj['/Height']), obj.get_data()
                    f = obj.get('/Filter', ''); fn = str(f[0] if isinstance(f, list) else f)
                    img = None
                    if fn in ('/DCTDecode','/JPXDecode'): img = Image.open(io.BytesIO(data))
                    elif fn == '/FlateDecode':
                        m, b = ('L',1) if str(obj.get('/ColorSpace','/DeviceRGB'))=='/DeviceGray' else ('RGB',3)
                        if len(data) >= w*h*b: img = Image.frombytes(m,(w,h),data[:w*h*b])
                    if img:
                        rgb = img.convert('RGB')
                        for r in pyzbar_decode(rgb):
                            if r.type == 'QRCODE': return rgb, r.data.decode('utf-8', errors='replace')
                except: continue
    except: pass
    return None, None


def _qr_render(pdf_bytes):
    try:
        for p in convert_from_bytes(pdf_bytes, dpi=200):
            for r in pyzbar_decode(p):
                if r.type == 'QRCODE':
                    rc, m = r.rect, 15
                    return p.crop((max(0,rc.left-m),max(0,rc.top-m),
                                   min(p.width,rc.left+rc.width+m),min(p.height,rc.top+rc.height+m))), \
                           r.data.decode('utf-8',errors='replace')
    except: pass
    return None, None


# ──────────────────────────────────────────
# 시험지 QR 제거 & 맨뒷장 합성
# ──────────────────────────────────────────

def remove_qr_from_pages(pdf_bytes, dpi=200):
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    removed = 0
    for p in pages:
        for r in pyzbar_decode(p):
            if r.type == 'QRCODE':
                rc = r.rect; m = int(max(rc.width, rc.height) * 0.15)
                ImageDraw.Draw(p).rectangle([rc.left-m, rc.top-m, rc.left+rc.width+m, rc.top+rc.height+m], fill='white')
                removed += 1
    return pages, removed


def compose_back_page(template_bytes, qr_images, num_groups):
    result = convert_from_bytes(template_bytes, dpi=RENDER_DPI)[0].copy().convert('RGB')
    boxes = detect_box_positions(result)
    draw = ImageDraw.Draw(result)
    for gnum in range(1, min(MAX_GROUPS+1, len(boxes)+1)):
        l,t,r,b = boxes[gnum-1]; bw,bh = r-l,b-t
        if gnum in qr_images:
            pad = int(min(bw,bh)*0.08); sz = min(bw-2*pad, bh-2*pad)
            qr = qr_images[gnum].convert('RGB').resize((sz,sz), Image.NEAREST)
            draw.rectangle([l+2,t+2,r-2,b-2], fill='white')
            result.paste(qr, (l+(bw-sz)//2, t+(bh-sz)//2))
        elif gnum > num_groups:
            draw.rectangle([l+2,t+2,r-2,b-2], fill='#F0F0F0')
            lw,m = max(3,int(bw*0.02)), int(bw*0.15)
            draw.line([(l+m,t+m),(r-m,b-m)], fill='#CCCCCC', width=lw)
            draw.line([(r-m,t+m),(l+m,b-m)], fill='#CCCCCC', width=lw)
    return result


def images_to_pdf_bytes(images, pw, ph):
    buf = io.BytesIO(); tmps = []
    try:
        c = rl_canvas.Canvas(buf, pagesize=(pw,ph))
        for img in images:
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(tmp.name, 'PNG', optimize=True); tmps.append(tmp.name)
            c.drawImage(tmp.name, 0,0, width=pw, height=ph, preserveAspectRatio=True, anchor='c')
            c.showPage()
        c.save()
    finally:
        for f in tmps:
            try: os.unlink(f)
            except: pass
    buf.seek(0); return buf.read()


def load_default_template():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_TEMPLATE)
    if os.path.exists(p):
        with open(p, 'rb') as f: return f.read()
    return None


# ──────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────

def main():
    st.set_page_config(page_title="족보 통합시험지", page_icon="📝", layout="wide")

    st.markdown(
        '<h2 style="margin-bottom:0;">족보닷컴 통합 시험지 PDF 생성기</h2>',
        unsafe_allow_html=True
    )
    st.caption("배포그룹 시험지 → QR 추출 → QR 제거 → 맨뒷장 합성 → 통합 PDF")

    default_template = load_default_template()

    # ── 사이드바: 템플릿만 ──
    with st.sidebar:
        st.header("⚙️ 설정")
        if default_template:
            st.success("✅ 기본 템플릿 적용됨")
            override = st.file_uploader("템플릿 변경 (선택사항)", type=['pdf'])
            template_bytes = override.read() if override else default_template
            if override: override.seek(0)
        else:
            st.warning("기본 템플릿 없음")
            uploaded_t = st.file_uploader("맨뒷장 템플릿 PDF", type=['pdf'])
            if not uploaded_t:
                st.info("맨뒷장_템플릿.pdf를 GitHub 레포에 넣으면 자동 로드됩니다.")
                template_bytes = None
            else:
                template_bytes = uploaded_t.read(); uploaded_t.seek(0)

        st.divider()
        st.markdown("""
        **자동 처리 항목**
        - 시험지 QR코드 제거
        - 정답/해설 페이지 자동 감지
        - 맨뒷장 QR코드 배치
        """)
        st.divider()
        if st.button("🔄 초기화", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── 메인 ──
    st.subheader("📁 배포그룹 시험지 업로드")

    uploaded_pdfs = st.file_uploader(
        "배포그룹 PDF 파일들 (최대 6개)",
        type=['pdf'],
        accept_multiple_files=True,
    )

    # 파일 변경 감지 → 이전 결과 초기화
    current_files = tuple(f.name for f in uploaded_pdfs) if uploaded_pdfs else ()
    if 'prev_files' not in st.session_state:
        st.session_state.prev_files = ()
    if current_files != st.session_state.prev_files:
        st.session_state.prev_files = current_files
        st.session_state.pop('results', None)

    if not uploaded_pdfs:
        st.info("👆 족보닷컴에서 다운로드한 배포그룹별 시험지 PDF를 업로드해주세요.")
        return

    if len(uploaded_pdfs) > MAX_GROUPS:
        st.error(f"최대 {MAX_GROUPS}개까지 업로드 가능합니다.")
        return

    if template_bytes is None:
        st.warning("⬅️ 사이드바에서 맨뒷장 템플릿을 업로드해주세요.")
        return

    # ── 그룹 배정 (접히는 형태) ──
    sorted_pdfs = sorted(uploaded_pdfs, key=lambda f: f.name)

    st.caption(f"✅ {len(sorted_pdfs)}개 배포그룹 감지됨 (파일명 순서대로 자동 배정)")

    group_assignments = {}
    with st.expander("배포그룹 순서 변경", expanded=False):
        cols = st.columns(min(len(sorted_pdfs), 3))
        for i, pdf_file in enumerate(sorted_pdfs):
            with cols[i % 3]:
                name = pdf_file.name
                if len(name) > 40: name = name[:18] + "..." + name[-18:]
                group_assignments[i] = st.selectbox(
                    f"📄 {name}", list(range(1, MAX_GROUPS+1)),
                    index=i, key=f"grp_{i}"
                )

    # 기본값 (expander 안 열었을 때)
    if not group_assignments:
        for i in range(len(sorted_pdfs)):
            group_assignments[i] = i + 1

    if len(set(group_assignments.values())) != len(group_assignments):
        st.error("⚠️ 배포그룹 번호가 중복됩니다!")
        return

    st.divider()

    # ── 정답지 분리 토글 ──
    # 토글 크기 키우는 CSS
    st.markdown("""
    <style>
    div[data-testid="stToggle"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 1.15rem;
        font-weight: 600;
    }
    div[data-testid="stToggle"] > label > div:last-child {
        transform: scale(1.3);
        transform-origin: center;
    }
    </style>
    """, unsafe_allow_html=True)

    separate_answers = st.toggle(
        "정답지 분리 출력",
        value=True,
        help="ON: 시험지 + 정답지 2개 파일 / OFF: 전부 합쳐서 1개 파일",
    )

    if separate_answers:
        st.caption("시험지(문제 + 맨뒷장)와 정답지가 **각각** 생성됩니다.")
    else:
        st.caption("문제 + QR안내 + 정답이 **하나의 PDF**로 생성됩니다.")

    st.divider()

    if st.button("🚀 통합 시험지 생성", type="primary", use_container_width=True):
        _run(sorted_pdfs, group_assignments, template_bytes, separate_answers)

    # ── 결과 ──
    if 'results' in st.session_state and st.session_state.results:
        _show_results(sorted_pdfs)


def _run(sorted_pdfs, group_assignments, template_bytes, separate_answers):
    n = len(sorted_pdfs)
    steps = n + 4
    progress = st.progress(0, text="시작...")

    # 1. QR 추출
    qr_images, errors = {}, []
    for i, f in enumerate(sorted_pdfs):
        gnum = group_assignments[i]
        progress.progress((i+1)/steps, text=f"배포그룹 {gnum} QR 추출... ({i+1}/{n})")
        data = f.read(); f.seek(0)
        qr, url = extract_qr_from_pdf(data)
        if qr: qr_images[gnum] = qr
        else: errors.append(str(gnum))

    if errors:
        st.error(f"❌ QR 추출 실패: 배포그룹 {', '.join(errors)}")
        progress.empty(); return

    st.subheader("🔍 추출된 QR 코드")
    qcols = st.columns(min(len(qr_images), 6))
    for idx, gnum in enumerate(sorted(qr_images)):
        with qcols[idx]:
            st.image(qr_images[gnum], caption=f"{gnum}그룹", width=120)

    # 2. 정답 감지
    step = n + 1
    progress.progress(step/steps, text="정답/해설 페이지 감지...")

    first_data = sorted_pdfs[0].read(); sorted_pdfs[0].seek(0)
    reader = PdfReader(io.BytesIO(first_data))
    total = len(reader.pages)
    pw, ph = float(reader.pages[0].mediabox.width), float(reader.pages[0].mediabox.height)

    ans_start = find_answer_start_page(first_data)
    if ans_start is None:
        ans_start = max(1, total - 1)
        st.caption(f"📄 전체 {total}p — 정답 키워드 미감지, 마지막 {total-ans_start}p를 정답으로 처리")
    else:
        st.caption(f"📄 전체 {total}p — 문제 {ans_start}p + 정답/해설 {total-ans_start}p")

    # 3. QR 제거
    step += 1
    progress.progress(step/steps, text="시험지 QR 제거...")
    all_imgs, removed = remove_qr_from_pages(first_data, dpi=RENDER_DPI)
    q_imgs, a_imgs = all_imgs[:ans_start], all_imgs[ans_start:]
    if removed: st.caption(f"✂️ QR코드 {removed}개 제거됨")

    # 4. 맨뒷장
    step += 1
    progress.progress(step/steps, text="맨뒷장 QR 합성...")
    back = compose_back_page(template_bytes, qr_images, n)

    # 5. PDF 생성
    step += 1
    progress.progress(step/steps, text="PDF 생성...")

    original_name = sorted_pdfs[0].name

    if separate_answers:
        exam_pdf = images_to_pdf_bytes(q_imgs + [back], pw, ph)
        answer_pdf = images_to_pdf_bytes(a_imgs, pw, ph)
        st.session_state.results = {
            'mode': 'separate',
            'exam_pdf': exam_pdf,
            'answer_pdf': answer_pdf,
            'original_name': original_name,
            'q_count': len(q_imgs),
            'a_count': len(a_imgs),
            'n_groups': n,
        }
    else:
        combined = images_to_pdf_bytes(q_imgs + [back] + a_imgs, pw, ph)
        st.session_state.results = {
            'mode': 'combined',
            'combined_pdf': combined,
            'original_name': original_name,
            'total': len(q_imgs) + len(a_imgs) + 1,
            'n_groups': n,
        }

    progress.progress(1.0, text="완료!")
    progress.empty()
    st.rerun()


def _show_results(sorted_pdfs):
    r = st.session_state.results
    name = r['original_name']

    st.divider()
    st.success("✅ 통합 시험지가 생성되었습니다!")

    if r['mode'] == 'separate':
        st.info(f"📄 시험지: 문제 {r['q_count']}p + 맨뒷장 1p | 📋 정답지: {r['a_count']}p | 배포그룹 {r['n_groups']}개")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"⬇️ 시험지 다운로드 ({len(r['exam_pdf'])/1024:.0f}KB)",
                data=r['exam_pdf'],
                file_name=make_download_name(name, '문제'),
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        with col2:
            st.download_button(
                label=f"⬇️ 정답지 다운로드 ({len(r['answer_pdf'])/1024:.0f}KB)",
                data=r['answer_pdf'],
                file_name=make_download_name(name, '정답'),
                mime="application/pdf",
                use_container_width=True,
            )

        with st.expander("📖 시험지 미리보기", expanded=True):
            try:
                prev = convert_from_bytes(r['exam_pdf'], dpi=100)
                pc = st.columns(len(prev))
                for i, pg in enumerate(prev):
                    with pc[i]: st.image(pg, caption=f"p.{i+1}", use_container_width=True)
            except: pass

        with st.expander("📖 정답지 미리보기"):
            try:
                prev = convert_from_bytes(r['answer_pdf'], dpi=100)
                pc = st.columns(max(1, len(prev)))
                for i, pg in enumerate(prev):
                    with pc[i]: st.image(pg, caption=f"정답 p.{i+1}", use_container_width=True)
            except: pass

    else:
        st.info(f"📄 합본: 총 {r['total']}p | 배포그룹 {r['n_groups']}개")
        st.download_button(
            label=f"⬇️ 다운로드 ({len(r['combined_pdf'])/1024:.0f}KB)",
            data=r['combined_pdf'],
            file_name=make_download_name(name, '문제+정답'),
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
        with st.expander("📖 미리보기", expanded=True):
            try:
                prev = convert_from_bytes(r['combined_pdf'], dpi=100)
                pc = st.columns(min(len(prev), 5))
                for i, pg in enumerate(prev):
                    with pc[i % len(pc)]: st.image(pg, caption=f"p.{i+1}", use_container_width=True)
            except: pass


if __name__ == "__main__":
    main()
