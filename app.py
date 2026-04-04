"""
📝 족보닷컴 통합 시험지 PDF 생성기 v4
- 배포그룹별 PDF에서 QR코드 자동 추출
- 시험지 QR코드 자동 제거
- 정답/해설 자동 감지 & 분리 출력
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

Image.MAX_IMAGE_PIXELS = 200_000_000

MAX_GROUPS = 6
RENDER_DPI = 200
DEFAULT_TEMPLATE = "맨뒷장_템플릿.pdf"

# 정답 페이지 감지 키워드
ANSWER_KEYWORDS = ['정답 및 해설', '정답및해설', '정답']


# ──────────────────────────────────────────
# 정답 페이지 감지
# ──────────────────────────────────────────

def find_answer_start_page(pdf_bytes):
    """
    PDF에서 정답/해설이 시작되는 페이지 인덱스를 찾음
    Returns: 정답 시작 페이지 인덱스 (0-based), 못찾으면 None
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # 공백 제거 후 비교
        text_clean = text.replace(" ", "")
        for kw in ANSWER_KEYWORDS:
            kw_clean = kw.replace(" ", "")
            if kw_clean in text_clean:
                return i
    return None


# ──────────────────────────────────────────
# 템플릿 박스 위치 자동 감지
# ──────────────────────────────────────────

def detect_box_positions(template_img):
    """템플릿 이미지에서 6개 QR 박스 border를 감지하여 내부 좌표 반환"""
    arr = np.array(template_img)
    h, w = arr.shape[:2]

    y_scan = int(0.70 * h)
    line_gray = np.mean(arr[y_scan, :, :], axis=1)
    vborders = _find_dark_lines(line_gray)
    vborders = [b for b in vborders if b[0] > w * 0.1 and b[1] < w * 0.95]

    x_scan = int(0.25 * w)
    col_gray = np.mean(arr[:, x_scan, :], axis=1)
    hborders = _find_dark_lines(col_gray)
    hborders = [b for b in hborders if b[0] > h * 0.60]

    col_ranges = _borders_to_inner(vborders)
    row_ranges = _borders_to_inner(hborders)

    if len(col_ranges) < 3:
        col_ranges = [(int(w*r), int(w*(r+0.14))) for r in [0.17, 0.43, 0.69]]
    if len(row_ranges) < 2:
        row_ranges = [(int(h*0.658), int(h*0.750)), (int(h*0.824), int(h*0.916))]

    col_ranges = sorted(sorted(col_ranges, key=lambda x: x[1]-x[0], reverse=True)[:3])
    row_ranges = sorted(sorted(row_ranges, key=lambda x: x[1]-x[0], reverse=True)[:2])

    boxes = []
    for rs, re_ in row_ranges:
        for cs, ce in col_ranges:
            boxes.append((cs, rs, ce, re_))
    return boxes


def _find_dark_lines(gray_array, max_width=60, threshold=100):
    dark = gray_array < threshold
    indices = np.where(dark)[0]
    if len(indices) == 0:
        return []
    groups, start = [], indices[0]
    for i in range(1, len(indices)):
        if indices[i] - indices[i-1] > 3:
            if indices[i-1] - start + 1 <= max_width:
                groups.append((start, indices[i-1]))
            start = indices[i]
    if indices[-1] - start + 1 <= max_width:
        groups.append((start, indices[-1]))
    return groups


def _borders_to_inner(borders):
    return [(b1[1]+1, b2[0]-1) for b1, b2 in zip(borders, borders[1:])
            if b2[0] - b1[1] > 100]


# ──────────────────────────────────────────
# QR 코드 추출
# ──────────────────────────────────────────

def extract_qr_from_pdf(pdf_bytes):
    qr, url = _qr_from_embedded(pdf_bytes)
    if qr:
        return qr, url
    return _qr_from_render(pdf_bytes)


def _qr_from_embedded(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            if '/Resources' not in page or '/XObject' not in page['/Resources']:
                continue
            for obj_name in page['/Resources']['/XObject'].get_object():
                try:
                    obj = page['/Resources']['/XObject'].get_object()[obj_name].get_object()
                    if obj.get('/Subtype') != '/Image':
                        continue
                    w, h = int(obj['/Width']), int(obj['/Height'])
                    data = obj.get_data()
                    f = obj.get('/Filter', '')
                    fn = str(f[0] if isinstance(f, list) else f)
                    img = None
                    if fn in ('/DCTDecode', '/JPXDecode'):
                        img = Image.open(io.BytesIO(data))
                    elif fn == '/FlateDecode':
                        cs = str(obj.get('/ColorSpace', '/DeviceRGB'))
                        mode, bpp = ('L', 1) if cs == '/DeviceGray' else ('RGB', 3)
                        if len(data) >= w * h * bpp:
                            img = Image.frombytes(mode, (w, h), data[:w*h*bpp])
                    if img:
                        rgb = img.convert('RGB')
                        for r in pyzbar_decode(rgb):
                            if r.type == 'QRCODE':
                                return rgb, r.data.decode('utf-8', errors='replace')
                except Exception:
                    continue
    except Exception:
        pass
    return None, None


def _qr_from_render(pdf_bytes):
    try:
        for page_img in convert_from_bytes(pdf_bytes, dpi=200):
            for r in pyzbar_decode(page_img):
                if r.type == 'QRCODE':
                    rect = r.rect
                    m = 15
                    crop = page_img.crop((
                        max(0, rect.left-m), max(0, rect.top-m),
                        min(page_img.width, rect.left+rect.width+m),
                        min(page_img.height, rect.top+rect.height+m)))
                    return crop, r.data.decode('utf-8', errors='replace')
    except Exception:
        pass
    return None, None


# ──────────────────────────────────────────
# 시험지 QR 코드 제거
# ──────────────────────────────────────────

def remove_qr_from_pages(pdf_bytes, dpi=200):
    """시험지 각 페이지에서 QR 코드를 흰색으로 제거"""
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    removed = 0
    for page_img in pages:
        for r in pyzbar_decode(page_img):
            if r.type == 'QRCODE':
                rect = r.rect
                margin = int(max(rect.width, rect.height) * 0.15)
                ImageDraw.Draw(page_img).rectangle([
                    rect.left - margin, rect.top - margin,
                    rect.left + rect.width + margin, rect.top + rect.height + margin
                ], fill='white')
                removed += 1
    return pages, removed


# ──────────────────────────────────────────
# 맨뒷장 합성 & PDF 생성
# ──────────────────────────────────────────

def compose_back_page(template_bytes, qr_images, num_groups):
    pages = convert_from_bytes(template_bytes, dpi=RENDER_DPI)
    result = pages[0].copy().convert('RGB')
    boxes = detect_box_positions(result)
    draw = ImageDraw.Draw(result)

    for gnum in range(1, min(MAX_GROUPS + 1, len(boxes) + 1)):
        l, t, r, b = boxes[gnum - 1]
        bw, bh = r - l, b - t

        if gnum in qr_images:
            padding = int(min(bw, bh) * 0.08)
            sz = min(bw - 2*padding, bh - 2*padding)
            qr = qr_images[gnum].convert('RGB').resize((sz, sz), Image.NEAREST)
            draw.rectangle([l+2, t+2, r-2, b-2], fill='white')
            result.paste(qr, (l + (bw-sz)//2, t + (bh-sz)//2))
        elif gnum > num_groups:
            draw.rectangle([l+2, t+2, r-2, b-2], fill='#F0F0F0')
            lw, m = max(3, int(bw*0.02)), int(bw*0.15)
            draw.line([(l+m,t+m),(r-m,b-m)], fill='#CCCCCC', width=lw)
            draw.line([(r-m,t+m),(l+m,b-m)], fill='#CCCCCC', width=lw)

    return result


def images_to_pdf_bytes(images, page_w, page_h):
    buf = io.BytesIO()
    tmps = []
    try:
        c = rl_canvas.Canvas(buf, pagesize=(page_w, page_h))
        for img in images:
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(tmp.name, 'PNG', optimize=True)
            tmps.append(tmp.name)
            c.drawImage(tmp.name, 0, 0, width=page_w, height=page_h,
                        preserveAspectRatio=True, anchor='c')
            c.showPage()
        c.save()
    finally:
        for f in tmps:
            try: os.unlink(f)
            except: pass
    buf.seek(0)
    return buf.read()


def parse_filename_info(filename):
    info = {}
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if m: info['date'] = m.group(1)
    m = re.search(r'([가-힣]+)T(?:R|eacher)?', filename)
    if m: info['teacher'] = m.group(1)
    for s in ['화학','물리','생물','생명','지구','통합과학','과학']:
        if s in filename:
            info['subject'] = s
            break
    return info


def make_output_name(info, n_groups, suffix=""):
    parts = ['통합시험지' if not suffix else suffix]
    if 'teacher' in info: parts.append(info['teacher'] + 'T')
    if 'subject' in info: parts.append(info['subject'])
    if 'date' in info: parts.append(info['date'])
    if not suffix:
        parts.append(f"{n_groups}그룹")
    return "_".join(parts) + ".pdf"


def load_default_template():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, DEFAULT_TEMPLATE)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    return None


# ──────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────

def main():
    st.set_page_config(page_title="족보 통합시험지", page_icon="📝", layout="wide")

    st.title("📝 족보닷컴 통합 시험지 PDF 생성기")
    st.caption("배포그룹 시험지 → QR 추출 → QR 제거 → 맨뒷장 합성 → 통합 PDF")

    default_template = load_default_template()

    with st.sidebar:
        st.header("⚙️ 설정")

        # 템플릿
        if default_template:
            st.success("✅ 기본 템플릿 적용됨")
            override = st.file_uploader("템플릿 변경 (선택사항)", type=['pdf'])
            template_bytes = override.read() if override else default_template
            if override: override.seek(0)
        else:
            st.warning("기본 템플릿 없음")
            uploaded_t = st.file_uploader("📄 맨뒷장 템플릿 PDF", type=['pdf'])
            if not uploaded_t:
                st.info("맨뒷장_템플릿.pdf를 GitHub 레포에 넣으면 자동 로드됩니다.")
                template_bytes = None
            else:
                template_bytes = uploaded_t.read()
                uploaded_t.seek(0)

        st.divider()

        # 정답지 분리 토글
        separate_answers = st.toggle(
            "📋 정답지 분리 출력",
            value=True,
            help="ON: 시험지 + 정답지 2개 파일 / OFF: 전부 합쳐서 1개 파일"
        )

        st.divider()
        st.markdown("""
        **자동 처리 항목**
        - 시험지 QR코드 제거
        - 정답/해설 페이지 자동 감지
        - 맨뒷장 QR코드 배치
        """)

    # ── 메인 ──
    st.subheader("📁 배포그룹 시험지 업로드")

    uploaded_pdfs = st.file_uploader(
        "배포그룹 PDF 파일들 (최대 6개)",
        type=['pdf'],
        accept_multiple_files=True,
    )

    if not uploaded_pdfs:
        st.info("👆 족보닷컴에서 다운로드한 배포그룹별 시험지 PDF를 업로드해주세요.")
        return

    if len(uploaded_pdfs) > MAX_GROUPS:
        st.error(f"최대 {MAX_GROUPS}개까지 업로드 가능합니다.")
        return

    if template_bytes is None:
        st.warning("⬅️ 사이드바에서 맨뒷장 템플릿을 업로드해주세요.")
        return

    # ── 그룹 배정 ──
    sorted_pdfs = sorted(uploaded_pdfs, key=lambda f: f.name)

    st.subheader(f"📊 {len(sorted_pdfs)}개 배포그룹")

    info = parse_filename_info(sorted_pdfs[0].name)
    parts = [f"{k}: {v}" for k, v in
             [('선생님', info.get('teacher')),
              ('과목', info.get('subject')),
              ('날짜', info.get('date'))] if v]
    if parts:
        st.caption(" | ".join(parts))

    group_assignments = {}
    cols = st.columns(min(len(sorted_pdfs), 3))
    for i, pdf_file in enumerate(sorted_pdfs):
        with cols[i % 3]:
            name = pdf_file.name
            if len(name) > 35:
                name = name[:15] + "..." + name[-15:]
            group_assignments[i] = st.selectbox(
                f"📄 {name}", list(range(1, MAX_GROUPS+1)),
                index=i, key=f"grp_{i}"
            )

    if len(set(group_assignments.values())) != len(group_assignments):
        st.error("⚠️ 배포그룹 번호가 중복됩니다!")
        return

    st.divider()

    if st.button("🚀 통합 시험지 생성", type="primary", use_container_width=True):
        _run(sorted_pdfs, group_assignments, template_bytes, separate_answers)

    # ── 결과 표시 ──
    if 'results' in st.session_state and st.session_state.results:
        _show_results(sorted_pdfs)


def _run(sorted_pdfs, group_assignments, template_bytes, separate_answers):
    """전체 파이프라인"""
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
        if qr:
            qr_images[gnum] = qr
        else:
            errors.append(str(gnum))

    if errors:
        st.error(f"❌ QR 추출 실패: 배포그룹 {', '.join(errors)}")
        progress.empty()
        return

    st.subheader("🔍 추출된 QR 코드")
    qcols = st.columns(min(len(qr_images), 6))
    for idx, gnum in enumerate(sorted(qr_images)):
        with qcols[idx]:
            st.image(qr_images[gnum], caption=f"{gnum}그룹", width=120)

    # 2. 정답 페이지 감지
    step = n + 1
    progress.progress(step/steps, text="정답/해설 페이지 감지...")

    first_data = sorted_pdfs[0].read(); sorted_pdfs[0].seek(0)
    reader = PdfReader(io.BytesIO(first_data))
    total_pages = len(reader.pages)
    pw = float(reader.pages[0].mediabox.width)
    ph = float(reader.pages[0].mediabox.height)

    answer_start = find_answer_start_page(first_data)

    if answer_start is not None:
        q_count = answer_start
        a_count = total_pages - answer_start
        st.caption(f"📄 전체 {total_pages}페이지: 문제 {q_count}p + 정답/해설 {a_count}p")
    else:
        # 정답 감지 실패 → 마지막 페이지를 정답으로 간주
        answer_start = max(1, total_pages - 1)
        q_count = answer_start
        a_count = total_pages - answer_start
        st.caption(f"📄 전체 {total_pages}페이지: 정답 키워드 미감지 → 마지막 {a_count}p를 정답으로 처리")

    # 3. 시험지 렌더 + QR 제거
    step += 1
    progress.progress(step/steps, text="시험지 QR 제거 중...")

    all_page_imgs, removed = remove_qr_from_pages(first_data, dpi=RENDER_DPI)
    question_imgs = all_page_imgs[:answer_start]
    answer_imgs = all_page_imgs[answer_start:]

    if removed > 0:
        st.caption(f"✂️ QR코드 {removed}개 제거됨")

    # 4. 맨뒷장 합성
    step += 1
    progress.progress(step/steps, text="맨뒷장 QR 합성...")
    back_img = compose_back_page(template_bytes, qr_images, n)

    # 5. 최종 PDF 생성
    step += 1
    progress.progress(step/steps, text="PDF 생성...")

    info = parse_filename_info(sorted_pdfs[0].name)

    if separate_answers:
        # 분리 모드: 시험지 PDF + 정답 PDF
        exam_pdf = images_to_pdf_bytes(question_imgs + [back_img], pw, ph)
        answer_pdf = images_to_pdf_bytes(answer_imgs, pw, ph)

        st.session_state.results = {
            'mode': 'separate',
            'exam_pdf': exam_pdf,
            'answer_pdf': answer_pdf,
            'info': info,
            'n_groups': n,
            'q_count': len(question_imgs),
            'a_count': len(answer_imgs),
        }
    else:
        # 합본 모드: 문제 + 정답 + 맨뒷장 전부 하나로
        combined_pdf = images_to_pdf_bytes(question_imgs + answer_imgs + [back_img], pw, ph)

        st.session_state.results = {
            'mode': 'combined',
            'combined_pdf': combined_pdf,
            'info': info,
            'n_groups': n,
            'total': len(question_imgs) + len(answer_imgs) + 1,
        }

    progress.progress(1.0, text="완료!")
    progress.empty()
    st.rerun()


def _show_results(sorted_pdfs):
    """결과 파일 표시"""
    r = st.session_state.results
    info = r['info']

    st.divider()
    st.success("✅ 통합 시험지가 생성되었습니다!")

    if r['mode'] == 'separate':
        # ── 분리 모드: 2개 파일 ──
        st.info(
            f"📄 시험지: 문제 {r['q_count']}p + 맨뒷장 1p | "
            f"📋 정답지: {r['a_count']}p | "
            f"배포그룹 {r['n_groups']}개"
        )

        col1, col2 = st.columns(2)

        with col1:
            exam_name = make_output_name(info, r['n_groups'])
            st.download_button(
                label=f"⬇️ 시험지 다운로드 ({len(r['exam_pdf'])/1024:.0f}KB)",
                data=r['exam_pdf'],
                file_name=exam_name,
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

        with col2:
            answer_name = make_output_name(info, r['n_groups'], suffix="정답해설")
            st.download_button(
                label=f"⬇️ 정답지 다운로드 ({len(r['answer_pdf'])/1024:.0f}KB)",
                data=r['answer_pdf'],
                file_name=answer_name,
                mime="application/pdf",
                use_container_width=True,
            )

        # 미리보기
        with st.expander("📖 시험지 미리보기", expanded=True):
            try:
                preview = convert_from_bytes(r['exam_pdf'], dpi=100)
                pcols = st.columns(len(preview))
                for i, pg in enumerate(preview):
                    with pcols[i]:
                        st.image(pg, caption=f"p.{i+1}", use_container_width=True)
            except Exception:
                pass

        with st.expander("📖 정답지 미리보기"):
            try:
                preview = convert_from_bytes(r['answer_pdf'], dpi=100)
                pcols = st.columns(max(1, len(preview)))
                for i, pg in enumerate(preview):
                    with pcols[i]:
                        st.image(pg, caption=f"정답 p.{i+1}", use_container_width=True)
            except Exception:
                pass

    else:
        # ── 합본 모드: 1개 파일 ──
        st.info(f"📄 합본: 총 {r['total']}p | 배포그룹 {r['n_groups']}개")

        combined_name = make_output_name(info, r['n_groups'])
        st.download_button(
            label=f"⬇️ 통합 시험지 다운로드 ({len(r['combined_pdf'])/1024:.0f}KB)",
            data=r['combined_pdf'],
            file_name=combined_name,
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )

        with st.expander("📖 미리보기", expanded=True):
            try:
                preview = convert_from_bytes(r['combined_pdf'], dpi=100)
                pcols = st.columns(min(len(preview), 5))
                for i, pg in enumerate(preview):
                    with pcols[i % len(pcols)]:
                        st.image(pg, caption=f"p.{i+1}", use_container_width=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
