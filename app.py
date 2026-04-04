"""
📝 족보닷컴 통합 시험지 PDF 생성기 v2
- 배포그룹별 PDF에서 QR코드 자동 추출
- 시험지 1페이지 QR코드 자동 제거
- 맨뒷장 템플릿에 QR코드 배치 (박스 위치 자동 감지)
- 시험 내용 + 맨뒷장을 하나의 PDF로 합침

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

# PIL 고해상도 이미지 허용
Image.MAX_IMAGE_PIXELS = 200_000_000

MAX_GROUPS = 6
RENDER_DPI = 200


# ──────────────────────────────────────────
# 템플릿 박스 위치 자동 감지
# ──────────────────────────────────────────

def detect_box_positions(template_img):
    """
    템플릿 이미지에서 6개 QR 박스의 위치를 자동 감지
    박스 border(어두운 선)를 찾아서 박스 내부 좌표를 반환
    
    Returns: [(left, top, right, bottom), ...] 6개 박스 좌표 (픽셀)
    """
    arr = np.array(template_img)
    h, w = arr.shape[:2]
    
    # 하단 영역(60%~)에서 수직 border 라인 찾기
    # 박스 중간 높이(약 70%)에서 수평 스캔
    y_scan = int(0.70 * h)
    line = np.mean(arr[y_scan, :, :], axis=1)
    dark_mask = line < 100
    
    # 연속된 어두운 픽셀 그룹 = border line
    vborders = _find_line_groups(dark_mask, max_width=60)
    # 외곽 border 제거 (페이지 테두리)
    vborders = [b for b in vborders if b[0] > w * 0.1 and b[1] < w * 0.95]
    
    # 박스 중간 폭(약 25%)에서 수직 스캔
    x_scan = int(0.25 * w)
    col = np.mean(arr[:, x_scan, :], axis=1)
    dark_mask_v = col < 100
    
    hborders = _find_line_groups(dark_mask_v, max_width=60)
    # 하단 60% 이후의 border만 (박스 영역)
    hborders = [b for b in hborders if b[0] > h * 0.60]
    
    # border 쌍으로 박스 구간 추출
    col_ranges = _borders_to_ranges(vborders)  # 박스 열 내부 범위
    row_ranges = _borders_to_ranges(hborders)  # 박스 행 내부 범위
    
    # 3열 x 2행 = 6개 박스 기대
    if len(col_ranges) < 3:
        col_ranges = _fallback_columns(w)
    if len(row_ranges) < 2:
        row_ranges = _fallback_rows(h)
    
    # 가장 큰 3개 열, 2개 행 선택
    col_ranges = sorted(col_ranges, key=lambda x: x[1]-x[0], reverse=True)[:3]
    col_ranges = sorted(col_ranges)
    row_ranges = sorted(row_ranges, key=lambda x: x[1]-x[0], reverse=True)[:2]
    row_ranges = sorted(row_ranges)
    
    boxes = []
    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            boxes.append((c_start, r_start, c_end, r_end))
    
    return boxes


def _find_line_groups(dark_mask, max_width=60):
    """연속된 True 구간을 찾아 (start, end) 리스트 반환"""
    groups = []
    indices = np.where(dark_mask)[0]
    if len(indices) == 0:
        return groups
    
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] - indices[i-1] > 3:
            w = indices[i-1] - start + 1
            if w <= max_width:
                groups.append((start, indices[i-1]))
            start = indices[i]
    w = indices[-1] - start + 1
    if w <= max_width:
        groups.append((start, indices[-1]))
    
    return groups


def _borders_to_ranges(borders):
    """인접한 border 쌍의 사이 구간 (박스 내부) 추출"""
    ranges = []
    for i in range(len(borders) - 1):
        gap_start = borders[i][1] + 1
        gap_end = borders[i+1][0] - 1
        gap_size = gap_end - gap_start
        if gap_size > 100:  # 최소 크기 필터
            ranges.append((gap_start, gap_end))
    return ranges


def _fallback_columns(w):
    """자동 감지 실패 시 기본 3열 위치"""
    margin = int(w * 0.17)
    box_w = int(w * 0.14)
    gap = int(w * 0.12)
    cols = []
    for i in range(3):
        left = margin + i * (box_w + gap)
        cols.append((left, left + box_w))
    return cols


def _fallback_rows(h):
    """자동 감지 실패 시 기본 2행 위치"""
    return [(int(h * 0.658), int(h * 0.750)),
            (int(h * 0.824), int(h * 0.916))]


# ──────────────────────────────────────────
# QR 코드 추출
# ──────────────────────────────────────────

def extract_qr_from_pdf(pdf_bytes):
    """PDF에서 QR 코드 이미지와 URL 추출 (2단계 시도)"""
    # 1차: 임베디드 이미지
    qr_img, qr_url = _extract_qr_from_embedded(pdf_bytes)
    if qr_img:
        return qr_img, qr_url
    # 2차: 렌더링
    return _extract_qr_by_rendering(pdf_bytes)


def _extract_qr_from_embedded(pdf_bytes):
    """PDF 임베디드 이미지에서 QR 코드 찾기"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            if '/Resources' not in page or '/XObject' not in page['/Resources']:
                continue
            xobjects = page['/Resources']['/XObject'].get_object()
            for obj_name in xobjects:
                try:
                    obj = xobjects[obj_name].get_object()
                    if obj.get('/Subtype') != '/Image':
                        continue
                    width = int(obj['/Width'])
                    height = int(obj['/Height'])
                    data = obj.get_data()
                    
                    img = None
                    filters = obj.get('/Filter', '')
                    fname = str(filters[0] if isinstance(filters, list) else filters)
                    
                    if fname in ('/DCTDecode', '/JPXDecode'):
                        img = Image.open(io.BytesIO(data))
                    elif fname == '/FlateDecode':
                        cs = str(obj.get('/ColorSpace', '/DeviceRGB'))
                        mode, bpp = ('L', 1) if cs == '/DeviceGray' else ('RGB', 3)
                        expected = width * height * bpp
                        if len(data) >= expected:
                            img = Image.frombytes(mode, (width, height), data[:expected])
                    
                    if img is None:
                        continue
                    img_rgb = img.convert('RGB')
                    for r in pyzbar_decode(img_rgb):
                        if r.type == 'QRCODE':
                            return img_rgb, r.data.decode('utf-8', errors='replace')
                except Exception:
                    continue
    except Exception:
        pass
    return None, None


def _extract_qr_by_rendering(pdf_bytes):
    """페이지 렌더링 후 QR 코드 추출"""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
        for page_img in pages:
            for r in pyzbar_decode(page_img):
                if r.type == 'QRCODE':
                    rect = r.rect
                    m = 15
                    crop = page_img.crop((
                        max(0, rect.left - m), max(0, rect.top - m),
                        min(page_img.width, rect.left + rect.width + m),
                        min(page_img.height, rect.top + rect.height + m)
                    ))
                    return crop, r.data.decode('utf-8', errors='replace')
    except Exception:
        pass
    return None, None


# ──────────────────────────────────────────
# 시험지 QR 코드 제거
# ──────────────────────────────────────────

def remove_qr_from_pages(pdf_bytes, dpi=200):
    """
    시험지 PDF의 각 페이지에서 QR 코드를 흰색으로 덮어서 제거
    Returns: 수정된 페이지 이미지 리스트
    """
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    modified = []
    removed_count = 0
    
    for page_img in pages:
        results = pyzbar_decode(page_img)
        qrs = [r for r in results if r.type == 'QRCODE']
        
        if qrs:
            draw = ImageDraw.Draw(page_img)
            for q in qrs:
                rect = q.rect
                margin = int(max(rect.width, rect.height) * 0.15)
                draw.rectangle([
                    rect.left - margin, rect.top - margin,
                    rect.left + rect.width + margin, rect.top + rect.height + margin
                ], fill='white')
                removed_count += 1
        
        modified.append(page_img)
    
    return modified, removed_count


def images_to_pdf_bytes(images, page_w, page_h):
    """여러 이미지를 하나의 PDF로 변환"""
    buf = io.BytesIO()
    tmp_files = []
    
    try:
        c = rl_canvas.Canvas(buf, pagesize=(page_w, page_h))
        
        for img in images:
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(tmp.name, 'PNG', optimize=True)
            tmp_files.append(tmp.name)
            
            c.drawImage(tmp.name, 0, 0, width=page_w, height=page_h,
                        preserveAspectRatio=True, anchor='c')
            c.showPage()
        
        c.save()
    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass
    
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────
# 맨뒷장 합성
# ──────────────────────────────────────────

def compose_back_page(template_bytes, qr_images, num_groups):
    """템플릿 이미지에 QR 코드 합성"""
    pages = convert_from_bytes(template_bytes, dpi=RENDER_DPI)
    result = pages[0].copy().convert('RGB')
    
    # 박스 위치 자동 감지
    boxes = detect_box_positions(result)
    
    if len(boxes) < MAX_GROUPS:
        st.warning(f"⚠️ 박스 {len(boxes)}개만 감지됨 (기대: {MAX_GROUPS}개)")
    
    draw = ImageDraw.Draw(result)
    
    for group_num in range(1, min(MAX_GROUPS + 1, len(boxes) + 1)):
        idx = group_num - 1
        l, t, r, b = boxes[idx]
        bw, bh = r - l, b - t
        
        if group_num in qr_images:
            # QR 코드 배치
            padding = int(min(bw, bh) * 0.08)
            target = min(bw - 2*padding, bh - 2*padding)
            qr_resized = qr_images[group_num].convert('RGB').resize(
                (target, target), Image.NEAREST
            )
            draw.rectangle([l+2, t+2, r-2, b-2], fill='white')
            px = l + (bw - target) // 2
            py = t + (bh - target) // 2
            result.paste(qr_resized, (px, py))
        elif group_num > num_groups:
            # 미사용 그룹: X 표시
            draw.rectangle([l+2, t+2, r-2, b-2], fill='#F0F0F0')
            lw = max(3, int(bw * 0.02))
            m = int(bw * 0.15)
            draw.line([(l+m, t+m), (r-m, b-m)], fill='#CCCCCC', width=lw)
            draw.line([(r-m, t+m), (l+m, b-m)], fill='#CCCCCC', width=lw)
    
    return result


def image_to_pdf_bytes(img, page_w, page_h):
    """단일 이미지 → PDF"""
    buf = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name, 'PNG', optimize=True)
        tmp_path = tmp.name
    try:
        c = rl_canvas.Canvas(buf, pagesize=(page_w, page_h))
        c.drawImage(tmp_path, 0, 0, width=page_w, height=page_h,
                    preserveAspectRatio=True, anchor='c')
        c.save()
    finally:
        os.unlink(tmp_path)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────
# PDF 합치기
# ──────────────────────────────────────────

def build_final_pdf(test_page_images, back_page_img, page_w, page_h):
    """시험 페이지 이미지들 + 맨뒷장 이미지 → 최종 PDF"""
    all_images = test_page_images + [back_page_img]
    return images_to_pdf_bytes(all_images, page_w, page_h)


def parse_filename_info(filename):
    """파일명에서 과목/선생님/날짜 추출"""
    info = {}
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if m:
        info['date'] = m.group(1)
    m = re.search(r'([가-힣]+)T(?:R|eacher)?', filename)
    if m:
        info['teacher'] = m.group(1)
    for s in ['화학', '물리', '생물', '생명', '지구', '통합과학', '과학']:
        if s in filename:
            info['subject'] = s
            break
    return info


# ──────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="족보 통합시험지 생성기",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 족보닷컴 통합 시험지 PDF 생성기")
    st.caption("배포그룹별 시험지 → QR 추출 → 1페이지 QR 제거 → 맨뒷장 합성 → 통합 PDF")
    
    # ── 사이드바 ──
    with st.sidebar:
        st.header("⚙️ 설정")
        
        template_file = st.file_uploader(
            "📄 맨뒷장 템플릿 PDF",
            type=['pdf'],
            help="족보닷컴 답안지 입력방법 템플릿"
        )
        
        st.divider()
        include_answers = st.checkbox("정답/해설 페이지 포함", value=False)
        
        st.divider()
        st.markdown("""
        **작동 방식**
        1. 배포그룹 PDF에서 QR 자동 추출
        2. 시험지 1페이지 QR코드 제거
        3. 맨뒷장에 그룹별 QR 배치
        4. 통합 PDF 생성
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
    
    if not template_file:
        st.warning("⬅️ 사이드바에서 맨뒷장 템플릿 PDF를 업로드해주세요.")
        return
    
    # ── 그룹 배정 ──
    sorted_pdfs = sorted(uploaded_pdfs, key=lambda f: f.name)
    
    st.subheader(f"📊 {len(sorted_pdfs)}개 배포그룹")
    
    info = parse_filename_info(sorted_pdfs[0].name)
    if info:
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
                f"📄 {name}", list(range(1, MAX_GROUPS + 1)),
                index=i, key=f"grp_{i}"
            )
    
    if len(set(group_assignments.values())) != len(group_assignments):
        st.error("⚠️ 배포그룹 번호가 중복됩니다!")
        return
    
    st.divider()
    
    if st.button("🚀 통합 시험지 생성", type="primary", use_container_width=True):
        _run_pipeline(sorted_pdfs, group_assignments, template_file, include_answers)
    
    # 결과 표시
    if 'final_pdf' in st.session_state and st.session_state.final_pdf:
        st.divider()
        st.success("✅ 통합 시험지가 생성되었습니다!")
        
        info = parse_filename_info(sorted_pdfs[0].name)
        parts = ['통합시험지']
        for key in ['teacher', 'subject', 'date']:
            if key in info:
                val = info[key] + ('T' if key == 'teacher' else '')
                parts.append(val)
        parts.append(f"{len(sorted_pdfs)}그룹")
        output_name = "_".join(parts) + ".pdf"
        
        st.download_button(
            label="⬇️ 통합 시험지 다운로드",
            data=st.session_state.final_pdf,
            file_name=output_name,
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
        
        with st.expander("📖 최종 PDF 미리보기", expanded=True):
            try:
                preview = convert_from_bytes(st.session_state.final_pdf, dpi=100)
                pcols = st.columns(len(preview))
                for i, pg in enumerate(preview):
                    with pcols[i]:
                        st.image(pg, caption=f"p.{i+1}", use_container_width=True)
            except Exception as e:
                st.warning(f"미리보기 실패: {e}")


def _run_pipeline(sorted_pdfs, group_assignments, template_file, include_answers):
    """전체 파이프라인"""
    steps = len(sorted_pdfs) + 4  # QR추출 + QR제거 + 합성 + 변환 + 합치기
    progress = st.progress(0, text="시작...")
    
    # ── 1. QR 추출 ──
    qr_images = {}
    errors = []
    
    for i, pdf_file in enumerate(sorted_pdfs):
        gnum = group_assignments[i]
        progress.progress((i+1) / steps, text=f"배포그룹 {gnum} QR 추출... ({i+1}/{len(sorted_pdfs)})")
        
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        
        qr_img, qr_url = extract_qr_from_pdf(pdf_bytes)
        if qr_img:
            qr_images[gnum] = qr_img
        else:
            errors.append(f"배포그룹 {gnum}")
    
    if errors:
        st.error(f"❌ QR 추출 실패: {', '.join(errors)}")
        progress.empty()
        return
    
    # QR 결과 표시
    st.subheader("🔍 추출된 QR 코드")
    qr_cols = st.columns(min(len(qr_images), 6))
    for idx, gnum in enumerate(sorted(qr_images.keys())):
        with qr_cols[idx]:
            st.image(qr_images[gnum], caption=f"{gnum}그룹", width=120)
    
    # ── 2. 시험지 1페이지 QR 제거 ──
    step = len(sorted_pdfs) + 1
    progress.progress(step / steps, text="시험지 QR 제거 중...")
    
    first_pdf_bytes = sorted_pdfs[0].read()
    sorted_pdfs[0].seek(0)
    
    reader = PdfReader(io.BytesIO(first_pdf_bytes))
    total_pages = len(reader.pages)
    page_w = float(reader.pages[0].mediabox.width)
    page_h = float(reader.pages[0].mediabox.height)
    
    # 포함할 페이지 결정
    if include_answers:
        page_count = total_pages
    else:
        page_count = max(1, total_pages - 1)
    
    # 선택된 페이지만 렌더링 + QR 제거
    test_pages_img, removed = remove_qr_from_pages(first_pdf_bytes, dpi=RENDER_DPI)
    test_pages_img = test_pages_img[:page_count]  # 선택된 페이지만
    
    if removed > 0:
        st.caption(f"✂️ 시험지에서 QR코드 {removed}개 제거됨")
    
    # ── 3. 맨뒷장 합성 ──
    step += 1
    progress.progress(step / steps, text="맨뒷장 QR 합성 중...")
    
    template_bytes = template_file.read()
    template_file.seek(0)
    
    back_page_img = compose_back_page(template_bytes, qr_images, len(sorted_pdfs))
    
    # ── 4. 최종 PDF 합치기 ──
    step += 1
    progress.progress(step / steps, text="최종 PDF 생성 중...")
    
    final_pdf = build_final_pdf(test_pages_img, back_page_img, page_w, page_h)
    
    progress.progress(1.0, text="완료!")
    progress.empty()
    
    # 결과 저장
    st.session_state.final_pdf = final_pdf
    
    final_reader = PdfReader(io.BytesIO(final_pdf))
    st.info(
        f"📄 총 {len(final_reader.pages)}페이지 "
        f"(시험 {page_count}p + 맨뒷장 1p) | "
        f"배포그룹 {len(qr_images)}개 | "
        f"{len(final_pdf)/1024:.0f}KB"
    )
    
    st.rerun()


if __name__ == "__main__":
    main()
