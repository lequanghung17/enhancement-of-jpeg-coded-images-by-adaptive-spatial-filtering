# app.py — Streamlit UI for JPEG Enhancement (no Gradio)
import streamlit as st

st.set_page_config(page_title="JPEG Enhancement — Adaptive Spatial Filtering", layout="wide")
st.title("Enhancement of JPEG Coded Images by Adaptive Spatial Filtering")

# --- hiển thị đường dẫn làm việc + python để dễ debug ---
import os, sys
st.caption(f"CWD: {os.getcwd()} | Python: {sys.executable}")

# --- import core + báo lỗi rõ ràng nếu fail ---
try:
    from PIL import Image
    import numpy as np
    from core import FullPipeline, merge_y_into_rgb
except Exception as e:
    st.error("❌ Lỗi khi import module. Kiểm tra `core.py` nằm cùng thư mục với `app.py`.")
    st.exception(e)
    st.stop()

# Sidebar: tham số
st.sidebar.header("Parameters")
noise_type   = st.sidebar.selectbox("Noise", ["None", "Gaussian", "Salt & Pepper"], index=0)
noise_level  = st.sidebar.slider("Noise level (Gaussian std / S&P prob*1000)", 0, 50, 25, 1)
jpeg_quality = st.sidebar.slider("JPEG quality (lower = more compression)", 1, 100, 35, 1)
tau1 = st.sidebar.slider("τ1 (range threshold)", 0, 255, 20, 1)
tau2 = st.sidebar.slider("τ2 (two-group split)", 1, 50, 16, 1)
tau3 = st.sidebar.slider("τ3 (final edge decision)", 0.0, 5.0, 2.3, 0.1)
beta = st.sidebar.slider("β (RNZC threshold ~12 for 5×5)", 0, 50, 12, 1)
run_btn = st.sidebar.button("Run Enhancement")

# Upload ảnh
uploaded = st.file_uploader("Upload a grayscale or color image", type=["png","jpg","jpeg","bmp","tif","tiff"])
if uploaded is not None:
    # Lưu trước để dùng lại chroma gốc khi hiển thị
    uploaded.seek(0)
    _preview_img = Image.open(uploaded)
    st.image(_preview_img, caption="Input image", use_container_width=True)
    # xác định có phải input màu không
    is_color_input = (_preview_img.mode in ("RGB", "RGBA"))
else:
    is_color_input = False

# Khi bấm Run
if run_btn:
    if uploaded is None:
        st.warning("Please upload an image first.")
        st.stop()

    try:
        # Lưu file gốc để pipeline dùng (giữ nguyên mode/color)
        uploaded.seek(0)             # quan trọng khi dùng file-like
        img_path = "uploaded_image.png"
        Image.open(uploaded).save(img_path)

        # Chạy pipeline (làm trên kênh Y)
        pipeline = FullPipeline(img_path, tau1=tau1, tau2=tau2, tau3=tau3, beta=beta)
        bitrate = pipeline.encode_image(quality=jpeg_quality, noise_type=noise_type, noise_level=noise_level)
        pipeline.label_edges()
        pipeline.apply_filters()
        metrics = pipeline.assess_quality()

        # Chuẩn bị ảnh hiển thị:
        # - Nếu input màu: ghép Y đã tăng cường vào chroma gốc (median 3×3 nhẹ, mặc định ON)
        # - Nếu input xám: hiển thị trực tiếp ảnh Y
        base_rgb = Image.open(img_path).convert("RGB") if is_color_input else None

        if is_color_input:
            img_orig = merge_y_into_rgb(pipeline.original_image,  base_rgb, smooth_chroma=True)
            img_comp = merge_y_into_rgb(pipeline.compressed_image, base_rgb, smooth_chroma=True)
            img_s1   = merge_y_into_rgb(pipeline.scheme1_image,    base_rgb, smooth_chroma=True)
            img_s2   = merge_y_into_rgb(pipeline.scheme2_image,    base_rgb, smooth_chroma=True)
        else:
            img_orig = pipeline.original_image
            img_comp = pipeline.compressed_image
            img_s1   = pipeline.scheme1_image
            img_s2   = pipeline.scheme2_image

        img_lbl  = pipeline.three_way_image  # label map luôn dạng xám để quan sát nhãn

    except Exception as e:
        st.error("❌ Lỗi khi chạy pipeline.")
        st.exception(e)
        st.stop()

    # Hiển thị
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("(a) Original/Noisy")
        st.image(img_orig, use_container_width=True, clamp=True)
        st.subheader("(b) Compressed")
        st.image(img_comp, use_container_width=True, clamp=True)
    with col2:
        st.subheader("(c) Enhanced — Scheme 2")
        st.image(img_s2, use_container_width=True, clamp=True)
        st.subheader("(d) Enhanced — Scheme 1")
        st.image(img_s1, use_container_width=True, clamp=True)
    with col3:
        st.subheader("(e) 3-way Pixel Labeling")
        st.image(img_lbl, use_container_width=True, clamp=True)

        st.markdown("### Bitrate")
        st.code(f"{bitrate:.3f} bit/pixel")

        st.markdown("### M₁ (QC regions)")
        st.code(
            f"Original  : {metrics['M1_original_qc']:.3f}\n"
            f"Compressed: {metrics['M1_compressed_qc']:.3f}\n"
            f"Scheme 1  : {metrics['M1_scheme1_qc']:.3f}\n"
            f"Scheme 2  : {metrics['M1_scheme2_qc']:.3f}"
        )

        st.markdown("### BRISQUE / PIQE")
        st.code(
            f"BRISQUE (Compressed): {metrics['brisque_compressed']:.3f}\n"
            f"BRISQUE (Scheme 1) : {metrics['brisque_scheme1']:.3f}\n"
            f"BRISQUE (Scheme 2) : {metrics['brisque_scheme2']:.3f}\n"
            f"PIQE (Compressed)  : {metrics['piqe_compressed']:.3f}\n"
            f"PIQE (Scheme 1)   : {metrics['piqe_scheme1']:.3f}\n"
            f"PIQE (Scheme 2)   : {metrics['piqe_scheme2']:.3f}"
        )

st.info("Tip: window 5×5 cho labeling; β≈12; η≈10–12. D-filter 3×3, median 5×5/7×7.")
