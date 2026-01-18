import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from streamlit_image_comparison import image_comparison

# --- 1. SECURITY GATEKEEPER ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Security Access Required", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Security Access Required", type="password", on_change=password_entered, key="password")
        st.error("❌ Access Denied")
        return False
    return True

# --- 2. IMAGE PROCESSING LOGIC ---
def process_image(image_bytes, threshold, radius, surgical_mode=False):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if surgical_mode:
        mask = np.zeros_like(gray)
        h, w = gray.shape
        roi = gray[int(h*0.6):h, int(w*0.6):w] 
        _, roi_mask = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
        mask[int(h*0.6):h, int(w*0.6):w] = roi_mask
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    result = cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- 3. THE USER INTERFACE ---
st.set_page_config(page_title="Diamond Eraser Pro", page_icon="💎", layout="wide")

if check_password():
    st.sidebar.success("✅ Authenticated")
    st.title("💎 Diamond Magic Eraser")
    
    # Define Settings FIRST
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Detection Sensitivity", 100, 255, 230)
        radius = st.slider("Healing Smoothness", 1, 20, 5)
        surgical = st.checkbox("Surgical Mode (Bottom-Right Only)", value=True)

    # Then the Uploader
    uploaded_files = st.file_uploader("Upload campaign images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        zip_buffer = io.BytesIO()

        # 1. Preview
        with st.expander("🔍 Calibration Preview", expanded=True):
            first_img_bytes = uploaded_files[0].read()
            uploaded_files[0].seek(0) 
            processed_first, original_first = process_image(first_img_bytes, threshold, radius, surgical)
            image_comparison(img1=original_first, img2=processed_first, label1="Original", label2="Cleaned")

        # 2. Batch Processing
        if st.button("🚀 Start Batch Processing"):
            progress_bar = st.progress(0)
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for idx, file in enumerate(uploaded_files):
                    img_bytes = file.read()
                    processed_img, _ = process_image(img_bytes, threshold, radius, surgical)
                    
                    result_pil = Image.fromarray(processed_img)
                    img_io = io.BytesIO()
                    result_pil.save(img_io, format='PNG')
                    zip_file.writestr(f"cleaned_{file.name}", img_io.getvalue())
                    progress_bar.progress((idx + 1) / len(uploaded_files))

            st.sidebar.download_button(
                label="💾 Download Cleaned Batch (.zip)",
                data=zip_buffer.getvalue(),
                file_name="research_ready_images.zip",
                mime="application/zip",
                use_container_width=True
            )
