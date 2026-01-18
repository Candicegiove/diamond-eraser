import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from streamlit_image_comparison import image_comparison

# --- 1. SECURITY ---
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

# --- 2. THE TWIN-SLAYER LOGIC ---
def process_image(image_bytes, threshold, radius, solidity_min, debug=False):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Surgical ROI (Bottom-Right)
    mask = np.zeros_like(gray)
    h, w = gray.shape
    roi = gray[int(h*0.6):h, int(w*0.6):w] 
    _, roi_mask = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
    mask[int(h*0.6):h, int(w*0.6):w] = roi_mask

    # Geometry Filter
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 5000: continue
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        # The 'Twin' star is around 0.75 - 0.85 solidity
        if solidity > solidity_min:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    if debug:
        return final_mask, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Expanded kernel to catch the star points
    kernel = np.ones((5,5), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    result = cv2.inpaint(img, final_mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- 3. UI ---
st.set_page_config(page_title="Diamond Eraser Pro", page_icon="💎", layout="wide")

if check_password():
    st.title("💎 Diamond Magic Eraser")
    with st.sidebar:
        st.header("⚙️ Settings")
        # Start at 160 to catch the gray star!
        threshold = st.slider("Detection Sensitivity", 50, 255, 160)
        solidity_min = st.slider("Solidity Check", 0.40, 0.99, 0.75)
        radius = st.slider("Healing Smoothness", 1, 15, 5)
        debug_mode = st.checkbox("🐞 Debug Mode: Show Mask")

    uploaded_files = st.file_uploader("Upload campaign images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        with st.expander("🔍 Calibration Preview", expanded=True):
            first_img_bytes = uploaded_files[0].read()
            uploaded_files[0].seek(0) 
            processed, original = process_image(first_img_bytes, threshold, radius, solidity_min, debug_mode)
            
            if debug_mode:
                st.image(processed, caption="White = To be erased", use_container_width=True)
            else:
                image_comparison(img1=original, img2=processed, label1="Original", label2="Cleaned")
