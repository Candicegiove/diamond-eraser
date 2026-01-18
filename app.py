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

# --- 2. FORENSIC IMAGE PROCESSING ---
def process_image(image_bytes, radius, debug=False):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Surgical ROI (Focus on Bottom-Right 40%)
    h, w = gray.shape
    roi = gray[int(h*0.6):h, int(w*0.6):w]
    
    # UNIVERSAL DETECTION: Otsu's Method handles both Light and Dark backgrounds
    # This automatically finds the best threshold for the specific image contrast
    _, mask_light = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_dark = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Merge both results to ensure the 'Twin' has nowhere to hide
    combined_roi_mask = cv2.bitwise_or(mask_light, mask_dark)
    
    full_mask = np.zeros_like(gray)
    full_mask[int(h*0.6):h, int(w*0.6):w] = combined_roi_mask

    # GEOMETRY AUDIT: Filter by the 'Star' footprint
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(full_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Target the specific pixel-area of the watermark
        if 200 < area < 4500:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area if hull_area > 0 else 0
            
            # The four-pointed star 'Twin' has a solidity 'sweet spot'
            # (Higher than text, but lower than a perfect square/circle)
            if 0.60 < solidity < 0.92:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    if debug:
        # Returns the mask so you can see exactly what the AI is targeting
        return final_mask, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HEALING: Expand the mask slightly to ensure no 'gray shadows' remain
    kernel = np.ones((5,5), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    result = cv2.inpaint(img, final_mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- 3. THE INTERFACE ---
st.set_page_config(page_title="Diamond Eraser Pro", page_icon="💎", layout="wide")

if check_password():
    st.sidebar.success("✅ Authenticated")
    st.title("💎 Diamond Magic Eraser")
    st.write("Forensic normalization of political campaign imagery for PhD research.")

    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        radius = st.slider("Healing Smoothness (Radius)", 1, 15, 5)
        debug_mode = st.checkbox("🐞 Debug Mode: Show Mask", help="Turn this on to see what the AI is targeting
