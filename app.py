import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from streamlit_image_comparison import image_comparison

# --- 1. SECURITY GATEKEEPER ---
def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # clear for security
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Security Access Required", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Security Access Required", type="password", on_change=password_entered, key="password")
        st.error("❌ Access Denied: Incorrect Credentials")
        return False
    else:
        return True

# --- 2. IMAGE PROCESSING LOGIC ---
def process_image(image_bytes, threshold, radius, surgical_mode=False):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Identify the Shape (The 'Mask')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if surgical_mode:
        mask = np.zeros_like(gray)
        h, w = gray.shape
        # Focus on the likely watermark area (Bottom Right)
        roi = gray[int(h*0.6):h, int(w*0.6):w] 
        _, roi_mask = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
        mask[int(h*0.6):h, int(w*0.6):w] = roi_mask
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Refine the Mask & Heal the Image
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    result = cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- 3. THE USER INTERFACE ---
st.set_page_config(page_title="Diamond Eraser Pro", page_icon="💎", layout="wide")

# This 'if' statement is the primary condition for the script to run
if check_password():
    st.title("💎 Diamond Magic Eraser")
    st.write("Professional batch watermark removal for research imagery.")

    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Detection Sensitivity", 100, 255, 230)
        radius = st
