import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from streamlit_image_comparison import image_comparison

def check_password():
    def p_entered():
        if st.session_state["pw"] == st.secrets["password"]:
            st.session_state["pw_ok"] = True
            del st.session_state["pw"]
        else:
            st.session_state["pw_ok"] = False
    if "pw_ok" not in st.session_state:
        st.text_input("Password", type="password", on_change=p_entered, key="pw")
        return False
    elif not st.session_state["pw_ok"]:
        st.text_input("Password", type="password", on_change=p_entered, key="pw")
        st.error("Denied")
        return False
    return True

def process_image(img_bytes, radius, debug=False):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[int(h*0.6):h, int(w*0.6):w]
    _, ml = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, md = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    comb = cv2.bitwise_or(ml, md)
    mask = np.zeros_like(gray)
    mask[int(h*0.6):h, int(w*0.6):w] = comb
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f_mask = np.zeros_like(mask)
    for c in cnts:
        area = cv2.contourArea(c)
        if 200 < area < 4500:
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            sol = float(area)/ha if ha > 0 else 0
            if 0.60 < sol < 0.92:
                cv2.drawContours(f_mask, [c], -1, 255, -1)
    if debug:
        return f_mask, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    k = np.ones((5,5), np.uint8)
    f_mask = cv2.dilate(f_mask, k, iterations=1)
    res = cv2.inpaint(img, f_mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

st.set_page_config(page_title="Eraser", layout="wide")
if check_password():
    st.title("Diamond Eraser")
    with st.sidebar:
        rad = st.slider("Radius", 1, 15, 5)
        db = st.checkbox("Debug Mode")
    
    up = st.file_uploader("Upload Ads", type=['jpg', 'png'], accept_multiple_files=True)
    
    if up:
        # --- PREVIEW SECTION ---
        with st.expander("🔍 Calibration Preview", expanded=True):
            first_file = up[0]
            first_bytes = first_file.getvalue()
            p, o = process_image(first_bytes, rad, db)
            
            if db:
                st.image(p, caption="Forensic Mask")
            else:
                image_comparison(img1=o, img2=p, label1="Original", label2="Cleaned")

        # --- BATCH SECTION ---
        if st.button("🚀 Start Batch Processing"):
            buf = io.BytesIO()
            progress_text = st.empty()
            bar = st.progress(0)
            
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for idx, f in enumerate(up):
                    progress_text.text(f"Processing: {f.name}")
                    res, _ = process_image(f.getvalue(), rad)
                    pi = Image.fromarray(res)
                    t = io.BytesIO()
                    pi.save(t, format='PNG')
                    z.writestr(f"clean_{f.name}", t.getvalue())
                    bar.progress((idx + 1) / len(up))
            
            st.success("✅ Batch Complete!")
            st.download_button(
                label="💾 Download Cleaned ZIP",
                data=buf.getvalue(),
                file_name="cleaned_research_images.zip",
                mime="application/zip"
            )
