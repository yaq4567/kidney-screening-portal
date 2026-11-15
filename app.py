import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64, io, os

st.set_page_config(page_title="Ministry of Health - AI Kidney Screening", page_icon="🇴🇲", layout="wide", initial_sidebar_state="collapsed")

def get_logo():
    for ext in ["png", "jpg", "jpeg", "jfif", "PNG", "JPG", "JPEG", "JFIF", "webp"]:
        if os.path.exists(f"logo.{ext}"):
            img = Image.open(f"logo.{ext}").convert("RGBA")
            new_data = [(255,255,255,0) if r>180 and g>180 and b>180 else ((220,30,30,255) if r>120 and g<90 and b<90 else (255,255,255,255)) for r,g,b,a in img.getdata()]
            img.putdata(new_data)
            if img.getbbox(): img = img.crop(img.getbbox())
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return None

def get_banner():
    for ext in ["png", "jpg", "jpeg", "jfif", "PNG", "JPG", "JPEG", "JFIF", "webp"]:
        if os.path.exists(f"banner.{ext}"):
            with open(f"banner.{ext}", "rb") as f:
                return f"data:image/{'png' if ext.lower()=='png' else 'jpeg'};base64,{base64.b64encode(f.read()).decode()}"
    return None

logo_uri, banner_uri = get_logo(), get_banner()
bg_css = f"linear-gradient(rgba(4,24,32,0.78), rgba(4,24,32,0.88)), url('{banner_uri}')" if banner_uri else "linear-gradient(135deg, #041c24, #072f3b)"

st.markdown(f"""<style>
.stApp {{ background: {bg_css}; background-size: cover; background-position: center; background-attachment: fixed; color: #fff; font-family: 'Segoe UI', sans-serif; }}
header, footer {{ visibility: hidden; }}
label, .stFileUploader label, .stSlider label {{ color: #fff !important; font-weight: 600 !important; }}
.center-logo {{ text-align: center; padding-bottom: 10px; }}
.center-logo img {{ height: 95px; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.75)); }}
div[data-baseweb="select"] > div, [data-testid="stSelectbox"] > div > div {{ background-color: rgba(0,105,92,0.55) !important; color: #fff !important; border: 1px solid #80cbc4 !important; border-radius: 8px !important; }}
[data-testid="stSelectbox"] div, [data-testid="stSelectbox"] span, [data-testid="stSelectbox"] svg {{ color: #fff !important; fill: #fff !important; }}
.hero-sec {{ text-align: center; padding: 15px 20px 30px; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.18); }}
.hero-sec h1 {{ color: #fff; font-size: 2.5rem; margin-bottom: 10px; text-shadow: 0 2px 8px rgba(0,0,0,0.8); }}
.hero-sec p {{ color: #e0f2f1; font-size: 1.15rem; max-width: 780px; margin: 0 auto; }}
.info-card {{ background: rgba(6,35,45,0.92); border-left: 5px solid #00e5ff; padding: 20px; border-radius: 12px; margin-bottom: 25px; }}
.portal-footer {{ text-align: center; color: #b2dfdb; padding: 20px; margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.15); font-size: 0.92rem; }}
div.stButton > button {{ background-color: rgba(0,105,92,0.55); color: white; border: 1px solid #80cbc4; border-radius: 8px; padding: 8px 16px; font-weight: 600; width: 100%; }}
div.stButton > button:hover {{ background-color: #00897b; color: white; }}
</style>""", unsafe_allow_html=True)

if "show_info" not in st.session_state: st.session_state.show_info = False
c1, c2, c3 = st.columns([2, 3, 1.5])
with c1:
    st.write("")
    if st.button("ℹ️ About Kidney Diseases & Project"): st.session_state.show_info = not st.session_state.show_info
with c2:
    st.markdown(f'<div class="center-logo"><img src="{logo_uri}"></div>' if logo_uri else '<div class="center-logo"><h3>🇴🇲 Sultanate of Oman | Ministry of Health</h3></div>', unsafe_allow_html=True)
with c3:
    st.write("")
    st.selectbox("🌐 Language", ["English (EN)", "العربية (Coming Soon)"], label_visibility="collapsed")

if st.session_state.show_info:
    st.markdown('<div class="info-card"><h3 style="color:#80cbc4;margin-top:0;">📖 About This AI Project & Kidney Conditions</h3><p>This medical diagnostic web portal assists healthcare professionals in automated screening of Kidney CT Scans using deep learning (<b>YOLO Object Detection</b>, achieving <b>79.7% mAP50</b> accuracy across 1,788 clinical CT images).</p><ul><li><b>Kidney Stones:</b> Hard mineral deposits inside the kidneys causing pain or urinary obstruction.</li><li><b>Kidney Tumors:</b> Abnormal cellular growths requiring early identification and clinical staging.</li><li><b>Kidney Cysts:</b> Fluid-filled sacs on or inside the kidneys requiring differentiation from solid masses.</li></ul></div>', unsafe_allow_html=True)

st.markdown('<div class="hero-sec"><h1>AI-Powered Kidney CT Scan Screening</h1><p>Digital Health & AI Diagnostics Portal — Empowering healthcare professionals in the Sultanate of Oman with rapid, accurate deep-learning detection for Kidney Stones, Tumors, and Cysts.</p></div>', unsafe_allow_html=True)
st.markdown("### 🔬 Smart Kidney CT Scan Diagnostic Unit")

@st.cache_resource
def load_model(): return YOLO("best.pt")

model = load_model()
disease_info = {
    "Kidney stone": ("Kidney Stone", "Hard mineral deposits detected inside the kidney."),
    "Kidney tumor": ("Kidney Tumor", "Abnormal tissue growth detected; specialist review advised."),
    "Kidney cyst": ("Kidney Cyst", "Fluid-filled sac detected on or inside the kidney.")
}

col_up, col_conf = st.columns([3, 1])
uploaded_file = col_up.file_uploader("Upload Kidney CT Scan Image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])
conf_threshold = col_conf.slider("AI Sensitivity (Confidence):", 0.10, 0.90, 0.25, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("⏳ Analyzing CT Scan with AI Model..."):
        results = model.predict(image, conf=conf_threshold)
        annotated_img, boxes = results[0].plot(), results[0].boxes

    img1, img2 = st.columns(2)
    img1.subheader("🖼️ Original CT Scan")
    img1.image(image, use_container_width=True)
    img2.subheader("🎯 AI Detection Output")
    img2.image(annotated_img[..., ::-1], use_container_width=True)

    st.subheader("📋 Preliminary Diagnostic Report")
    if len(boxes) == 0:
        st.success("✅ **Normal Finding:** No kidney stones, tumors, or cysts detected.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Detected Findings", f"{len(boxes)} Anomaly(ies)")
        m2.metric("Highest Confidence", f"{max([float(b.conf[0])*100 for b in boxes]):.1f}%")
        for i, box in enumerate(boxes, 1):
            title, desc = disease_info.get(model.names[int(box.cls[0])], (model.names[int(box.cls[0])], ""))
            st.warning(f"**{i}. Condition Detected:** {title} | **Confidence:** `{float(box.conf[0])*100:.1f}%` — {desc}")

st.markdown('<div class="portal-footer">🩺 Sultanate of Oman — Ministry of Health | Digital Health & AI Diagnostics | Muscat, Oman</div>', unsafe_allow_html=True)