"""
frontend/app.py
CXR AI-PACS — Clinical Workstation v4
Based on real PACS system standards:
- Intelligent worklist with priority sorting
- DICOM viewer with W/L controls
- Side-by-side comparison
- AI-assisted triage (PubMedCLIP)
- Structured report generation
- PDF export
- System stability monitoring
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import os
import json
import streamlit as st
import requests
from PIL import Image, ImageEnhance
import io
import numpy as np
from datetime import datetime

DEFAULT_API_URL = os.environ.get("API_URL", "https://pampered-enrage-girdle.ngrok-free.dev")

DISEASE_INFO = {
    "Atelectasis"       : "Partial or complete collapse of a lung or lobe.",
    "Cardiomegaly"      : "Enlargement of the cardiac silhouette beyond normal limits.",
    "Consolidation"     : "Lung parenchyma replaced by fluid, producing opacity.",
    "Edema"             : "Excess fluid accumulation in the lung interstitium.",
    "Effusion"          : "Fluid accumulation in the pleural space.",
    "Emphysema"         : "Permanent enlargement of air spaces distal to terminal bronchioles.",
    "Fibrosis"          : "Scarring and thickening of connective tissue in the lung.",
    "Hernia"            : "Protrusion of abdominal contents through the diaphragm.",
    "Infiltration"      : "Radiographic opacity suggesting infection or inflammation.",
    "Mass"              : "Focal opacity greater than 3 cm in diameter.",
    "Nodule"            : "Focal opacity 3 cm or less in diameter.",
    "Pleural_Thickening": "Fibrotic thickening of the pleural surfaces.",
    "Pneumonia"         : "Consolidation pattern consistent with infectious process.",
    "Pneumothorax"      : "Air in the pleural space causing lung collapse.",
    "No Finding"        : "No significant radiographic abnormality identified.",
}

URGENCY = {
    "Pneumothorax" : ("STAT", "#e07070"),
    "Pneumonia"    : ("URGENT", "#d4a050"),
    "Effusion"     : ("ROUTINE", "#6abe6a"),
    "Cardiomegaly" : ("ROUTINE", "#6abe6a"),
    "Mass"         : ("URGENT", "#d4a050"),
    "Nodule"       : ("ROUTINE", "#6abe6a"),
    "Edema"        : ("URGENT", "#d4a050"),
    "No Finding"   : ("NORMAL", "#7a8394"),
}

st.set_page_config(
    page_title="CXR AI-PACS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }

html, body, .stApp {
    background: #1c2028 !important;
    color: #dde1ea !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

#MainMenu, footer, [data-testid="stToolbar"],
.stDeployButton, header { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: #232830 !important;
    border-bottom: 1px solid #2e3442 !important;
    padding: 0 12px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6a7385 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 11px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s !important;
    letter-spacing: 0.1px !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #b0b8c8 !important; }
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #4d8ef0 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #1c2028 !important;
    padding: 20px 0 0 !important;
}

/* BUTTONS */
.stButton > button {
    background: #282e3c !important;
    color: #c8d0e0 !important;
    border: 1px solid #363d4e !important;
    border-radius: 5px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    letter-spacing: 0.1px !important;
    transition: all 0.12s !important;
}
.stButton > button:hover {
    background: #303748 !important;
    color: #ffffff !important;
    border-color: #4d8ef0 !important;
}

/* INPUTS */
.stTextInput input {
    background: #232830 !important;
    color: #dde1ea !important;
    border: 1px solid #363d4e !important;
    border-radius: 5px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}
.stTextInput input:focus {
    border-color: #4d8ef0 !important;
    box-shadow: 0 0 0 2px rgba(77,142,240,0.15) !important;
}
.stTextInput label {
    color: #8a93a6 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* SLIDERS */
.stSlider > label {
    color: #8a93a6 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploadDropzone"] {
    background: #232830 !important;
    border: 2px dashed #363d4e !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: #4d8ef0 !important; }
[data-testid="stFileUploadDropzone"] p { color: #6a7385 !important; font-size: 13px !important; }

/* METRICS */
[data-testid="metric-container"] {
    background: #232830 !important;
    border: 1px solid #2e3442 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #6a7385 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* EXPANDER */
details > summary {
    background: #232830 !important;
    color: #c8d0e0 !important;
    border: 1px solid #2e3442 !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
details > summary:hover { border-color: #4d8ef0 !important; color: #fff !important; }

/* ALERTS */
.stSuccess { background:#162316 !important; border-color:#265626 !important;
             color:#72c472 !important; font-size:13px !important; border-radius:6px !important; }
.stError   { background:#231616 !important; border-color:#562626 !important;
             color:#e07878 !important; font-size:13px !important; border-radius:6px !important; }
.stWarning { background:#23200e !important; border-color:#564f1e !important;
             color:#d4aa50 !important; font-size:13px !important; border-radius:6px !important; }
.stInfo    { background:#0e1623 !important; border-color:#1e3456 !important;
             color:#70a8e0 !important; font-size:13px !important; border-radius:6px !important; }

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    background: #162336 !important; color: #4d8ef0 !important;
    border: 1px solid #263e56 !important; border-radius: 5px !important;
    font-size: 13px !important; font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: #4d8ef0 !important; color: #000 !important;
}

/* CHECKBOX */
.stCheckbox > label { color: #c8d0e0 !important; font-size: 13px !important; font-weight: 500 !important; }

/* PROGRESS */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4d8ef0, #74aaff) !important;
    border-radius: 4px !important;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1c2028; }
::-webkit-scrollbar-thumb { background: #363d4e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #464d5e; }

[data-testid="column"] { padding: 0 6px !important; }

/* ── CXR animated loading / buffering visual ─────────────── */
@keyframes cxrSpin  { to { transform: rotate(360deg); } }
@keyframes cxrSpinR { to { transform: rotate(-360deg); } }
@keyframes cxrBar   { 0% { left:-45%; } 100% { left:115%; } }
@keyframes cxrFloat { 0%,100% { transform:translateY(0); }
                      50% { transform:translateY(-4px); } }
@keyframes cxrGlow  { 0%,100% { box-shadow:0 0 0 0 rgba(77,142,240,0.0); }
                      50% { box-shadow:0 0 22px 3px rgba(77,142,240,0.30); } }
@keyframes cxrCore  { 0%,100% { transform:scale(0.62); opacity:0.55; }
                      50% { transform:scale(1.05); opacity:1; } }
@keyframes cxrDots  { 0%,20% { content:''; } 40% { content:'.'; }
                      60% { content:'..'; } 80%,100% { content:'...'; } }

.cxr-load-wrap { display:flex; flex-direction:column; align-items:center;
    justify-content:center; gap:18px; padding:34px 0 26px; }

/* Dual concentric rings, floating with a soft glow */
.cxr-spin { position:relative; width:60px; height:60px; border-radius:50%;
    animation:cxrFloat 2.6s ease-in-out infinite,
              cxrGlow  2.6s ease-in-out infinite; }
.cxr-spin::before, .cxr-spin::after { content:""; position:absolute;
    border-radius:50%; border:3px solid transparent; }
.cxr-spin::before { inset:0; border-top-color:#4d8ef0; border-right-color:#74aaff;
    animation:cxrSpin 1s cubic-bezier(0.5,0.15,0.35,0.9) infinite; }
.cxr-spin::after  { inset:10px; border-width:2px; border-bottom-color:#74aaff;
    border-left-color:#3a6fc0; animation:cxrSpinR 1.5s linear infinite; }

/* Pulsing core */
.cxr-core { position:absolute; top:50%; left:50%; width:12px; height:12px;
    margin:-6px 0 0 -6px; border-radius:50%;
    background:radial-gradient(circle at 35% 35%,#bcd8ff,#4d8ef0);
    box-shadow:0 0 10px rgba(116,170,255,0.7);
    animation:cxrCore 1.4s ease-in-out infinite; }

/* Shimmer progress track */
.cxr-track { position:relative; width:240px; height:3px;
    background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden; }
.cxr-track::after { content:""; position:absolute; top:0; width:45%; height:100%;
    border-radius:4px; filter:drop-shadow(0 0 6px rgba(77,142,240,0.6));
    background:linear-gradient(90deg,transparent,#4d8ef0 35%,#bcd8ff 50%,
              #4d8ef0 65%,transparent);
    animation:cxrBar 1.4s cubic-bezier(0.45,0.05,0.55,0.95) infinite; }

/* Caption with animated typing dots */
.cxr-txt { font-family:'JetBrains Mono',monospace; font-size:11px;
    letter-spacing:2.5px; color:#8fb8f0; text-transform:uppercase;
    font-weight:500; text-align:center; opacity:0.92; }
.cxr-txt::after { content:'...'; display:inline-block; width:1.4em;
    text-align:left; animation:cxrDots 1.6s steps(1,end) infinite; }

/* Full-screen popup overlay that dims + blurs the whole app */
@keyframes cxrFade { from { opacity:0; } to { opacity:1; } }
@keyframes cxrRise { from { opacity:0; transform:translateY(10px) scale(0.98); }
                     to   { opacity:1; transform:translateY(0) scale(1); } }
.cxr-overlay { position:fixed; inset:0; z-index:99999;
    display:flex; align-items:center; justify-content:center;
    background:rgba(10,13,18,0.74);
    backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
    animation:cxrFade 0.2s ease both; }
.cxr-card { background:linear-gradient(160deg,#1f2530,#181d26);
    border:1px solid #2b3240; border-radius:16px; padding:30px 50px 28px;
    box-shadow:0 24px 64px rgba(0,0,0,0.55), 0 0 0 1px rgba(77,142,240,0.06);
    animation:cxrRise 0.28s cubic-bezier(0.2,0.8,0.2,1) both; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for k, v in {
    "api"        : DEFAULT_API_URL,
    "page"       : "cover",
    "study_id"   : None,
    "report"     : None,
    "ok"         : False,
    "ww"         : 160,
    "wl_val"     : 120,
    "inv"        : False,
    "brightness" : 1.0,
    "contrast"   : 1.0,
    "search"     : "",
    "last_ids"   : [],
    "retry_count": 0,
    "last_error" : "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API helpers ───────────────────────────────────────────────
HDR = {"ngrok-skip-browser-warning": "true"}

@st.cache_data(ttl=30, show_spinner=False)
def fetch_reports(api_url):
    try:
        r = requests.get(
            api_url.rstrip("/") + "/reports",
            headers=HDR, timeout=15
        )
        return r.json() if r.ok else {"total":0,"reports":[]}
    except:
        return {"total":0,"reports":[]}

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_image_bytes_cached(api_url, image_id):
    """Download raw image bytes — only successful results reach this cache.
    Raises on failure so st.cache_data does NOT store a None (which would
    otherwise make a study's image stay blank for the whole TTL)."""
    r = requests.get(
        api_url.rstrip("/") + f"/images/{image_id}",
        headers=HDR, timeout=30
    )
    if not r.ok or not r.content:
        raise RuntimeError(f"image fetch failed: HTTP {r.status_code}")
    return r.content

def fetch_image_bytes(api_url, image_id):
    """Cached fetch with safe fallback — failures are never cached, so a
    previously-uploaded study shows its image as soon as it is reachable."""
    try:
        return _fetch_image_bytes_cached(api_url, image_id)
    except Exception:
        return None

@st.cache_data(ttl=30, show_spinner=False)
def fetch_health(api_url):
    try:
        r = requests.get(
            api_url.rstrip("/") + "/health",
            headers=HDR, timeout=5
        )
        return r.json() if r.ok else None
    except:
        return None

def req(path, method="GET", t=10, no_cache=False, **kw):
    try:
        url = st.session_state.api.rstrip("/") + path
        r   = requests.request(method, url,
                                headers=HDR, timeout=t, **kw)
        if no_cache:
            fetch_reports.clear()
            fetch_health.clear()
        if r.ok:
            st.session_state.last_error = ""
            return r
        # Surface the backend's real error (FastAPI puts it in `detail`) so the
        # UI can show *why* a call failed instead of a generic message.
        try:
            st.session_state.last_error = str(r.json().get("detail",
                                                            f"HTTP {r.status_code}"))
        except Exception:
            st.session_state.last_error = f"HTTP {r.status_code}: {r.text[:300]}"
        return None
    except requests.exceptions.Timeout:
        st.session_state.retry_count += 1
        st.session_state.last_error = (
            f"Request timed out after {t}s (the server may still be loading "
            "models or generating — try again in a moment)."
        )
        return None
    except requests.exceptions.ConnectionError:
        st.session_state.ok = False
        st.session_state.last_error = "Could not connect to the API."
        return None
    except Exception as e:
        st.session_state.last_error = f"{type(e).__name__}: {e}"
        return None

def wl_img(raw):
    try:
        a  = np.array(
            Image.open(io.BytesIO(raw)).convert("L"),
            dtype=np.float32
        )
        lo = st.session_state.wl_val - st.session_state.ww / 2
        a  = np.clip((a - lo) / (st.session_state.ww + 1e-6) * 255,
                     0, 255)
        o  = Image.fromarray(a.astype(np.uint8)).convert("RGB")
        o  = ImageEnhance.Brightness(o).enhance(st.session_state.brightness)
        o  = ImageEnhance.Contrast(o).enhance(st.session_state.contrast)
        if st.session_state.inv:
            o = Image.fromarray(255 - np.array(o))
        b = io.BytesIO(); o.save(b, "PNG"); return b.getvalue()
    except:
        return raw

def add_recent(image_id):
    ids = st.session_state.last_ids
    if image_id and image_id not in ids:
        ids.insert(0, image_id)
    st.session_state.last_ids = ids[:5]

from contextlib import contextmanager

@contextmanager
def cxr_spinner(text="Loading"):
    """Animated buffering visual used in place of st.spinner everywhere."""
    # The trailing dots are supplied by the animated CSS, so strip any the
    # caller passed to avoid doubling them up (e.g. "Opening study...").
    label = text.rstrip(" .…")
    holder = st.empty()
    holder.markdown(f"""
    <div class="cxr-overlay">
      <div class="cxr-card">
        <div class="cxr-load-wrap">
            <div class="cxr-spin"><div class="cxr-core"></div></div>
            <div class="cxr-track"></div>
            <div class="cxr-txt">{label}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    try:
        yield
    finally:
        holder.empty()

def section(label):
    st.markdown(f"""
    <div style="font-family:'Inter',sans-serif; font-size:11px;
                font-weight:600; color:#6a7385; text-transform:uppercase;
                letter-spacing:0.8px; padding-bottom:8px;
                border-bottom:1px solid #2e3442; margin-bottom:14px;">
        {label}
    </div>
    """, unsafe_allow_html=True)

def data_row(label, value, color="#dde1ea", border=True):
    bdr = "border-bottom:1px solid #262c38;" if border else ""
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                font-family:'JetBrains Mono',monospace; font-size:12px;
                padding:6px 0; {bdr}">
        <span style="color:#6a7385;">{label}</span>
        <span style="color:{color}; font-weight:500;">{value}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Health check ──────────────────────────────────────────────
h   = fetch_health(st.session_state.api)
st.session_state.ok = h is not None
ok  = st.session_state.ok
db  = h.get("mongodb","—") if h else "—"
cnt = h.get("total_reports",0) if h else 0
now = datetime.now().strftime("%d %b %Y  %H:%M")

# ══════════════════════════════════════════════════════════════
# COVER / LANDING PAGE  (renders alone — brand bar is below it)
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "cover":
    _stat_col = "#5fb98f" if ok else "#d98c8c"
    _stat_txt = "System Online" if ok else "Backend Offline"
    _today    = datetime.now().strftime("%A, %d %B %Y")

    st.markdown(f"""
    <style>
      @keyframes lpFade {{ from {{ opacity:0; transform:translateY(14px); }}
                          to {{ opacity:1; transform:translateY(0); }} }}
      .lp-wrap {{ max-width:1080px; margin:18px auto 0;
                  animation:lpFade 0.6s ease both; }}
      .lp-rule {{ height:1px; background:linear-gradient(90deg,
                  transparent,#2e3442 18%,#2e3442 82%,transparent); }}
      .lp-kicker {{ font-family:'JetBrains Mono',monospace; font-size:12px;
                    letter-spacing:3px; color:#74aaff; text-transform:uppercase;
                    font-weight:500; display:inline-block;
                    background:rgba(77,142,240,0.08); border:1px solid #283a55;
                    padding:6px 12px; border-radius:20px; }}
      .lp-h1 {{ font-family:'Inter',sans-serif; font-size:52px; line-height:1.05;
                font-weight:700; color:#f2f5fb; letter-spacing:-1.2px;
                margin:20px 0 0; }}
      .lp-h1 .accent {{ background:linear-gradient(90deg,#5b9bf0,#74aaff);
                        -webkit-background-clip:text; background-clip:text;
                        -webkit-text-fill-color:transparent; }}
      .lp-sub {{ font-family:'Inter',sans-serif; font-size:17px; line-height:1.6;
                 color:#9aa3b5; max-width:520px; margin-top:18px; font-weight:400; }}
      .lp-panel {{ background:linear-gradient(160deg,#1f2530,#1a1f29);
                   border:1px solid #2b3240; border-radius:12px; padding:22px 24px;
                   transition:transform 0.18s ease, border-color 0.18s ease,
                              box-shadow 0.18s ease; }}
      .lp-panel:hover {{ transform:translateY(-3px); border-color:#3f6aa5;
                         box-shadow:0 10px 28px rgba(0,0,0,0.35); }}
      .lp-ico {{ width:34px; height:34px; border-radius:9px; display:flex;
                 align-items:center; justify-content:center; margin-bottom:12px;
                 background:rgba(77,142,240,0.12); border:1px solid #2f4a6e;
                 color:#74aaff; font-family:'JetBrains Mono'; font-weight:600;
                 font-size:15px; }}
    </style>

    <div class="lp-wrap">
      <div style="display:flex; align-items:center; justify-content:space-between;
                  padding-bottom:18px;">
        <div style="display:flex; align-items:center; gap:11px;">
          <div style="width:30px; height:30px; border-radius:7px;
                      background:linear-gradient(135deg,#4d8ef0,#3a6fc0);
                      display:flex; align-items:center; justify-content:center;
                      font-family:'Inter'; font-weight:700; color:#fff;
                      font-size:15px;
                      box-shadow:0 4px 12px rgba(77,142,240,0.35);">C</div>
          <span style="font-family:'Inter'; font-weight:600; font-size:15px;
                       color:#dfe4ee; letter-spacing:0.2px;">CXR AI-PACS</span>
        </div>
        <span style="font-family:'JetBrains Mono',monospace; font-size:11px;
                     color:#566074;">{_today}</span>
      </div>
      <div class="lp-rule"></div>

      <div style="text-align:center; padding:54px 0 40px;">
        <div class="lp-kicker">Chest Radiograph Analysis Workstation</div>
        <div class="lp-h1">Read chest X-rays<br>with <span class="accent">AI assistance.</span></div>
        <div class="lp-sub" style="margin-left:auto; margin-right:auto;">
          A PACS-style workstation that pairs PubMedCLIP vision analysis
          with Ollama report generation — upload, review, and export
          structured radiograph observations in one place.
        </div>
        <div style="display:inline-flex; align-items:center; gap:8px;
                    margin-top:22px; background:#1c2230; border:1px solid #2b3240;
                    padding:7px 14px; border-radius:20px;">
          <span style="width:8px; height:8px; border-radius:50%;
                       background:{_stat_col};
                       box-shadow:0 0 8px {_stat_col};"></span>
          <span style="font-family:'JetBrains Mono',monospace; font-size:12px;
                       color:{_stat_col};">{_stat_txt}</span>
          <span style="font-family:'JetBrains Mono',monospace; font-size:12px;
                       color:#48515f;">·&nbsp; {cnt} studies in archive</span>
        </div>
      </div>
      <div class="lp-rule"></div>

      <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:18px;
                  padding:28px 0 8px;">
        <div class="lp-panel">
          <div class="lp-ico">▣</div>
          <div style="font-family:'Inter'; font-weight:600; font-size:14px;
                      color:#e3e8f0;">DICOM Viewer</div>
          <div style="font-family:'Inter'; font-size:12.5px; color:#7c8595;
                      margin-top:6px; line-height:1.6;">
            Window/level, brightness and contrast controls with live adjustment.</div>
        </div>
        <div class="lp-panel">
          <div class="lp-ico">◈</div>
          <div style="font-family:'Inter'; font-weight:600; font-size:14px;
                      color:#e3e8f0;">AI Triage</div>
          <div style="font-family:'Inter'; font-size:12.5px; color:#7c8595;
                      margin-top:6px; line-height:1.6;">
            15-class disease scoring with priority flags from STAT to routine.</div>
        </div>
        <div class="lp-panel">
          <div class="lp-ico">≣</div>
          <div style="font-family:'Inter'; font-weight:600; font-size:14px;
                      color:#e3e8f0;">Structured Reports</div>
          <div style="font-family:'Inter'; font-size:12.5px; color:#7c8595;
                      margin-top:6px; line-height:1.6;">
            Generated observations, exportable as a formatted PDF report.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Centered call-to-action — kept directly beneath the hero content
    _lc1, _lc2, _lc3 = st.columns([1, 1.4, 1])
    with _lc2:
        if st.button("Enter Workstation  →", use_container_width=True,
                     type="primary", key="enter_btn"):
            # Show an animated buffering visual while the first study loads,
            # so the user knows the system is working.
            _loader = st.empty()
            _loader.markdown("""
            <div class="cxr-overlay">
              <div class="cxr-card">
                <div class="cxr-load-wrap">
                  <div class="cxr-spin"><div class="cxr-core"></div></div>
                  <div class="cxr-track"></div>
                  <div class="cxr-txt">Loading workstation</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Preload the first study so the viewer opens with data ready.
            _rj_load = fetch_reports(st.session_state.api)
            _reps    = _rj_load.get("reports", [])
            _an_load = [r for r in _reps if r.get("status") == "analyzed"] or _reps
            if _an_load:
                _f = _an_load[-1]          # newest = top of worklist
                st.session_state.study_id = _f["image_id"]
                _rx = req(f"/report/{_f['image_id']}", t=15)
                if _rx:
                    st.session_state.report = _rx.json()

            _loader.empty()
            # Land directly on the Viewer with the first study loaded.
            st.session_state.page = "viewer"
            st.rerun()

    st.stop()

# ── Top navigation bar ────────────────────────────────────────
conn_bg  = "#162316" if ok else "#231616"
conn_bdr = "#265626" if ok else "#562626"
conn_col = "#72c472" if ok else "#e07878"
conn_dot = "#4abe4a" if ok else "#e07070"
conn_lbl = "Connected" if ok else "Offline"

st.markdown(f"""
<div style="background:#232830; border-bottom:1px solid #2e3442;
            padding:10px 24px; display:flex; align-items:center;
            justify-content:space-between; margin:-20px -20px 0 -20px;">
    <div style="display:flex; align-items:center; gap:20px;">
        <div>
            <div style="font-family:'Inter',sans-serif; font-size:15px;
                         font-weight:700; color:#ffffff; letter-spacing:-0.2px;">
                CXR AI-PACS
            </div>
            <div style="font-family:'Inter',sans-serif; font-size:11px;
                         color:#4a5368; margin-top:1px;">
                AI-Assisted Chest X-Ray Analysis System
            </div>
        </div>
        <div style="width:1px; height:28px; background:#2e3442;"></div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                    color:#4a5368;">
            PubMedCLIP · NIH ChestX-ray14
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:18px;">
        <div style="display:inline-flex; align-items:center; gap:7px;
                    background:{conn_bg}; border:1px solid {conn_bdr};
                    padding:4px 12px; border-radius:20px;
                    font-family:'Inter',sans-serif; font-size:12px;
                    font-weight:500; color:{conn_col};">
            <span style="width:7px; height:7px; border-radius:50%;
                         background:{conn_dot}; display:inline-block;"></span>
            {conn_lbl}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;
                    font-size:11px; color:#4a5368;">
            DB: {db.split()[0]}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;
                    font-size:11px; color:#4a5368;">
            {cnt} studies
        </div>
        <div style="font-family:'JetBrains Mono',monospace;
                    font-size:11px; color:#363d4e;">
            {now}
        </div>
    </div>
</div>
<div style="height:16px;"></div>
""", unsafe_allow_html=True)

# ── Offline warning ───────────────────────────────────────────
if not ok:
    st.warning(
        "API is offline. Go to Settings, update the API URL, "
        "and click Save. Make sure Colab Cell 8 is running."
    )

# ── Page registry ─────────────────────────────────────────────
PAGES = ["worklist", "viewer", "upload", "history", "performance", "settings"]
PAGE_LABELS = {
    "worklist": "Worklist", "viewer": "Viewer", "upload": "Upload",
    "history": "History", "performance": "Performance", "settings": "Settings",
}

# ── Page navigation bar ───────────────────────────────────────
_nav = st.columns(len(PAGES))
for _i, _pg in enumerate(PAGES):
    with _nav[_i]:
        if st.button(PAGE_LABELS[_pg], key=f"nav_{_pg}",
                     use_container_width=True,
                     type="primary" if st.session_state.page == _pg
                          else "secondary"):
            st.session_state.page = _pg
            st.rerun()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
PAGE = st.session_state.page

# ══════════════════════════════════════════════════════════
# WORKLIST
# ══════════════════════════════════════════════════════════
if PAGE == "worklist":
    rj  = fetch_reports(st.session_state.api)
    rps = rj.get("reports",[])
    ana = sum(1 for r in rps if r.get("status")=="analyzed")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Studies", rj.get("total",0))
    c2.metric("Analyzed",      ana)
    c3.metric("Pending",       rj.get("total",0) - ana)
    c4.metric("DICOM Studies", sum(1 for r in rps if r.get("is_dicom")))
    c5.metric("Modality",      "CXR")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Search + sort
    sc1, sc2, sc3 = st.columns([3,1,1])
    with sc1:
        search = st.text_input(
            "Search",
            placeholder="Search by filename or finding...",
            value=st.session_state.search,
            label_visibility="collapsed",
            key="wl_s"
        )
        st.session_state.search = search
    with sc2:
        sort_by = st.selectbox(
            "Sort",
            ["Date (newest)", "Date (oldest)", "Finding A-Z", "Priority"],
            label_visibility="collapsed",
            key="wl_sort"
        )
    with sc3:
        if st.button("Refresh", use_container_width=True):
            fetch_reports.clear()
            fetch_health.clear()
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Filter + sort
    filtered = [
        r for r in rps
        if not search or
        search.lower() in r.get("filename","").lower() or
        search.lower() in (r.get("disease_label") or "").lower()
    ]

    if sort_by == "Date (oldest)":
        filtered = list(reversed(filtered))
    elif sort_by == "Finding A-Z":
        filtered = sorted(filtered, key=lambda r: r.get("disease_label",""))
    elif sort_by == "Priority":
        # STAT first, then URGENT, then ROUTINE, then NORMAL
        priority_order = {"STAT":0,"URGENT":1,"ROUTINE":2,"NORMAL":3,"":4}
        filtered = sorted(
            filtered,
            key=lambda r: priority_order.get(
                URGENCY.get(r.get("disease_label",""),("",))[0], 4
            )
        )
    else:
        filtered = list(reversed(filtered))

    # Table header
    st.markdown("""
    <div style="display:grid;
                grid-template-columns:32px 1fr 100px 140px 72px 80px 120px 56px 64px;
                background:#232830; border:1px solid #2e3442;
                border-bottom:2px solid #363d4e; padding:8px 12px;
                font-family:'Inter',sans-serif; font-size:11px; font-weight:600;
                color:#6a7385; text-transform:uppercase; letter-spacing:0.6px;
                align-items:center; gap:6px;">
        <div>#</div><div>Filename</div><div>Study ID</div>
        <div>Finding</div><div>Priority</div><div>Status</div>
        <div>Date</div><div>Time</div><div>Action</div>
    </div>
    """, unsafe_allow_html=True)

    if not filtered:
        st.markdown(f"""
        <div style="background:#1e2430; border:1px solid #252d3a; border-top:none;
                    padding:32px; text-align:center; color:#4a5368;
                    font-family:'Inter',sans-serif; font-size:14px;
                    border-radius:0 0 6px 6px;">
            {'No results for "' + search + '"' if search else
             'No studies. Upload a chest X-ray to begin.'}
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, r in enumerate(filtered, 1):
            sel   = st.session_state.study_id == r.get("image_id")
            find  = r.get("disease_label") or "—"
            urg, uc = URGENCY.get(find, ("ROUTINE","#6abe6a"))
            sc    = "#72c472" if r.get("status")=="analyzed" else "#d4aa50"
            bg    = "#162232" if sel else "#1a2030"
            bc    = "#264060" if sel else "#222830"
            fn    = r.get("filename","")[:26]
            sid   = r.get("image_id","")[:10]
            dt    = (r.get("analyzed_at") or r.get("uploaded_at",""))[:16]
            tt    = r.get("total_time","—")

            rc, btn = st.columns([9,1])
            with rc:
                st.markdown(f"""
                <div style="display:grid;
                            grid-template-columns:32px 1fr 100px 140px 72px 80px 120px 56px;
                            background:{bg}; border:1px solid {bc}; border-top:none;
                            padding:8px 12px; align-items:center; gap:6px;
                            font-family:'Inter',sans-serif; font-size:13px;
                            color:#c8d0e0; transition:background 0.1s;">
                    <div style="color:#4a5368; font-size:12px;">{i:02d}</div>
                    <div style="font-weight:500; overflow:hidden;
                                text-overflow:ellipsis; white-space:nowrap;"
                         title="{r.get('filename','')}">{fn}</div>
                    <div style="font-family:'JetBrains Mono',monospace;
                                font-size:11px; color:#4a5368;">{sid}...</div>
                    <div style="color:{'#72c472' if find!='—' else '#4a5368'};
                                font-weight:600; overflow:hidden;
                                text-overflow:ellipsis; white-space:nowrap;"
                         title="{DISEASE_INFO.get(find,'')}">{find}</div>
                    <div style="color:{uc}; font-size:11px;
                                font-weight:600;">{urg}</div>
                    <div style="color:{sc}; font-size:12px; font-weight:600;">
                        {r.get('status','').upper()}</div>
                    <div style="font-family:'JetBrains Mono',monospace;
                                font-size:11px; color:#6a7385;">{dt}</div>
                    <div style="font-family:'JetBrains Mono',monospace;
                                font-size:12px; color:#6a7385;">{tt}s</div>
                </div>
                """, unsafe_allow_html=True)
            with btn:
                if st.button("Open", key=f"w{r.get('image_id')}",
                             use_container_width=True):
                    with cxr_spinner("Opening study..."):
                        st.session_state.study_id = r.get("image_id")
                        rx = req(f"/report/{r.get('image_id')}")
                        if rx:
                            st.session_state.report = rx.json()
                        add_recent(r.get("image_id"))
                    st.session_state.page = "viewer"
                    st.rerun()

    # Recently opened
    if st.session_state.last_ids:
        st.markdown("""
        <div style="margin-top:16px; font-family:'Inter',sans-serif;
                    font-size:11px; font-weight:600; color:#6a7385;
                    text-transform:uppercase; letter-spacing:0.6px;
                    margin-bottom:8px;">
            Recently Opened
        </div>
        """, unsafe_allow_html=True)
        rcols = st.columns(min(len(st.session_state.last_ids),5))
        for i, rid in enumerate(st.session_state.last_ids[:5]):
            with rcols[i]:
                if st.button(f"...{rid[-8:]}", key=f"rec{rid}",
                             use_container_width=True):
                    with cxr_spinner("Opening study..."):
                        st.session_state.study_id = rid
                        rx = req(f"/report/{rid}")
                        if rx:
                            st.session_state.report = rx.json()
                    st.session_state.page = "viewer"
                    st.rerun()


# ══════════════════════════════════════════════════════════
# VIEWER
# ══════════════════════════════════════════════════════════
if PAGE == "viewer":
    # Auto-load the study at the TOP of the worklist (newest) if none selected,
    # so the viewer opens on the most recent study without picking first.
    # /reports is oldest→newest, so the newest analyzed study is the last one.
    if not st.session_state.study_id:
        _rj_auto = fetch_reports(st.session_state.api)
        _analyzed = [r for r in _rj_auto.get("reports", [])
                     if r.get("status") == "analyzed"]
        if not _analyzed:
            _analyzed = _rj_auto.get("reports", [])
        if _analyzed:
            _first = _analyzed[-1]          # newest = top of worklist
            st.session_state.study_id = _first["image_id"]
            _rx = req(f"/report/{_first['image_id']}", t=15)
            if _rx:
                st.session_state.report = _rx.json()
            st.rerun()

    if not st.session_state.study_id:
        st.markdown("""
        <div style="height:360px; display:flex; align-items:center;
                    justify-content:center; flex-direction:column; gap:12px;">
            <div style="width:64px; height:64px; border-radius:50%;
                        background:#232830; border:1px solid #2e3442;
                        display:flex; align-items:center;
                        justify-content:center; font-size:28px; opacity:0.4;">
                X
            </div>
            <div style="font-family:'Inter',sans-serif; font-size:15px;
                        font-weight:500; color:#4a5368;">
                No study loaded
            </div>
            <div style="font-family:'Inter',sans-serif; font-size:13px;
                        color:#363d4e;">
                Open a study from the Worklist or upload a new image
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        rep = st.session_state.report or {}
        sid = st.session_state.study_id
        fn  = rep.get("filename","—")
        dis = rep.get("disease_label","—")
        sta = rep.get("status","")
        urg, uc = URGENCY.get(dis, ("ROUTINE","#6abe6a"))

        # ── PACS Toolbar (like Synapse) ───────────────────────
        dm_pre = rep.get("dicom_metadata", {})
        patient_name = dm_pre.get("patient_name", "—")
        study_date   = dm_pre.get("study_date", "—")
        modality     = dm_pre.get("modality", "CR")
        view_pos     = dm_pre.get("view_position", "PA")

        st.markdown(f"""
        <div style="background:#0a0c10; border:1px solid #1e2430;
                    border-radius:6px 6px 0 0; padding:6px 14px;
                    display:flex; align-items:center; gap:16px;
                    font-family:'JetBrains Mono',monospace; font-size:11px;
                    color:#5a6a7a;">
            <span style="color:#8a9aaa; font-weight:600;">{fn}</span>
            <span>|</span>
            <span>W: {st.session_state.ww}</span>
            <span>C: {st.session_state.wl_val}</span>
            <span>|</span>
            <span style="color:{uc}; font-weight:600;">{urg}</span>
            <span>|</span>
            <span style="color:#72c472;">{dis if sta=='analyzed' else 'Pending'}</span>
            <span style="flex:1;"></span>
            <span>{modality} · {view_pos} · {study_date}</span>
        </div>
        """, unsafe_allow_html=True)

        left, right = st.columns([11,10], gap="large")

        with left:
            # Cached fetch — only hits network once per study, so slider
            # adjustments (brightness/contrast/W-L) reprocess locally and stay instant
            raw_img = fetch_image_bytes(st.session_state.api, sid)
            if raw_img:
                # Apply W/L processing locally (fast, no network)
                processed = wl_img(raw_img)

                # DICOM-style info strip above the image
                p_id  = dm_pre.get("patient_id","—")
                p_age = dm_pre.get("patient_age","—")
                p_sex = dm_pre.get("patient_sex","—")
                acq_time = datetime.now().strftime("%d %b %Y  %H:%M")

                st.markdown(f"""
                <div style="background:#0a0c10; border:1px solid #1e2430;
                            border-top:none; padding:6px 12px; display:flex;
                            justify-content:space-between;
                            font-family:'JetBrains Mono',monospace; font-size:10px;
                            color:#00b0c0;">
                    <span>{patient_name} · {p_id} · {p_age}/{p_sex}</span>
                    <span>{modality} {view_pos} · {'DICOM' if rep.get('is_dicom') else 'PNG'} · {acq_time}</span>
                </div>
                """, unsafe_allow_html=True)

                st.image(
                    Image.open(io.BytesIO(processed)),
                    use_container_width=True,
                    caption=None
                )
            else:
                st.markdown("""
                <div style="background:#000000; border:1px solid #1e2430;
                            border-top:none; min-height:320px; display:flex;
                            align-items:center; justify-content:center;
                            flex-direction:column; gap:10px; padding:24px;
                            text-align:center; color:#5a6a7a;
                            font-family:'JetBrains Mono',monospace;
                            font-size:12px; line-height:1.7;">
                    <div style="font-size:13px; color:#8a9aaa;">
                        Image file not available on the server
                    </div>
                    <div style="color:#46546b; max-width:360px;">
                        The report is stored, but the image isn't on the current
                        backend. This happens when the study was uploaded in a
                        previous Colab session (the image files are not kept in
                        MongoDB, only the report). Re-upload the X-ray, or check
                        the API URL in Settings.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Retry loading image",
                             use_container_width=True, key="retry_img"):
                    _fetch_image_bytes_cached.clear()
                    st.rerun()

            # W/L Controls panel — like Synapse bottom bar
            st.markdown("""
            <div style="background:#0e1218; border:1px solid #1e2430;
                        border-top:1px solid #2a3442; padding:10px 14px 4px;
                        border-radius:0 0 6px 6px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                            color:#3a4a5a; text-transform:uppercase;
                            letter-spacing:0.5px; margin-bottom:8px;">
                    Window / Level &nbsp;&nbsp; Brightness / Contrast
                </div>
            </div>
            """, unsafe_allow_html=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                w = st.slider("Window", 20, 255, st.session_state.ww, key="sw")
                st.session_state.ww = w
                b = st.slider("Brightness", 0.3, 2.0,
                              st.session_state.brightness, step=0.05, key="sb")
                st.session_state.brightness = b
            with cc2:
                l = st.slider("Level", 0, 255,
                              st.session_state.wl_val, key="sl")
                st.session_state.wl_val = l
                cv = st.slider("Contrast", 0.5, 2.0,
                               st.session_state.contrast, step=0.05, key="sc2")
                st.session_state.contrast = cv

            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                inv = st.checkbox("Invert image",
                                  value=st.session_state.inv, key="si")
                st.session_state.inv = inv
            with ic2:
                if st.button("Reset controls",
                             use_container_width=True, key="rst"):
                    for k, v in [("ww",160),("wl_val",120),("inv",False),
                                  ("brightness",1.0),("contrast",1.0)]:
                        st.session_state[k] = v
                    st.rerun()
            with ic3:
                if raw_img:
                    st.download_button(
                        "Save image",
                        data=wl_img(raw_img),
                        file_name=f"CXR_{sid[:8]}.png",
                        mime="image/png",
                        use_container_width=True,
                        key="save_img"
                    )

        with right:
            if sta != "analyzed":
                st.markdown("""
                <div style="background:#232830; border:1px solid #2e3442;
                            border-radius:8px; padding:24px; text-align:center;
                            color:#4a5368; font-size:14px; margin-bottom:14px;">
                    This study has not been analyzed yet
                </div>
                """, unsafe_allow_html=True)
                if st.button("Run AI Analysis",
                             use_container_width=True, key="anb"):
                    with cxr_spinner("Analyzing: PubMedCLIP → LLaMA report → Database…"):
                        rx = req(f"/analyze/{sid}",
                                 method="POST", t=180, no_cache=True)
                    if rx:
                        st.session_state.report = rx.json()
                        st.rerun()
                    else:
                        st.error(
                            "Analysis failed: "
                            f"{st.session_state.get('last_error') or 'unknown error'}"
                        )
            else:
                # ── Study Information panel — single summary of the study,
                #    visible throughout the analyzed-study workflow ─────────
                _top      = rep.get("top_diseases") or []
                _conf     = f"{_top[0][1]*100:.0f}%" if _top else "—"
                _exported = rep.get("report_exported")
                _exp_txt  = "Exported" if _exported else "Not exported"
                _exp_col  = "#72c472" if _exported else "#8a93a6"
                section("Study Information")
                data_row("Study ID",    (sid or "")[:18] + "...")
                data_row("Filename",    fn)
                data_row("Status",      sta.upper(), color="#72c472")
                data_row("Uploaded",    (rep.get("uploaded_at") or "—")[:16])
                data_row("Analyzed",    (rep.get("analyzed_at") or "—")[:16])
                data_row("Predicted",   dis, color="#72c472")
                data_row("Confidence",  _conf, color=uc)
                data_row("Report time", f"{rep.get('total_time',0)}s")
                data_row("Export",      _exp_txt, color=_exp_col, border=False)
                st.markdown("<div style='height:14px'></div>",
                            unsafe_allow_html=True)

                # DICOM patient info
                dm = rep.get("dicom_metadata",{})
                if dm:
                    section("Patient Information")
                    for k,v in list(dm.items())[:5]:
                        data_row(k.replace("_"," ").title(), v)
                    st.markdown("<div style='height:14px'></div>",
                                unsafe_allow_html=True)

                # Primary finding
                section("AI Analysis — PubMedCLIP")
                dis_info = DISEASE_INFO.get(dis,"")
                urg2, uc2 = URGENCY.get(dis,("ROUTINE","#6abe6a"))

                st.markdown(f"""
                <div style="background:#162316; border:1px solid #265626;
                            border-left:4px solid #4abe4a; border-radius:6px;
                            padding:12px 16px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:4px;">
                        <div style="font-family:'Inter',sans-serif; font-size:18px;
                                    font-weight:700; color:#72c472;">{dis}</div>
                        <div style="font-family:'Inter',sans-serif; font-size:11px;
                                    font-weight:600; color:{uc2};
                                    background:#1c2028; padding:2px 8px;
                                    border-radius:10px; border:1px solid #363d4e;">
                            {urg2}
                        </div>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;
                                font-size:11px; color:#3a6a3a;">
                        PubMedCLIP · image-text matching
                    </div>
                    <div style="font-family:'Inter',sans-serif; font-size:12px;
                                color:#5a8a5a; margin-top:6px; line-height:1.5;">
                        {dis_info}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence scores — rank 1 is the primary prediction, the
                # rest are alternative possibilities (PubMedCLIP Top-3).
                section("Confidence Scores — Top 3 Predictions")
                for _rank, (d, s) in enumerate(rep.get("top_diseases",[])):
                    p    = int(s * 100)
                    col  = "#72c472" if p>50 else ("#d4aa50" if p>20 else "#e07878")
                    info = DISEASE_INFO.get(d,"")
                    if _rank == 0:
                        tag_txt, tag_col, tag_bg = "PRIMARY", "#72c472", "#162316"
                    else:
                        tag_txt, tag_col, tag_bg = "ALTERNATIVE", "#8a93a6", "#20242e"
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between;
                                    align-items:center; margin-bottom:4px;">
                            <span style="font-family:'Inter',sans-serif;
                                         font-size:13px; font-weight:500;
                                         color:#c8d0e0;">
                                {d}
                                <span style="font-family:'Inter',sans-serif;
                                             font-size:9px; font-weight:600;
                                             color:{tag_col}; background:{tag_bg};
                                             border:1px solid #2e3442; padding:1px 6px;
                                             border-radius:8px; margin-left:6px;
                                             letter-spacing:0.5px;">{tag_txt}</span>
                            </span>
                            <span style="font-family:'JetBrains Mono',monospace;
                                         font-size:12px; font-weight:600;
                                         color:{col};">{p}%</span>
                        </div>
                        <div style="background:#2a3040; height:5px;
                                    border-radius:3px; overflow:hidden;">
                            <div style="width:{max(p,1)}%; height:100%;
                                        background:{col}; border-radius:3px;">
                            </div>
                        </div>
                        <div style="font-family:'Inter',sans-serif; font-size:11px;
                                    color:#4a5368; margin-top:3px;">{info}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Observational report
                section("Observational Report")
                report_text = rep.get("llm_report","—")
                st.markdown(f"""
                <div style="background:#232830; border:1px solid #2e3442;
                            border-left:3px solid #4d8ef0; border-radius:6px;
                            padding:14px 16px; font-family:'Inter',sans-serif;
                            font-size:13px; color:#d8dce8; line-height:1.8;">
                    {report_text}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                            color:#363d4e; margin-top:6px;">
                    VLM {rep.get('vlm_time',0)}s
                    &nbsp;·&nbsp; Report {rep.get('llm_time',0)}s
                    &nbsp;·&nbsp; Total {rep.get('total_time',0)}s
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:14px'></div>",
                            unsafe_allow_html=True)

                # Actions
                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button("Export PDF",
                                 use_container_width=True, key="epdf"):
                        with cxr_spinner("Generating PDF..."):
                            pr = req(f"/export/{sid}", t=30)
                        if pr:
                            # Backend now records the export in the audit trail;
                            # mirror it locally so the Study Information panel
                            # reflects "Exported" on the next rerun.
                            if isinstance(st.session_state.report, dict):
                                st.session_state.report["report_exported"] = True
                            fetch_reports.clear()
                            st.download_button(
                                "Download PDF",
                                data=pr.content,
                                file_name=f"CXR_{dis}_{sid[:8]}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="dl_pdf"
                            )
                        else:
                            st.error(
                                "PDF generation failed. "
                                "Run in Colab: !pip install fpdf2 -q"
                            )
                with ac2:
                    if st.button("Analysis",
                                 use_container_width=True, key="reanalyze"):
                        with cxr_spinner("Analyzing: PubMedCLIP → LLaMA report → Database…"):
                            rx = req(f"/analyze/{sid}",
                                     method="POST", t=180, no_cache=True)
                        if rx:
                            st.session_state.report = rx.json()
                            st.rerun()
                        else:
                            st.error(
                                "Analysis failed: "
                                f"{st.session_state.get('last_error') or 'unknown error'}"
                            )

                if st.button("Delete study",
                             use_container_width=True, key="del"):
                    with cxr_spinner("Deleting study..."):
                        dx = req(f"/report/{sid}",
                                 method="DELETE", no_cache=True)
                    if dx:
                        st.session_state.study_id = None
                        st.session_state.report   = None
                        st.session_state.page     = "worklist"
                        st.rerun()
                    else:
                        st.error("Delete failed")

                st.markdown("""
                <div style="margin-top:12px; padding:8px 12px;
                            background:#201c0e; border:1px solid #3a3210;
                            border-radius:6px; font-family:'Inter',sans-serif;
                            font-size:11px; color:#7a6a30; line-height:1.5;">
                    For demonstration purposes only. Not intended for clinical
                    or diagnostic use. Consult a qualified radiologist for
                    medical interpretation.
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════
if PAGE == "upload":
    uc1, uc2 = st.columns([1,1], gap="large")

    with uc1:
        section("Upload Chest X-Ray")

        ups = st.file_uploader(
            "Choose file(s)",
            type=["png","jpg","jpeg","dcm"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="PNG, JPG, JPEG, or DICOM (.dcm) — select one or many, max 20 MB each"
        )

        if ups:
            total_kb = sum(len(f.getvalue()) for f in ups) / 1024
            st.markdown(f"""
            <div style="background:#232830; border:1px solid #2e3442;
                        border-radius:8px; padding:14px 16px; margin:12px 0;">
            """, unsafe_allow_html=True)
            data_row("Files selected", str(len(ups)))
            data_row("Total size", f"{total_kb:.0f} KB")
            data_row("Preprocessing", "Resize to 224 x 224 RGB", color="#72c472")
            data_row("Processing", "Sequential upload + analysis",
                     color="#72c472", border=False)
            st.markdown("</div>", unsafe_allow_html=True)

            btn_label = ("Upload and Analyze" if len(ups) == 1
                         else f"Upload and Analyze {len(ups)} images")
            if st.button(btn_label, use_container_width=True, key="ub"):
                prog    = st.progress(0)
                results = []
                errors  = []
                n       = len(ups)

                for idx, up in enumerate(ups):
                    base = int(idx / n * 100)
                    prog.progress(
                        min(base + 4, 99),
                        text=f"[{idx+1}/{n}] Image Upload + Preprocessing — {up.name}"
                    )
                    ru = req(
                        "/upload", method="POST", t=30, no_cache=True,
                        files={"file": (up.name, up.getvalue(),
                                        up.type or "image/png")}
                    )
                    if not ru:
                        errors.append(f"{up.name}: upload failed")
                        continue

                    uid = ru.json().get("image_id")
                    prog.progress(
                        min(base + 12, 99),
                        text=f"[{idx+1}/{n}] PubMedCLIP Analysis → LLaMA Report → "
                             f"DB Storage — {up.name}"
                    )
                    ra = req(f"/analyze/{uid}", method="POST",
                             t=180, no_cache=True)
                    if not ra:
                        errors.append(
                            f"{up.name}: "
                            f"{st.session_state.get('last_error') or 'analysis failed'}"
                        )
                        continue

                    an = ra.json()
                    results.append((uid, an))
                    # Make the FIRST successful study the active one so the
                    # viewer opens on it directly.
                    if st.session_state.study_id is None or \
                       len(results) == 1:
                        st.session_state.study_id = results[0][0]
                        st.session_state.report   = results[0][1]
                    add_recent(uid)

                prog.progress(100, text="Report Generation Completed")

                # No errors → jump straight to the viewer (first study loaded),
                # so the user doesn't have to open studies one by one.
                if results and not errors:
                    st.session_state.study_id = results[0][0]
                    st.session_state.report   = results[0][1]
                    st.session_state.page      = "viewer"
                    st.rerun()

                if results:
                    st.markdown(f"""
                    <div style="background:#162316; border:1px solid #265626;
                                border-left:4px solid #4abe4a; border-radius:6px;
                                padding:14px 16px; margin-top:12px;">
                        <div style="font-family:'Inter',sans-serif; font-size:14px;
                                    font-weight:600; color:#72c472; margin-bottom:8px;">
                            {len(results)} of {n} analyzed successfully
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    for uid, an in results:
                        dis5 = an.get("disease_label","—")
                        urg5, uc5 = URGENCY.get(dis5,("ROUTINE","#6abe6a"))
                        _rcol, _bcol = st.columns([4, 1])
                        with _rcol:
                            data_row(an.get("filename", uid[:8]),
                                     f"{dis5}  ·  {urg5}  ·  {an.get('total_time',0)}s",
                                     color=uc5)
                        with _bcol:
                            if st.button("Open", key=f"upopen_{uid}",
                                         use_container_width=True):
                                st.session_state.study_id = uid
                                st.session_state.report   = an
                                add_recent(uid)
                                st.session_state.page = "viewer"
                                st.rerun()

                if errors:
                    for e in errors:
                        st.error(e)

                if results:
                    if st.button("Open in Viewer", use_container_width=True,
                                 type="primary", key="upload_to_viewer"):
                        st.session_state.page = "viewer"
                        st.rerun()

    with uc2:
        section("Preview")
        if ups:
            cols = st.columns(2)
            for i, up in enumerate(ups[:6]):
                with cols[i % 2]:
                    if up.name.lower().endswith(".dcm"):
                        st.markdown(f"""
                        <div style="background:#161a22; border:1px solid #2a3040;
                                    border-radius:6px; height:130px; display:flex;
                                    align-items:center; justify-content:center;
                                    flex-direction:column; gap:6px; color:#5a6a7a;
                                    font-family:'JetBrains Mono',monospace;
                                    font-size:11px; margin-bottom:8px;">
                            DICOM<br>{up.name[:18]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        try:
                            up.seek(0)
                            st.image(Image.open(up),
                                     use_container_width=True,
                                     caption=up.name[:20])
                        except:
                            st.warning(f"Cannot preview {up.name[:18]}")
            if len(ups) > 6:
                st.caption(f"+ {len(ups) - 6} more not shown")
        else:
            st.markdown("""
            <div style="background:#161a22; border:2px dashed #2a3040;
                        border-radius:8px; height:280px; display:flex;
                        align-items:center; justify-content:center;
                        flex-direction:column; gap:8px; color:#363d4e;
                        font-family:'Inter',sans-serif; font-size:14px;">
                <div style="font-size:28px; opacity:0.3; margin-bottom:4px;">
                    [ ]
                </div>
                No file selected
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════
if PAGE == "history":
    rj3 = fetch_reports(st.session_state.api)
    rp3 = rj3.get("reports",[])

    st.markdown(f"""
    <div style="font-family:'Inter',sans-serif; font-size:13px;
                font-weight:500; color:#6a7385; margin-bottom:16px;">
        {rj3.get('total',0)} studies in archive
    </div>
    """, unsafe_allow_html=True)

    # ── Search by Study ID / filename + filter by disease ─────
    hs1, hs2 = st.columns([2, 1])
    with hs1:
        hist_q = st.text_input(
            "Search history",
            placeholder="Search by Study ID or filename...",
            label_visibility="collapsed",
            key="hist_q",
        )
    with hs2:
        _diseases = sorted({r.get("disease_label") for r in rp3
                            if r.get("disease_label")})
        disease_filter = st.selectbox(
            "Filter by finding",
            ["All findings"] + _diseases,
            label_visibility="collapsed",
            key="hist_disease",
        )

    q = (hist_q or "").lower().strip()
    rp3 = [
        r for r in rp3
        if (not q
            or q in (r.get("image_id") or "").lower()
            or q in (r.get("filename") or "").lower())
        and (disease_filter == "All findings"
             or r.get("disease_label") == disease_filter)
    ]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if not rp3:
        st.markdown("""
        <div style="background:#232830; border:1px solid #2e3442;
                    border-radius:8px; padding:32px; text-align:center;
                    color:#4a5368; font-size:14px;">
            No studies match the current search or filter.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Collect which studies are checked this run
        selected_ids = []

        # ── Bulk action bar ───────────────────────────────────
        ba1, ba2 = st.columns([3, 1])
        with ba2:
            delete_selected = st.button(
                "Delete Selected", use_container_width=True,
                key="bulk_del", type="primary"
            )

        for r in reversed(rp3):
            iid  = r.get("image_id")
            dis4 = r.get("disease_label","—")
            urg4, uc4 = URGENCY.get(dis4,("ROUTINE","#6abe6a"))

            row_chk, row_exp = st.columns([1, 11])
            with row_chk:
                if st.checkbox(" ", key=f"chk_{iid}",
                               label_visibility="collapsed"):
                    selected_ids.append(iid)
            with row_exp:
                lbl = (
                    f"{r.get('filename','?')}  |  "
                    f"{dis4}  |  {urg4}  |  "
                    f"{r.get('status','').upper()}"
                )
                with st.expander(lbl):
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        sc2 = "#72c472" if r.get("status")=="analyzed" else "#d4aa50"
                        data_row("Study ID",   (iid or "")[:14]+"...")
                        data_row("Status",     r.get("status","").upper(),
                                 color=sc2)
                        data_row("DICOM",      "Yes" if r.get("is_dicom") else "No")
                        data_row("Total time", f"{r.get('total_time',0)}s",
                                 border=False)
                    with hc2:
                        data_row("Finding",   dis4,  color="#72c472")
                        data_row("Priority",  urg4,  color=uc4)
                        data_row("Uploaded",  (r.get("uploaded_at") or "")[:16])
                        data_row("Analyzed",  (r.get("analyzed_at") or "")[:16],
                                 border=False)

                    hb1, hb2 = st.columns(2)
                    with hb1:
                        if st.button("Open in Viewer",
                                     key=f"ho{iid}",
                                     use_container_width=True):
                            with cxr_spinner("Opening study..."):
                                st.session_state.study_id = iid
                                rx = req(f"/report/{iid}")
                                if rx:
                                    st.session_state.report = rx.json()
                                add_recent(iid)
                            st.session_state.page = "viewer"
                            st.rerun()
                    with hb2:
                        if st.button("Delete",
                                     key=f"hd{iid}",
                                     use_container_width=True):
                            with cxr_spinner("Deleting..."):
                                req(f"/report/{iid}",
                                    method="DELETE", no_cache=True)
                            st.rerun()

        # ── Handle bulk delete ────────────────────────────────
        if delete_selected:
            if not selected_ids:
                st.warning("No studies selected. Tick the boxes first.")
            else:
                with cxr_spinner(f"Deleting {len(selected_ids)} studies..."):
                    for iid in selected_ids:
                        req(f"/report/{iid}", method="DELETE", no_cache=True)
                st.success(f"Deleted {len(selected_ids)} studies")
                st.rerun()


# ══════════════════════════════════════════════════════════
# PERFORMANCE  — processing-metrics dashboard
# ══════════════════════════════════════════════════════════
if PAGE == "performance":
    rjp = fetch_reports(st.session_state.api)
    # Only analyzed studies carry timing metrics.
    # /reports returns records oldest → newest, so the trend already reads
    # left-to-right chronologically.
    studies = [r for r in rjp.get("reports", [])
               if r.get("status") == "analyzed"]

    if not studies:
        st.markdown("""
        <div style="background:#232830; border:1px solid #2e3442;
                    border-radius:8px; padding:32px; text-align:center;
                    color:#4a5368; font-size:14px;">
            No analyzed studies yet. Analyze an image to collect metrics.
        </div>
        """, unsafe_allow_html=True)
    else:
        def _vals(key):
            return [float(r.get(key, 0) or 0) for r in studies]

        totals = _vals("total_time")
        vlms   = _vals("vlm_time")
        llms   = _vals("llm_time")
        dbs    = _vals("db_time")
        n      = len(studies)

        def _avg(v):
            return sum(v) / len(v) if v else 0.0

        # ── Headline metrics ──────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Studies",       n)
        m2.metric("Avg Total",     f"{_avg(totals):.1f}s")
        m3.metric("Fastest Total", f"{min(totals):.1f}s")
        m4.metric("Slowest Total", f"{max(totals):.1f}s")

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── Stage averages ────────────────────────────────────
        s1, s2, s3 = st.columns(3)
        s1.metric("Avg PubMedCLIP (VLM)", f"{_avg(vlms):.1f}s")
        s2.metric("Avg LLaMA (LLM)",      f"{_avg(llms):.1f}s")
        s3.metric("Avg DB Storage",       f"{_avg(dbs):.2f}s")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Trend chart ───────────────────────────────────────
        section("Processing-Time Trend (per study, chronological)")
        st.line_chart(
            {"Total (s)": totals, "VLM (s)": vlms, "LLM (s)": llms},
            height=260,
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Per-study table ───────────────────────────────────
        section("Per-Study Metrics")
        st.markdown("""
        <div style="display:grid;
                    grid-template-columns:32px 1fr 130px 72px 72px 64px 72px 80px;
                    background:#232830; border:1px solid #2e3442;
                    border-bottom:2px solid #363d4e; padding:8px 12px;
                    font-family:'Inter',sans-serif; font-size:11px; font-weight:600;
                    color:#6a7385; text-transform:uppercase; letter-spacing:0.6px;
                    gap:6px;">
            <div>#</div><div>Filename</div><div>Finding</div><div>VLM</div>
            <div>LLM</div><div>DB</div><div>Total</div><div>Exported</div>
        </div>
        """, unsafe_allow_html=True)
        for i, r in enumerate(studies, 1):
            exp_txt = "Yes" if r.get("report_exported") else "—"
            exp_col = "#72c472" if r.get("report_exported") else "#4a5368"
            st.markdown(f"""
            <div style="display:grid;
                        grid-template-columns:32px 1fr 130px 72px 72px 64px 72px 80px;
                        background:#1a2030; border:1px solid #222830; border-top:none;
                        padding:8px 12px; align-items:center; gap:6px;
                        font-family:'JetBrains Mono',monospace; font-size:12px;
                        color:#c8d0e0;">
                <div style="color:#4a5368;">{i:02d}</div>
                <div style="overflow:hidden; text-overflow:ellipsis;
                            white-space:nowrap; font-family:'Inter',sans-serif;"
                     title="{r.get('filename','')}">{r.get('filename','')[:28]}</div>
                <div style="color:#72c472; overflow:hidden; text-overflow:ellipsis;
                            white-space:nowrap;">{r.get('disease_label','—')}</div>
                <div>{float(r.get('vlm_time',0) or 0):.1f}s</div>
                <div>{float(r.get('llm_time',0) or 0):.1f}s</div>
                <div>{float(r.get('db_time',0) or 0):.2f}s</div>
                <div style="color:#74aaff;">{float(r.get('total_time',0) or 0):.1f}s</div>
                <div style="color:{exp_col};">{exp_txt}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════
if PAGE == "settings":
    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        section("API Connection")

        nu = st.text_input(
            "API Base URL",
            value=st.session_state.api,
            help="Paste your Google Colab ngrok URL — "
                 "changes every time Colab restarts"
        )
        
        if st.button("Save and test connection",
                     use_container_width=True, key="sv"):
            st.session_state.api = nu.rstrip("/")
            fetch_reports.clear()
            fetch_health.clear()
            with cxr_spinner("Testing connection..."):
                hh = fetch_health(st.session_state.api)
            if hh:
                st.success(
                    f"Connected   "
                    f"DB: {hh.get('mongodb','—').split()[0]}   "
                    f"{hh.get('total_reports',0)} studies"
                )
            else:
                st.error(
                    "Cannot connect. Verify the URL is correct "
                    "and Colab Cell 8 is running."
                )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        section("System Status")

        sr = req("/status")
        if sr:
            sj = sr.json()
            data_row("API",          "Running",                      "#72c472")
            data_row("Uptime",       sj.get("uptime","—")[:18])
            data_row("Requests",     str(sj.get("total_requests",0)))
            data_row("Errors",       str(sj.get("error_count",0)))
            data_row("Error rate",   sj.get("error_rate","—"))
            data_row("Database",     sj.get("mongodb","—"),          "#72c472")
            data_row("Client retries", str(st.session_state.retry_count))
            data_row("Mode",         "AI-assisted analysis",         "#c8d0e0",
                     border=False)
        else:
            st.markdown("""
            <div style="color:#4a5368; font-size:13px; padding:8px 0;">
                Cannot reach status endpoint
            </div>
            """, unsafe_allow_html=True)

    with sc2:
        section("AI Models")

        for lbl, val, note in [
            ("Vision Language Model",
             "PubMedCLIP",
             "flaviagiammarino/pubmed-clip-vit-base-patch32"),
            ("Report Generator",
             "Qwen2.5-1.5B (in-process)",
             "Prompt-engineered, disease-aware structured report"),
            ("Input Resolution",
             "224 x 224 RGB",
             "Preprocessed on upload from any input size"),
            ("Disease Classes",
             "15 categories",
             "Based on NIH ChestX-ray14 dataset labels"),
            ("Analysis Mode",
             "Pretrained weights",
             "No training — using published model weights"),
        ]:
            st.markdown(f"""
            <div style="padding:10px 0; border-bottom:1px solid #262c38;">
                <div style="display:flex; justify-content:space-between;
                            font-family:'Inter',sans-serif; font-size:13px;">
                    <span style="color:#6a7385; font-weight:500;">{lbl}</span>
                    <span style="color:#c8d0e0; font-weight:600;">{val}</span>
                </div>
                <div style="font-family:'Inter',sans-serif; font-size:11px;
                            color:#4a5368; margin-top:3px;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        section("Disease Reference")

        for dis5, info5 in DISEASE_INFO.items():
            urg5, uc5 = URGENCY.get(dis5,("ROUTINE","#6a7385"))
            st.markdown(f"""
            <div style="padding:8px 0; border-bottom:1px solid #262c38;">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:3px;">
                    <span style="font-family:'Inter',sans-serif; font-size:13px;
                                 font-weight:600; color:#72c472;">{dis5}</span>
                    <span style="font-family:'Inter',sans-serif; font-size:11px;
                                 font-weight:600; color:{uc5};">{urg5}</span>
                </div>
                <div style="font-family:'Inter',sans-serif; font-size:12px;
                            color:#4a5368;">{info5}</div>
            </div>
            """, unsafe_allow_html=True)
