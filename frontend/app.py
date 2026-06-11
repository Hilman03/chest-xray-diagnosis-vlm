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
import streamlit as st
import requests
from PIL import Image, ImageEnhance
import io
import numpy as np
from datetime import datetime

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

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
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for k, v in {
    "api"        : DEFAULT_API_URL,
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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API helpers ───────────────────────────────────────────────
HDR = {"ngrok-skip-browser-warning": "true"}

@st.cache_data(ttl=5, show_spinner=False)
def fetch_reports(api_url):
    try:
        r = requests.get(
            api_url.rstrip("/") + "/reports",
            headers=HDR, timeout=8
        )
        return r.json() if r.ok else {"total":0,"reports":[]}
    except:
        return {"total":0,"reports":[]}

@st.cache_data(ttl=8, show_spinner=False)
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
        return r if r.ok else None
    except requests.exceptions.Timeout:
        st.session_state.retry_count += 1
        return None
    except requests.exceptions.ConnectionError:
        st.session_state.ok = False
        return None
    except:
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
        "and click Save. Make sure Colab Cell 10 is running."
    )

# ── Tabs ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "Worklist",
    "Viewer",
    "Compare",
    "Upload",
    "History",
    "Settings",
])

# ══════════════════════════════════════════════════════════
# WORKLIST
# ══════════════════════════════════════════════════════════
with t1:
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
                    st.session_state.study_id = r.get("image_id")
                    rx = req(f"/report/{r.get('image_id')}")
                    if rx:
                        st.session_state.report = rx.json()
                    add_recent(r.get("image_id"))
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
                    st.session_state.study_id = rid
                    rx = req(f"/report/{rid}")
                    if rx:
                        st.session_state.report = rx.json()
                    st.rerun()


# ══════════════════════════════════════════════════════════
# VIEWER
# ══════════════════════════════════════════════════════════
with t2:
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
                Select from Worklist or upload a new image
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
            ir = req(f"/images/{sid}")
            if ir:
                # Apply W/L processing
                processed = wl_img(ir.content)

                # Build DICOM-style patient overlay
                p_id  = dm_pre.get("patient_id","—")
                p_age = dm_pre.get("patient_age","—")
                p_sex = dm_pre.get("patient_sex","—")
                s_date= dm_pre.get("study_date","—")

                # Format date display
                acq_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")

                # Overlay HTML on top of image
                st.markdown(f"""
                <div style="position:relative; background:#000000;
                            border:1px solid #1e2430; border-top:none;">
                    <div style="position:absolute; top:8px; left:10px;
                                z-index:10; pointer-events:none;
                                font-family:'JetBrains Mono',monospace;
                                font-size:11px; color:#00e0ff;
                                line-height:1.7; text-shadow:1px 1px 2px #000;">
                        {patient_name}<br>
                        {p_id}<br>
                        {s_date}<br>
                        {p_age} / {p_sex}
                    </div>
                    <div style="position:absolute; top:8px; right:10px;
                                z-index:10; pointer-events:none;
                                font-family:'JetBrains Mono',monospace;
                                font-size:11px; color:#00e0ff; text-align:right;
                                line-height:1.7; text-shadow:1px 1px 2px #000;">
                        CXR AI-PACS<br>
                        {modality} · {view_pos}<br>
                        {acq_time}<br>
                        AI: {dis if sta=='analyzed' else '—'}
                    </div>
                    <div style="position:absolute; bottom:8px; left:10px;
                                z-index:10; pointer-events:none;
                                font-family:'JetBrains Mono',monospace;
                                font-size:10px; color:#5a8a6a; line-height:1.7;
                                text-shadow:1px 1px 2px #000;">
                        W: {st.session_state.ww} &nbsp; C: {st.session_state.wl_val}<br>
                        S: 1/1 &nbsp; IM: 1
                    </div>
                    <div style="position:absolute; bottom:8px; right:10px;
                                z-index:10; pointer-events:none;
                                font-family:'JetBrains Mono',monospace;
                                font-size:10px; color:#5a8a6a; text-align:right;
                                line-height:1.7; text-shadow:1px 1px 2px #000;">
                        224 x 224<br>
                        {'DICOM' if rep.get('is_dicom') else 'PNG'}
                    </div>
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
                            border-top:none; height:320px; display:flex;
                            align-items:center; justify-content:center;
                            color:#2a3040; font-family:'JetBrains Mono',
                            monospace; font-size:13px;">
                    Image unavailable
                </div>
                """, unsafe_allow_html=True)

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
                if ir:
                    st.download_button(
                        "Save image",
                        data=wl_img(ir.content),
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
                    with st.spinner("Running PubMedCLIP analysis..."):
                        rx = req(f"/analyze/{sid}",
                                 method="POST", t=120, no_cache=True)
                    if rx:
                        st.session_state.report = rx.json()
                        st.rerun()
                    else:
                        st.error(
                            "Analysis failed or timed out. "
                            "Check API connection and try again."
                        )
            else:
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

                # Confidence scores
                section("Confidence Scores")
                for d, s in rep.get("top_diseases",[]):
                    p    = int(s * 100)
                    col  = "#72c472" if p>50 else ("#d4aa50" if p>20 else "#e07878")
                    info = DISEASE_INFO.get(d,"")
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between;
                                    margin-bottom:4px;">
                            <span style="font-family:'Inter',sans-serif;
                                         font-size:13px; font-weight:500;
                                         color:#c8d0e0;">{d}</span>
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
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    if st.button("Export PDF",
                                 use_container_width=True, key="epdf"):
                        with st.spinner("Generating PDF..."):
                            pr = req(f"/export/{sid}", t=30)
                        if pr:
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
                    if st.button("Copy report",
                                 use_container_width=True, key="cpy"):
                        st.code(report_text, language=None)
                        st.caption("Select all and copy")
                with ac3:
                    if st.button("Run tests",
                                 use_container_width=True, key="tst"):
                        with st.spinner("Running functional tests..."):
                            tr = req(f"/test/{sid}", t=120)
                        if tr:
                            tj = tr.json()
                            sm = tj.get("summary",{})
                            if tj.get("overall")=="PASS":
                                st.success(
                                    f"{sm.get('passed')}/{sm.get('total')} "
                                    f"tests passed")
                            else:
                                st.error(
                                    f"{sm.get('passed')}/{sm.get('total')} "
                                    f"tests passed")

                if st.button("Delete study",
                             use_container_width=True, key="del"):
                    dx = req(f"/report/{sid}",
                             method="DELETE", no_cache=True)
                    if dx:
                        st.session_state.study_id = None
                        st.session_state.report   = None
                        st.success("Study deleted successfully")
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
# COMPARE
# ══════════════════════════════════════════════════════════
with t3:
    section("Side-by-Side Study Comparison")

    rj2  = fetch_reports(st.session_state.api)
    opts = {
        f"{r.get('filename','?')[:30]} ({r.get('disease_label','—')})":
        r.get("image_id")
        for r in rj2.get("reports",[])
        if r.get("status")=="analyzed"
    }

    if len(opts) < 2:
        st.markdown("""
        <div style="background:#232830; border:1px solid #2e3442;
                    border-radius:8px; padding:32px; text-align:center;
                    color:#4a5368; font-size:14px;">
            At least 2 analyzed studies are required for comparison.
        </div>
        """, unsafe_allow_html=True)
    else:
        cp1, cp2 = st.columns(2)
        with cp1:
            s1 = st.selectbox("Study A", list(opts.keys()), key="cmp1")
        with cp2:
            s2 = st.selectbox(
                "Study B",
                [k for k in opts.keys() if k != s1],
                key="cmp2"
            )

        if s1 and s2:
            cl, cr = st.columns(2, gap="large")
            for col, iid, lbl in [(cl,opts[s1],s1),(cr,opts[s2],s2)]:
                with col:
                    ir  = req(f"/images/{iid}")
                    rp  = req(f"/report/{iid}")
                    rj3 = rp.json() if rp else {}
                    dis3 = rj3.get("disease_label","—")
                    urg3, uc3 = URGENCY.get(dis3, ("ROUTINE","#6abe6a"))

                    st.markdown(f"""
                    <div style="background:#232830; border:1px solid #2e3442;
                                border-bottom:none; border-radius:8px 8px 0 0;
                                padding:8px 14px; display:flex;
                                justify-content:space-between; align-items:center;">
                        <span style="font-family:'Inter',sans-serif; font-size:13px;
                                     font-weight:500; color:#c8d0e0;
                                     overflow:hidden; text-overflow:ellipsis;
                                     white-space:nowrap; max-width:180px;">
                            {lbl[:28]}
                        </span>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <span style="font-family:'Inter',sans-serif;
                                         font-size:11px; font-weight:600;
                                         color:{uc3};">{urg3}</span>
                            <span style="font-family:'Inter',sans-serif;
                                         font-size:13px; font-weight:600;
                                         color:#72c472;">{dis3}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if ir:
                        st.image(
                            Image.open(io.BytesIO(ir.content)),
                            use_container_width=True
                        )
                    rep3 = rj3.get("llm_report","—")
                    st.markdown(f"""
                    <div style="background:#232830; border:1px solid #2e3442;
                                border-top:none; border-radius:0 0 8px 8px;
                                padding:10px 14px; font-family:'Inter',sans-serif;
                                font-size:12px; color:#8a93a6; line-height:1.7;">
                        {rep3[:200]}{'...' if len(rep3)>200 else ''}
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════
with t4:
    uc1, uc2 = st.columns([1,1], gap="large")

    with uc1:
        section("Upload Chest X-Ray")

        up = st.file_uploader(
            "Choose file",
            type=["png","jpg","jpeg","dcm"],
            label_visibility="collapsed",
            help="PNG, JPG, JPEG, or DICOM (.dcm) — max 20 MB"
        )

        if up:
            kb  = len(up.getvalue()) / 1024
            dcm = up.name.lower().endswith(".dcm")

            st.markdown(f"""
            <div style="background:#232830; border:1px solid #2e3442;
                        border-radius:8px; padding:14px 16px; margin:12px 0;">
            """, unsafe_allow_html=True)
            data_row("Filename", up.name)
            data_row("File size", f"{kb:.0f} KB")
            data_row("Format",
                     "DICOM" if dcm else up.type.split("/")[-1].upper())
            data_row("Preprocessing",  "Resize to 224 x 224 RGB",
                     color="#72c472")
            data_row("DICOM metadata",
                     "Will be extracted" if dcm else "Not applicable",
                     color="#72c472" if dcm else "#4a5368",
                     border=False)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Upload and Analyze",
                         use_container_width=True, key="ub"):
                prog = st.progress(0)

                prog.progress(10, text="Step 1 of 3 — Uploading image...")
                ru = req(
                    "/upload", method="POST", t=30, no_cache=True,
                    files={"file":(
                        up.name, up.getvalue(),
                        up.type or "image/png"
                    )}
                )
                if not ru:
                    prog.empty()
                    st.error(
                        "Upload failed. Check API connection "
                        "and make sure the file is valid."
                    )
                    st.stop()

                uid = ru.json().get("image_id")
                st.session_state.study_id = uid
                prog.progress(35, text="Step 2 of 3 — Running AI analysis...")

                ra = req(f"/analyze/{uid}", method="POST",
                         t=120, no_cache=True)
                if not ra:
                    prog.empty()
                    st.error(
                        "Analysis timed out. "
                        "This is normal on first run. Please try again."
                    )
                    st.stop()

                prog.progress(90, text="Step 3 of 3 — Saving to database...")
                an = ra.json()
                st.session_state.report = an
                add_recent(uid)
                prog.progress(100, text="Complete")

                dis5 = an.get("disease_label","—")
                urg5, uc5 = URGENCY.get(dis5,("ROUTINE","#6abe6a"))
                st.markdown(f"""
                <div style="background:#162316; border:1px solid #265626;
                            border-left:4px solid #4abe4a; border-radius:6px;
                            padding:14px 16px; margin-top:12px;">
                    <div style="font-family:'Inter',sans-serif; font-size:14px;
                                font-weight:600; color:#72c472; margin-bottom:8px;">
                        Analysis complete
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;
                                font-size:12px; color:#5a8a5a; line-height:2;">
                        Finding  &nbsp; {dis5}<br>
                        Priority &nbsp; <span style="color:{uc5};">{urg5}</span><br>
                        Time     &nbsp; {an.get('total_time',0)}s<br>
                        Study ID &nbsp; {uid[:16]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.info("Switch to the Viewer tab to see the full report")

    with uc2:
        section("Preview")
        if up and not up.name.endswith(".dcm"):
            try:
                up.seek(0)
                st.image(Image.open(up),
                         use_container_width=True, caption=None)
            except:
                st.warning("Cannot preview this file")
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
with t5:
    rj3 = fetch_reports(st.session_state.api)
    rp3 = rj3.get("reports",[])

    st.markdown(f"""
    <div style="font-family:'Inter',sans-serif; font-size:13px;
                font-weight:500; color:#6a7385; margin-bottom:16px;">
        {rj3.get('total',0)} studies in archive
    </div>
    """, unsafe_allow_html=True)

    if not rp3:
        st.markdown("""
        <div style="background:#232830; border:1px solid #2e3442;
                    border-radius:8px; padding:32px; text-align:center;
                    color:#4a5368; font-size:14px;">
            No history. Upload and analyze an image to begin.
        </div>
        """, unsafe_allow_html=True)
    else:
        for r in reversed(rp3):
            dis4 = r.get("disease_label","—")
            urg4, uc4 = URGENCY.get(dis4,("ROUTINE","#6abe6a"))
            lbl = (
                f"{r.get('filename','?')}  |  "
                f"{dis4}  |  {urg4}  |  "
                f"{r.get('status','').upper()}"
            )
            with st.expander(lbl):
                hc1, hc2 = st.columns(2)
                with hc1:
                    sc2 = "#72c472" if r.get("status")=="analyzed" else "#d4aa50"
                    data_row("Study ID",   r.get("image_id","")[:14]+"...")
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
                                 key=f"ho{r.get('image_id')}",
                                 use_container_width=True):
                        st.session_state.study_id = r.get("image_id")
                        rx = req(f"/report/{r.get('image_id')}")
                        if rx:
                            st.session_state.report = rx.json()
                        add_recent(r.get("image_id"))
                        st.info("Switch to the Viewer tab")
                with hb2:
                    if st.button("Delete",
                                 key=f"hd{r.get('image_id')}",
                                 use_container_width=True):
                        req(f"/report/{r.get('image_id')}",
                            method="DELETE", no_cache=True)
                        st.rerun()


# ══════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════
with t6:
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
                    "and Colab Cell 10 is running."
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

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        section("How to update API URL")
        st.markdown("""
        <div style="font-family:'Inter',sans-serif; font-size:13px;
                    color:#6a7385; line-height:2.2;">
            1. Go to Google Colab<br>
            2. Run Cell 10<br>
            3. Copy the ngrok URL from the output<br>
            4. Paste it in the field above and click Save
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        section("AI Models")

        for lbl, val, note in [
            ("Vision Language Model",
             "PubMedCLIP",
             "flaviagiammarino/pubmed-clip-vit-base-patch32"),
            ("Report Generator",
             "Template-based",
             "Structured clinical observation format"),
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