"""
frontend/app.py
────────────────────────────────────────────────────────────
VQA Guider — Streamlit frontend.
Run with:
    streamlit run frontend/app.py
Session state keys:
    video_id      str | None   — UUID returned after upload
    video_name    str | None   — Original filename (display only)
    last_result   dict | None  — Last successful /ask_question response
"""
from __future__ import annotations
import os
import sys
import streamlit as st
# Allow importing api_client from the same frontend/ directory
sys.path.insert(0, os.path.dirname(__file__))
from api_client import APIError, ask_question, health_check, upload_video
# ══════════════════════════════════════════════════════════════
# Page config — must be the first Streamlit call
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VQA Guider",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)
# ══════════════════════════════════════════════════════════════
# Custom CSS — clean, minimal styling
# ══════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
        /* Main title */
        .vqa-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 0.2rem;
        }
        .vqa-subtitle {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        /* Answer card */
        .answer-card {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 4px solid #0284c7;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
        }
        .answer-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #0369a1;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }
        .answer-text {
            font-size: 1.15rem;
            color: #1e293b;
            font-weight: 500;
            line-height: 1.6;
        }
        /* Routing section */
        .routing-header {
            font-size: 0.85rem;
            font-weight: 600;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 1.5rem 0 0.8rem 0;
        }
        .routing-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: #374151;
            margin-bottom: 0.2rem;
        }
        .cache-badge {
            background: #d1fae5;
            color: #065f46;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            display: inline-block;
        }
        .fresh-badge {
            background: #ede9fe;
            color: #4c1d95;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            display: inline-block;
        }
        /* Divider */
        hr { border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
        /* Status dot */
        .dot-green { color: #22c55e; }
        .dot-red   { color: #ef4444; }
        .dot-amber { color: #f59e0b; }
    </style>
    """,
    unsafe_allow_html=True,
)
# ══════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
# ══════════════════════════════════════════════════════════════
# Sidebar — configuration + server status
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    backend_url = st.text_input(
        "Backend URL",
        value=os.getenv("BACKEND_URL", "http://localhost:8000"),
        help="Base URL of the FastAPI backend server.",
    )
    st.markdown("---")
    st.markdown("### 🖥️ Server Status")
    if st.button("Check Status", use_container_width=True):
        with st.spinner("Pinging server…"):
            try:
                info = health_check(backend_url)
                model_ok = info.get("model_ready", False)
                device = info.get("device", "unknown")
                if model_ok:
                    st.success(f"✅ Online · Model ready · {device}")
                else:
                    st.warning("⚠️ Online · Model still loading…")
                st.caption(f"PyTorch {info.get('torch_version','?')} · CUDA: {info.get('cuda_available','?')}")
            except APIError as e:
                st.error(f"❌ {e.message}")
    st.markdown("---")
    st.markdown("### 📋 Session")
    if st.session_state.video_id:
        st.info(f"**Video loaded**\n\n`{st.session_state.video_name}`")
        if st.button("🗑️ Clear video", use_container_width=True):
            st.session_state.video_id = None
            st.session_state.video_name = None
            st.session_state.last_result = None
            st.rerun()
    else:
        st.caption("No video loaded yet.")
    st.markdown("---")
    st.caption("VQA Guider v1.0 · Inference only")
# ══════════════════════════════════════════════════════════════
# Main header
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="vqa-title">🎬 VQA Guider</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="vqa-subtitle">Upload a video · Ask a question · Get an AI-powered answer</p>',
    unsafe_allow_html=True,
)
# ══════════════════════════════════════════════════════════════
# Section 1 — Video upload
# ══════════════════════════════════════════════════════════════
st.markdown("#### 📤 Step 1 — Upload Video")
uploaded_file = st.file_uploader(
    label="Choose a video file",
    type=["mp4", "avi", "webm", "mov", "mkv"],
    help="Supported formats: MP4, AVI, WebM, MOV, MKV · Max 500 MB",
    label_visibility="collapsed",
)
# Upload to backend when a new file is selected
if uploaded_file is not None:
    is_new_file = (st.session_state.video_name != uploaded_file.name)
    if is_new_file:
        with st.spinner(f"Uploading **{uploaded_file.name}**…"):
            try:
                result = upload_video(
                    base_url=backend_url,
                    video_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                )
                st.session_state.video_id = result["video_id"]
                st.session_state.video_name = uploaded_file.name
                st.session_state.last_result = None   # clear previous answer
                st.success(f"✅ Uploaded successfully · `{result['video_id'][:8]}…`")
            except APIError as e:
                st.error(f"❌ Upload failed: {e.message}")
                st.session_state.video_id = None
                st.session_state.video_name = None
    else:
        # Same file re-selected — show current status without re-uploading
        if st.session_state.video_id:
            st.success(f"✅ Video ready · `{st.session_state.video_id[:8]}…`")
# ══════════════════════════════════════════════════════════════
# Section 2 — Question input + submit
# ══════════════════════════════════════════════════════════════
st.markdown("#### 💬 Step 2 — Ask a Question")
video_ready = st.session_state.video_id is not None
question = st.text_area(
    label="Your question",
    placeholder="e.g. What are the people in the video doing?",
    height=90,
    disabled=not video_ready,
    label_visibility="collapsed",
    help="Upload a video first, then type your question here.",
)
if not video_ready:
    st.caption("⬆️ Upload a video above to enable this field.")
submit_disabled = not video_ready or not (question and question.strip())
submitted = st.button(
    "🔍 Get Answer",
    use_container_width=True,
    disabled=submit_disabled,
    type="primary",
)
# ══════════════════════════════════════════════════════════════
# Inference — called when submit is pressed
# ══════════════════════════════════════════════════════════════
if submitted and not submit_disabled:
    with st.spinner("🧠 Analysing video and generating answer…"):
        try:
            result = ask_question(
                base_url=backend_url,
                video_id=st.session_state.video_id,
                question=question.strip(),
            )
            st.session_state.last_result = result
        except APIError as e:
            st.error(f"❌ {e.message}")
            st.session_state.last_result = None
# ══════════════════════════════════════════════════════════════
# Section 3 — Results display
# ══════════════════════════════════════════════════════════════
result = st.session_state.last_result
if result is not None:
    st.markdown("---")
    st.markdown("#### 📊 Results")
    # ── Answer card ───────────────────────────────────────────
    from_cache = result.get("from_cache", False)
    badge_html = (
        '<span class="cache-badge">⚡ From cache</span>'
        if from_cache else
        '<span class="fresh-badge">✨ Fresh inference</span>'
    )
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-label">Answer &nbsp; {badge_html}</div>
            <div class="answer-text">{result.get("answer", "—")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ── Task routing bars ─────────────────────────────────────
    routing = result.get("task_routing", {})
    action   = float(routing.get("action",   0.0))
    tracking = float(routing.get("tracking", 0.0))
    scene    = float(routing.get("scene",    0.0))
    st.markdown('<p class="routing-header">🧭 Task Routing</p>', unsafe_allow_html=True)
    # Three labelled progress bars with percentage
    _bars = [
        ("🏃 Action",   action,   "#0284c7"),
        ("🎯 Tracking", tracking, "#7c3aed"),
        ("🌄 Scene",    scene,    "#059669"),
    ]
    for label, value, _ in _bars:
        col_label, col_pct = st.columns([5, 1])
        with col_label:
            st.markdown(f"**{label}**")
        with col_pct:
            st.markdown(
                f"<div style='text-align:right; font-weight:600; color:#374151;'>"
                f"{value * 100:.1f}%</div>",
                unsafe_allow_html=True,
            )
        st.progress(value)
    # ── Dominant task highlight ───────────────────────────────
    task_names = {"action": "🏃 Action", "tracking": "🎯 Tracking", "scene": "🌄 Scene"}
    dominant_key = max(routing, key=routing.get) if routing else "—"
    dominant_name = task_names.get(dominant_key, dominant_key)
    st.caption(f"Dominant task: **{dominant_name}** ({routing.get(dominant_key, 0) * 100:.1f}%)")
    # ── Metadata ──────────────────────────────────────────────
    with st.expander("🔎 Response metadata"):
        st.json({
            "result_id": result.get("result_id"),
            "video_id":  result.get("video_id"),
            "question":  result.get("question"),
            "from_cache": from_cache,
            "task_routing": routing,
        })
# ══════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "VQA Guider · Powered by CLIP · DistilBERT · VQAGuiderCore · Phi-2"
)