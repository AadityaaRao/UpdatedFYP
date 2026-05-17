"""
frontend/edu_app.py
────────────────────────────────────────────────────────────
Edu-VQAGuider Streamlit Frontend.

Run with:
    streamlit run frontend/edu_app.py

UI Flow:
    1. Upload educational video
    2. Choose: auto-transcribe (Whisper) or paste manual transcript
    3. Wait for processing (status polling)
    4. Ask questions
    5. View grounded answers with evidence

Session state keys:
    video_id        str | None
    video_name      str | None
    video_status    str | None
    last_result     dict | None
    history         list[dict]
"""
from __future__ import annotations

import os
import sys
import time

import streamlit as st

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(__file__))
from edu_api_client import (
    APIError,
    ask_edu_question,
    auto_transcribe,
    get_video_history,
    get_video_status,
    health_check,
    upload_edu_video,
    upload_transcript,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# ══════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Edu-VQAGuider",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }

    /* Title */
    .edu-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .edu-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* Answer cards */
    .direct-answer-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .detailed-answer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        line-height: 1.7;
    }
    .answer-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
        color: #64748b;
    }

    /* Route badge */
    .route-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.8rem;
        border-radius: 999px;
        margin-right: 0.5rem;
    }
    .route-concept   { background: #dbeafe; color: #1d4ed8; }
    .route-procedure { background: #dcfce7; color: #16a34a; }
    .route-temporal  { background: #fef3c7; color: #d97706; }
    .route-visual    { background: #ede9fe; color: #7c3aed; }
    .route-summary   { background: #fce7f3; color: #db2777; }

    .source-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: #f1f5f9;
        color: #64748b;
    }

    /* Evidence */
    .evidence-card {
        background: #fefce8;
        border-left: 3px solid #eab308;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .evidence-time {
        font-weight: 600;
        color: #92400e;
        font-size: 0.8rem;
    }
    .evidence-score {
        font-size: 0.75rem;
        color: #a16207;
    }

    /* Status indicators */
    .status-ready  { color: #16a34a; font-weight: 600; }
    .status-pending { color: #d97706; font-weight: 600; }
    .status-error  { color: #dc2626; font-weight: 600; }

    /* Processing status */
    .processing-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }

    hr { border-top: 1px solid #e5e7eb; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════
for key, default in {
    "video_id": None,
    "video_name": None,
    "video_status": None,
    "last_result": None,
    "history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Settings")
    backend_url = st.text_input(
        "Backend URL",
        value=os.getenv("BACKEND_URL", "http://localhost:8000"),
    )

    st.markdown("---")
    st.markdown("### Server Status")
    if st.button("Check Status", use_container_width=True):
        with st.spinner("Checking..."):
            try:
                info = health_check(backend_url)
                edu_ready = info.get("v2_edu_ready", False)
                has_planner = info.get("v2_has_planner", False)
                has_qwen = info.get("v2_has_qwen", False)
                has_clip = info.get("v2_has_clip", False)

                if edu_ready:
                    st.success("Edu-VQAGuider Online")
                    cols = st.columns(3)
                    cols[0].metric("Planner", "Yes" if has_planner else "No")
                    cols[1].metric("Qwen", "Yes" if has_qwen else "No")
                    cols[2].metric("CLIP", "Yes" if has_clip else "No")
                else:
                    st.warning("Server online, models loading...")

                st.caption(
                    f"Device: {info.get('device', '?')} | "
                    f"CUDA: {info.get('cuda_available', '?')}"
                )
            except APIError as e:
                st.error(f"Offline: {e.message}")

    st.markdown("---")
    st.markdown("### Session")
    if st.session_state.video_id:
        st.info(f"**{st.session_state.video_name}**\n\n"
                f"`{st.session_state.video_id[:8]}...`\n\n"
                f"Status: {st.session_state.video_status or 'unknown'}")
        if st.button("Clear Video", use_container_width=True):
            for key in ["video_id", "video_name", "video_status", "last_result", "history"]:
                st.session_state[key] = None if key != "history" else []
            st.rerun()
    else:
        st.caption("No video loaded.")

    st.markdown("---")
    st.caption("Edu-VQAGuider v2.0")

# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="edu-title">Edu-VQAGuider</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="edu-subtitle">Grounded question answering for long educational videos</p>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
# Step 1: Upload
# ══════════════════════════════════════════════════════════════
st.markdown("#### 1. Upload Educational Video")

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "webm", "mov", "mkv"],
    help="Supported: MP4, AVI, WebM, MOV, MKV | Max 500 MB",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    is_new = st.session_state.video_name != uploaded_file.name
    if is_new:
        with st.spinner(f"Uploading **{uploaded_file.name}**..."):
            try:
                result = upload_edu_video(
                    backend_url, uploaded_file.getvalue(), uploaded_file.name,
                )
                st.session_state.video_id = result["video_id"]
                st.session_state.video_name = uploaded_file.name
                st.session_state.video_status = result.get("status", "pending")
                st.session_state.last_result = None
                st.session_state.history = []

                duration = result.get("duration_sec", 0)
                chunks = result.get("num_chunks", 0)
                st.success(
                    f"Uploaded! Duration: {duration:.0f}s | "
                    f"Chunks: {chunks} | ID: `{result['video_id'][:8]}...`"
                )
            except APIError as e:
                st.error(f"Upload failed: {e.message}")
    else:
        if st.session_state.video_id:
            st.success(f"Video ready: `{st.session_state.video_id[:8]}...`")

# ══════════════════════════════════════════════════════════════
# Step 2: Transcript
# ══════════════════════════════════════════════════════════════
if st.session_state.video_id and st.session_state.video_status in ("pending", None):
    st.markdown("---")
    st.markdown("#### 2. Provide Transcript")
    st.info(
        "The video needs a transcript before questions can be asked. "
        "Choose auto-transcription (Whisper) or paste your own."
    )

    tab_auto, tab_manual = st.tabs(["Auto-Transcribe (Whisper)", "Paste Transcript"])

    with tab_auto:
        st.markdown("Automatically transcribe using **faster-whisper**. "
                     "Requires ffmpeg and GPU for best speed.")
        if st.button("Start Auto-Transcription", type="primary", use_container_width=True):
            with st.spinner("Transcribing with Whisper... This may take a few minutes."):
                try:
                    result = auto_transcribe(backend_url, st.session_state.video_id)
                    st.session_state.video_status = result.get("status", "ready")
                    st.success(
                        f"Transcription complete! "
                        f"{result.get('num_chunks_with_text', 0)} chunks with text."
                    )
                    st.rerun()
                except APIError as e:
                    st.error(f"Transcription failed: {e.message}")

    with tab_manual:
        transcript_text = st.text_area(
            "Paste transcript here",
            height=200,
            placeholder="Paste the full transcript of the video here...",
        )
        if st.button("Submit Transcript", use_container_width=True):
            if not transcript_text or not transcript_text.strip():
                st.warning("Transcript cannot be empty.")
            else:
                with st.spinner("Processing transcript..."):
                    try:
                        result = upload_transcript(
                            backend_url, st.session_state.video_id, transcript_text,
                        )
                        st.session_state.video_status = result.get("status", "ready")
                        st.success(
                            f"Transcript processed! "
                            f"{result.get('num_chunks_with_text', 0)} chunks with text."
                        )
                        st.rerun()
                    except APIError as e:
                        st.error(f"Failed: {e.message}")

# ══════════════════════════════════════════════════════════════
# Step 3: Ask Question
# ══════════════════════════════════════════════════════════════
video_ready = (
    st.session_state.video_id is not None
    and st.session_state.video_status == "ready"
)

if video_ready:
    st.markdown("---")
    st.markdown("#### 3. Ask a Question")

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is the main concept explained in this lecture?",
        height=80,
        label_visibility="collapsed",
    )

    submit = st.button(
        "Get Answer",
        type="primary",
        use_container_width=True,
        disabled=not (question and question.strip()),
    )

    if submit and question.strip():
        with st.spinner("Analyzing video and generating answer..."):
            try:
                result = ask_edu_question(
                    backend_url, st.session_state.video_id, question.strip(),
                )
                st.session_state.last_result = result
            except APIError as e:
                st.error(f"Error: {e.message}")
                st.session_state.last_result = None

# ══════════════════════════════════════════════════════════════
# Results Display
# ══════════════════════════════════════════════════════════════
result = st.session_state.last_result

if result is not None:
    st.markdown("---")
    st.markdown("#### Results")

    # ── Route badge ───────────────────────────────────────────
    route_data = result.get("route", {})
    route_name = route_data.get("route", "concept")
    confidence = route_data.get("confidence", 0)
    planner_source = route_data.get("planner_source", "fallback")

    route_emoji = {
        "concept": "📚", "procedure": "📋", "temporal": "⏱️",
        "visual": "👁️", "summary": "📝",
    }

    st.markdown(
        f'<span class="route-badge route-{route_name}">'
        f'{route_emoji.get(route_name, "🏷️")} {route_name.upper()}</span>'
        f'<span class="source-badge">{planner_source} | {confidence:.0%}</span>',
        unsafe_allow_html=True,
    )

    # ── Direct answer ─────────────────────────────────────────
    st.markdown(
        f'<div class="direct-answer-card">'
        f'<div class="answer-label">Direct Answer</div>'
        f'{result.get("direct_answer", "---")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Detailed answer ───────────────────────────────────────
    with st.expander("Detailed Answer", expanded=True):
        st.markdown(result.get("detailed_answer", "---"))

    # ── Evidence chunks ───────────────────────────────────────
    evidence = result.get("evidence_chunks", [])
    if evidence:
        with st.expander(f"Evidence ({len(evidence)} chunks)", expanded=False):
            for i, chunk in enumerate(evidence):
                start = chunk.get("start_time", 0)
                end = chunk.get("end_time", 0)
                score = chunk.get("relevance_score", 0)
                text = chunk.get("transcript_text", "")
                frame = chunk.get("selected_frame_path", None)

                st.markdown(
                    f'<div class="evidence-card">'
                    f'<span class="evidence-time">'
                    f'{_fmt_time(start)} - {_fmt_time(end)}'
                    f'</span> '
                    f'<span class="evidence-score">score: {score:.3f}</span>'
                    f'<br/>{text}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Show frame if available
                if frame and os.path.exists(frame):
                    st.image(frame, caption=f"Frame from {_fmt_time(start)}", width=300)

    # ── Route scores ──────────────────────────────────────────
    all_scores = route_data.get("all_scores", {})
    if all_scores:
        with st.expander("Route Scores"):
            for route_label, score in sorted(
                all_scores.items(), key=lambda x: x[1], reverse=True
            ):
                st.progress(min(score, 1.0), text=f"{route_label}: {score:.3f}")

    # ── Metadata ──────────────────────────────────────────────
    with st.expander("Response Metadata"):
        st.json({
            "result_id": result.get("result_id"),
            "video_id": result.get("video_id"),
            "question": result.get("question"),
            "route": route_data,
        })

# ══════════════════════════════════════════════════════════════
# Question History
# ══════════════════════════════════════════════════════════════
if video_ready:
    st.markdown("---")
    with st.expander("Question History"):
        if st.button("Refresh History"):
            try:
                hist = get_video_history(backend_url, st.session_state.video_id)
                st.session_state.history = hist.get("items", [])
            except APIError:
                pass

        for item in st.session_state.history:
            route_name = item.get("route", "concept")
            st.markdown(
                f'<span class="route-badge route-{route_name}">{route_name}</span> '
                f'**{item.get("question", "")}**\n\n'
                f'{item.get("direct_answer", "")}\n\n'
                f'<small>{item.get("created_at", "")}</small>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

# ══════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Edu-VQAGuider | Powered by DistilBERT Planner "
    "| CLIP Visual Evidence | Qwen Answer Generation"
)
