# Edu-VQAGuider

A state-of-the-art **Retrieval-Augmented Multimodal Video Question Answering** system specifically designed for long, real-world educational videos and lectures (30-60 minutes).

Edu-VQAGuider intelligently routes questions based on their intent (e.g., conceptual, procedural, visual, temporal) to extract relevant transcript chunks and visual frames, passing them as grounded evidence to a local LLM to generate highly accurate, tutor-like answers.

## Architecture Highlights
- **Audio Extraction:** Uses `ffmpeg` to rip audio directly from video uploads.
- **Transcription:** Uses `faster-whisper` for offline, high-speed chunked transcription.
- **Planner (Routing):** Custom DistilBERT classification head trained on 200 educational query patterns to route questions into 5 categories (`concept`, `procedure`, `temporal`, `visual`, `summary`).
- **Retrieval Engine:** Combines FAISS cosine-similarity search (for text transcripts) with CLIP ViT-B/32 (for visual frame grounding).
- **Generation:** Utilizes `Qwen2.5-3B-Instruct` in 4-bit quantization (via `bitsandbytes`) to generate highly detailed, educational responses.
- **VRAM Lifecycle Management:** Dynamically loads and unloads Whisper and Qwen models on the fly to fit the entire pipeline inside a single **24GB RTX 3090 GPU**.

*(Note: The legacy v1 NExT-QA / Phi-2 short-video baseline is still retained in the codebase for evaluation comparison).*

## 1. System Requirements

- **OS:** Linux (Recommended for `bitsandbytes` 4-bit support)
- **GPU:** NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **System Dependencies:** `ffmpeg` (Required for Whisper transcription)

## 2. Environment Setup

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install ffmpeg (Ubuntu/Debian)
sudo apt update
sudo apt install ffmpeg -y

# 3. Install Python dependencies
pip install -r requirements.txt
```

## 3. Running the Application

The system uses a FastAPI backend and a Streamlit frontend. You will need **two separate terminal windows**.

### Terminal 1: Start the Backend (FastAPI)
The backend dynamically manages model VRAM loading. DistilBERT and CLIP are loaded at startup. Whisper and Qwen are loaded on demand.

```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Wait until you see `Server ready. v1 + v2 endpoints available.`*

### Terminal 2: Start the Frontend (Streamlit)
Start the frontend with an increased upload size limit to support 1GB+ educational videos.

```bash
source venv/bin/activate
streamlit run frontend/edu_app.py --server.maxUploadSize 1000
```
*This will start the UI at `http://localhost:8501` or your server's local IP.*

## 4. Usage Flow

1. Open the UI in your web browser.
2. **Upload** an `.mp4` educational lecture (e.g., from MIT OCW or Khan Academy).
3. Click **Start Auto-Transcription**. The system will use Whisper to transcribe and chunk the lecture, and extract visual frames.
4. Once processed, type a question (e.g., *"What is the intuition behind the row picture?"* or *"What are the steps to solve this system?"*).
5. The system will intelligently route your question, retrieve the exact transcript text and visual frames, and generate a tutor-like response with timestamp citations.
