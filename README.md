# VQA Guider

A state-of-the-art Video Question Answering (Video-LLM) system that bridges visual features from CLIP with textual context from DistilBERT, routing them through a custom Task Planner (Action/Tracking/Scene) into Microsoft's Phi-2 generative LLM.

## 1. Environment Setup

Ensure you have Python 3.10+ installed.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Training the Model (RTX 3090 / 24GB VRAM)

Before running the application, you need to train the model on your dataset. The training script will first encode all videos into a frame cache (this takes about 20-30 minutes), and then run the generative training for 20 epochs.

Make sure you are in the project root directory (`VQA-Guider-main`), then run:

```bash
python train.py \
    --csv_path "/home/nmit/Desktop/Abhijith FYP/train.csv" \
    --video_root "/home/nmit/Desktop/Abhijith FYP/NExTVideo" \
    --save_path "./models/vqa_model_generative.pt" \
    --epochs 20 \
    --batch_size 4
```

> **Note:** The best model weights will automatically be saved to `models/vqa_model_generative.pt`.

## 3. Running the Application

The application runs in two parts: a FastAPI backend for inference and a Streamlit frontend for the UI. You will need **two separate terminal windows**.

### Terminal 1: Start the Backend (FastAPI)

This will load the trained models (CLIP, DistilBERT, Phi-2, and your custom VQA weights) into GPU memory. This takes about 30-60 seconds to start.

```bash
# Ensure you are in the VQA-Guider-main directory
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
*Wait until you see `Application startup complete.` in the terminal.*

### Terminal 2: Start the Frontend (Streamlit)

In a new terminal window, run the Streamlit UI:

```bash
# Ensure you are in the VQA-Guider-main directory
streamlit run frontend/app.py
```

This will automatically open your web browser to `http://localhost:8501`.

## 4. Usage

1. Open the UI in your browser.
2. Upload an `.mp4` video.
3. Type a question (e.g., "What is the boy in the blue shirt doing?").
4. Click **Ask Question**.
5. The UI will display the generated English answer along with the Task Routing probabilities (Action, Tracking, Scene) showing how the model reasoned about the video.
