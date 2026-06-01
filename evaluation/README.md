# 📊 Model Comparison & Evaluation Framework

This directory contains the **Evaluation and Comparison Framework** for comparing your **Edu-VQAGuider** project against a standard **Baseline model** (Qwen2.5-3B answering directly without any video context/RAG).

The evaluation automatically calculates lexical metrics and performance indicators, exporting the results to a beautifully formatted, color-coded **Excel spreadsheet** with three analysis sheets.

---

## 🚀 How to Run the Evaluation

### Step 1: Install Dependencies
Make sure you install the required evaluation libraries:
```bash
pip install -r requirements.txt
```
*(This installs `rouge-score`, `nltk`, and `openpyxl` for metrics computation and Excel generation)*

### Step 2: Start the FastAPI Backend
Before running the evaluator, your backend server must be running and ready:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Ensure that the models load successfully.

### Step 3: Run the Comparison Evaluator
You have two ways to run the evaluation:

#### Option A: With a Local Video File (Automatic Ingestion)
This will automatically upload the video, run Whisper transcription, build the chunk indices, and then run all questions against both models:
```bash
python evaluation/evaluate_comparison.py \
    --csv evaluation/test_questions.csv \
    --video_path evaluation/sample_lecture.mp4 \
    --backend http://localhost:8000
```

#### Option B: With an Existing `video_id` (Already Processed)
If you have already uploaded and transcribed a video, you can bypass the ingestion stage by providing its `video_id`:
```bash
python evaluation/evaluate_comparison.py \
    --csv evaluation/test_questions.csv \
    --video_id "your-video-uuid-here" \
    --backend http://localhost:8000
```

---

## 📈 What Metrics Are Computed?

For every test question, the framework runs:
1. **BASELINE**: Queries the local Qwen2.5-3B model with **NO** video context (simulating asking the LLM without watching the video).
2. **EDU-VQAGUIDER**: Runs the complete RAG pipeline (transcribes audio, creates chunk embeddings, routes intent, retrieves top-3 chunks, selects CLIP keyframes, and generates a grounded response).

Then, it computes:
* **ROUGE-L F1 Score**: Measures phrase and word-ordering overlap against the reference answer.
* **BLEU-4 Score**: Measures precision of 4-gram overlaps (strict accuracy check).
* **Word Count**: Measures the length and richness of the detailed answer.
* **Response Time**: Measures latency in seconds.
* **Winner Verdict**: Declares whether Edu-VQAGuider, the Baseline, or a Tie won the question (based on a margin of $0.05$ in ROUGE-L).

---

## 📊 Styled Excel Output Sheets

The results are saved in `evaluation/results/comparison_YYYYMMDD_HHMMSS.xlsx`. The workbook is styled with professional dark-blue headers, color-coded cells, and contains **three interactive sheets**:

### 1. `Comparison` (Main Sheet)
* Shows question-by-question side-by-side answers.
* Color-coded **Winner** verdict column:
  * 🟢 **Green**: Edu-VQAGuider won (RAG pipeline significantly outperformed the baseline).
  * 🔴 **Red**: Baseline won.
  * 🟡 **Yellow**: Tie.
* Color-coded **EDU Route** column showing how the *EduPlanner* classified the question (Concept, Procedure, Temporal, Visual, Summary).
* Color-coded **Improvement** column showing the exact margin of improvement.

### 2. `Summary` (Statistics)
* Aggregated metrics (average ROUGE-L, average BLEU-4, average response time, word counts) for both models.
* Total win count and win-percentage breakdown.
* **Route distribution** representing which kinds of questions were asked most.

### 3. `By Category` (Deep Dive)
* Performance breakdown grouped by category.
* Helps you immediately see which question types (e.g., temporal vs. conceptual) benefit the most from the RAG pipeline.

---

## 📝 Customizing Test Questions

You can easily add your own questions to the test suite by editing [test_questions.csv](file:///c:/Users/Darshini/OneDrive/Desktop/final%20year%20project/UpdatedFYP/evaluation/test_questions.csv). Keep the following columns intact:
* `video_url`: The URL/source of the educational video.
* `video_local_path`: Local path to save/cache the video.
* `question`: The specific question to evaluate.
* `reference_answer`: The gold-standard ground-truth answer.
* `category`: The category (choose from: `concept`, `procedure`, `temporal`, `visual`, `summary`).
