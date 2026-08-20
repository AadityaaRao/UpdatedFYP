# Edu-VQAGuider Automated Evaluation

This folder contains an isolated answer-quality evaluation pipeline. It does not modify the core backend or frontend.

## What Was Discovered

- Active Edu-VQAGuider API: `POST /api/v2/videos/{video_id}/ask`
- Ask request body: `{"question": "..."}`
- Ask response fields used by evaluation: `direct_answer`, `detailed_answer`, `route`, `evidence_chunks`, `confidence_level`
- Upload endpoint: `POST /api/v2/videos`
- Auto-transcription endpoint: `POST /api/v2/videos/{video_id}/transcribe`
- Manual transcript endpoint: `POST /api/v2/videos/{video_id}/transcript`
- Video chunks, timestamps, transcripts, frame paths, and visual summaries are persisted in SQLite through `backend.edu.db_edu`
- Uploaded videos and extracted frames are stored by the application under `uploads/<video_id>/`, with frames under `uploads/<video_id>/frames/`

## College Workflow

Run commands from the project root:

```bash
cd UpdatedFYP
```

1. Put your five videos here:

```text
evaluation/videos/
```

2. Start the backend:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

3. Generate draft questions:

```bash
python evaluation/generate_questions.py
```

This uploads and transcribes each discovered video, reads the generated chunks/evidence, and writes:

```text
evaluation/generated/<video_name>/questions.json
evaluation/generated/all_questions.json
```

Each video gets exactly 10 questions: 3 factual, 2 reasoning, 1 temporal, 2 visual/multimodal, and 2 unanswerable.

4. Review the JSON files manually.

Generated questions are drafts. They start with:

```json
"verified": false
```

Change reviewed questions to:

```json
"verified": true
```

5. Run evaluation:

```bash
python evaluation/run_evaluation.py
```

For testing before review:

```bash
python evaluation/run_evaluation.py --allow-unverified
```

6. Read the reports:

```text
evaluation/report.json
evaluation/report.md
evaluation/generated/<video_name>/results.json
```

## Dry Run Without Videos Or Backend

Use this now on your laptop:

```bash
python evaluation/run_evaluation.py --dry-run
```

Dry run checks imports, JSON structure, evaluation logic, report generation, and resumable result writing. It does not call the real application and the report is marked as dry-run.

## Resuming

The runner writes each video's `results.json` after every question. If a run stops, rerun the same command and completed questions with evaluations are reused.

## If The Backend Was Restarted

The real QA endpoint keeps processed videos in memory. If the backend is restarted after question generation, the saved `video_id` may no longer be askable even though the database still contains rows. In that case, rerun:

```bash
python evaluation/generate_questions.py --force-upload
```

Then review/run evaluation again.

## Existing Comparison Scripts

The older comparison scripts are still available:

```text
evaluation/evaluate_comparison.py
evaluation/generate_sample_report.py
evaluation/test_questions.csv
```

They were left in place for reference.
