"""
evaluation/generate_sample_report.py
────────────────────────────────────────────────────────────
A utility script to generate a beautiful pre-filled sample comparison report.
This allows you to see the exact structure, styling, and color coding of the
Excel comparison report immediately without running the backend model or local videos.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluate_comparison import export_excel, export_csv

def main():
    print("🎨 Generating beautiful pre-filled sample comparison report...")

    sample_rows = [
        {
            "no": 1,
            "video_url": "https://www.youtube.com/watch?v=ZK3O402wf1c",
            "category": "concept",
            "question": "What is a matrix and how is it represented?",
            "reference_answer": "A matrix is a rectangular array of numbers arranged in rows and columns. It is represented by capital letters and enclosed in square brackets.",
            "baseline_answer": "A matrix is a grid-like structure used in mathematics and computer science. It consists of numbers and is represented using brackets, typically square or round brackets, containing rows and columns of numeric values.",
            "baseline_rouge_l": 0.4552,
            "baseline_bleu4": 0.2845,
            "baseline_words": 28,
            "baseline_time": 2.45,
            "edu_direct_answer": "A matrix is a rectangular array of numbers in rows and columns, denoted by a capital letter (e.g., A) and enclosed in brackets.",
            "edu_detailed_answer": "According to the lecture video at [00:45], a matrix is formally defined as a rectangular array of numbers or symbols arranged in horizontal rows and vertical columns. The instructor notes that matrices are denoted using capital letters (like A, B, or C) and their dimensions are given as row-by-column (m x n). The array elements are enclosed in square brackets [ ] or parentheses.",
            "edu_rouge_l": 0.8125,
            "edu_bleu4": 0.6582,
            "edu_words": 61,
            "edu_time": 6.82,
            "edu_route": "concept",
            "edu_confidence": 0.9850,
            "edu_planner_source": "learned",
            "winner": "Edu-VQAGuider ✓",
            "rouge_improvement": 0.3573,
        },
        {
            "no": 2,
            "video_url": "https://www.youtube.com/watch?v=ZK3O402wf1c",
            "category": "procedure",
            "question": "How do you calculate the determinant of a 2x2 matrix?",
            "reference_answer": "For a 2x2 matrix with elements [[a, b], [c, d]], the determinant is calculated by subtracting the product of the secondary diagonal from the main diagonal: ad - bc.",
            "baseline_answer": "To find the determinant of a 2x2 matrix, you take the top-left number multiplied by the bottom-right number, and subtract the product of the top-right and bottom-left numbers. It's written as ad minus bc.",
            "baseline_rouge_l": 0.6842,
            "baseline_bleu4": 0.5214,
            "baseline_words": 31,
            "baseline_time": 3.12,
            "edu_direct_answer": "Multiply the main diagonal elements and subtract the product of the secondary diagonal elements: det = ad - bc.",
            "edu_detailed_answer": "At [03:15], the instructor outlines the step-by-step procedure for finding the determinant of a 2x2 matrix A = [[a, b], [c, d]]: \n1. Identify the main diagonal elements (a and d) and multiply them to get 'ad'.\n2. Identify the secondary diagonal elements (b and c) and multiply them to get 'bc'.\n3. Subtract the product of the secondary diagonal from the main diagonal: det(A) = ad - bc.\nThe instructor provides an example with [[3, 5], [1, 2]], showing det = (3*2) - (5*1) = 6 - 5 = 1.",
            "edu_rouge_l": 0.8947,
            "edu_bleu4": 0.7812,
            "edu_words": 87,
            "edu_time": 7.45,
            "edu_route": "procedure",
            "edu_confidence": 0.9620,
            "edu_planner_source": "learned",
            "winner": "Edu-VQAGuider ✓",
            "rouge_improvement": 0.2105,
        },
        {
            "no": 3,
            "video_url": "https://www.youtube.com/watch?v=ZK3O402wf1c",
            "category": "temporal",
            "question": "What topic does the instructor cover immediately after explaining scalar multiplication?",
            "reference_answer": "Immediately after explaining scalar multiplication, the instructor transitions to matrix addition, demonstrating that matrices must be of the same dimensions.",
            "baseline_answer": "Typically in linear algebra courses, after scalar multiplication, instructors will teach matrix multiplication or sometimes dot products. However, without watching the specific video, I cannot confirm what this instructor did next.",
            "baseline_rouge_l": 0.1250,
            "baseline_bleu4": 0.0000,
            "baseline_words": 33,
            "baseline_time": 1.98,
            "edu_direct_answer": "The instructor covers matrix addition, explaining that matrices must have the same dimensions to be added.",
            "edu_detailed_answer": "Right after scalar multiplication is explained at [05:40], the instructor immediately transitions to demonstrating matrix addition at [06:12]. The video highlights that two matrices can only be added if they share the exact same dimensions (same number of rows and columns), adding corresponding elements one by one.",
            "edu_rouge_l": 0.8571,
            "edu_bleu4": 0.7125,
            "edu_words": 49,
            "edu_time": 5.92,
            "edu_route": "temporal",
            "edu_confidence": 0.9410,
            "edu_planner_source": "learned",
            "winner": "Edu-VQAGuider ✓",
            "rouge_improvement": 0.7321,
        },
        {
            "no": 4,
            "video_url": "https://www.youtube.com/watch?v=ZK3O402wf1c",
            "category": "visual",
            "question": "What visual aid was shown on the blackboard when explaining matrix multiplication?",
            "reference_answer": "The blackboard shows a visual color-coded diagram with blue highlighting for the rows of the first matrix and orange highlighting for the columns of the second matrix, illustrating how they combine.",
            "baseline_answer": "When explaining matrix multiplication, teachers often write matrices on a blackboard and draw arrows connecting rows and columns. I cannot see what visual aids or colors were used on this particular blackboard.",
            "baseline_rouge_l": 0.1818,
            "baseline_bleu4": 0.0520,
            "baseline_words": 32,
            "baseline_time": 2.15,
            "edu_direct_answer": "The blackboard shows a color-coded diagram using blue for rows of the first matrix and orange for columns of the second matrix.",
            "edu_detailed_answer": "At [08:25], the video uses CLIP visual indexing to match the whiteboard frame. The blackboard displays a detailed, color-coded diagram showing how rows are multiplied by columns. Specifically, the rows of the first matrix are highlighted in blue, and the columns of the second matrix are highlighted in orange, showing the step-by-step dot product mapping.",
            "edu_rouge_l": 0.8000,
            "edu_bleu4": 0.6845,
            "edu_words": 59,
            "edu_time": 6.10,
            "edu_route": "visual",
            "edu_confidence": 0.9150,
            "edu_planner_source": "learned",
            "winner": "Edu-VQAGuider ✓",
            "rouge_improvement": 0.6182,
        },
        {
            "no": 5,
            "video_url": "https://www.youtube.com/watch?v=ZK3O402wf1c",
            "category": "summary",
            "question": "Summarize the key topics covered in the lecture video.",
            "reference_answer": "The video covers three key topics: 1. The definition and representation of a matrix, 2. Determinant calculations for 2x2 matrices, and 3. Basic matrix operations including scalar multiplication, matrix addition, and multiplication.",
            "baseline_answer": "This general lecture on linear algebra likely covers basic topics such as what matrices are, dimensions, rows and columns, addition, subtraction, scalar multiplication, and possibly determinants.",
            "baseline_rouge_l": 0.5128,
            "baseline_bleu4": 0.3541,
            "baseline_words": 26,
            "baseline_time": 2.85,
            "edu_direct_answer": "The lecture covers the definition of a matrix, 2x2 determinant calculation, scalar multiplication, matrix addition, and matrix multiplication.",
            "edu_detailed_answer": "The comprehensive lecture spans several key segments: \n- [00:00 - 02:00] Introduction, formal definition, and notation/enclosures of matrices.\n- [02:00 - 05:00] Determinant of a 2x2 matrix with clear manual arithmetic examples.\n- [05:00 - 07:30] Basic operations, focusing on scalar multiplication and dimensional rules for matrix addition.\n- [07:30 - 10:00] In-depth procedural guide and visual breakdown of matrix multiplication.",
            "edu_rouge_l": 0.8387,
            "edu_bleu4": 0.7250,
            "edu_words": 65,
            "edu_time": 6.95,
            "edu_route": "summary",
            "edu_confidence": 0.9780,
            "edu_planner_source": "learned",
            "winner": "Edu-VQAGuider ✓",
            "rouge_improvement": 0.3259,
        }
    ]

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = str(output_dir / "sample_comparison.xlsx")
    csv_path = str(output_dir / "sample_comparison.csv")

    try:
        export_excel(sample_rows, excel_path)
        export_csv(sample_rows, csv_path)
        print("\n🎉 Success! Beautiful pre-filled reports generated successfully:")
        print(f"   🟢 Excel: {excel_path}")
        print(f"   🟢 CSV:   {csv_path}")
        print("\nYou can open 'evaluation/results/sample_comparison.xlsx' directly in Excel to view your styled formatting!")
    except Exception as e:
        print(f"❌ Error generating report: {e}")

if __name__ == "__main__":
    main()
