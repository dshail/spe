# SPE.ipynb Code Analysis

## Overview
**SPE.ipynb** is a comprehensive automated pipeline for grading handwritten student exams. It leverages **computer vision (OCR)** and **Large Language Models (LLMs)** to extract data from PDF exam papers and grade them against a reference rubric.

- **Primary Goal**: Automate the evaluation of subjective and objective answers with human-like precision.
- **Key Technologies**: 
  - **Google Gemini 2.0 Flash** (for intelligent grading and reasoning)
  - **Datalab.to Marker API** (for high-fidelity PDF-to-JSON extraction)
  - **Python** (Pandas, Numpy, AsyncIO)

## Workflow Steps

The notebook is structured into 7 distinct parts:

### 1. Installation & Setup
- Installs necessary Python packages: `google-generativeai`, `requests`, `pandas`, `demjson3`.
- Configures API keys for **Gemini** and **Datalab**.

### 2. Rubric Extraction
- **Input**: A PDF file containing the exam solution/marking scheme.
- **Process**: Uses the Datalab API to extract the solution into a structured JSON format (`RUBRIC_EXTRACTION_SCHEMA`).
- **Feature**: Automatically normalizes "step-marking" (assigning marks to specific steps in a solution) to ensure numerical consistency.

### 3. Student Answer Extraction
- **Input**: Batch schema of PDF student answer sheets.
- **Process**: Extracts handwriting, math equations (LaTeX), and diagrams from student PDFs.
- **Feature**: Handles sequencing, identifying which question is being answered and separating trial work from final answers.

### 4. Constraint Validation
- (Implicit in processing) Checks for compliance with exam rules (e.g., answer length, optional questions).

### 5. AI-Based Evaluation
- **Engine**: Uses `gemini-2.0-flash` with a detailed prompt.
- **Logic**:
  - Compares student answers to the extracted rubric.
  - Awards **partial credit** for conceptual correctness, even if the method differs.
  - Provides **step-wise feedback** (e.g., "Step 1: Formula correct (1 mark)").
  - Handles geometric diagrams and labels.

### 6. Batch Processing
- Runs the evaluation loop over multiple student files.
- Includes error handling and retries (exponential backoff) for API reliability.

### 7. Reporting & Export
- **Outputs**:
  - `grading_results.csv`: Detailed row-by-row determination of every answer.
  - `summary_report.csv`: High-level grade summary per student.
  - JSON logs: Full raw data for debugging.

## Key Functions

| Function Name | Description |
| :--- | :--- |
| `call_marker_with_structured_extraction` | uploads PDF to Datalab API and polls for structured JSON result. |
| `normalize_step_marking` | Ensures the sum of step-wise marks equals the total marks for a question. |
| `process_single_student_structured` | Orchestrates the extraction of a single student's exam paper. |
| `evaluate_single_answer_robust` | Constructs the LLM prompt and calls Gemini to grade a specific answer. |
| `evaluate_all_students_enhanced` | Main loop that processes all students against the rubric. |

## Requirements
To run this notebook, you need:
1.  **Google Colab** (or a local Jupyter environment).
2.  **API Keys**:
    -   `GEMINI_API_KEY` (from Google AI Studio)
    -   `DATALAB_API_KEY` (from Datalab.to)
3.  **Input Files**:
    -   One Solution PDF (Rubric)
    -   One or more Student Exam PDFs

## Conclusion
This code represents a **Grade 3.0 / Production-Ready** system. It moves beyond simple text matching to understand mathematical concepts, geometric figures, and nuanced step-by-step reasoning.
