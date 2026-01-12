import os
import re
import json
import time
import requests
import pandas as pd

import google.generativeai as genai
import asyncio
import datetime
import glob

# --- HISTORY MANAGEMENT ---

HISTORY_DIR = "history"

def save_to_history(data, category, filename_prefix):
    """
    Save data to history/{category}/timestamp_{filename_prefix}.json
    Returns the absolute path of the saved file.
    """
    try:
        # Create category directory
        cat_dir = os.path.join(HISTORY_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prefix = "".join(x for x in filename_prefix if x.isalnum() or x in ('-', '_')).strip()
        filename = f"{timestamp}_{safe_prefix}.json"
        filepath = os.path.join(cat_dir, filename)
        
        # Save JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        return filepath
    except Exception as e:
        print(f"Error saving to history: {e}")
        return None

def list_history_files(category):
    """
    List all JSON files in history/{category} sorted by newest first.
    Returns list of dicts: {'filename': str, 'path': str, 'timestamp': str}
    """
    cat_dir = os.path.join(HISTORY_DIR, category)
    if not os.path.exists(cat_dir):
        return []
        
    files = []
    # Use glob to find json files
    for filepath in glob.glob(os.path.join(cat_dir, "*.json")):
        filename = os.path.basename(filepath)
        # Try to extract timestamp
        parts = filename.split('_')
        timestamp_str = "Unknown"
        if len(parts) >= 2:
            try:
                # format YYYYMMDD_HHMMSS
                ts_part = parts[0] + "_" + parts[1]
                dt = datetime.datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_str = "Unknown"
                
        files.append({
            'filename': filename,
            'path': filepath,
            'timestamp': timestamp_str,
            'mod_time': os.path.getmtime(filepath)
        })
    
    # Sort by modification time, newest first
    files.sort(key=lambda x: x['mod_time'], reverse=True)
    return files

def load_history_file(category, filename):
    """Load content of a specific history file"""
    filepath = os.path.join(HISTORY_DIR, category, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None

def delete_history_file(category, filename):
    """Delete a specific history file"""
    filepath = os.path.join(HISTORY_DIR, category, filename)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False

def clear_history_category(category):
    """Delete all files in a category"""
    cat_dir = os.path.join(HISTORY_DIR, category)
    if not os.path.exists(cat_dir):
        return True
        
    try:
        for filepath in glob.glob(os.path.join(cat_dir, "*.json")):
            os.remove(filepath)
        return True
    except Exception as e:
        print(f"Error clearing category: {e}")
        return False

RUBRIC_EXTRACTION_SCHEMA = {
    "type": "object",
    "description": "Complete grading rubric with step-wise marking breakdown",
    "properties": {
        "exam_metadata": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject name"},
                "grade": {"type": "string", "description": "Grade level"},
                "exam_name": {"type": "string"},
                "total_marks": {"type": "string"},
                "total_questions": {"type": "string"},
                "duration": {"type": "string"},
                "instructions": {"type": "string"}
            }
        },
        "section_info": {
            "type": "array",
            "description": "Detailed section-wise metadata and evaluation rules",
            "items": {
                "type": "object",
                "properties": {
                    "section_name": {"type": "string"},
                    "section_instructions": {"type": "string", "description": "Specific instructions like 'Attempt any 2 questions' or 'All questions are compulsory'."},
                    "attempt_count_required": {"type": "integer", "description": "Number of questions student is REQUIRED to attempt in this section. If not specified, assume all."},
                    "total_marks_section": {"type": "number", "description": "Total marks allocated to this entire section"},
                    "marks_per_question": {"type": "number", "description": "Marks for each question in this section (if uniform)"},
                    "has_internal_choice": {"type": "boolean", "description": "True if there are 'OR' choices between questions within this section"}
                }
            }
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_no": {"type": "string"},
                    "question_hierarchy": {
                        "type": "object",
                        "description": "Structure for sub-questions (e.g. 1a, 1b)",
                        "properties": {
                            "is_sub_question": {"type": "boolean", "description": "True if this is a part of a larger question (e.g. 1a)"},
                            "main_question_no": {"type": "string", "description": "The parent question number (e.g. '1' if this is '1a')"},
                            "sub_part_label": {"type": "string", "description": "The part label (e.g. 'a', 'ii')"}
                        }
                    },
                    "is_part_of_choice": {"type": "boolean", "description": "True if this question is part of an 'Either/Or' choice grouping"},
                    "section": {"type": "string"},
                    "question_type": {"type": "string"},
                    "difficulty_level": {"type": "string"},
                    "question_text_plain": {"type": "string"},
                    "question_math_latex": {"type": "string", "description": "Question text with math formatted in LaTeX, wrapped in $ symbols (e.g. $x^2$)"},
                    "figure_summary_rubric": {"type": "string"},
                    "correct_answer_plain": {"type": "string"},
                    "correct_answer_latex": {"type": "string", "description": "Answer with math formatted in LaTeX, wrapped in $ symbols"},
                    "max_marks": {"type": "string"},
                    "marking_scheme": {
                        "type": "string",
                        "description": "Free-text marking guide (kept for reference)"
                    },
                    "step_marking": {
                        "type": "array",
                        "description": "Step-wise marking rubric. Each element represents a logical concept/step.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "marksplit": {"type": "number", "description": "Marks allocated to this concept/step"},
                                "step_wise_answer": {"type": "string", "description": "Concept description. Use $...$ for all match expressions (e.g. $x=5$)."},
                                "diagram_description": {"type": "string", "description": "Optional: diagram/label requirement for this step"}
                            }
                        }
                    },
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "Key concepts to check"},
                    "diagram_labeling_requirements": {"type": "string"}
                }
            }
        }
    }
}

STUDENT_EXTRACTION_SCHEMA = {
    "type": "object",
    "description": "Student exam answers with complete text, math, figure, and metadata support",
    "properties": {
        "student_metadata": {
            "type": "object",
            "properties": {
                "student_name": {"type": "string"},
                "roll_number": {"type": "string"},
                "class_section": {"type": "string"},
                "exam_date": {"type": "string"}
            }
        },
        "answers": {
            "type": "array",
            "description": "Complete student answers with text & math. CRITICAL: STRICTLY IGNORE and EXCLUDE any text that is crossed out, struck through, or explicitly cancelled by the student.",
            "items": {
                "type": "object",
                "properties": {
                    "question_no": {"type": "string", "description": "Normalized question number"},
                    "page_number": {"type": "string", "description": "Page where answer appears"},
                    "answer_sequence_position": {"type": "string", "description": "Position in student's writing order"},
                    "section_group": {"type": "string", "description": "Detected section grouping"},
                    "answer_text_plain": {
                        "type": "string",
                        "description": "Student answer. WRAP ALL MATH EXPRESSIONS IN $...$ delimiters. Pay close attention to small details like fractions (e.g. /2), subscripts, and superscripts. DO NOT include the text that has been crossed out or struck through."
                    },
                    "figure_summary_student": {"type": "string", "description": "Detailed textual description of student-drawn diagram"},
                    "status": {"type": "string", "description": "Attempted, Blank, or Partial"}
                }
            }
        }
    }
}

# --- HELPER FUNCTIONS ---

def normalize_qno(qno: str) -> str:
    """Normalize question numbers: Q1., 1., 1, 1), 1i, 1I, 1ii, 1ii), Q1I, Q1i), i) -> 1"""
    if not qno:
        return ""
    q = str(qno).strip()
    q = q.lstrip("Qq").rstrip(".").strip()
    return q

def safe_get_string(obj, key, default=""):
    """Safely get string/list from dict"""
    if not obj:
        return default
    value = obj.get(key, default)
    if value is None:
        return default
    if isinstance(value, (str, list)):
        return value
    return str(value)

def normalize_step_marking(reference_rubric):
    """Normalize step_marking so that sum of marksplit equals max_marks"""
    if not reference_rubric:
        return reference_rubric

    for q in reference_rubric.get("questions", []):
        try:
            max_marks = float(str(q.get("max_marks", "0")))
        except:
            max_marks = 0.0

        steps = q.get("step_marking") or []
        if not steps or max_marks <= 0:
            continue

        total_step_marks = 0.0
        for step in steps:
            try:
                ms = float(step.get("marksplit", 0))
            except:
                ms = 0.0
            step["marksplit"] = ms
            total_step_marks += ms

        if total_step_marks <= 0:
            continue

        scale = max_marks / total_step_marks
        for step in steps:
            step["marksplit"] = round(step["marksplit"] * scale, 2)
            
    return reference_rubric

# --- API CLIENTS ---

def call_marker_with_structured_extraction(filepath, api_key, page_schema, max_retries=3):
    """Call Datalab Marker API"""
    DATALAB_MARKER_ENDPOINT = "https://www.datalab.to/api/v1/marker"
    
    for attempt in range(max_retries):
        try:
            with open(filepath, 'rb') as f:
                form_data = {
                    'file': (os.path.basename(filepath), f, 'application/pdf'),
                    'page_schema': (None, json.dumps(page_schema)),
                    'output_format': (None, 'json'),
                    'use_llm': (None, 'true'),
                    'force_ocr': (None, 'true'),
                }
                headers = {'X-Api-Key': api_key}
                response = requests.post(DATALAB_MARKER_ENDPOINT, files=form_data, headers=headers)
                data = response.json()

                if not data.get('success'):
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue

                check_url = data.get('request_check_url')
                
                # Poll for completion
                for i in range(600): #150
                    time.sleep(2)
                    resp = requests.get(check_url, headers=headers)
                    result = resp.json()
                    
                    if result.get('status') == 'complete':
                        return result, None
                    elif result.get('status') == 'error':
                        return None, f"Extraction Error: {result.get('error', 'Unknown error')}"
                    
                return None, "Timeout: Extraction took too long." # Timeout

        except Exception as e:
            if attempt < max_retries - 1:
                 time.sleep(2 ** attempt)
            else:
                return None, f"Request Failed: {str(e)}"
    return None, "Max retries exceeded or API unreachable."

def extract_structured_json(marker_result):
    """Extract and parse JSON from Marker result"""
    if not marker_result or not marker_result.get('success'):
        return None, None

    extraction_json_str = marker_result.get('extraction_schema_json')
    if not extraction_json_str:
        return None, None

    try:
        extracted_data = json.loads(extraction_json_str)
        citations = marker_result.get('json')
        return extracted_data, citations
    except:
        return None, None

def extract_json_robust(text):
    """Clean JSON from LLM response"""
    text = text.strip()
    json_match = re.search(r'```json?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
        
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        return text[start:end+1]
    return text.strip()

def parse_json_fallbacks(json_str):
    """Parse JSON with fallbacks, specifically handling LaTeX backslashes"""
    # 1. Try standard clean
    json_clean = re.sub(r'\\n|\\t', ' ', json_str)
    json_clean = re.sub(r'\s+', ' ', json_clean).strip()
    
    try:
        return json.loads(json_clean)
    except:
        pass

    # 2. Try raw string as sometimes regex hurts more than helps
    try:
        return json.loads(json_str)
    except:
        pass

    # 3. Try to fix single backslashes for LaTeX (e.g. \frac -> \\frac)
    # This looks for backslashes that are NOT followed by specific JSON control chars
    # We want to target things like \t, \f, \r, \n which ARE valid JSON escapes if intended as control chars
    # But often LLM outputs \frac which is invalid.
    # A simple heuristic: Replace single \ with \\, but respect \\ already existing.
    
    # Negative lookbehind to ensure we don't start with a backslash (i.e. strictly odd number of backslashes)
    # actually, simpler approach: use the raw string and re-escape.
    
    try:
        # crude fix: replace all single backslashes with double, 
        # BUT this ruins valid escapes like \" or \n if we aren't careful.
        # Let's try to specifically target common LaTeX patterns if simple parse fails.
        # This is hard to do perfectly with regex.
        # Alternative: use a more lenient parser or just try double escaping everything that looks like a latex command
        
        # Heuristic: Replace \ with \\ if it's followed by a letter
        fixed_str = re.sub(r'\\([a-zA-Z])', r'\\\\\1', json_str)
        return json.loads(fixed_str)
    except:
        pass

    return None

def call_gemini_with_retries(model, eval_prompt, question_ref, max_retries=3):
    """Call Gemini API for evaluation"""
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                eval_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.05,
                    max_output_tokens=1024
                )
            )
            json_str = extract_json_robust(response.text)
            result = parse_json_fallbacks(json_str)
            if result:
                return result
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            if "429" in str(e):
                delay *= 2
            time.sleep(delay)
            # Capture last error
            if attempt == max_retries - 1:
                return {
                    "question_no": safe_get_string(question_ref, "question_no"),
                    "marks_awarded": "ERROR",
                    "max_marks": safe_get_string(question_ref, "max_marks"),
                    "feedback": f"Failed to evaluate. Error: {str(e)}",
                    "status": "Error",
                    "error_details": f"Error: {str(e)}\n\nRaw Response:\n{response.text if hasattr(response, 'text') else 'No response text'}"
                }

    return {
        "question_no": safe_get_string(question_ref, "question_no"),
        "marks_awarded": "ERROR",
        "max_marks": safe_get_string(question_ref, "max_marks"),
        "feedback": "Failed to evaluate (Max Retries)",
        "status": "Error",
        "error_details": "Max retries exceeded during Gemini call."
    }

# --- EVALUATION LOGIC ---

def evaluate_single_answer(model, question_ref, student_answer_text, student_status, student_figures=""):
    """Evaluate a single answer using Gemini"""
    fig_clean = (student_figures or "").strip()
    if not (student_answer_text or "").strip() and not fig_clean:
        return {
            "question_no": safe_get_string(question_ref, "question_no"),
            "marks_awarded": "0",
            "max_marks": safe_get_string(question_ref, "max_marks", "0"),
            "feedback": "Answer not attempted",
            "stepwise_feedback": [],
            "diagram_feedback": "N/A",
            "status": "Blank"
        }

    eval_prompt = f"""You are an STEM and the Humanities subject expert examiner grading subjective & objective student answer for questions using detailed rubric.

==QUESTION DETAILS==
Question No: {safe_get_string(question_ref, 'question_no')}
Question Type: {safe_get_string(question_ref, 'question_type')}
Question Text: {safe_get_string(question_ref, 'question_text_plain')}

==RUBRIC REQUIREMENTS==
Max Marks: {safe_get_string(question_ref, 'max_marks', '5')}
Marking Scheme (STEPWISE):
{safe_get_string(question_ref, 'marking_scheme', 'Grade on correctness, completeness, and reasoning')}

Keywords to Check: {', '.join(safe_get_string(question_ref, 'keywords', []))}

==REFERENCE SOLUTION==
Plain Text Answer:
{safe_get_string(question_ref, 'correct_answer_plain')}

==STUDENT'S ANSWER==
Plain Text:
{student_answer_text}

Diagrams/Figures:
{student_figures}

==EVALUATION REQUIREMENTS==
1. Check for conceptual correctness.
2. Award partial credit for correct steps even if final answer is wrong.
3. Ignore minor spelling mistakes.

==OUTPUT FORMAT==
Provide evaluation in this EXACT JSON format:
{{
  "question_no": "{safe_get_string(question_ref, 'question_no')}",
  "marks_awarded": <number>,
  "max_marks": "{safe_get_string(question_ref, 'max_marks')}",
  "stepwise_feedback": [
    {{
      "step_id": 1,
      "description": "<concept>",
      "marks_awarded": <number>,
      "max_marks": <number>,
      "feedback": "<short feedback. IMPORTANT: Use double backslashes for ALL LaTeX math, e.g. \\\\frac, \\\\tanh.>"
    }}
  ],
  "overall_feedback": "<summary>",
  "status": "Attempted" | "Correct" | "Partial" | "Incorrect"
}}"""

    return call_gemini_with_retries(model, eval_prompt, question_ref)

def postprocess_evaluation(eval_result, max_marks):
    """Normalize status based on marks"""
    if not eval_result:
        return {"status": "Error"}
    
    marks_str = str(eval_result.get("marks_awarded", "0")).strip()
    if marks_str == "ERROR":
        return eval_result
        
    try:
        marks = float(marks_str)
        max_m = float(max_marks or 0)
        if max_m > 0 and marks >= max_m * 0.9:
            eval_result["status"] = "Correct"
        elif marks > 0:
            eval_result["status"] = "Attempted"
        else:
            eval_result["status"] = "Blank"
    except:
        eval_result["status"] = "Error"
    return eval_result

def apply_grading_rules(student_evals, rubric):
    """
    Apply section-wise constraints (e.g. "Attempt 3 of 5") to student evaluations.
    Marks extra questions as 'is_excluded=True' and status='Extra'.
    """
    if not student_evals or not rubric:
        return student_evals

    # 1. Build Question Registry (Map QNo -> Section, MainID)
    q_map = {}
    for q in rubric.get("questions", []):
        q_no = normalize_qno(q.get("question_no"))
        
        # Hierarchy info
        hierarchy = q.get("question_hierarchy", {})
        is_sub = hierarchy.get("is_sub_question", False)
        main_id = hierarchy.get("main_question_no")
        
        # Fallback if hierarchy missing but "1a" pattern exists? 
        # For now rely on schema. if missing, MainID = QNo
        if not main_id:
             main_id = q_no
             
        q_map[q_no] = {
            "section": q.get("section", "Default"),
            "main_id": normalize_qno(main_id),
            "is_sub": is_sub
        }

    # 2. Group Student Evals by Section -> MainID
    # Structure: sections[section_name][main_id] = [eval1, eval2...]
    sections = {}
    
    for eval_item in student_evals:
        q_no = normalize_qno(eval_item.get("question_no"))
        
        # Default to unknown if not in rubric (shouldn't happen often)
        info = q_map.get(q_no, {"section": "Default", "main_id": q_no})
        
        sec_name = info["section"]
        main_id = info["main_id"]
        
        if sec_name not in sections:
            sections[sec_name] = {}
        if main_id not in sections[sec_name]:
            sections[sec_name][main_id] = []
            
        sections[sec_name][main_id].append(eval_item)

    # 3. Process Each Section
    rubric_sections = rubric.get("section_info", [])
    # Flatten rubric sections for easy lookup
    sec_rules = {s.get("section_name"): s for s in rubric_sections}

    for sec_name, main_questions in sections.items():
        pass_rule = sec_rules.get(sec_name)
        
        # Determine allowed count
        allowed_count = 9999
        if pass_rule:
             # Look for attempt_count_required
             try:
                 req = int(pass_rule.get("attempt_count_required", 0))
                 if req > 0:
                     allowed_count = req
             except:
                 pass
        
        # Calculate score for each Main Question
        # List of (main_id, total_score, evaluatons_list)
        scored_mains = []
        
        for m_id, evals in main_questions.items():
            m_score = 0
            for e in evals:
                try:
                    m_score += float(e.get("marks_awarded", 0))
                except:
                    pass
            scored_mains.append({
                "main_id": m_id,
                "score": m_score,
                "evals": evals
            })
            
        # Sort by Score Descending
        scored_mains.sort(key=lambda x: x["score"], reverse=True)
        
        # Mark Best N as Counted, Rest as Extra
        for idx, item in enumerate(scored_mains):
            is_counted = idx < allowed_count
            
            for e in item["evals"]:
                e["is_excluded"] = not is_counted
                if not is_counted:
                    e["original_status"] = e.get("status")
                    e["status"] = "Extra (Not Counted)"
                    
    return student_evals
