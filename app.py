import streamlit as st
import pandas as pd
import os
import json
import time
import tempfile
import google.generativeai as genai
from utils import (
    RUBRIC_EXTRACTION_SCHEMA,
    STUDENT_EXTRACTION_SCHEMA,
    normalize_step_marking,
    call_marker_with_structured_extraction,
    extract_structured_json,
    evaluate_single_answer,
    postprocess_evaluation,
    normalize_qno,
    safe_get_string,
    save_to_history,
    list_history_files,
    load_history_file,
    delete_history_file,
    clear_history_category,
    convert_datalab_to_markdown,
    extract_rubric_with_gemini_native
)

st.set_page_config(page_title="Auto-Grader AI", layout="wide")

# --- STATE MANAGEMENT ---
if "rubric_data" not in st.session_state:
    st.session_state.rubric_data = None
if "student_data_list" not in st.session_state:
    st.session_state.student_data_list = []
if "grading_results" not in st.session_state:
    st.session_state.grading_results = []

# --- CONFIGURATION ---
# Try to get keys from secrets (local dev) or env vars (deployment)
try:
    datalab_key = st.secrets.get("DATALAB_API_KEY")
    gemini_key = st.secrets.get("GEMINI_API_KEY")
except FileNotFoundError:
    datalab_key = None
    gemini_key = None

# Fallback to os.environ if not in secrets
if not datalab_key:
    datalab_key = os.environ.get("DATALAB_API_KEY", "")
if not gemini_key:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

if not datalab_key or not gemini_key:
    st.warning("⚠️ API Keys missing. Please set DATALAB_API_KEY and GEMINI_API_KEY in .streamlit/secrets.toml or as Environment Variables.")

if gemini_key:
    genai.configure(api_key=gemini_key)
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
    except:
        st.error("Invalid Gemini Key in secrets")
        model = None
else:
    model = None
    st.warning("GEMINI_API_KEY not found in secrets.")

if not datalab_key:
    st.warning("DATALAB_API_KEY not found in secrets.")

# --- SIDEBAR: SESSION MANAGEMENT ---
with st.sidebar:
    st.header("💾 Session Manager")
    st.info("Upload previously downloaded JSON files to restore your session.")
    
    uploaded_states = st.file_uploader("Restore Session Data", type=["json"], accept_multiple_files=True)
    
    if uploaded_states:
        for state_file in uploaded_states:
            try:
                data = json.load(state_file)
                
                # Detect Rubric
                if isinstance(data, dict) and "exam_metadata" in data and "questions" in data:
                    st.session_state.rubric_data = data
                    st.toast(f"✅ Restored Rubric: {data.get('exam_metadata', {}).get('exam_name')}")
                
                # Detect Student List
                elif isinstance(data, list) and len(data) > 0 and "student_metadata" in data[0]:
                     st.session_state.student_data_list = data
                     st.toast(f"✅ Restored {len(data)} Student Scripts")
                
                # Detect Single Student (if uploaded individually)
                elif isinstance(data, dict) and "student_metadata" in data:
                     # Check if exists
                     exists = any((s.get("student_metadata") or {}).get("roll_number") == (data.get("student_metadata") or {}).get("roll_number") for s in st.session_state.student_data_list)
                     if not exists:
                         st.session_state.student_data_list.append(data)
                         st.toast(f"✅ Restored Student: {(data.get('student_metadata') or {}).get('student_name')}")
                     
            except Exception as e:
                st.error(f"Error reading {state_file.name}: {e}")
    
    
    if st.button("Clear All Session Data"):
        st.session_state.rubric_data = None
        st.session_state.student_data_list = []
        st.session_state.grading_results = []
        st.rerun()

    st.divider()
    
    # --- HISTORY VIEWER ---
    st.header("📜 History Viewer")
    st.info("View and download auto-saved files from this server session. (Cloud: Cleared on restart)")
    
    hist_type = st.selectbox("Select Category", ["Rubrics", "Students", "Grading"], key="hist_cat")
    
    category_map = {
        "Rubrics": "rubrics",
        "Students": "students",
        "Grading": "grading"
    }
    
    selected_cat = category_map[hist_type]
    files = list_history_files(selected_cat)
    
    if files:
        file_options = {f"{f['timestamp']} - {f['filename']}": f['filename'] for f in files}
        selected_file_label = st.selectbox("Select File", list(file_options.keys()), key="hist_file_sel")
        
        if selected_file_label:
            real_filename = file_options[selected_file_label]
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"Load {hist_type} Preview"):
                    data = load_history_file(selected_cat, real_filename)
                    if data:
                        st.json(data, expanded=False)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json.dumps(data, indent=2),
                            file_name=real_filename,
                            mime="application/json"
                        )
                    else:
                        st.error("File not found or empty.")
            
            with c2:
                if st.button("🗑️ Delete File", type="primary"):
                    if delete_history_file(selected_cat, real_filename):
                        st.toast(f"Deleted {real_filename}")
                        st.rerun()
                    else:
                        st.error("Failed to delete file.")
                        
        st.divider()
        if st.button(f"🗑️ Clear All {hist_type} History", type="primary"):
             if clear_history_category(selected_cat):
                 st.toast(f"Cleared all {hist_type} history.")
                 st.rerun()
             else:
                 st.error("Failed to clear history.")
                 
    else:
        st.caption("No history found for this category.")

    
# --- MAIN APP ---
st.title("📝 AI Exam Auto-Grader")
st.markdown("Upload a solution rubric and student answer scripts to automatically grade them.")

tab1, tab2, tab3 = st.tabs(["1. Upload Rubric", "2. Upload Students", "3. Grading & Results"])

# --- TAB 1: RUBRIC ---
with tab1:
    st.header("Step 1: Upload Reference Solution (Rubric)")
    uploaded_rubric = st.file_uploader("Upload Solution PDF", type=["pdf"])
    
    if uploaded_rubric and datalab_key:
        if st.button("Extract Rubric"):
            with st.spinner("Analyzing Rubric PDF..."):
                # Save to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_rubric.getbuffer())
                    rubric_path = tmp.name
                
                # Call API - PARSE ONLY (No Schema)
                result, error_msg = call_marker_with_structured_extraction(
                    rubric_path, datalab_key, None
                )
                
                if result:
                    # 1. Convert Datalab JSON to Text
                    st.toast("✅ PDF Parsed. Extracting Rubric with Gemini...")
                    rubric_text = convert_datalab_to_markdown(result)
                    
                    # 2. Extract with Gemini
                    rubric, gemini_err = extract_rubric_with_gemini_native(model, rubric_text, RUBRIC_EXTRACTION_SCHEMA)
                    
                    if rubric:
                        rubric = normalize_step_marking(rubric)
                        st.session_state.rubric_data = rubric
                        
                        # Auto-save Rubric
                        saved_path = save_to_history(rubric, "rubrics", "rubric_extracted")
                        if saved_path:
                            st.toast(f"💾 Rubric auto-saved to history.")

                        st.success("✅ Rubric Extracted and Parsed Successfully!")
                    else:
                        st.error(f"Gemini Extraction Failed: {gemini_err}")
                else:
                    st.error(f"Rubric parsing failed: {error_msg}")
                
                # Cleanup
                if os.path.exists(rubric_path):
                    os.remove(rubric_path)

    if st.session_state.rubric_data:
        st.success(f"Active Rubric: {st.session_state.rubric_data.get('exam_metadata', {}).get('exam_name', 'Unnamed')}")
        
        # Download Option
        rubric_json = json.dumps(st.session_state.rubric_data, indent=2)
        st.download_button(
            label="📥 Download Rubric JSON",
            data=rubric_json,
            file_name="rubric_extracted.json",
            mime="application/json"
        )
        
        # Persistent Display
        rubric = st.session_state.rubric_data
        st.json(rubric.get("exam_metadata", {}))
        
        questions = rubric.get("questions", [])
        st.info(f"Found {len(questions)} questions in rubric.")

        with st.expander("View Extracted Rubric Details"):
            for q in questions:
                st.markdown(f"**Q{q.get('question_no')} ({q.get('max_marks')} marks)**")
                # Prefer LaTeX if available, else plain text
                q_text = q.get("question_math_latex") or q.get("question_text_plain")
                st.markdown(q_text)
                
                st.caption("Reference Answer:")
                ans_text = q.get("correct_answer_latex") or q.get("correct_answer_plain")
                st.markdown(ans_text)

                if q.get("step_marking"):
                    st.caption("Step-wise Marking Scheme:")
                    for step in q.get("step_marking"):
                        marks = step.get("marksplit", 0)
                        desc = step.get("step_wise_answer", "")
                        if desc:
                            st.markdown(f"• **[{marks} Marks]** {desc}")
                st.divider()

# --- TAB 2: STUDENTS ---
with tab2:
    st.header("Step 2: Upload Student Answer Scripts")
    uploaded_students = st.file_uploader("Upload Student PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_students and datalab_key and st.session_state.rubric_data:
        if st.button(f"Process {len(uploaded_students)} Student Files"):
            progress_bar = st.progress(0)
            st.session_state.student_data_list = [] # Reset on new process
            
            for idx, stu_file in enumerate(uploaded_students):
                status_text = st.empty()
                status_text.text(f"Processing {stu_file.name}...")
                
                # Save temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(stu_file.getbuffer())
                    stu_path = tmp.name
                
                # Call API
                result, error_msg = call_marker_with_structured_extraction(
                    stu_path, datalab_key, STUDENT_EXTRACTION_SCHEMA
                )
                
                if result:
                    data, _ = extract_structured_json(result)
                    if data:
                        data['filename'] = stu_file.name
                        st.session_state.student_data_list.append(data)
                        
                        # Auto-save Student
                        s_name = (data.get('student_metadata') or {}).get('student_name', 'unknown')
                        save_to_history(data, "students", f"student_{s_name}")
                        
                        st.write(f"✅ Extracted: {(data.get('student_metadata') or {}).get('student_name', 'Unknown')}")
                    else:
                        st.warning(f"Failed to parse {stu_file.name}")
                else:
                    st.error(f"Failed to process {stu_file.name}: {error_msg}")
                
                # Cleanup
                if os.path.exists(stu_path):
                    os.remove(stu_path)
                
                progress_bar.progress((idx + 1) / len(uploaded_students))
                
            st.success(f"Processed {len(st.session_state.student_data_list)} students.")

    if st.session_state.student_data_list:
        st.info(f"{len(st.session_state.student_data_list)} students ready for grading.")
        
        # Download Option
        students_json = json.dumps(st.session_state.student_data_list, indent=2)
        st.download_button(
            label="📥 Download Students JSON",
            data=students_json,
            file_name="students_extracted.json",
            mime="application/json"
        )

        st.divider()
        st.subheader("Student Answer Previews")
        for s_data in st.session_state.student_data_list:
            s_name = (s_data.get('student_metadata') or {}).get('student_name', 'Unknown')
            s_file = s_data.get('filename', 'Unknown File')
            
            with st.expander(f"📄 {s_name} ({s_file})"):
                for ans in s_data.get("answers", []):
                    st.markdown(f"**Q{ans.get('question_no')}**")
                    st.markdown(ans.get("answer_text_plain", "No text extracted"))
                    if ans.get("figure_summary_student"):
                        st.caption("Figure Summary:")
                        st.info(ans.get("figure_summary_student"))
                    st.divider()

# --- TAB 3: GRADING ---
with tab3:
    st.header("Step 3: AI Grading")
    
    if not st.session_state.rubric_data:
        st.warning("Please upload specific rubric first.")
    elif not st.session_state.student_data_list:
        st.warning("Please upload student scripts.")
    elif not model:
        st.error("Gemini API Key required.")
    else:
        if st.button("🚀 Start Grading"):
            all_evals = []
            rubric = st.session_state.rubric_data
            students = st.session_state.student_data_list
            
            total_ops = len(students) * len(rubric.get("questions", []))
            bar = st.progress(0)
            completed_ops = 0
            
            for student in students:
                student_evals = []
                stu_answers = {normalize_qno(a.get("question_no")): a for a in student.get("answers", [])}
                
                st.write(f"Grading **{(student.get('student_metadata') or {}).get('student_name', 'Unknown')}**...")
                
                for q_ref in rubric.get("questions", []):
                    qno = normalize_qno(q_ref.get("question_no"))
                    
                    # Match answer
                    # Handles duplicates poorly, just takes last one for now or first
                    # Improvements can be made here
                    ans_data = stu_answers.get(qno)
                    
                    ans_text = ""
                    ans_figs = ""
                    status = "Blank"
                    
                    if ans_data:
                        ans_text = ans_data.get("answer_text_plain", "")
                        ans_figs = ans_data.get("figure_summary_student", "")
                        status = ans_data.get("status", "Attempted")
                    
                    # Evaluate
                    eval_res = evaluate_single_answer(model, q_ref, ans_text, status, ans_figs)
                    eval_res = postprocess_evaluation(eval_res, q_ref.get("max_marks"))
                    
                    # Add metadata
                    eval_res["student_name"] = (student.get("student_metadata") or {}).get("student_name")
                    eval_res["student_roll"] = (student.get("student_metadata") or {}).get("roll_number")
                    eval_res["filename"] = student.get("filename", "Unknown") # Add filename
                    eval_res["question_text"] = q_ref.get("question_text_plain")
                    eval_res["student_answer"] = ans_text
                    eval_res["figure_summary_student"] = ans_figs
                    
                    all_evals.append(eval_res)
                    
                    completed_ops += 1
                    bar.progress(min(completed_ops / total_ops, 1.0))
            
            st.session_state.grading_results = all_evals
            
            # Auto-save Results
            save_to_history(all_evals, "grading", "grading_results")
            st.toast("💾 Grading results auto-saved to history.")
            
            st.success("Grading Complete!")
            
    # RESULTS DISPLAY
    if st.session_state.grading_results:
        # Separate results by student for clearer display
        df = pd.DataFrame(st.session_state.grading_results)
        
        # Prepare Rubric Data
        rubric_questions = st.session_state.rubric_data.get("questions", []) if st.session_state.rubric_data else []
        
        # Deduplicate questions for calculation
        unique_questions = {}
        for q in rubric_questions:
            qno = normalize_qno(q.get("question_no"))
            if qno not in unique_questions:
                unique_questions[qno] = q
        
        # 1. Try to get Total Marks from Metadata
        metadata_total = 0
        rubric_metadata = st.session_state.rubric_data.get("exam_metadata", {}) if st.session_state.rubric_data else {}
        try:
             metadata_total = float(rubric_metadata.get("total_marks", 0))
        except:
             metadata_total = 0
             
        # 2. Calculate Sum from Questions (Fallback & Check)
        calculated_total = 0
        for qno, q in unique_questions.items():
            try:
                calculated_total += float(q.get("max_marks", 0))
            except:
                pass
        
        # Decide which total to show
        display_total = metadata_total if metadata_total > 0 else calculated_total
        
        # Warning if mismatch (only if both exist and differ significantly)
        if metadata_total > 0 and calculated_total > 0 and abs(metadata_total - calculated_total) > 1.0:
             st.warning(f"⚠️ Note: Rubric Metadata states Total Marks = {metadata_total}, but sum of extracted questions = {calculated_total}. Using Metadata value for display.")
        
        # Get unique filenames (more reliable than names)
        unique_filenames = df["filename"].unique()
        
        for filename in unique_filenames:
            student_df = df[df["filename"] == filename]
            
            # Get student name from the first record for this file
            student_name_extracted = student_df.iloc[0].get("student_name", "Unknown")
            
            
            # Calculate Student Score
            student_score = 0
            for marks in student_df["marks_awarded"]:
                try:
                    student_score += float(marks)
                except:
                    pass
            
            st.divider()
            st.divider()
            st.markdown(f"### 🎓 File: {filename}")
            st.caption(f"**Student Name Detected:** {student_name_extracted}")
            
            # Score Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Score", f"{student_score} / {display_total}")
            with col2:
                csv = student_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{filename}_results.csv",
                    mime="text/csv",
                    key=f"csv_{filename}"
                )
            with col3:
                 # Prepare nested JSON for download
                 student_json = student_df.to_json(orient="records", indent=2)
                 st.download_button(
                    label="📥 Download JSON",
                    data=student_json,
                    file_name=f"{filename}_results.json",
                    mime="application/json",
                    key=f"json_{filename}"
                 )

            # Detailed Breakdown (Readable UI)
            st.subheader("Detailed Breakdown")
            
            for index, row in student_df.iterrows():
                with st.expander(f"Q{row['question_no']}: {row['status']} ({row['marks_awarded']} / {row['max_marks']} Marks)"):
                    
                    # Columns for side-by-side view
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown("**📝 Question:**")
                        st.markdown(row.get("question_text", "N/A"))
                        
                        st.markdown("**✍️ Student Answer:**")
                        st.markdown(row.get("student_answer", "N/A"))
                    
                    with c2:
                        st.markdown("**✅ Reference Criteria:**")
                        # Find matching rubric question for context if needed, 
                        # but we can rely on what's captured in eval_res if extended.
                        # For now, we show the result's feedback
                        
                        st.markdown("**🤖 AI Feedback:**")
                        if isinstance(row.get("stepwise_feedback"), list):
                            for step in row.get("stepwise_feedback"):
                                color = "green" if step.get('marks_awarded', 0) > 0 else "red"
                                st.markdown(f":{color}[• {step.get('feedback')} ({step.get('marks_awarded')}/{step.get('max_marks')})]")
                        else:
                            st.write(row.get("feedback"))
                            if row.get("error_details"):
                                st.error(f"Error Details: {row.get('error_details')}")
                    
                    # Raw JSON Viewer for this question (toggle)
                    st.caption("Raw Data")
                    st.json(row.to_dict())

            st.divider()

        # Global Downloads (All Students)
        st.subheader("📦 Bulk Actions")
        
        # Prepare Summary Data
        summary_rows = []
        for filename in unique_filenames:
            subset = df[df["filename"] == filename]
            if not subset.empty:
                s_meta = subset.iloc[0]
                
                # Calculate totals safely
                tot_score = 0
                tot_max = 0
                for _, r in subset.iterrows():
                    try: tot_score += float(r.get("marks_awarded", 0))
                    except: pass
                    try: tot_max += float(r.get("max_marks", 0))
                    except: pass
                
                summary_rows.append({
                    "Filename": filename,
                    "Student Name": s_meta.get("student_name", "Unknown"),
                    "Roll Number": s_meta.get("student_roll", ""),
                    "Total Score": tot_score,
                    "Max Marks": tot_max,
                    "Percentage": f"{(tot_score/tot_max*100):.2f}%" if tot_max > 0 else "0%"
                })
        
        summary_df = pd.DataFrame(summary_rows)

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            all_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Detailed CSV", all_csv, "all_results_detailed.csv", "text/csv")
        with col_b2:
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Summary Report", summary_csv, "summary_report.csv", "text/csv")
        with col_b3:
            all_json = df.to_json(orient="records", indent=2)
            st.download_button("Download All JSON", all_json, "all_results.json", "application/json")
        
