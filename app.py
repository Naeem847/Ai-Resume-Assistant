import os
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# Page Configuration
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text safely from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(uploaded_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as error:
        st.error(f"Error reading PDF file: {error}")
        return ""

def generate_ats_analysis(api_key: str, resume_text: str, job_description: str) -> str:
    """Invokes Gemini 2.5 Flash to evaluate the resume against ATS criteria."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an experienced HR Director and ATS (Applicant Tracking System) Optimization Expert.
    Analyze the provided candidate resume text against the target job description.

    ---
    ### TARGET JOB DESCRIPTION:
    {job_description if job_description.strip() else "No specific job description provided. Perform a general evaluation based on high-standard industry standards for the core role identified in the resume."}

    ---
    ### CANDIDATE RESUME:
    {resume_text}

    ---
    ### REQUIRED OUTPUT FORMAT:
    Please provide a detailed report with the following exact structure:

    1. **Overall ATS Match Score**: State an exact percentage (0% - 100%).
    2. **Profile Fit Summary**: A 2-3 sentence overview assessing candidate alignment.
    3. **Key Matching Strengths**: Bullet points detailing relevant skills, experiences, and technical keywords present.
    4. **Missing Keywords & Gaps**: Critical hard/soft skills, tools, or qualifications missing from the resume.
    5. **Actionable Improvement Recommendations**: 3-5 bullet points offering specific advice on phrasing, formatting, quantifiable metrics, and impact.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )
    return response.text

# Header Section
st.title("📄 AI ATS Resume Analyzer")
st.write("Upload your resume (PDF) to evaluate its ATS compatibility, receive an overall match score, and view tailored improvement points.")

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    
    # Automatically check Secrets or Environment Variables, or allow manual input
    default_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=default_api_key,
        type="password",
        help="Obtain your API key from Google AI Studio (https://aistudio.google.com/)"
    )
    
    st.markdown("---")
    st.caption("Engine: `gemini-2.5-flash` via `google-genai` SDK")

# Main Input Section
uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])
job_description = st.text_area("Job Description (Optional, but recommended for specific matching)", height=150)

# Execution Action
if st.button("Run ATS Analysis", type="primary"):
    if not api_key_input:
        st.error("Please enter a valid Gemini API key in the sidebar or configure it in secrets.")
    elif uploaded_file is None:
        st.warning("Please upload a PDF resume to proceed.")
    else:
        with st.spinner("Extracting PDF text and analyzing with Gemini AI..."):
            extracted_resume_text = extract_text_from_pdf(uploaded_file)

            if not extracted_resume_text:
                st.error("Unable to extract text from the PDF. Please ensure it is not a scanned image or locked file.")
            else:
                try:
                    result = generate_ats_analysis(
                        api_key=api_key_input,
                        resume_text=extracted_resume_text,
                        job_description=job_description
                    )
                    st.success("Analysis Complete!")
                    st.markdown(result)
                except Exception as ex:
                    st.error(f"Failed to communicate with Gemini API: {ex}")