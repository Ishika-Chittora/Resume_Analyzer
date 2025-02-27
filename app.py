import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model for information extraction
nlp = spacy.load("en_core_web_sm")

# Set the title of the app with emojis
st.title("📄 Resume Ranking and Analysis App 🚀")
st.write("Upload a job description and resumes (PDF or text) to rank them and extract key information.")

# Add a colorful header
st.markdown(
    """
    <style>
    .header {
        font-size: 24px;
        color: #ffffff;
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    <div class="header">
        🎯 Find the Best Candidates with AI-Powered Resume Analysis!
    </div>
    """,
    unsafe_allow_html=True,
)

# Load the saved TF-IDF vectorizer
@st.cache_resource  # Cache the loaded objects for better performance
def load_objects():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return tfidf

tfidf = load_objects()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract skills using a predefined skill list
def extract_skills(text):
    # Predefined list of skills (customize this based on your needs)
    skills_list = [
        "python", "java", "machine learning", "data analysis", "sql", "c++", "software development",
        "deep learning", "artificial intelligence", "data science", "tensorflow", "pytorch", "scikit-learn",
        "natural language processing", "computer vision", "big data", "spark", "hadoop", "aws", "azure",
        "docker", "kubernetes", "git", "linux", "statistics", "mathematics", "excel", "tableau", "power bi"
    ]
    
    # Create a PhraseMatcher object to match skills
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp(skill) for skill in skills_list]
    matcher.add("SKILLS", None, *patterns)
    
    # Match skills in the text
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        skills.add(doc[start:end].text)
    
    return list(skills)

# Function to extract qualifications (e.g., degrees)
def extract_qualifications(text):
    # Regex pattern to extract qualifications
    degree_pattern = re.compile(
        r"\b(bachelor|master|phd|bs|ms|mtech|btech|mca|bca|b\.\s*sc|m\.\s*sc|b\.\s*tech|m\.\s*tech)\b", 
        re.IGNORECASE
    )
    qualifications = set()
    for match in degree_pattern.findall(text):
        qualifications.add(match)
    return list(qualifications)

# Function to extract the entire "Experience" section
def extract_experience_section(text):
    # Look for the keyword "Experience" (case-insensitive)
    experience_keywords = ["experience", "work experience", "professional experience", "employment history"]
    for keyword in experience_keywords:
        start_index = text.lower().find(keyword)
        if start_index != -1:
            # Extract the text after the keyword
            experience_section = text[start_index + len(keyword):].strip()
            # Stop at the next section (e.g., "Education", "Skills")
            next_section_keywords = ["education", "skills", "projects", "certifications"]
            for next_keyword in next_section_keywords:
                next_index = experience_section.lower().find(next_keyword)
                if next_index != -1:
                    experience_section = experience_section[:next_index].strip()
                    break
            return experience_section
    return "No experience section found."

# Input: Text area for job description
st.markdown("### 📝 Job Description")
job_description = st.text_area("Paste the job description here:", height=150)

# File uploader for resumes (PDF or text)
st.markdown("### 📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload resumes (PDF or text files):", type=["pdf", "txt"], accept_multiple_files=True)

# Predict button with animation
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("🚀 Rank Resumes and Extract Information"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        # Preprocess the job description
        job_tfidf = tfidf.transform([job_description]).toarray()

        # Process each resume
        results = []
        for uploaded_file in uploaded_files:
            # Extract text from the uploaded file
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode("utf-8")
            
            # Preprocess the resume text
            resume_tfidf = tfidf.transform([resume_text]).toarray()

            # Calculate cosine similarity between job description and resume
            similarity = cosine_similarity(job_tfidf, resume_tfidf)[0][0]

            # Extract skills, qualifications, and experience section
            skills = extract_skills(resume_text)
            qualifications = extract_qualifications(resume_text)
            experience_section = extract_experience_section(resume_text)

            # Store the result
            results.append({
                'file_name': uploaded_file.name,
                'resume_text': resume_text,
                'similarity': similarity,
                'skills': skills,
                'qualifications': qualifications,
                'experience_section': experience_section
            })

        # Sort resumes by similarity score (descending order)
        results.sort(key=lambda x: x['similarity'], reverse=True)

        # Display ranked resumes and extracted information
        st.subheader("🏆 Ranked Resumes")
        for i, result in enumerate(results):
            st.markdown(
                f"""
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                    <h3>Rank {i + 1}: {result['file_name']} (Similarity: {result['similarity']:.2f})</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"🔍 View Resume {i + 1} Details"):
                st.write("**Skills:**")
                st.write(", ".join(result['skills']) if result['skills'] else "No skills found.")
                st.write("**Qualifications:**")
                st.write(", ".join(result['qualifications']) if result['qualifications'] else "No qualifications found.")
                st.write("**Experience Section:**")
                st.write(result['experience_section'])
                st.write("**Full Resume Text:**")
                st.write(result['resume_text'])

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app ranks resumes based on their relevance to a given job description and extracts key information. "
    "Upload a job description and resumes (PDF or text) to get started."
)

# Add a footer
st.markdown(
    """
    <style>
    .footer {
        font-size: 14px;
        color: #777777;
        text-align: center;
        padding: 10px;
        margin-top: 20px;
    }
    </style>
    <div class="footer">
        Made by AlphaBytes
    </div>
    """,
    unsafe_allow_html=True,
)