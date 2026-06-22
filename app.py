import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

# --------------------
# Page Config
# --------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

st.title("📄 AI Resume Screening & Candidate Ranking System")

st.markdown(
"""
This system compares resumes against a job description,
calculates similarity scores, ranks candidates,
and identifies missing skills.
"""
)

# --------------------
# Load Dataset
# --------------------

df = pd.read_csv("Resume.csv")

# Change this if your dataset uses another column name
resume_column = "Resume_str"

# --------------------
# Skill Database
# --------------------

skills_db = [
    "python",
    "machine learning",
    "deep learning",
    "sql",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "power bi",
    "excel",
    "tableau",
    "data analysis",
    "data visualization"
]

# --------------------
# Text Cleaning
# --------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

# --------------------
# Skill Extraction
# --------------------

def extract_skills(text):

    text = str(text).lower()

    found = []

    for skill in skills_db:
        if skill in text:
            found.append(skill)

    return found

# --------------------
# Job Description Input
# --------------------

st.sidebar.header("Job Description")

job_description = st.sidebar.text_area(
    "Paste Job Description",
    height=250
)

# --------------------
# Screening Button
# --------------------

if st.sidebar.button("Screen Resumes"):

    if job_description.strip() == "":
        st.warning("Please enter a Job Description.")
        st.stop()

    resumes = df[resume_column].astype(str).tolist()

    documents = resumes + [job_description]

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_matrix = vectorizer.fit_transform(documents)

    job_vector = tfidf_matrix[-1]

    resume_vectors = tfidf_matrix[:-1]

    scores = cosine_similarity(
        resume_vectors,
        job_vector
    ).flatten()

    df["Match Score"] = scores * 100

    ranked = df.sort_values(
        by="Match Score",
        ascending=False
    )

    # --------------------
    # Skill Gap Analysis
    # --------------------

    job_skills = extract_skills(job_description)

    missing_skills_list = []

    for resume in ranked[resume_column]:

        candidate_skills = extract_skills(resume)

        missing = list(
            set(job_skills) -
            set(candidate_skills)
        )

        missing_skills_list.append(
            ", ".join(missing)
        )

    ranked["Missing Skills"] = missing_skills_list

    # --------------------
    # Display Results
    # --------------------

    st.subheader("🏆 Top Candidates")

    st.dataframe(
        ranked[
            [
                "Category",
                "Match Score",
                "Missing Skills"
            ]
        ].head(20)
    )

    # --------------------
    # Visualization
    # --------------------

    st.subheader("📊 Match Scores")

    top10 = ranked.head(10)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        range(len(top10)),
        top10["Match Score"]
    )

    ax.set_xticks(range(len(top10)))

    ax.set_xticklabels(
        top10["Category"],
        rotation=45
    )

    ax.set_ylabel("Score")

    st.pyplot(fig)

    # --------------------
    # Top Candidate
    # --------------------

    best = ranked.iloc[0]

    st.subheader("🥇 Best Candidate")

    st.write("Category:", best["Category"])

    st.write(
        "Match Score:",
        round(best["Match Score"],2)
    )

    st.write(
        "Missing Skills:",
        best["Missing Skills"]
    )

    # --------------------
    # Download Results
    # --------------------

    csv = ranked.to_csv(index=False)

    st.download_button(
        label="⬇ Download Ranking CSV",
        data=csv,
        file_name="ranking_results.csv",
        mime="text/csv"
    )
