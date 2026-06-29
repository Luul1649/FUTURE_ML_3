# AI-Powered Resume Screening and Candidate Ranking System

## Overview

The AI-Powered Resume Screening and Candidate Ranking System is a Machine Learning and Natural Language Processing (NLP) application that automates the process of screening resumes against a job description. The system extracts relevant skills, measures resume-job similarity, identifies skill gaps, and ranks candidates based on their suitability for a given role.

This project demonstrates how Artificial Intelligence can support recruiters and HR professionals by reducing manual screening time and improving consistency in candidate evaluation.

---

## Features

* Upload and analyze a job description
* Automatically process multiple resumes
* Text preprocessing using NLP techniques
* Skill extraction from resumes and job descriptions
* Resume-to-job similarity scoring using TF-IDF and Cosine Similarity
* Candidate ranking based on job relevance
* Missing skill identification
* Interactive Streamlit dashboard
* Download ranked candidates as a CSV file

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Gdown

---

## Machine Learning Workflow

1. Load resume dataset
2. Input job description
3. Clean and preprocess text
4. Extract relevant skills
5. Convert text into TF-IDF vectors
6. Calculate cosine similarity between resumes and the job description
7. Rank candidates based on similarity score
8. Identify missing skills
9. Display results through an interactive dashboard

---

## Project Structure

```text
Resume-Screening-System/
│
├── app.py
├── Resume.csv (downloaded automatically)
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
```

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Dataset

This project uses the Resume Dataset from Kaggle.

Download the dataset from:

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

The deployed application automatically downloads the dataset from Google Drive when it is first launched.

---

## Sample Output

The dashboard provides:

* Candidate ranking
* Resume match score
* Missing skills analysis
* Bar chart of top candidates
* Downloadable CSV report

---

## Future Improvements

* PDF resume upload
* Resume parsing using spaCy
* Named Entity Recognition (NER)
* AI-powered resume feedback
* Semantic similarity using Sentence Transformers
* Multi-job comparison
* Recruiter authentication
* Interview recommendation engine

---

## Learning Outcomes

Through this project I gained practical experience in:

* Natural Language Processing (NLP)
* Text preprocessing
* Feature extraction using TF-IDF
* Cosine similarity
* Machine Learning for decision support
* Streamlit application development
* Data visualization
* Model deployment
* Git and GitHub

---

## Author

Developed by **Haji**

Machine Learning | Data Science | Artificial Intelligence

This project was completed as part of the Future Interns Machine Learning Internship Program.
