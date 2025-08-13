import streamlit as st
import os
import time
import requests
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
load_dotenv()

import pandas as pd
import json
from io import StringIO


# A complete, self-contained Streamlit app for a data science project.
# This app demonstrates:
# 1. Manual user input for job postings.
# 2. Using the Gemini API with a few-shot learning prompt to extract
#    data science-specific skills and confidence scores.
# 3. Handling API calls with exponential backoff for robustness.
# This version focuses solely on skill extraction and displays the results in a table.

# --- 1. Set Up API Configuration ---
# IMPORTANT: The API key is left as an empty string. The Canvas environment will
# automatically provide an API key during runtime.
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"


# --- 2. Define the LLM Skill Extraction Function ---
@st.cache_data(show_spinner=False)
def extract_skills_with_gemini(description, max_retries=5, initial_delay=1.0):
    """
    Extracts a list of data science-related skills and confidence scores from a job description
    using the Gemini API, with exponential backoff for retries.
    """
    prompt = f"""
    You are a data science career coach. Your task is to extract all data science-related skills, tools, and technologies from the following job description. Focus on hard skills like programming languages, libraries, databases, cloud platforms, and methodologies. For each skill, also provide a confidence score (from 0 to 100) indicating how certain you are that this is a relevant skill. A score of 100 means you are highly certain, while a lower score indicates less certainty.

    Output the skills and confidence scores as a JSON array of objects. Do not include any other text, explanations, or formatting outside of the JSON array.

    --- Few-Shot Learning Examples ---

    Job Description:
    "The Platform Emerging Technology team is a collection of engineers focused on providing enablement and acceleration for other teams that are delivering products which leverage emergent technologies. The Emerging Technology team enhances high quality outcomes through the delivery of foundational products that meet the scale and agility of the business while maintaining its robust security posture. Drive the implementation and refinement of a cutting-edge framework, deployed on Google Kubernetes Engine (GKE) within Google Cloud Platform (GCP), empowering development teams to seamlessly integrate with Generative AI technologies. Champion best practices in software engineering, including code quality, testing, and documentation, to ensure a robust and reliable framework for internal and external developers."

    JSON Output:
    [
      {{"skill": "Generative AI", "confidence_score": 98}},
      {{"skill": "Google Kubernetes Engine (GKE)", "confidence_score": 95}},
      {{"skill": "Google Cloud Platform (GCP)", "confidence_score": 95}},
      {{"skill": "software engineering", "confidence_score": 90}},
      {{"skill": "code quality", "confidence_score": 85}},
      {{"skill": "documentation", "confidence_score": 80}}
    ]

    Job Description:
    "We are seeking a highly skilled and motivated Human Performance Data Scientist to join our team. The ideal candidate will be experienced in extracting insights from complex datasets, visualizing data through compelling reports, and supporting human-performance initiatives. This position will play a critical role in designing and delivering actionable insights through data-driven storytelling. Use Python and/or R for advanced data analysis, statistical modeling, and automation. Develop dashboards and reports using Power BI, Teamworks AMS, or similar data visualization tools."

    JSON Output:
    [
      {{"skill": "data analysis", "confidence_score": 95}},
      {{"skill": "statistical modeling", "confidence_score": 95}},
      {{"skill": "Python", "confidence_score": 98}},
      {{"skill": "R", "confidence_score": 98}},
      {{"skill": "Power BI", "confidence_score": 90}},
      {{"skill": "data visualization", "confidence_score": 90}},
      {{"skill": "Teamworks AMS", "confidence_score": 85}}
    ]

    --- User Input ---

    Job Description:
    {description}
    """

    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "skill": { "type": "STRING" },
                    "confidence_score": {
                        "type": "NUMBER",
                        "description": "A score from 0 to 100 representing the confidence that the extracted skill is relevant to the job description."
                    }
                },
                "propertyOrdering": ["skill", "confidence_score"]
            }
        }
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config
    }

    headers = {'Content-Type': 'application/json'}
    delay = initial_delay

    for i in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            if result and 'candidates' in result and result['candidates']:
                text_part = result['candidates'][0]['content']['parts'][0]['text']
                skills_with_scores = json.loads(text_part)
                return skills_with_scores
            else:
                return []

        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            st.error(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            st.error(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            st.error(f"An unexpected error occurred: {err}")
        except json.JSONDecodeError:
            st.error("Failed to decode JSON response from API. Retrying...")

        time.sleep(delay)
        delay *= 2

    st.error("Failed to extract skills after multiple retries.")
    return []

# --- NEW: Function to generate project ideas using the LLM ---
def generate_project_ideas(skills_list):
    """
    Generates a list of project ideas based on a list of skills using the Gemini API.
    """
    # The prompt has been updated to specifically request 3 project ideas.
    skills_string = ", ".join(skills_list)
    prompt = f"""
    You are a career coach and project advisor. Based on the following 3 most important data science skills, generate 3 project ideas for a portfolio. Each project should integrate multiple skills from the list. Provide a project title and a short description for each idea.

    Skills: {skills_string}

    Format the output as a JSON array of objects, where each object has a "title" and a "description". Do not include any other text, explanations, or formatting outside of the JSON array.
    """
    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": { "type": "STRING" },
                    "description": { "type": "STRING" }
                }
            }
        }
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result and 'candidates' in result and result['candidates']:
            text_part = result['candidates'][0]['content']['parts'][0]['text']
            project_ideas = json.loads(text_part)
            return project_ideas
        else:
            return []
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        st.error(f"Error generating project ideas: {e}")
        return []

# --- 3. Streamlit App Layout and Logic ---

# Add a descriptive header for the user
st.header("AI-Powered Job Skill Extractor")
st.markdown("""
This app helps you extract key data science skills and their confidence scores from job descriptions. Simply enter a job title, company, and a detailed description, then click 'Add Job to List'. Once you have added at least one job, click 'Extract Skills' to see a structured list of skills and how confident the AI is about each one. You can then generate project ideas based on the extracted skills.
""")


# Initialize session state for storing job data and project ideas
if 'job_data' not in st.session_state:
    st.session_state.job_data = pd.DataFrame(columns=['Title', 'Company', 'Description'])
if 'project_ideas' not in st.session_state:
    st.session_state.project_ideas = None

with st.form("job_input_form"):
    st.write("### Add a New Job Posting")
    title = st.text_input("Job Title")
    company = st.text_input("Company Name")
    description = st.text_area("Job Description", height=200)
    submitted = st.form_submit_button("Add Job to List")

    if submitted and title and company and description:
        new_job = pd.DataFrame([{'Title': title, 'Company': company, 'Description': description}])
        st.session_state.job_data = pd.concat([st.session_state.job_data, new_job], ignore_index=True)
        st.success(f"Job posting for '{title}' added!")
        st.session_state.project_ideas = None # Reset ideas on new job added

st.write("---")
if not st.session_state.job_data.empty:
    st.write("### 📜 Current Job Postings")
    st.dataframe(st.session_state.job_data)

    if st.button("Extract Skills"):
        with st.spinner("Extracting skills from job descriptions..."):
            extracted_skills_list = []
            for index, row in st.session_state.job_data.iterrows():
                print(f"Processing job posting {index+1}: {row['Title']}")
                skills = extract_skills_with_gemini(row['Description'])
                extracted_skills_list.append(skills)
            
            st.session_state.job_data['Extracted_Skills_With_Scores'] = extracted_skills_list
        
        st.success("Skill extraction complete!")
        st.session_state.project_ideas = None # Reset ideas
        
    if 'Extracted_Skills_With_Scores' in st.session_state.job_data and not st.session_state.job_data.empty:
        st.write("---")
        st.write("### 📝 Extracted Skills")
        
        def format_skills(skills_list):
            if isinstance(skills_list, list):
                return ", ".join([f"{item['skill']} ({item['confidence_score']}%)" for item in skills_list])
            return str(skills_list)

        df_display = st.session_state.job_data.copy()
        df_display['Formatted_Skills'] = df_display['Extracted_Skills_With_Scores'].apply(format_skills)

        st.dataframe(df_display[['Title', 'Company', 'Formatted_Skills']])
        
        # --- NEW: Generate and display project ideas ---
        st.write("---")
        st.write("### 💡 Project Ideas")
        
        # Get all unique skills and calculate their average confidence score
        skill_scores = {}
        for skills_list in st.session_state.job_data['Extracted_Skills_With_Scores']:
            for skill_item in skills_list:
                skill = skill_item['skill']
                confidence = skill_item['confidence_score']
                if skill not in skill_scores:
                    skill_scores[skill] = {'total_score': 0, 'count': 0}
                skill_scores[skill]['total_score'] += confidence
                skill_scores[skill]['count'] += 1
        
        # Calculate average confidence and sort
        avg_scores = {skill: data['total_score'] / data['count'] for skill, data in skill_scores.items()}
        top_skills = sorted(avg_scores, key=avg_scores.get, reverse=True)[:3]
        
        if st.button("Generate Project Ideas"):
            if not top_skills:
                st.info("No skills extracted. Please add job postings and extract skills first.")
            else:
                with st.spinner("Generating project ideas..."):
                    st.session_state.project_ideas = generate_project_ideas(top_skills)
        
        if st.session_state.project_ideas:
            if st.session_state.project_ideas:
                for idea in st.session_state.project_ideas:
                    st.subheader(idea.get('title', 'No Title'))
                    st.write(idea.get('description', 'No description provided.'))
            else:
                st.info("No project ideas were generated. Please try again.")

