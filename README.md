🤖 AI Career Coach for Data Science

https://huggingface.co/spaces/jiyachachan/ai_career_coach

🌟 **Overview**:
The AI Career Coach for Data Science is a full-stack generative AI application designed to help aspiring data professionals navigate their career paths. The tool analyzes real-world job descriptions, extracts key skills, and provides personalized portfolio project recommendations. This project serves as an end-to-end demonstration of a modern data science application, from data collection and model interaction to deployment in a cloud environment.


✨ **Features**


Job Description Analysis: Analyze raw job descriptions to identify and extract crucial technical skills.

Gemini API Integration: Leverages the power of the Google Gemini API with a few-shot learning prompt for highly accurate and consistent skill extraction.

Confidence Scoring: Each extracted skill is assigned a confidence score, which is used to identify the most important skills for a given role.

Project Idea Generation: Based on the top skills, the application generates relevant and creative project ideas to help users build a strong portfolio.

Interactive Web App: A user-friendly interface built with Streamlit allows for manual input of job descriptions and real-time visualization of results.




🛠️ **Technology Stack**


Language: Python

Web Framework: Streamlit

GenAI/LLM: Google Gemini API

Data Analysis: Pandas, Scikit-learn

API Interaction: Requests

Containerization: Docker

Deployment: Hugging Face Spaces 




🚀 **Getting Started**

Prerequisites: A Gemini API key

Local Setup
Clone the repository:

git clone https://github.com/JiyaChachan/ai_career_coach.git

cd ai-career-coach

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Configure your API key:
Create a .env file in the root directory and add your Gemini API key.

API_KEY="your-gemini-api-key-here"

Run the application:

streamlit run app.py

The app will be available at http://localhost:8501.

🐳 **Deployment**

This project is designed for seamless deployment using Docker.

Hugging Face Spaces
Push your code to a GitHub repository.

Create a new Hugging Face Space, selecting the Streamlit SDK.

Link your GitHub repository to the Space.

In the Space's settings, add a new secret named API_KEY and paste your Gemini API key.

Hugging Face will automatically build and deploy your app.

🤝 **Contributions**

Contributions, issues, and feature requests are welcome.

📄 **License**

This project is licensed under the MIT License.
