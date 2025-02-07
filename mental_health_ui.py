import streamlit as st
import joblib
import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

st.image("C:\\Users\\Dell\\OneDrive\\Desktop\\Assignment\\background.jpg", use_container_width=True)

if "ml_model" not in st.session_state:
    st.session_state.model = joblib.load("C:\\Users\\Dell\\OneDrive\\Desktop\\Assignment\\best_random_forest_model.pkl")

if "llm" not in st.session_state:
    st.session_state.llm = pipeline("text-generation", model="microsoft/phi-2", device=0 if device == "cuda" else -1)

feature_questions = {
    "self_employed_numeric": "Are you self-employed?",
    "family_history_numeric": "Do you have a family history of mental illness?",
    "age_group": "What is your age group?",
    "company_support_score": "How many mental health support benefits does your company provide?",
    "work_interfere": "If you have a mental health condition, how much does it interfere with work?",
    "no_employees": "How many employees does your company have?",
    "remote_work": "Do you work remotely at least 50% of the time?",
    "tech_company": "Is your employer primarily a tech company?",
    "anonymity": "Is your anonymity protected if you seek mental health treatment?",
    "leave": "How easy is it for you to take medical leave for mental health?",
    "mental_health_consequence": "Do you think discussing mental health at work has negative consequences?",
    "phys_health_consequence": "Do you think discussing physical health at work has negative consequences?",
    "coworkers": "Would you discuss mental health with coworkers?",
    "supervisor": "Would you discuss mental health with your supervisor?",
    "mental_health_interview": "Would you bring up mental health in a job interview?",
    "phys_health_interview": "Would you bring up physical health in a job interview?",
    "mental_vs_physical": "Does your employer take mental health as seriously as physical health?",
    "obs_consequence": "Have you observed negative consequences for coworkers with mental health conditions?"
}

if 'page' not in st.session_state:
    st.session_state.page = "welcome"
if 'responses' not in st.session_state:
    st.session_state.responses = {}
    st.session_state.responses_text = {}
if 'question_idx' not in st.session_state:
    st.session_state.question_idx = 0

def restart():
    st.session_state.page = "welcome"
    st.session_state.responses = {}
    st.session_state.responses_text = {}
    st.session_state.question_idx = 0
    st.session_state.clear()
    st.rerun()

if st.session_state.page == "welcome":
    st.title("Welcome to the Mental Health Risk Survey")
    st.write("This survey assesses potential mental health risks based on workplace conditions and personal history.")
    if st.button("Start Survey"):
        st.session_state.page = "survey"
        st.rerun()
    if st.button("Exit"):
        st.session_state.page = "thank_you"
        st.rerun()

elif st.session_state.page == "thank_you":
    st.title("Thank You")
    st.write("We appreciate your time. Have a great day!")
    if st.button("Restart"):
        restart()

elif st.session_state.page == "survey":
    questions = list(feature_questions.items())
    total_questions = len(questions)
    idx = st.session_state.question_idx
    feature, question = questions[idx]

    st.write(f"**Question {idx + 1} of {total_questions}**")

    if feature in ["self_employed_numeric", "family_history_numeric", "remote_work", "tech_company"]:
        options = ["Yes", "No"]
        response = st.radio(question, options)
        rf_response = 1 if response == "Yes" else 0 
    elif feature == "age_group":
        options = ["10-25", "26-40", "41-60", "61+"]
        response = st.selectbox(question, options, index=0)
        rf_response = options.index(response)
    elif feature == "leave":
        options = ["Very Difficult", "Difficult", "Normal", "Easy", "Very Easy"]
        response = st.selectbox(question, options, index=2)
        rf_response = options.index(response)
    elif feature == "company_support_score":
        response = st.slider(question, min_value=0, max_value=4, step=1)
        rf_response = response 
    elif feature == "no_employees":
        options = ["1-5", "6-25", "26-100", "100+"]
        response = st.selectbox(question, options, index=0)
        rf_response = options.index(response)
    elif feature == "anonymity":
        options = ["Yes", "No", "Not Sure"]
        response = st.radio(question, options)
        rf_response = options.index(response)
    else:
        options = ["Never", "Rarely", "Sometimes", "Often", "Always"]
        response = st.selectbox(question, options, index=2)
        rf_response = options.index(response)

    st.session_state.responses[feature] = rf_response  
    st.session_state.responses_text[feature] = response  

    col1, col2 = st.columns(2)
    with col1:
        if idx > 0 and st.button("Previous"):
            st.session_state.question_idx -= 1
            st.rerun()
    with col2:
        if idx < total_questions - 1 and st.button("Next"):
            st.session_state.question_idx += 1
            st.rerun()
        elif idx == total_questions - 1 and st.button("Submit"):
            st.session_state.page = "comments"
            st.rerun()

elif st.session_state.page == "comments":
    st.title("Additional Comments")
    comments = st.text_area("Any additional comments or concerns regarding mental health in the workplace? (Optional)")
    st.session_state.responses_text["comments"] = comments
    if st.button("Submit"):
        st.session_state.page = "results"
        st.rerun()

elif st.session_state.page == "results":
    st.title("Survey Results")
    input_df = pd.DataFrame([st.session_state.responses])
    
    if "comments" in input_df.columns:
        input_df = input_df.drop(columns=["comments"])

    prediction = st.session_state.model.predict(input_df)
    risk_status = "At Risk" if prediction[0] == 1 else "Not at Risk"
    
    user_answers = "\n".join([f"{feature_questions[key]}: {value}" for key, value in st.session_state.responses_text.items() if key in feature_questions])
    user_comments = st.session_state.responses_text.get("comments", "No additional comments provided.")

    explanation_prompt = (f"The user provided the following responses:\n{user_answers}\n\n"
                          f"Based on their responses, the predicted mental health risk status is: {risk_status}.\n\n"
                          f"User's additional comments: {user_comments}\n\n"
                          f"Give a short explanation about the user's mental health risks based on their responses, along with coping mechanisms and advice."
                          f"Do NOT repeat the user's responses verbatim."
                          f"Do NOT generate meaningless or repetitive text."
                          f"Keep the response concise and structured."
                        )

    response = st.session_state.llm(
        explanation_prompt, 
        max_new_tokens=150,
        pad_token_id=st.session_state.llm.tokenizer.eos_token_id
    )[0]['generated_text']

    llm_output = response.replace(explanation_prompt, "").strip()

    st.info(f"**Prediction: {risk_status}**")
    st.write("### Explanation & Recommendations")
    st.write(llm_output)

    if st.button("Restart"):
        restart()
