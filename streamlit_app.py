import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# Streamlit app
st.title("🧠 AI Text to Personality Predictor")    #display app title
st.write("Enter a paragraph about yourself and see your predicted personality type!")    #display app description to guide user
st.write("This app uses AI to analyze your text and predict your personality type based on the MBTI model.")

user_input = st.text_area("📝 Your Text Here:")    #creates a text box input    

if st.button("Predict Personality"):
    if user_input.strip() == "":     # Check if input is empty
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(user_input)   #clean user's input
        vect_text = vectorizer.transform([cleaned])    #convert cleaned text to vector using the loaded vectorizer
        # Make prediction using the loaded model
        prediction = model.predict(vect_text)[0]
        
        st.success(f"🎯 Predicted Personality Type: **{prediction}**")   #display prediction result to user
        
        # Display personality explanation
        personality_info = {
            "INTJ": "Architect: Imaginative and strategic thinkers.",
            "INFP": "Mediator: Poetic, kind and altruistic.",
            "ENTP": "Debater: Smart and curious thinkers.",
            "ISFJ": "Defender: Very dedicated and warm protectors.",
            "ESTJ": "Executive: Excellent administrators.",
            "ESFJ": "Consul: Extraordinarily caring, social and popular.",
            "INTP": "Logician: Innovative inventors with an unquenchable thirst for knowledge.",
            "ENFP": "Campaigner: Enthusiastic, creative and sociable free spirits.",
            "INFJ": "Advocate: Quiet and mystical, yet very inspiring and tireless idealists.",
            "ISFP": "Adventurer: Flexible and charming artists.",
            "ESTP": "Entrepreneur: Smart, energetic and very perceptive people.",
            "ESFP": "Entertainer: Spontaneous, energetic and enthusiastic people.",
            "ISTJ": "Logistician: Practical and fact-minded individuals.",
            #can add more personality types and their descriptions here
        }
        if prediction in personality_info:
            st.info(personality_info[prediction])

