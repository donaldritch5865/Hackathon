import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

import cv2
import streamlit as st
from deepface import DeepFace
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model and vectorizer for text analysis
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

port_stem = PorterStemmer()

# Emotion detection function
def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30))
    results = []

    for (x, y, w, h) in faces:
        padding = 2
        x, y, w, h = x - padding, y - padding, w + 2 * padding, h + 2 * padding
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            # Detect multiple emotions from the face
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']  # Returns a dictionary of emotions and their respective probabilities
            
            # Sorting emotions by probability to get the top emotions
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            results.append((x, y, w, h, sorted_emotions))
        except:
            continue
    return results

# Check if the webcam is available
def is_webcam_available(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False
    cap.release()
    return True

# Text processing for sentiment analysis
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="wide", page_icon="🎥")

# Title and Description
st.title("🎥 Sentiment Analysis")
st.markdown("""
Welcome to the Sentiment Analysis! 
This app uses advanced AI to help you analyze emotions during interviews. 
Select a mode below to get started.
""")

# State for webcam control
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# Cards Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Emotion Detection")
    st.markdown("Analyze emotions in real-time through your webcam!")
    
    if st.button("Start Interview"):
        st.session_state.webcam_active = True

    if st.session_state.webcam_active:
        st.markdown("#### Live Camera Feed")
        
        # Try default camera (index 0), if not available, try other indices
        webcam_available = False
        for index in range(0, 3):  # Try indices 0 to 2
            if is_webcam_available(index):
                cap = cv2.VideoCapture(index)
                webcam_available = True
                break
        
        if not webcam_available:
            st.error("Webcam not available. Please check your camera and try again.")
        else:
            stframe = st.empty()
            stop_button = st.button("Stop Webcam")

            while st.session_state.webcam_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam.")
                    break

                # Detect emotions in the frame
                results = detect_emotions(frame)

                # Draw results on the frame and display multiple emotions
                for (x, y, w, h, sorted_emotions) in results:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    y_offset = y - 10  # Start drawing emotions above the face

                    # Display the top emotions
                    for emotion, prob in sorted_emotions:
                        cv2.putText(frame, f"{emotion}: {prob:.2f}", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        y_offset -= 30  # Move down for the next emotion

                # Display frame in Streamlit
                stframe.image(frame, channels="BGR")

                # Stop webcam if button is clicked
                if stop_button:
                    st.session_state.webcam_active = False

            cap.release()

with col2:
    st.markdown("### Text Analysis")
    st.markdown("Analyze the sentiment of text-based responses.")
    
    if "text_analysis_active" not in st.session_state:
        st.session_state.text_analysis_active = False

    if st.button("Start Text Analysis"):
        st.session_state.text_analysis_active = True

    if st.session_state.text_analysis_active:
        st.title("Twitter Sentiment Analysis")
        st.write("This application predicts the sentiment of a given text as Positive or Negative.")

        # Input text
        user_input = st.text_area("Enter the text for sentiment analysis:")

        if st.button("Analyze Sentiment"):
            if user_input:
                processed_text = stemming(user_input)
                transformed_text = vectorizer.transform([processed_text])  # Use the loaded vectorizer
                prediction = model.predict(transformed_text)

                if prediction[0] == 0:
                    st.error("The sentiment is Negative.")
                else:
                    st.success("The sentiment is Positive.")
            else:
                st.warning("Please enter some text for analysis.")
        if st.button("Close Text Analysis"):
            st.session_state.text_analysis_active = False

with col3:
    st.markdown("### Voice Emotion Detection")
    st.markdown("Analyze emotions from voice data (Coming Soon!).")
    st.button("Start Voice Analysis")
