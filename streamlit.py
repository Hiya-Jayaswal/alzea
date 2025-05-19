import streamlit as st
import json
import datetime
import os
import cohere
from fuzzywuzzy import fuzz
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import tempfile
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from PIL import Image

co = cohere.Client("key")
PROFILES_FILE = "alzea_profiles.json"
recognizer = sr.Recognizer()
model = load_model("CNN_model.h5")

# -------------------- Utilities -------------------- #

def ask_cohere(question):
    try:
        prompt = (
            "You are Alzea, a medical Alzheimer's AI. Respond to user queries with a medical lens. "
            "Even if non-medical, revert back to the topic and answer the question through an Alzheimer's lens. "
            "Do not deviate from the topic and also give advice based on it. "
            "Answer the user query, providing full but short and crisp information. " + question + " in 100 words."
        )
        response = co.chat(message=prompt, model="command-r-plus", temperature=0.5)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

def list_doctors_by_state(state):
    question = f"List 10 top doctors in {state} who treat Alzheimer's, with their hospital and city."
    return ask_cohere(question)

def predict_image_class(image_bytes):
    try:
        img = Image.open(image_bytes).resize((176, 176)).convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        class_names = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
        return class_names[np.argmax(prediction)]
    except Exception as e:
        return f"Prediction error: {str(e)}"

# -------------------- WebRTC Audio Recorder -------------------- #
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.recorded_frames.append(audio)
        return frame

def save_audio(frames, sample_rate=48000):
    audio_np = np.concatenate(frames, axis=1)[0]
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_segment = AudioSegment(
        audio_np.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )
    audio_segment.export(temp_wav.name, format="wav")
    return temp_wav.name

# -------------------- Streamlit UI -------------------- #

st.set_page_config(page_title="Alzea - Alzheimer's Assistant", layout="centered")
st.title("🧠 Alzea - Alzheimer's Symptom Assessment")

if "assessment_started" not in st.session_state:
    st.session_state["assessment_started"] = False

username = st.text_input("Enter your name")
input_mode = st.radio("Choose input method", ["Text", "Speech (record)"])
input_method = 'speech' if input_mode.startswith("Speech") else 'text'

profiles = load_profiles()

if username:
    if username not in profiles:
        st.info(f"Creating new profile for {username}")
        age = st.number_input("Enter your age", min_value=1, max_value=120)
        profiles[username] = {"name": username, "age": age, "assessment_history": []}
        save_profiles(profiles)
    else:
        st.success(f"Welcome back, {username}!")

    user_profile = profiles[username]

    st.subheader("📝 Alzheimer's Symptom Questionnaire")

    if st.button("Start Assessment"):
        st.session_state["assessment_started"] = True

    if st.session_state["assessment_started"]:
        symptoms = [
            "Do you experience memory loss?",
            "Have you gotten lost in familiar places?",
            "Do you struggle with daily tasks?",
            "Do you forget common words?",
            "Do you repeat questions or stories?",
            "Do you show mood swings or agitation?",
            "Do you resist change?",
            "Trouble recognizing friends or family?",
            "Experiencing delusions or paranoia?",
            "Trouble solving problems?"
        ]
        score = 0
        for symptom in symptoms:
            answer = st.radio(symptom, ["Yes", "No"], key=symptom)
            if answer == "Yes":
                score += 1

        percentage = (score / len(symptoms)) * 100
        if percentage >= 60:
            result = "You may have severe Alzheimer's symptoms."
        elif percentage >= 30:
            result = "You may have moderate symptoms."
        elif percentage >= 10:
            result = "You may have mild symptoms."
        else:
            result = "You likely do not show Alzheimer's symptoms."

        st.markdown(f"### Alzea says: {result}")
        st.info("This is not a medical diagnosis. Please consult a healthcare professional.")

    st.subheader("💬 Ask Alzea a Question")

    if input_method == 'text':
        user_query = st.text_input("Type your question")
        if st.button("Ask") and user_query:
            response = ask_cohere(user_query)
            st.markdown(f"**Alzea:** {response}")
    else:
        st.info("Click start to record your question")
        ctx = webrtc_streamer(
            key="speech",
            audio_receiver_size=256,
            desired_playing_state=True,
            sendback_audio=False,
            audio_processor_factory=AudioProcessor,
        )

        if ctx.audio_processor:
            if st.button("Process Speech"):
                audio_path = save_audio(ctx.audio_processor.recorded_frames)
                with sr.AudioFile(audio_path) as source:
                    audio = recognizer.record(source)
                    try:
                        query = recognizer.recognize_google(audio)
                        st.success(f"You said: {query}")
                        response = ask_cohere(query)
                        st.markdown(f"**Alzea:** {response}")
                    except sr.UnknownValueError:
                        st.error("Sorry, I couldn't understand what you said.")
                    except sr.RequestError:
                        st.error("Speech recognition service failed.")

    st.subheader("👨‍⚕️ Find Doctors by State")
    state = st.text_input("Enter Indian State")
    if st.button("Find Doctors") and state:
        doctors = list_doctors_by_state(state)
        st.markdown(doctors)

    st.subheader("🧠 Upload MRI Image for Classification")
    uploaded_image = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded MRI Image", use_container_width=True)
        result = predict_image_class(uploaded_image)
        st.success(f"Predicted Category: {result}")

else:
    st.info("Please enter your name to continue.")