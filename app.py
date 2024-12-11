import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.title("Face & Emotion Detection")
st.write("Capture or upload a photo for emotion detection.")

# Haar Cascade for face detection
# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load model
model = tf.keras.models.load_model("emotion_model.h5")


# Emotion prediction function
def predict_emotion(face_image):
    face_resized = cv2.resize(face_image, (48, 48))
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_normalized = face_gray / 255.0
    face_array = np.expand_dims(face_normalized, axis=0)
    face_array = np.expand_dims(face_array, axis=-1)
    emotion_prediction = model.predict(face_array)
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    emotion_percentages = {emotion: round(prob * 100, 2) for emotion, prob in
                           zip(emotion_labels, emotion_prediction[0])}
    return emotion_percentages


# Emotion colors
emotion_colors = {
    "Angry": "red",
    "Disgust": "green",
    "Fear": "#ff4d4d",
    "Happy": "yellow",
    "Neutral": "pink",
    "Sad": "blue",
    "Surprise": "purple"
}


# Detect faces and analyze
def detect_faces_and_analyze(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    feedbacks = []
    for idx, (x, y, w, h) in enumerate(faces):
        face = image_array[y:y + h, x:x + w]
        emotion_percentages = predict_emotion(face)
        emotion_label = max(emotion_percentages, key=emotion_percentages.get)

        # Only show non-zero percentages
        filtered_emotions = {emotion: percentage for emotion, percentage in emotion_percentages.items() if
                             percentage > 0}
        filtered_colors = [emotion_colors[emotion] for emotion in filtered_emotions.keys()]

        # Draw rectangle and label
        cv2.rectangle(image_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label_pos = (x, y - 10 if y - 10 > 10 else y + h + 10)
        cv2.putText(image_array, f"{emotion_label} - Face {idx + 1}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (36, 255, 12), 3)


        # Sidebar for feedback
        st.sidebar.write(f"Face {idx + 1} - Emotion Analysis:")
        for emotion, percentage in filtered_emotions.items():
            st.sidebar.write(f"{emotion}: {percentage}%")

        # Pie chart and color labels
        st.write("### Analysis Results for Each Face")
        col1, col2 = st.columns([3, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(
                filtered_emotions.values(),
                startangle=90,
                colors=filtered_colors
            )
            ax.axis('equal')
            st.pyplot(fig)

        with col2:
            st.subheader("Emotion Color Mapping")
            for emotion, color in emotion_colors.items():
                st.markdown(
                    f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
                    f"<div style='width: 20px; height: 20px; background-color:{color}; margin-right: 10px;'></div>"
                    f"<span style='font-size: 16px;'>{emotion}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Feedback section
        st.subheader("Feedback on the Analysis")
        feedback = st.radio(f"Is the analysis correct for Face {idx + 1}?", ("Yes", "No"), key=f"feedback_{idx}")

        correct_emotion = None
        if feedback == "No":
            correct_emotion = st.selectbox(
                "Please select the correct emotion or specify if it's not a face.",
                options=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "This is not a face"],
                key=f"correct_emotion_{idx}"
            )

        if st.button("Submit Feedback", key=f"submit_{idx}"):
            feedback_text = f"Thank you for your feedback! {'Corrected emotion: ' + correct_emotion if correct_emotion else 'Confirmed as correct.'}"
            st.success(feedback_text)
            feedbacks.append((idx + 1, feedback, correct_emotion))

    return image_array, feedbacks


# Capture or upload image
st.write("Capture photo from camera:")
img_file_buffer = st.camera_input("")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Display captured or uploaded image first, then analyses below
if img_file_buffer:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)
    st.image(img_array, caption="Captured Photo", use_column_width=True)
    result_img, feedbacks = detect_faces_and_analyze(img_array)

if uploaded_file:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    st.image(img_array, caption="Uploaded Photo", use_column_width=True)
    result_img, feedbacks = detect_faces_and_analyze(img_array)
