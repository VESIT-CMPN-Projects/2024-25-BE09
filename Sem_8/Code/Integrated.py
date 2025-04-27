

import os
import joblib
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from retinaface import RetinaFace  # Face detection
from deepface import DeepFace
from retinaface import RetinaFace  # Face detection
from deepface import DeepFace  # Age & gender prediction
from PIL import Image
from reportlab.pdfgen import canvas
import io
import google.generativeai as genai
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from PIL import Image
import pandas as pd

import streamlit as st
import cv2
import face_recognition
import pickle
from PIL import Image

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import base64
import torch.nn.functional as F


# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Crime Detection System", page_icon="🕵️", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Suspect Details","About","Face recog real time","Video summariser","OCR"])

if "report_data" not in st.session_state:
    st.session_state["report_data"] = {}

# ✅ Home Page
if page == "Home":
    st.title("🔍 Integrated Multimodal Crime Detection and Prediction System")
    st.write("An AI-powered system to analyze crime-related data from text, images, audio, and videos.")

    st.header("✨ Features")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📖 Crime Category Classification")
        st.write("Classifies crime descriptions into predefined categories using a *BERT model*.")

        st.subheader("👤 Suspect Identification")
        st.write("Detects *age, gender, and height* of suspects from images.")
    with col2:
        st.subheader("🔫 Weapon Detection")
        st.write("Identifies weapons in images using object detection models.")

        st.subheader("🎥 Violence Detection")
        st.write("Analyzes videos to detect violent activities using deep learning.")

    st.markdown("### 🚀 Get Started")
    st.write("Upload crime-related data for AI-based analysis.")
    st.button("Upload Data")

# ✅ Upload Data Page
elif page == "Upload Data":

    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>Crime Classification & Summarisation</h1>
        <h3 style='text-align: center; color: #555;'>Automatically classify crime reports into relevant categories using AI-powered models. Quickly generate concise summaries, highlighting key details such as crime type, location, and suspects, enabling faster analysis and decision-making.</h3>
        <hr style='border: 1px solid #ddd;'>
        """,
        unsafe_allow_html=True
    )
    checkpoint = "LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_path = "bert_finetuned.pth"
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    bert_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    bert_model.eval()

    def predict_crime_category(text):
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        category_labels = ["Drug Crimes", "Property Crimes", "Violent Crimes", "Traffic Offences", "Commercial Crimes", "Other Offences"]
        predicted_categories = [category_labels[i] for i in range(len(probabilities[0])) if probabilities[0][i] > 0.5]
        return predicted_categories

    def file_preprocessing(file):
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = ""
        for text in texts:
            final_texts += text.page_content
        return final_texts

    def llm_pipeline(uploaded_file):
        pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
        input_text = file_preprocessing(uploaded_file)
        result = pipe_sum(input_text)
        return result[0]['summary_text']

    @st.cache_data
    def displayPDF(file):
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    if uploaded_file is not None:
        if st.button("Summarize & Classify"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)
            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)
                crime_categories = predict_crime_category(summary)
                st.session_state["report_data"]["Crime Summary"] = summary
                st.session_state["report_data"]["Predicted Crime Categories"] = crime_categories
                st.info("Predicted Crime Categories")
                st.success(f"Crime Types: {', '.join(crime_categories)}")


# ✅ Suspect Details Page
elif page == "Suspect Details":

    # Header
    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>🎥 Suspect Profiling and Report Generation</h1>
        <h3 style='text-align: center; color: #555;'>Effortlessly analyze crime data and generate detailed reports using AI. Identify suspect profiles, extract key information, and simplify legal terms for clear, accurate reporting.</h3>
        <hr style='border: 1px solid #ddd;'>
        """,
        unsafe_allow_html=True
    )

    # Load weight and BMI models
    weight_model = joblib.load("model_weight.pkl")
    bmi_model = joblib.load("model_BMI.pkl")
        
    class HeightRegressor(nn.Module):
        def __init__(self, feature_extractor):
            super(HeightRegressor, self).__init__()
            self.feature_extractor = feature_extractor
            self.regressor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )

        def forward(self, x):
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1)
            return self.regressor(features)

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    resnet.fc = nn.Identity()
    model = HeightRegressor(resnet).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load("best_height_model.pth", map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Function to extract face encodings using face_recognition
    def get_face_encoding(image):
        image_np = np.array(image)
        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            return [], []
        encodings = face_recognition.face_encodings(image_np, face_locations)
        return encodings, face_locations

    # Function to predict weight and BMI from a face encoding
    def predict_weight_bmi(face_encoding):
        if len(face_encoding) == 0:
            return None, None
        test_array = np.expand_dims(np.array(face_encoding), axis=0)
        weight = np.exp(weight_model.predict(test_array)).item()
        bmi = np.exp(bmi_model.predict(test_array)).item()
        return round(weight, 2), round(bmi, 2)

    # Function to detect faces using Haar Cascade
    def detect_faces(image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_crops = []
        for (x, y, w, h) in faces:
            face = image.crop((x, y, x + w, y + h))
            face_crops.append((face, (x, y, w, h)))
        return face_crops, faces

    # Function to predict height from face image
    def predict_height(image, model, transform, device):
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_height = model(image).item()
        return round(predicted_height, 2)
    
    # age gender
    
    genai.configure(api_key="AIzaSyCt8Rua8K5aDv010hbWktFrvRX7XAA8MFQ")  # Replace with your actual API Key

    def categorize_age(age):
        if age < 13:
            return "Child"
        elif age < 25:
            return "Young"
        elif age < 45:
            return "Middle-aged"
        else:
            return "Old"

    def describe_image(image):
        """Generate a detailed description of the image using Gemini Vision"""
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Include weapon detection in the prompt
        prompt = """
        Describe this image with a focus on:
        - People (gender, age, attire, facial expressions)
        - Objects (vehicles, bags, etc.)
        - Crime-related elements (suspicious activities, illegal items)
        - Detect if there are any weapons (knives, guns, or other weapons) and specify the type if possible.
        """
        
        response = model.generate_content([prompt, image])
        return response.text


    def generate_summary(num_people, annotations, image_description,recognised_person_details):
        summary = f"""
        Crime Investigation Report - Image Analysis

        Total People Detected: {num_people}

        Subjects Identified:
        """
        for face_data in annotations:
                summary += f"""
        Person {face_data.get('Face')}:
        - Gender: {face_data.get('Gender')}
        - Age Group: {face_data.get('Age')}
        - Height (cm): {face_data.get('Height (cm)')}
        - Weight (kg): {face_data.get('Weight (kg)')}
        - BMI: {face_data.get('BMI')}
        """
                
        summary += f"""

        Person Identified:
        """
        for face_data in recognised_person_details:
                summary += f"""
        Person Name {face_data.get('Full Name')}:
        - Age: {face_data.get('Age')}
        - Gender: {face_data.get('Gender')}
        - Nationality: {face_data.get('Nationality')}
        - Crime History: {face_data.get('Crime History')}
        - Known Associates: {face_data.get('Known Associates')}
        - Criminal Status: {face_data.get('Criminal Status')}
        - Description: {face_data.get('Description')}
        """
        
        summary += f"""
        Image Description:
        {image_description}

        Conclusions Based on Analysis:
        - The presence of {num_people} individuals can help in identifying potential suspects or witnesses.
        - The estimated age and gender distribution may provide investigative leads.
        - Objects and surroundings in the image could offer additional crime-related context.
        """
        
        return summary
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet


    def create_pdf(summary):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()

        content = []
        for line in summary.strip().split("\n"):
            content.append(Paragraph(line.strip(), styles["BodyText"]))
            content.append(Spacer(1, 12))  # Add space between paragraphs

        doc.build(content)
        buffer.seek(0)
        return buffer

    def generate_pdf_report(num_people, annotations, summary):
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer)
        pdf.setTitle("Crime Investigation Report")
        
        pdf.drawString(100, 800, "Crime Investigation Report - AI Image Analysis")
        pdf.drawString(100, 780, f"Total People Detected: {num_people}")
        
        y_position = 750
        for face_data in annotations:
            pdf.drawString(100, y_position, f"Person {face_data.get('Face')}: Gender - {face_data.get('Gender')}, Age - {face_data.get('Age')}, Height-{face_data.get('Height (cm)')}, Weight- {face_data.get('Weight (kg)')}, BMI - {face_data.get('BMI')}")
            y_position -= 20
        
        pdf.drawString(100, y_position - 20, "AI-Generated Summary:")
        text_lines = summary.split("\n")
        y_position -= 40
        for line in text_lines:
            pdf.drawString(100, y_position, line)
            y_position -= 20
        
        pdf.save()
        buffer.seek(0)
        return buffer

    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca_model.pkl")
    le = joblib.load("label_encoder.pkl")
    svm_model = joblib.load("svm_classifier.pkl")

    # VGG-Face model
    def vgg_face():
        model = Sequential()
        model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        return model

    # Initialize VGG-Face model
    vgg_face_model = vgg_face()
    vgg_face_model.load_weights("vgg_face_model.h5")

    # Extract features from last fully connected layer
    vgg_face_descriptor = Model(inputs=vgg_face_model.input, outputs=vgg_face_model.layers[-2].output)

    # Function to process image
    def process_image(img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        embedding_vector = vgg_face_descriptor.predict(img)[0]
        return embedding_vector

    

    
    criminal_df = pd.read_csv("criminal_details.csv")
    uploaded_image = st.file_uploader("Upload an image of suspects", type=["jpg", "jpeg", "png"])
    annotations = []

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        # Detect faces in the image
        face_crops, faces = detect_faces(image)

        if not face_crops:
            st.warning("No faces detected. Please upload a clearer image.")
        else:
            st.markdown("### Predicted Measurements:")
            img_with_boxes = np.array(image)

            for i, (face, (x, y, w, h)) in enumerate(face_crops):
                # Predict height
                predicted_height = predict_height(face, model, transform, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                # Get face encoding for weight and BMI prediction
                encodings, _ = get_face_encoding(face)
                if encodings:
                    weight, bmi = predict_weight_bmi(encodings[0])
                else:
                    weight, bmi = None, None

                # Draw bounding box and label on image
                # cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # label_text = f"Face {i+1}"
                # cv2.putText(img_with_boxes, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # st.success(f"Face {i+1}: {predicted_height} cm, {weight} kg, BMI: {bmi}")

                # Initialize a dictionary for the face data
                face_data = {
                    "Face": i+1,
                    "Height (cm)": predicted_height,
                    "Weight (kg)": weight,
                    "BMI": bmi
                }
                annotations.append(face_data)

            # st.image(img_with_boxes, caption="Detected Faces with Measurements", channels="RGB", width=600)

        # Use another face detection method for age and gender analysis
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        faces_detected = RetinaFace.detect_faces(image_np)
        num_people = len(faces_detected) if faces_detected else 0

        # Loop through detected faces for age and gender analysis
        for i, key in enumerate(faces_detected.keys(), start=1):
            face_info = faces_detected[key]
            x1, y1, x2, y2 = face_info["facial_area"]
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Person {i}"
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            face_crop = image_bgr[y1:y2, x1:x2]

            if face_crop.size > 0:
                with st.spinner(f"Analyzing Person {i}..."):
                    predictions = DeepFace.analyze(face_crop, actions=["age", "gender"], enforce_detection=False)
                    # If you already have an annotation for this face from the previous loop,
                    # merge the age and gender data.
                    age = predictions[0]["age"]
                    gender = predictions[0]["dominant_gender"].capitalize()

                    # Update the corresponding annotation if it exists
                    # Here we assume that the order of faces in both detections is the same.
                    if i-1 < len(annotations):
                        annotations[i-1].update({
                            "Age": age,
                            "Gender": gender
                        })
                    else:
                        # If not, create a new annotation entry
                        annotations.append({
                            "Face": i,
                            "Age": age,
                            "Gender": gender
                        })

        st.image(image_np, caption="Processed Image", width=500)
        st.subheader(f"👥 Number of People Detected: {num_people}")

        image_fr = Image.open(uploaded_image) 
        recognised_person_details=[]
        # Preprocess and predict
        try:
            embedding_vector = process_image(image_fr)
            
            # Apply preprocessing
            X_test_std = scaler.transform([embedding_vector])
            X_test_pca = pca.transform(X_test_std)

            # Predict with SVM
            y_pred = svm_model.predict(X_test_pca)
            predicted_name = le.inverse_transform(y_pred)[0]

            # Display result

            predicted_name = predicted_name.replace('pins_', '')

            criminal_record = criminal_df[criminal_df['Full Name'].str.lower() == predicted_name.lower()]



            if not criminal_record.empty:

                for index, row in criminal_record.iterrows():
                        # If not, create a new annotation entry
                            recognised_person_details.append({
                                "Full Name": row['Full Name'],
                                "Alias": row['Alias'],
                                "Age": row['Age'],
                                "Gender": row['Gender'],
                                "Nationality": row['Nationality'],
                                "Crime History": row['Crime History'],
                                "Known Associates": row['Known Associates'],
                                "Criminal Status": row['Criminal Status'],
                                "Description": row['Description'],
                            })
            else:
                st.warning("⚠️ No details found for the recognized person.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

        # Display all annotations together
        if annotations:
            st.success("✅ Analysis Complete!")
            st.write("### Details of the Detected Faces:")
            # Option 1: Display as individual cards
            for face_data in annotations:
                st.markdown(
                    f"<div style='padding:10px; border-radius:10px; background-color:#f0f2f6; margin-bottom:10px;'>"
                    f"<b>Face {face_data.get('Face')}</b><br>"
                    f"<b>Height:</b> {face_data.get('Height (cm)', 'N/A')} cm<br>"
                    f"<b>Weight:</b> {face_data.get('Weight (kg)', 'N/A')} kg<br>"
                    f"<b>BMI:</b> {face_data.get('BMI', 'N/A')}<br>"
                    f"<b>Age:</b> {face_data.get('Age', 'N/A')}<br>"
                    f"<b>Gender:</b> {face_data.get('Gender', 'N/A')}<br>"
                    
                    "</div>", unsafe_allow_html=True)
            
            
            if recognised_person_details:
                recognised_names = ", ".join([face_detail.get('Full Name', 'N/A') for face_detail in recognised_person_details])
    
                # Display the recognised person's name
                st.markdown(f"<h3 style='color:green;'>✅ Person Recognised: {recognised_names}</h3>", unsafe_allow_html=True)

                st.markdown("<h4>🔍 Details of the Person:</h4>", unsafe_allow_html=True)

                for face_detail in recognised_person_details:
                    st.markdown(
                        f"<div style='padding:15px; border-radius:10px; background-color:#f0f2f6; margin-bottom:15px;'>"
                        f"<b>Full Name:</b> {face_detail.get('Full Name', 'N/A')}<br>"
                        f"<b>Alias:</b> {face_detail.get('Alias', 'N/A')}<br>"
                        f"<b>Age:</b> {face_detail.get('Age', 'N/A')}<br>"
                        f"<b>Gender:</b> {face_detail.get('Gender', 'N/A')}<br>"
                        f"<b>Nationality:</b> {face_detail.get('Nationality', 'N/A')}<br>"
                        f"<b>Crime History:</b> {face_detail.get('Crime History', 'N/A')}<br>"
                        f"<b>Known Associates:</b> {face_detail.get('Known Associates', 'N/A')}<br>"
                        f"<b>Criminal Status:</b> {face_detail.get('Criminal Status', 'N/A')}<br>"
                        f"<b>Description:</b> {face_detail.get('Description', 'N/A')}<br>"
                        "</div>", unsafe_allow_html=True
                    )
            else:
                st.markdown("<h3 style='color:red;'>❌ No Person Recognised.</h3>", unsafe_allow_html=True)
                            

            # Option 2: Display as a table using pandas
            # import pandas as pd
            # df = pd.DataFrame(annotations)
            # st.table(df)

            image_description = describe_image(image)
            summary = generate_summary(num_people, annotations, image_description,recognised_person_details)
            st.write("### 📜 AI-Generated Crime Investigation Report:")
            st.info(summary)

            # pdf_buffer = generate_pdf_report(num_people, annotations, summary)
            pdf_buffer=create_pdf(summary)

            st.download_button(label="📄 Download Report", data=pdf_buffer, file_name="crime_investigation_report.pdf", mime="application/pdf")
        else:
            st.warning("No faces detected.")


        st.session_state["report_data"]["Suspect Summary"] = summary
    


elif page=="Face recog real time":
    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>🎥 Real Time Face Recognition of Suspects</h1>
        <h3 style='text-align: center; color: #555;'>Instantly identify suspects using AI-powered face recognition, enhancing security with accurate, real-time detection and verification.</h3>
        <hr style='border: 1px solid #ddd;'>
        """,
        unsafe_allow_html=True
    )

    # Load the CSV and face encodings
    csv_file = "crime_record.csv"
    df = pd.read_csv(csv_file)

    # Load face encodings from the pickle file
    with open("encodings.pkl", "rb") as f:
        image_encodings = pickle.load(f)

    # Set to track detected criminals
    detected_criminals = set()

    # Function to get criminal details by name
    def get_criminal_details(name):
        result = df[df['name'].str.contains(name, case=False, na=False)]
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    # Display criminal details in Streamlit
    def display_criminal_info(name):
        criminal_info = get_criminal_details(name)
        
        if criminal_info and name not in detected_criminals:
            st.markdown(f"### 🛑 **Criminal Detected: {name}**")
            st.write(f"📌 **Crime:** {criminal_info['crime']}")
            st.write(f"⚠️ **Threat Level:** {criminal_info['threat_level']}")
            st.write(f"📍 **Location:** {criminal_info['location']}")
            st.write(f"🧑‍🤝‍🧑 **Gender:** {criminal_info['gender']}")
            st.write(f"🎂 **Age:** {criminal_info['age']}")
            st.write(f"🔗 **Known Associates:** {criminal_info['known_associates']}")

            # Display the criminal's image
            image_path = criminal_info['image_path']
            try:
                image = Image.open(image_path)
                st.image(image, caption=f"{name}", width=300)
            except Exception as e:
                st.warning(f"❌ Image not found for {name}.")
            
            # Add the criminal to the detected set to prevent repeated display
            detected_criminals.add(name)

    # Real-time face recognition
    def real_time_recognition(image_encodings):
        stframe = st.empty()
        video_capture = cv2.VideoCapture(0)

        # Sidebar stop button
        st.sidebar.header("Control Panel")
        stop = st.sidebar.button("Stop Recognition")

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_loc in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(image_encodings["encodings"], face_encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = image_encodings["names"][match_index]

                    # Draw rectangle around detected face
                    top, right, bottom, left = face_loc
                    color = (0, 255, 0) if get_criminal_details(name) else (255, 0, 0)
                    label = f"{name} (Detected)" if get_criminal_details(name) else "Unknown"
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Display details only once
                    if get_criminal_details(name) and name not in detected_criminals:
                        display_criminal_info(name)

            stframe.image(frame, channels="BGR")

            # Stop recognition
            if stop:
                break

        video_capture.release()


    # Add criminal to CSV
    def add_criminal_to_csv(name, crime, threat_level, location, gender, age, associates, image_path):
        new_entry = pd.DataFrame([[name, crime, threat_level, location, gender, age, associates, image_path]], 
                                columns=["name", "crime", "threat_level", "location", "gender", "age", "known_associates", "image_path"])
        new_entry.to_csv(csv_file, mode='a', header=False, index=False)
        st.success(f"✅ Added {name} to the database!")


    # Display options
    option = st.selectbox("Choose an action", ["Real-Time Recognition", "Add Criminal"])

    if option == "Real-Time Recognition":
        st.write("Starting Real-Time Recognition...")
        real_time_recognition(image_encodings)

    elif option == "Add Criminal":
        with st.form("add_criminal_form"):
            name = st.text_input("Name")
            gender = st.selectbox("Gender", ["male", "female", "other"])
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            crime = st.text_input("Crime")
            threat_level = st.selectbox("Threat Level", ["low", "medium", "high"])
            location = st.text_input("Last Known Location")
            associates = st.text_input("Known Associates")
            image_path = st.text_input("Image Path")

            if st.form_submit_button("Add Criminal"):
                add_criminal_to_csv(name, crime, threat_level, location, gender, age, associates, image_path)

elif page=="Video summariser":
    import streamlit as st
    from phi.agent import Agent
    from phi.model.google import Gemini
    from phi.tools.duckduckgo import DuckDuckGo
    from google.generativeai import upload_file, get_file
    import google.generativeai as genai
    import time
    from pathlib import Path
    import tempfile
    from dotenv import load_dotenv
    import os
    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if API_KEY:
        genai.configure(api_key=API_KEY)

   

    # Header
    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>🎥 AI-Powered Video Summarizer</h1>
        <h3 style='text-align: center; color: #555;'>Powered by Gemini 2.0 Flash Exp</h3>
        <hr style='border: 1px solid #ddd;'>
        """,
        unsafe_allow_html=True
    )

    @st.cache_resource
    def initialize_agent():
        return Agent(
            name="Video AI Summarizer",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    # Initialize the agent
    multimodal_Agent = initialize_agent()

    # File uploader
    st.markdown("### 📤 Upload a Video File")
    video_file = st.file_uploader(
        "Supported formats: MP4, MOV, AVI",
        type=['mp4', 'mov', 'avi'],
        help="Upload a video for AI analysis"
    )

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)
        st.markdown("---")

        if st.button("🔍 Analyze Video", key="analyze_video_main"):
            with st.spinner("Processing video and gathering insights..."):
                try:
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                    
                    analysis_prompt = """
                    Analyze the uploaded video for content and context.
                    Respond to the following queries using video insights and supplementary web research:
                    - 📌 Is there any violent activity in the video?
                    - ⏳ Timeframes where violence occurs.
                    - 🔫 Are there any weapons detected?
                    Provide a detailed, user-friendly, and actionable response.
                    """
                    
                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])
                    
                    st.markdown("## 📊 Analysis Report")
                    st.markdown("""
    <div style='padding: 15px; border-radius: 10px; background-color: #F8F9FA;'>
        <p style='font-size: 18px; color: #333;'>
            {}
        </p>
    </div>
""".format(response.content.replace('\n', '<br>')), unsafe_allow_html=True)


                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    Path(video_path).unlink(missing_ok=True)

        st.markdown("## 🎯 Additional Analysis")
        user_query = st.text_area(
            "", placeholder="Ask anything about the video content...",
            help="Provide specific questions or insights you want from the video."
        )
        
        if st.button("🔍 Get Insights", key="analyze_video_query"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                with st.spinner("Analyzing your query..."):
                    try:
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)
                        analysis_prompt = f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}
                        Provide a detailed, user-friendly, and actionable response.
                        """
                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])
                        st.subheader("Analysis Result")
                        st.markdown(response.content)
                        st.session_state["report_data"]["video"] = response.content
                    except Exception as error:
                        st.error(f"An error occurred: {error}")

    else:
        st.info("📂 Upload a video file to begin analysis.")

    # Custom styles
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 120px;
            font-size: 16px;
        }
        div[data-testid="stSidebar"] {
            background-color: #F5F7FA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif page=="OCR":
    import google.generativeai as genai
    import streamlit as st
    from PIL import Image
    import io

    # Configure the Gemini model (Ensure you have API access)
    API_KEY = "AIzaSyBvsLvsVb4kZT_uqKysW5KmOM8unLUIM-E"
    genai.configure(api_key=API_KEY)

    # Function to perform OCR using Gemini
    def gemini_ocr(image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([image, "Extract text from this image."])
            return response.text
        except Exception as e:
            return str(e)

    # Function for Q&A chatbot
    def gemini_chat(context, question):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Based on the following FIR text, answer the question:\n\n{context}\n\nQuestion: {question}"
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return str(e)

    # Streamlit UI
    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>FIR Report Extraction & Q&A Chatbot</h1>
        <h3 style='text-align: center; color: #555;'>Efficiently extract and analyze information from FIR reports using OCR technology. The integrated Q&A chatbot allows you to quickly retrieve details, answer queries, and summarize key points, making crime investigation faster and more effective.</h3>
        <hr style='border: 1px solid #ddd;'>
        """,
        unsafe_allow_html=True
    )


    uploaded_files = st.file_uploader("Upload FIR Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    extracted_text = ""

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            image_bytes = uploaded_file.read()
            text = gemini_ocr(image_bytes)
            extracted_text += text + "\n\n"
        
        st.subheader("Extracted Text from FIR:")
        st.text_area("", extracted_text, height=200)

    # Q&A Chatbot Section
    st.subheader("Ask Questions about the FIR Report")
    user_question = st.text_input("Enter your question:")

    if user_question and extracted_text:
        answer = gemini_chat(extracted_text, user_question)
        st.write("### Answer:")
        st.write(answer)
    print(st.session_state["report_data"])
  
