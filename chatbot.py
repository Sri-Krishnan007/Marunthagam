import streamlit as st
import pandas as pd
import numpy as np
import pickle
import spacy
import nltk
import urllib.parse
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
from nltk.tag import pos_tag
from flashtext import KeywordProcessor
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from flask import session
from datetime import datetime
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import json

def read_csv_with_encoding(file_path, encodings=['utf-8', 'iso-8859-1', 'cp1252']):
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the provided encodings: {encodings}")

input_data = read_csv_with_encoding(r"datasets\Training.csv")
with open(r"models\gbm_model.pkl", "rb") as file:
    gbm = pickle.load(file)

# Load all other datasets
symptoms_severity_df = read_csv_with_encoding(r"datasets\Symptom-severity.csv")
description_df = read_csv_with_encoding(r"datasets\description.csv")
diets_df = read_csv_with_encoding(r"datasets\diets.csv")
medications_df = read_csv_with_encoding(r"datasets\medications.csv")
symptoms_df = read_csv_with_encoding(r"datasets\symtoms_df.csv")
precautions_df = read_csv_with_encoding(r"datasets\precautions_df.csv")
workouts_df = read_csv_with_encoding(r"datasets\workout_df.csv")
doctors_df = read_csv_with_encoding(r"datasets\doc.csv")
# Setup features and initialize keyword processor
features = input_data.columns[:-1]
prognosis = input_data['prognosis']
feature_dict = {feature: idx for idx, feature in enumerate(features)}
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(features.tolist())


# Initialize Medical Preprocessor
class MedicalPreprocessor:
    def __init__(self, diseases, symptoms):
        self.nlp = spacy.load('en_core_web_sm')
        self.diseases = [d.lower() for d in diseases]
        self.symptoms = [s.lower() for s in symptoms]
        self.chunk_grammar = r"""
            Medical: {<JJ.*>*<NN.*>+}
            Symptom: {<VB.*>?<JJ.*>*<NN.*>+}
        """
        self.chunk_parser = RegexpParser(self.chunk_grammar)

    def normalize_text(self, text):
        return text.lower().strip()

    def find_matches(self, text):
        text = self.normalize_text(text)
        disease_matches = [d for d in self.diseases if d in text]
        symptom_matches = [s for s in self.symptoms if s in text]
        return disease_matches, symptom_matches

    def determine_intent(self, text, disease_matches, symptom_matches):
        text = self.normalize_text(text)
        symptom_indicators = {'symptom', 'symptoms', 'experiencing', 'feel', 'feeling', 'suffer', 'suffering'}
        disease_indicators = {'do i have', 'is this', 'diagnosed', 'disease', 'condition'}
        
        if any(indicator in text for indicator in disease_indicators) and disease_matches:
            return 'disease'
        if any(indicator in text for indicator in symptom_indicators):
            return 'symptom'
        if disease_matches and not symptom_matches:
            return 'disease'
        return 'symptom'

    def process_query(self, query):
        normalized_query = self.normalize_text(query)
        disease_matches, symptom_matches = self.find_matches(normalized_query)
        intent = self.determine_intent(normalized_query, disease_matches, symptom_matches)
        extracted = disease_matches + symptom_matches
        
        if "symptoms of" in normalized_query and disease_matches:
            extracted = disease_matches
            intent = "symptom"
        
        return {'intent': intent, 'extracted': extracted}

# Helper functions from original code
def predict_disease_from_query(query):
    # Extract keywords (symptoms) from the query
    selected_symptoms = keyword_processor.extract_keywords(query)
    
    # Initialize a sample array with zeros for all features
    sample_x = np.zeros(len(features))
    
    # Map the selected symptoms to the corresponding feature index
    for symptom in selected_symptoms:
        if symptom in feature_dict:
            sample_x[feature_dict[symptom]] = 1  # Set the corresponding feature value to 1
    
    # Reshape the input to match the expected shape for the model
    sample_x = sample_x.reshape(1, -1)
    
    # Use the model to predict the disease
    predicted_result = gbm.predict(sample_x)[0]
    
    # Check if the prediction is a disease name or an index
    if isinstance(predicted_result, str):
        # If it's a string, assume it's the disease name
        predicted_disease = predicted_result
    else:
        # If it's an integer index, look up the disease from the prognosis DataFrame
        predicted_index = int(predicted_result)
        try:
            predicted_disease = prognosis.iloc[predicted_index]
        except IndexError:
            # Handle the case where the predicted index is out of bounds
            predicted_disease = "Unknown Disease"  # Or any default value or handling
    
    # Return the predicted disease and the selected symptoms
    return predicted_disease, selected_symptoms



def get_distinct_symptoms(disease):
    disease_symptoms = symptoms_df[symptoms_df['Disease'].str.lower() == disease.lower()]
    if disease_symptoms.empty:
        print(f"No symptoms found for disease: '{disease}'")
        return []
    symptom_columns = disease_symptoms.columns[1:]
    all_symptoms = disease_symptoms[symptom_columns].values.flatten()
    distinct_symptoms = list(set([str(symptom).strip() for symptom in all_symptoms if pd.notna(symptom)]))
    return distinct_symptoms

def get_disease_info(disease):
    print(f"Searching for disease: '{disease}'")
    print(f"Available diseases: {description_df['Disease'].unique()}")
    
    # Convert to lowercase for case-insensitive matching
    disease_lower = disease.lower()
    
    # Find the disease in the dataframe (case-insensitive)
    disease_row = description_df[description_df['Disease'].str.lower() == disease_lower]
    
    if disease_row.empty:
        print(f"Disease '{disease}' not found in description_df")
        raise ValueError(f"No information found for disease '{disease}'")
    
    description = disease_row['Description'].values[0]
    
    # Similar case-insensitive matching for other dataframes
    diet = diets_df[diets_df['Disease'].str.lower() == disease_lower]['Diet'].values[0]
    medication = medications_df[medications_df['Disease'].str.lower() == disease_lower]['Medication'].values[0]
    
    distinct_symptoms = get_distinct_symptoms(disease)
    
    precautions = precautions_df[precautions_df['Disease'].str.lower() == disease_lower].drop('Disease', axis=1).values.flatten()
    precautions = list(set([str(precaution) for precaution in precautions if pd.notna(precaution)]))
    
    workouts = workouts_df[workouts_df['disease'].str.lower() == disease_lower]['workout'].values
    workouts = list(set([str(workout) for workout in workouts if pd.notna(workout)]))
    
    return description, diet, medication, distinct_symptoms, precautions, workouts

def get_color_for_severity(weight):
    if weight <= 2: return 'green'
    elif weight <= 4: return 'yellow'
    else: return 'red'

diseases_list = prognosis.unique().tolist()
symptoms_list = features.tolist()
medical_preprocessor = MedicalPreprocessor(diseases_list, symptoms_list)

def get_doctors_for_disease(disease):
    # Normalize the input disease name
    disease_lower = disease.lower().strip()
    
    # First, try an exact match
    doctors = doctors_df[doctors_df['Disease'].str.lower().str.strip() == disease_lower]
    
    # If no results, try a partial match
    if doctors.empty:
        doctors = doctors_df[doctors_df['Disease'].str.lower().str.strip().str.contains(disease_lower)]
    
    # If still no results, try matching individual words
    if doctors.empty:
        disease_words = set(disease_lower.split())
        doctors = doctors_df[doctors_df['Disease'].str.lower().str.strip().apply(lambda x: set(x.split()).intersection(disease_words))]
    
    # Print debugging information
    print(f"Searching for disease: '{disease_lower}'")
    print(f"Available diseases in doctors_df: {doctors_df['Disease'].str.lower().str.strip().unique()}")
    print(f"Number of doctors found: {len(doctors)}")
    
    return doctors

# New function to calculate travel time (simplified version)
def calculate_travel_time(start_lat, start_lon, end_lat, end_lon):
    distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
    # Assuming an average speed of 40 km/h
    time_hours = distance / 40
    return f"{time_hours:.2f} hours"

def get_lat_lon(address):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        st.error("Could not find the location. Please try another address.")
        return None, None

def main():
    st.title("Marunthagam Medical Assistant")
    st.title("User Profile")
    with st.sidebar:
        if st.button("Logout"):
            st.markdown(f'<meta http-equiv="refresh" content="0;url=http://127.0.0.1:5000/">', unsafe_allow_html=True)
    
    st.sidebar.header("Enter your location")

    # Input field for the address
    address = st.sidebar.text_input("Enter your address:")

    # Option to enter manually if needed
    manual_entry = st.sidebar.checkbox("Enter coordinates manually")

    # If the user enters an address, geocode it to get lat and lon
    if address and not manual_entry:
        user_lat, user_lon = get_lat_lon(address)
    else:
    # If manual input is preferred or no address, let them enter lat/lon
        user_lat = st.sidebar.number_input("Enter your latitude:", value=10.0)
        user_lon = st.sidebar.number_input("Enter your longitude:", value=78.0)

    st.write(f"Latitude: {user_lat}, Longitude: {user_lon}")
    

    # Get query parameters from the URL
    query_params = st.experimental_get_query_params()
    user_name = query_params.get("name", [""])[0]
    user_email = query_params.get("email", [""])[0]

    if user_name and user_email:
        st.write(f"**Name:** {user_name}")
        st.write(f"**Email:** {user_email}")
        
        # Display history in sidebar
        with st.sidebar:
            st.header("Your Query History")
            if "query_history" not in st.session_state:
                st.session_state.query_history = []
            
            # Display history from session state
            for query in st.session_state.query_history:
                st.text(f" {query}")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None
    if 'disease_info' not in st.session_state:
        st.session_state.disease_info = None
    if 'show_options' not in st.session_state:
        st.session_state.show_options = False
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
    
    # Get user location
    
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if 'speech_text' not in st.session_state:
        st.session_state.speech_text = ""
    
    if st.button("🎤 Click to Speak"):
        try:
            with st.spinner("Listening..."):
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("Listening... Speak now!")
                    audio = r.listen(source, timeout=5)
                    try:
                        text = r.recognize_google(audio)
                        st.session_state.speech_text = text
                        st.success(f"Recognized: {text}")
                    except sr.UnknownValueError:
                        st.error("Could not understand audio. Please try again.")
                    except sr.RequestError as e:
                        st.error(f"Could not request results from speech recognition service; {e}")
        except Exception as e:
            st.error(f"Error accessing microphone: {e}")
            
    placeholder_text = st.session_state.speech_text if st.session_state.speech_text else "Enter your symptoms or describe your condition:"
    query = st.chat_input(placeholder_text)        
    
    if not query and st.session_state.speech_text:
        query = st.session_state.speech_text
        st.session_state.speech_text = "" 
    # Chat input
    if query :
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
        st.session_state.query_history.append(query)
        # Reset options when a new query is entered
        st.session_state.show_options = False
        st.session_state.current_disease = None
        st.session_state.selected_option = None
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process query using NLP
        processed_result = medical_preprocessor.process_query(query)
        intent = processed_result['intent']
        extracted_terms = processed_result['extracted']
        
        # Add assistant response to chat history
        assistant_response = f"Detected Intent: {intent}\nExtracted Terms: {', '.join(extracted_terms)}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        if intent == 'symptom':
            # Predict the disease from the query
            predicted_disease, selected_symptoms = predict_disease_from_query(query)
            disease_message = f"Predicted Disease: {predicted_disease}"
            st.session_state.messages.append({"role": "assistant", "content": disease_message})
            
            # Store current disease and its information
            st.session_state.current_disease = predicted_disease
            st.session_state.disease_info = get_disease_info(predicted_disease)
            
            # Display symptom severities
            severity_message = "### Symptom Severities:\n"
            for symptom in extracted_terms:
                try:
                    weight = symptoms_severity_df[symptoms_severity_df['Symptom'] == symptom]['weight'].values[0]
                    color = get_color_for_severity(weight)
                    severity_message += f"<span style='color:{color}'>{symptom} (Severity: {weight})</span>\n"
                except IndexError:
                    severity_message += f"Severity information not found for symptom: {symptom}\n"
            st.markdown(severity_message, unsafe_allow_html=True)        
            st.session_state.messages.append({"role": "assistant", "content": severity_message})
            
            # Display disease description
            description = st.session_state.disease_info[0]
            st.session_state.messages.append({"role": "assistant", "content": f"### Description\n{description}"})
        
        elif intent == 'disease':
            disease = extracted_terms[0].lower() if extracted_terms else query.lower()
            try:
                st.session_state.current_disease = disease
                st.session_state.disease_info = get_disease_info(disease)
                description = st.session_state.disease_info[0]
                st.session_state.messages.append({"role": "assistant", "content": f"### Description\n{description}"})
            except IndexError:
                error_message = f"No information found for disease '{disease}'"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        # Set show_options to True after processing the query
        st.session_state.show_options = True
    
    # Options for further information
    if st.session_state.show_options and st.session_state.current_disease:
        with st.form("options_form"):
            option = st.selectbox(
            "Select an option for more information:",
            ["Recommended Diet", "Medications", "Other Symptoms", "Precautions", "Suggested Workouts", "Show recommended doctors", "exit(to end)"],
            key="option_select"
        )
            submitted = st.form_submit_button("Enter")
        if submitted:
            if option != st.session_state.selected_option:
                st.session_state.selected_option = option
            
                if option == "exit(to end)":
                    st.session_state.show_options = False
                    st.session_state.current_disease = None
                    st.session_state.messages.append({"role": "assistant", "content": "Thank you for using Marunthagam Medical Assistant. Goodbye!"})
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍"):
                            st.success("Thank you for your positive feedback!")
                        # Here you could save the feedback to a database
                    with col2:
                        if st.button("👎"):
                            st.error("We're sorry to hear that. Your feedback helps us improve!")
                elif option == "Show recommended doctors":
                    doctors = get_doctors_for_disease(st.session_state.current_disease)
                    st.write("### Recommended Doctors")
                    st.dataframe(doctors[['Doctor\'s Name', 'Specialist', 'ADDRESS']])
                
                    m = folium.Map(location=[user_lat, user_lon], zoom_start=10)
                    folium.Marker(
                        [user_lat, user_lon],
                        popup="Your Location",
                        icon=folium.Icon(color="red", icon="info-sign"),
                    ).add_to(m)
                
                    for _, doctor in doctors.iterrows():
                        folium.Marker(
                            [doctor['LAT'], doctor['LON']],
                            popup=f"{doctor['Doctor\'s Name']} - {doctor['Specialist']}",
                            icon=folium.Icon(color="green", icon="plus"),
                        ).add_to(m)
                    
                        travel_time = calculate_travel_time(user_lat, user_lon, doctor['LAT'], doctor['LON'])
                        st.write(f"Travel time to {doctor['Doctor\'s Name']}: {travel_time}")
                
                    folium_static(m)
                else:
                    info_index = {
                    "Recommended Diet": 1,
                    "Medications": 2,
                    "Other Symptoms": 3,
                    "Precautions": 4,
                    "Suggested Workouts": 5
                    }
                    info = st.session_state.disease_info[info_index[option]]
                    if isinstance(info, str):
                        info = eval(info)
                    info_message = f"### {option}\n" + ", ".join(info)
                    st.session_state.messages.append({"role": "assistant", "content": info_message})
                
                
        

if __name__ == "__main__":
    main()
    
    
    
#nlp tech, voice, response,logout  