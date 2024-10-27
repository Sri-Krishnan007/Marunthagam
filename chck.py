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
from spellchecker import SpellChecker 
from nltk.corpus import wordnet
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googlesearch import search


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


kb_data = pd.read_csv('datasets/kb.csv')
medical_terms = list(set(kb_data['Symptoms'].dropna().tolist() + kb_data['Diseases'].dropna().tolist()))


class EnhancedMedicalPreprocessor:
    def __init__(self, diseases, symptoms, kb_path='datasets/kb.csv'):
        self.nlp = spacy.load('en_core_web_sm')
        self.diseases = [d.lower() for d in diseases]
        self.symptoms = [s.lower() for s in symptoms]
        
        # Load knowledge base
        self.kb_data = pd.read_csv(kb_path)
        self.medical_terms = list(set(
            self.kb_data['Symptoms'].dropna().tolist() + 
            self.kb_data['Diseases'].dropna().tolist()
        ))
        
        # Initialize TF-IDF vectorizer for similarity matching
        self.vectorizer = TfidfVectorizer()
        self.term_vectors = self.vectorizer.fit_transform(self.medical_terms)
        
        # Chunking grammar
        self.chunk_grammar = r"""
            Medical: {<JJ.*>*<NN.*>+}
            Symptom: {<VB.*>?<JJ.*>*<NN.*>+}
            Question: {<W.*>|<MD>}
            NegPhrase: {<RB>*<NOT>|<RB>*<DT>*<VB.*>}
        """
        self.chunk_parser = RegexpParser(self.chunk_grammar)
        
        # Keyword processor for exact matches
        self.keyword_processor = KeywordProcessor()
        self.keyword_processor.add_keywords_from_list(self.medical_terms)

    def find_similar_terms(self, term, threshold=0.5):
    # Optional: Use a spell checker to handle common misspellings
        spell = SpellChecker()
        corrected_term = spell.correction(term)
    
    # Now use the corrected term for further processing
        term_vector = self.vectorizer.transform([corrected_term])
        similarities = cosine_similarity(term_vector, self.term_vectors).flatten()
    
    # Get the top 5 similar terms based on cosine similarity
        similar_indices = similarities.argsort()[-5:][::-1]  
        best_match = None
    
        for idx in similar_indices:
            if similarities[idx] >= threshold:
                best_match = self.medical_terms[idx]
                break  # Stop after finding the first high-confidence match
    
        if best_match:
        # If a match was found with cosine similarity, return the corrected term
            return best_match
    
    # If no cosine similarity match, use Levenshtein distance for close matches
        leven_matches = get_close_matches(corrected_term, self.medical_terms, n=1, cutoff=0.7)
    
        if leven_matches:
        # Return the best Levenshtein match if found
            return leven_matches[0]
    
    # If no similar terms were found, return the original term
        return corrected_term


    def extract_medical_entities(self, text):
        """Extract medical entities using spaCy NER and custom rules"""
        doc = self.nlp(text)
        medical_entities = []
        
        # Extract entities from spaCy NER
        for ent in doc.ents:
            if ent.label_ in ['DISEASE', 'SYMPTOM']:
                medical_entities.append(ent.text)
        
        # Extract using keyword processor
        keywords = self.keyword_processor.extract_keywords(text)
        medical_entities.extend(keywords)
        
        # Custom rule-based extraction
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = self.chunk_parser.parse(pos_tags)
        
        for subtree in chunks.subtrees(filter=lambda t: t.label() in ['Medical', 'Symptom']):
            entity = ' '.join([word for word, tag in subtree.leaves()])
            medical_entities.append(entity)
        
        return list(set(medical_entities))

    def check_negation(self, text):
        """Check for negation in the text"""
        doc = self.nlp(text)
        for token in doc:
            if token.dep_ == 'neg':
                return True
        return False

    def detect_question_type(self, text):
        """Detect if the text is a question and its type"""
        doc = self.nlp(text)
        
        # Check for question words
        question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which'}
        first_word = doc[0].text.lower()
        
        if first_word in question_words or text.endswith('?'):
            # Analyze question intent
            if any(word.text.lower() in {'what', 'how'} for word in doc):
                return 'information'
            elif any(word.text.lower() in {'is', 'are', 'do', 'does'} for word in doc):
                return 'verification'
            return 'general'
        return None

    def process_query(self, query):
        """Main method to process medical queries"""
        # Normalize text
        normalized_query = query.lower().strip()
        
        # Extract medical entities
        medical_entities = self.extract_medical_entities(normalized_query)
        
        # Check for similar terms if no exact matches
        if not medical_entities:
            words = word_tokenize(normalized_query)
            for word in words:
                similar_terms = self.find_similar_terms(word)
                if similar_terms:
                    return {
                        'status': 'clarification_needed',
                        'similar_terms': similar_terms,
                        'original_term': word
                    }
        
        # Check for negation
        has_negation = self.check_negation(normalized_query)
        
        # Detect question type
        question_type = self.detect_question_type(normalized_query)
        
        # Determine intent
        intent = self.determine_intent(normalized_query, medical_entities)
        
        # Get sentiment
        sentiment = TextBlob(normalized_query).sentiment.polarity
        
        return {
            'status': 'processed',
            'intent': intent,
            'extracted_terms': medical_entities,
            'has_negation': has_negation,
            'question_type': question_type,
            'sentiment': sentiment
        }

    def determine_intent(self, text, extracted_terms):
        """Determine whether the query is about symptoms or disease"""
        symptom_indicators = {'symptom', 'symptoms', 'experiencing', 'feel', 'feeling', 'suffer', 'suffering'}
        disease_indicators = {'do i have', 'is this', 'diagnosed', 'disease', 'condition'}
        
        # Check for disease mentions
        disease_matches = [term for term in extracted_terms if term.lower() in self.diseases]
        symptom_matches = [term for term in extracted_terms if term.lower() in self.symptoms]
        
        if any(indicator in text for indicator in disease_indicators) or disease_matches:
            return 'disease'
        if any(indicator in text for indicator in symptom_indicators) or symptom_matches:
            return 'symptom'
        if disease_matches and not symptom_matches:
            return 'disease'
        return 'symptom'

    def get_knowledge_base_response(self, query_type, term):
    
        if query_type == 'information':
        # Check if the term exists in the medical knowledge base (Symptoms or Diseases)
            matches = self.kb_data[
            (self.kb_data['Symptoms'].str.contains(term, na=False, case=False)) |
            (self.kb_data['Diseases'].str.contains(term, na=False, case=False))
        ]
        
        # If the term exists in the knowledge base, fetch description from Google
            if not matches.empty:
                print(f"Fetching description from Google for: {term}")
            
            # Use Google Search to find the description of the term
                google_search_results = list(search(term + " medical description", num_results=1))
            
                if google_search_results:
                # Optionally, you can fetch content from the first search result link
                    return f"Description found on Google: {google_search_results[0]}"
                else:
                    return f"No Google results found for {term}."
        
        # If the term is not found in the kb_data
        return f"{term} not found in the medical knowledge base."
    
        return None





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
medical_preprocessor = EnhancedMedicalPreprocessor(diseases_list, symptoms_list)

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
    
    # Sidebar and logout functionality
    with st.sidebar:
        if st.button("Logout"):
            st.markdown(f'<meta http-equiv="refresh" content="0;url=http://127.0.0.1:5000/">', unsafe_allow_html=True)
    
    # Location handling
    st.sidebar.header("Enter your location")
    address = st.sidebar.text_input("Enter your address:")
    manual_entry = st.sidebar.checkbox("Enter coordinates manually")

    if address and not manual_entry:
        user_lat, user_lon = get_lat_lon(address)
    else:
        user_lat = st.sidebar.number_input("Enter your latitude:", value=10.0)
        user_lon = st.sidebar.number_input("Enter your longitude:", value=78.0)

    st.write(f"Latitude: {user_lat}, Longitude: {user_lon}")

    # User profile information from URL
    query_params = st.experimental_get_query_params()
    user_name = query_params.get("name", [""])[0]
    user_email = query_params.get("email", [""])[0]

    if user_name and user_email:
        st.write(f"**Name:** {user_name}")
        st.write(f"**Email:** {user_email}")
        
        # Query history in sidebar
    with st.sidebar:
        st.header("Your Query History")
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
        for query in st.session_state.query_history:
            st.text(f" {query}")
    
    # Initialize session states
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
    if 'clarification_needed' not in st.session_state:
        st.session_state.clarification_needed = False
    if 'suggested_terms' not in st.session_state:
        st.session_state.suggested_terms = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Voice input handling
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
    
    # Chat input handling
    placeholder_text = st.session_state.speech_text if st.session_state.speech_text else "Enter your symptoms or describe your condition:"
    query = st.chat_input(placeholder_text)
    
    if not query and st.session_state.speech_text:
        query = st.session_state.speech_text
        st.session_state.speech_text = ""
    
    # Process user input
    if query:
        # Add to query history
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
        st.session_state.query_history.append(query)
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process query using enhanced NLP
        processed_result = medical_preprocessor.process_query(query)
        
        # Handle similar term suggestions
        if processed_result['status'] == 'clarification_needed':
            st.session_state.clarification_needed = True
            st.session_state.suggested_terms = processed_result['similar_terms']
            clarification_msg = f"Did you mean one of these terms: {', '.join(processed_result['similar_terms'])}?"
            st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
            
            # Create buttons for suggested terms
            cols = st.columns(len(processed_result['similar_terms']))
            for idx, term in enumerate(processed_result['similar_terms']):
                with cols[idx]:
                    if st.button(term):
                        # Process the selected term
                        query = term
                        st.session_state.clarification_needed = False
                        st.experimental_rerun()
            return
        
        # Handle negation
        if processed_result['has_negation']:
            extracted_terms = processed_result['extracted_terms']
            if extracted_terms:
                positive_msg = f"Good news! Based on your description, you don't seem to have {', '.join(extracted_terms)}. However, if you're concerned, please consult a healthcare professional."
                st.session_state.messages.append({"role": "assistant", "content": positive_msg})
                return
        
        # Handle questions
        if processed_result['question_type']:
            kb_response = medical_preprocessor.get_knowledge_base_response(
                processed_result['question_type'],
                processed_result['extracted_terms'][0] if processed_result['extracted_terms'] else query
            )
            if kb_response:
                st.session_state.messages.append({"role": "assistant", "content": kb_response})
                
                # Add follow-up suggestion
                follow_up_msg = "Would you like to:\n1. Learn more about treatment options\n2. Find nearby specialists\n3. Get diet and lifestyle recommendations"
                st.session_state.messages.append({"role": "assistant", "content": follow_up_msg})
                return
        
        # Process regular symptom/disease query
        intent = processed_result['intent']
        extracted_terms = processed_result['extracted_terms']
        sentiment = processed_result['sentiment']
        
        # Add intent and extraction info to chat
        assistant_response = f"I detected: {', '.join(extracted_terms)}\nAnalysis type: {intent}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        if intent == 'symptom':
            # Predict disease and get severity information
            predicted_disease, selected_symptoms = predict_disease_from_query(query)
            
            # Add severity analysis
            severity_message = "### Symptom Analysis:\n"
            total_severity = 0
            symptom_count = 0
            
            for symptom in extracted_terms:
                try:
                    weight = symptoms_severity_df[symptoms_severity_df['Symptom'] == symptom]['weight'].values[0]
                    total_severity += weight
                    symptom_count += 1
                    color = get_color_for_severity(weight)
                    severity_message += f"<span style='color:{color}'>{symptom} (Severity: {weight})</span>\n"
                except IndexError:
                    severity_message += f"{symptom} (Severity information not available)\n"
            
            if symptom_count > 0:
                avg_severity = total_severity / symptom_count
                severity_message += f"\nOverall Severity Level: {avg_severity:.1f}/10"
                
            st.markdown(severity_message, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": severity_message})
            
            # Store disease information and show description
            st.session_state.current_disease = predicted_disease
            try:
                st.session_state.disease_info = get_disease_info(predicted_disease)
                description = st.session_state.disease_info[0]
                st.session_state.messages.append({"role": "assistant", "content": f"### Potential Condition\n{description}"})
            except Exception as e:
                st.error(f"Error retrieving disease information: {e}")
                
        elif intent == 'disease':
            disease = extracted_terms[0].lower() if extracted_terms else query.lower()
            try:
                st.session_state.current_disease = disease
                st.session_state.disease_info = get_disease_info(disease)
                description = st.session_state.disease_info[0]
                st.session_state.messages.append({"role": "assistant", "content": f"### Disease Information\n{description}"})
            except Exception as e:
                st.error(f"Error retrieving disease information: {e}")
        
        # Set show_options to True after processing
        st.session_state.show_options = True
    
    # Show options for additional information
    if st.session_state.show_options and st.session_state.current_disease:
        with st.form("options_form"):
            option = st.selectbox(
                "What would you like to know more about?",
                [
                    "Recommended Diet",
                    "Medications",
                    "Other Symptoms",
                    "Precautions",
                    "Suggested Workouts",
                    "Show recommended doctors",
                    "exit(to end)"
                ],
                key="option_select"
            )
            submitted = st.form_submit_button("Get Information")
            
        if submitted:
            if option != st.session_state.selected_option:
                st.session_state.selected_option = option
                
                if option == "exit(to end)":
                    st.session_state.show_options = False
                    st.session_state.current_disease = None
                    farewell_msg = "Thank you for using Marunthagam Medical Assistant. Take care!"
                    st.session_state.messages.append({"role": "assistant", "content": farewell_msg})
                    
                    # Feedback buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍"):
                            st.success("Thank you for your positive feedback!")
                    with col2:
                        if st.button("👎"):
                            st.error("We're sorry to hear that. Your feedback helps us improve!")
                
                elif option == "Show recommended doctors":
                    doctors = get_doctors_for_disease(st.session_state.current_disease)
                    
                    # Display doctor information
                    st.write("### Recommended Specialists")
                    st.dataframe(doctors[['Doctor\'s Name', 'Specialist', 'ADDRESS']])
                    
                    # Create and display map
                    m = folium.Map(location=[user_lat, user_lon], zoom_start=10)
                    
                    # Add user location marker
                    folium.Marker(
                        [user_lat, user_lon],
                        popup="Your Location",
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(m)
                    
                    # Add doctor markers and calculate travel times
                    for _, doctor in doctors.iterrows():
                        folium.Marker(
                            [doctor['LAT'], doctor['LON']],
                            popup=f"{doctor['Doctor\'s Name']} - {doctor['Specialist']}",
                            icon=folium.Icon(color="green", icon="plus")
                        ).add_to(m)
                        
                        travel_time = calculate_travel_time(user_lat, user_lon, doctor['LAT'], doctor['LON'])
                        st.write(f"Estimated travel time to {doctor['Doctor\'s Name']}: {travel_time}")
                    
                    folium_static(m)
                
                else:
                    # Get additional information based on selection
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
