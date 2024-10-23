import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load the spacy model for NER and POS tagging
nlp = spacy.load('en_core_web_sm')

# Load the KB.csv file (Symptom and Disease data)
kb_data = pd.read_csv('datasets/kb.csv')

# Combine symptoms and disease into a single list
medical_terms = list(set(kb_data['Symptoms'].dropna().tolist() + kb_data['Diseases'].dropna().tolist()))

# Common words to ignore in matching
ignore_words = {'i', 'have', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

def extract_potential_terms(doc):
    """Extract potential medical terms using linguistic patterns"""
    potential_terms = set()  # Use set to avoid duplicates
    
    # Extract noun phrases that might be symptoms
    for chunk in doc.noun_chunks:
        # Skip chunks that only contain ignore words
        if not all(token.text.lower() in ignore_words for token in chunk):
            potential_terms.add(chunk.text)
    
    # Extract individual nouns and adjectives that might be symptoms
    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ'] and token.text.lower() not in ignore_words:
            potential_terms.add(token.text)
    
    return list(potential_terms)

def find_medical_terms(text, medical_terms, threshold=80):
    """Find distinct medical terms with improved fuzzy matching"""
    doc = nlp(text)
    potential_terms = extract_potential_terms(doc)
    found_terms = {}  # Use dictionary to track unique terms
    
    # First pass: Look for exact matches
    for term in potential_terms:
        term_lower = term.lower()
        for med_term in medical_terms:
            if term_lower == med_term.lower():
                found_terms[med_term] = {
                    'original': term,
                    'matched': med_term,
                    'score': 100,
                    'exact': True
                }
                break
    
    # Second pass: Look for fuzzy matches for terms that weren't exact matches
    remaining_terms = [term for term in potential_terms 
                      if not any(ft['original'].lower() == term.lower() 
                               for ft in found_terms.values())]
    
    for term in remaining_terms:
        if len(term.split()) >= 2:  # For multi-word terms
            min_score = threshold - 5
        else:
            min_score = threshold
        
        best_match, score = process.extractOne(term, medical_terms)
        if score >= min_score:
            # Only add if we don't already have this medical term
            if best_match not in found_terms:
                found_terms[best_match] = {
                    'original': term,
                    'matched': best_match,
                    'score': score,
                    'exact': False
                }
    
    return list(found_terms.values())

def process_medical_query(user_query):
    # Process the user query
    doc = nlp(user_query)
    
    # Detect medical terms
    found_terms = find_medical_terms(user_query, medical_terms)
    
    print("\n=== Detected Medical Terms ===")
    confirmed_terms = set()  # Use set to ensure uniqueness
    
    # Process exact matches first
    exact_matches = [term for term in found_terms if term['exact']]
    for term in exact_matches:
        confirmed_terms.add(term['matched'])
        print(f"Found exact match: {term['matched']}")
    
    # Then process fuzzy matches
    fuzzy_matches = [term for term in found_terms if not term['exact']]
    for term in fuzzy_matches:
        if (term['original'].lower() != term['matched'].lower() and  # Only ask if they're different
            term['matched'] not in confirmed_terms):  # And we haven't confirmed this term yet
            confirmation = input(f"Did you mean '{term['matched']}' for '{term['original']}'? (yes/no): ").strip().lower()
            if confirmation == 'yes':
                confirmed_terms.add(term['matched'])
    
    # Process sentences with confirmed terms
    sentences = list(doc.sents)
    results = []

    for sentence in sentences:
        sentence_doc = nlp(str(sentence))
        is_negated = any(token.dep_ == 'neg' for token in sentence_doc)
        
        for term in confirmed_terms:
            if is_negated:
                results.append(f"Don't worry, you are healthy (regarding '{term}').")
            else:
                results.append(f"I will tell you about '{term}'.")

    # Print results
    print("\n=== Query Results ===")
    if confirmed_terms:
        for result in sorted(set(results)):  # Sort and deduplicate results
            print(result)
    else:
        print("No medical terms found.")

# Example Usage
user_query = input("Enter your query: ")
process_medical_query(user_query)