from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re

# Define question keywords related to medical queries
QUESTION_WORDS = ['describe', 'define', 'what is', 'how is', 'what is like', 'tell me about', 'explain']

def is_medical_query(user_input):
    """Check if the user input contains a valid medical-related question."""
    for word in QUESTION_WORDS:
        if re.search(r'\b' + word + r'\b', user_input.lower()):
            return True
    return False

def extract_medical_term(user_input):
    """Extract possible medical term from the user input by removing question words."""
    for word in QUESTION_WORDS:
        user_input = re.sub(r'\b' + word + r'\b', '', user_input.lower())
    return user_input.strip()

def google_search(query):
    """Fetch top Google search results for the query."""
    try:
        for result in search(query, num_results=1):
            return result  # Return the top result
    except Exception as e:
        return f"Error occurred while searching: {str(e)}"

def extract_text_from_url(url):
    """Extract the main paragraph from the URL."""
    try:
        # Send HTTP request to the page
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Parse the page content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text from <p> (paragraph) tags
            paragraphs = soup.find_all('p')

            # Filter out short or irrelevant paragraphs
            for para in paragraphs:
                text = para.get_text()
                if len(text) > 300:  # Return the first meaningful paragraph
                    return text.strip()
        else:
            return "Unable to retrieve content from the page."
    except Exception as e:
        return f"Error occurred while extracting content: {str(e)}"

def handle_query():
    """Prompt user for input and fetch relevant paragraph."""
    user_input = input("Please enter your medical query: ")

    if is_medical_query(user_input):
        medical_term = extract_medical_term(user_input)

        # Perform a Google search for the medical term and get the top URL
        url = google_search(medical_term)
        if url:
            # Extract and return the text content from the URL
            paragraph = extract_text_from_url(url)
            if paragraph:
                print(paragraph)  # Directly output the extracted paragraph
            else:
                print("No relevant content found.")
        else:
            print("No results found.")
    else:
        print("This is not a recognized medical question.")

# Run the system
handle_query()