# main.py
from syntaxx import MedicalSentenceCorrector

def main():
    corrector = MedicalSentenceCorrector()
    
    print("Welcome to the Medical Sentence Corrector!")
    print("Type your sentences below (type 'exit' to quit):")
    
    while True:
        # Get user input
        sentence = input("\nEnter a medical sentence: ")
        
        # Exit condition
        if sentence.lower() == 'exit':
            print("Exiting the Medical Sentence Corrector. Goodbye!")
            break
        
        # Correct the sentence
        corrected = corrector.correct_sentence(sentence)
        
        # Display the corrected sentence
        print(f"\nOriginal: {sentence}")
        print(f"Corrected: {corrected}")

if __name__ == "__main__":
    main()
