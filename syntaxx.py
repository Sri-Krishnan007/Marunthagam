class MedicalSentenceCorrector:
    def __init__(self):
        # Common patterns and their corrections
        self.patterns = {
            'present_continuous': {
                'patterns': [
                    (r'i am having (.+) from', r'I have been having \1 for'),
                    (r'i having (.+)', r'I am having \1'),
                    (r'i not sleeping', r'I am not sleeping'),
                    (r'i have not sleeping', r'I have not been sleeping'),
                    (r'i getting (.+)', r'I am getting \1'),
                ],
            },
            'symptoms': {
                'patterns': [
                    (r'having pain (.+)', r'having pain in \1'),
                    (r'got fever from (.+)', r'have had fever for \1'),
                    (r'having fever from', r'having fever for'),
                    (r'getting headache from', r'having headaches for'),
                ],
            },
            'duration': {
                'patterns': [
                    (r'from (\d+) days', r'for \1 days'),
                    (r'from yesterday', r'since yesterday'),
                    (r'from last (.+)', r'since last \1'),
                ],
            }
        }
        
        # Common medical symptoms and their proper forms
        self.symptoms = {
            'headache': 'headache',
            'stomachache': 'stomach ache',
            'fever': 'fever',
            'cold': 'cold',
            'caugh': 'cough',
            'throatpain': 'throat pain'
        }
        
        # Verb tense corrections
        self.verb_corrections = {
            'am having': 'have been having',
            'is having': 'has been having',
            'not eating': 'have not been eating',
            'not sleeping': 'have not been sleeping',
            'not working': 'has not been working'
        }

    def correct_basic_grammar(self, text):
        """Fix basic grammatical issues in the text"""
        text = text.lower().strip()
        
        # Capitalize first letter of sentences
        text = '. '.join(s.capitalize() for s in text.split('. '))
        
        # Capitalize 'I'
        text = text.replace(' i ', ' I ')
        if text.startswith('i '):
            text = 'I ' + text[2:]
            
        return text

    def correct_medical_terms(self, text):
        """Correct common medical terms and symptoms"""
        for incorrect, correct in self.symptoms.items():
            text = text.replace(incorrect, correct)
        return text

    def correct_verb_tense(self, text):
        """Correct verb tenses in the sentence"""
        for incorrect, correct in self.verb_corrections.items():
            if incorrect in text:
                text = text.replace(incorrect, correct)
        return text

    def apply_pattern_corrections(self, text):
        """Apply all pattern-based corrections"""
        import re
        
        for category in self.patterns.values():
            for pattern, replacement in category['patterns']:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def correct_sentence(self, text):
        """Main function to correct the entire sentence"""
        # Apply corrections in sequence
        text = self.correct_basic_grammar(text)
        text = self.correct_medical_terms(text)
        text = self.correct_verb_tense(text)
        text = self.apply_pattern_corrections(text)
        
        return text

# Example usage and testing
def main():
    corrector = MedicalSentenceCorrector()
    
    test_sentences = [
        "i have not sleeping",
        "i having headache from 2 days",
        "i getting fever from yesterday",
        "patient having stomachache from last week",
        "i am having cold from 3 days",
        "i got fever from morning",
        "having pain stomach since morning",
        "i not eating properly",
    ]
    
    print("Medical Sentence Grammar Correction Examples:")
    print("-" * 50)
    
    for sentence in test_sentences:
        corrected = corrector.correct_sentence(sentence)
        print(f"\nOriginal: {sentence}")
        print(f"Corrected: {corrected}")

if __name__ == "__main__":
    main()