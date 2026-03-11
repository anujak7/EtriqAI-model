import re

class LanguageProcessor:
    def __init__(self, llm_service):
        self.llm = llm_service

    def is_hindi(self, text):
        # Using unicode range for Devanagari (Hindi)
        return bool(re.search('[\u0900-\u097F]', text))

    def process_user_input(self, text):
        """
        Translates Hindi input to English for internal processing if needed.
        Returns (original_text, processed_text, is_hindi)
        """
        is_hi = self.is_hindi(text)
        if is_hi:
            print("Hindi detected. Translating to English for internal use...")
            processed_text = self.llm.translate(text, target_language="English")
            return text, processed_text, True
        return text, text, False

    def format_response(self, response_text, target_language_hindi=True):
        """
        Ensures the response is in the desired language.
        If the response from LLM is English but we need Hindi, it translates.
        """
        is_hi = self.is_hindi(response_text)
        
        if target_language_hindi and not is_hi:
            print("Translating response to Hindi...")
            return self.llm.translate(response_text, target_language="Hindi")
        elif not target_language_hindi and is_hi:
            print("Translating response to English...")
            return self.llm.translate(response_text, target_language="English")
            
        return response_text
