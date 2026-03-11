from azure_speech_service import AzureSpeechService
from grok_llm_service import GrokLLMService
from knowledge_base_module import KnowledgeBaseModule
from language_processor import LanguageProcessor

class AIReceptionist:
    def __init__(self):
        print("Initializing AI Receptionist...")
        self.speech_service = AzureSpeechService()
        self.llm_service = GrokLLMService()
        self.knowledge_base = KnowledgeBaseModule()
        self.lang_processor = LanguageProcessor(self.llm_service)
        
        self.system_prompt_base = (
            "You are a polite, helpful, and professional Hospital Receptionist at Etriq Multispeciality Hospital. "
            "Use the following information to answer user queries accurately. "
            "If the information is not in the context, politely say that you don't have it and ask them to contact the hospital management. "
            "Keep answers concise and actionable. "
            "\n\nContext Information:\n{context}"
        )

    def handle_voice_input(self, audio_file_path, session_history=None):
        # 1. Speech To Text (Azure)
        user_text = self.speech_service.transcribe_from_file(audio_file_path)
        if not user_text:
            return "Could not understand audio.", None

        return self.handle_text_input(user_text, session_history)

    def handle_text_input(self, user_text, session_history=None):
        if session_history is None:
            session_history = []

        # 2. Language Processing (Hindi -> English internal)
        original_query, processed_query, is_hindi = self.lang_processor.process_user_input(user_text)

        # 3. LLM Processing (Grok)
        context = self.knowledge_base.get_context()
        system_prompt = self.system_prompt_base.format(context=context)
        
        # If input was Hindi, we want response in Hindi too (usually)
        # But Grok can generate either. Let's see what it does.
        # We can explicitly ask for the same language if needed.
        if is_hindi:
            processed_query += " (Please respond in Hindi)"

        raw_response = self.llm_service.get_response(system_prompt, processed_query, session_history)

        # 4. Final Language Formatting
        final_response = self.lang_processor.format_response(raw_response, target_language_hindi=is_hindi)

        return user_text, final_response

    def generate_speech(self, text, output_path, is_hindi=True):
        lang = "hi-IN" if is_hindi else "en-US"
        return self.speech_service.synthesize_to_file(text, output_path, language=lang)
