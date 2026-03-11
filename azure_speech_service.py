import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

class AzureSpeechService:
    def __init__(self):
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.service_region = os.getenv("AZURE_SPEECH_REGION")
        
        if not self.speech_key or not self.service_region:
            print("Warning: Azure Speech keys not found in environment variables.")

        self.speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.service_region)
        self.speech_config.speech_recognition_language = "hi-IN"  # Default to Hindi
        
        # Set up voice for TTS
        # Madhur (Hindi-India) or Swara (Hindi-India) are common choices
        self.speech_config.speech_synthesis_voice_name = "hi-IN-MadhurNeural"

    def transcribe_from_file(self, audio_file_path):
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

        print("Transcribing with Azure...")
        result = speech_recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"Recognized: {result.text}")
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
        
        return ""

    def synthesize_to_file(self, text, output_file_path, language="hi-IN"):
        # Select voice based on language
        if language == "en-US":
            self.speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
        else:
            self.speech_config.speech_synthesis_voice_name = "hi-IN-MadhurNeural"

        audio_config = speechsdk.audio.AudioConfig(filename=output_file_path)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

        print(f"Synthesizing to {output_file_path}...")
        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized for text [{text}]")
            return True
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
        
        return False
