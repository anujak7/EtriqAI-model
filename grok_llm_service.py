import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class GrokLLMService:
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")
        
        # Detecting if it's a Grok (x.ai) or Groq key
        if self.api_key and self.api_key.startswith("gsk_"):
            print("Groq key detected. Using Groq speed-optimized endpoint.")
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = "llama-3.3-70b-versatile" # Premium fast model for empathy
        else:
            self.base_url = "https://api.x.ai/v1"
            self.model = "grok-beta"
        
        if not self.api_key:
            print("Warning: API key not found in environment variables.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_response(self, system_prompt, user_query, history=None):
        if history is None:
            history = []
            
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_query})
        
        try:
            print("Calling Grok AI API...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Grok API: {e}")
            return "I am sorry, I am having trouble connecting to my brain right now. Please try again in a moment."

    def translate(self, text, target_language="English"):
        system_prompt = f"You are a professional translator. Translate the following text into {target_language}. Only provide the translated text without any extra comments."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during translation: {e}")
            return text
