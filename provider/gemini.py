import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from logger import logger

class GeminiProvider:
    def __init__(self, api_key: str = None):
        logger.info("Initializing GeminiProvider...")
        self.api_key = os.environ.get("GENAI_API_KEY", api_key)
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, model: str = "gemma-4-31b-it", max_tokens: int = 100, temperature: float = 0.8) -> str:
        logger.info("Generating content...")
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                temperature=temperature,
            )
        )
        return response.text
    