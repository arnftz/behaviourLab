import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from logger import logger

load_dotenv()


class GeminiProvider:

    def __init__(self, api_key: str = None):
        logger.info("Initializing GeminiProvider...")

        self.api_key = os.environ.get("GENAI_API_KEY", api_key)
        self.client = genai.Client(api_key=self.api_key)

        self.max_retries = 3

    def generate(
        self,
        prompt: str,
        model: str = "gemma-4-31b-it",
        max_tokens: int = 100,
        temperature: float = 0.8,
    ) -> str:

        for attempt in range(1, self.max_retries + 1):

            try:

                logger.info(
                    f"Generating content (attempt {attempt}/{self.max_retries})..."
                )

                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        temperature=temperature,
                    ),
                )

                logger.info("Generation successful.")

                return response.text

            except Exception as e:

                if attempt == self.max_retries:

                    logger.error(
                        f"Generation failed after {self.max_retries} attempts."
                    )

                    raise

                wait_time = 2 ** (attempt - 1)

                logger.warning(
                    f"Generation failed ({e}). "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)