from transformers import pipeline
import torch

from utils.logger import logger


class HuggingFaceProvider:

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    ):
        logger.info(f"Loading model: {model_name}")

        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        logger.info("Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:

        logger.info("Generating response...")

        outputs = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False
        )

        return outputs[0]["generated_text"].strip()