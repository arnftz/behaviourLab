from pathlib import Path
import json
from provider.gemini import GeminiProvider
from logger import logger


class BenchmarkBuilder:

    def __init__(self, provider):
        logger.info("Initializing BenchmarkBuilder...")
        self.provider = provider
        self.root = Path(__file__).parent

    def load_seeds(self, behaviour: str) -> list[str]:
        logger.info(f"Loading seeds for '{behaviour}'...")

        seed_path = (
            self.root
            / "behaviours"
            / behaviour
            / "seeds.json"
        )

        with open(seed_path, "r", encoding='utf-8') as f:
            seeds = json.load(f)
        return seeds
    
    def generate_variations(self, behaviour: str, seeds: list[str], num_variations: int = 10, batch: int = 5, max_tokens: int = 1000) -> list[str]:
        logger.info(f"Generating variations for '{behaviour}' with {len(seeds)} seeds...")
        variations = []
        for i in range(0, len(seeds), batch):
            logger.info(f"Processing batch {i // batch + 1} for '{behaviour}'...")
            seed_batch_text = ""
            seed_batch = seeds[i:i+batch]
            for seed in seed_batch:
                seed_batch_text += f"\n{seed}"

            prompt_path = (
                Path(__file__).parent
                / "prompts"
                / "benchmark_builder.md"
            )   

            definition_path = (
                Path(__file__).parent
                / "behaviours"
                / behaviour
                / "definition.md"
            )

            definition = definition_path.read_text(encoding="utf-8")

            template = prompt_path.read_text(encoding="utf-8")

            prompt = (
                template
                .replace("{BEHAVIOUR_NAME}", behaviour)
                .replace("{SEED_PROMPT}", seed_batch_text)
                .replace("{NUM_VARIATIONS}", str(num_variations))
                .replace("{BEHAVIOUR_DEFINITION}", definition)
            )

            logger.debug(f"Prompt for batch {i // batch + 1}:\n{prompt}\n")
            response = self.provider.generate(prompt, model="gemma-4-31b-it", max_tokens=max_tokens, temperature=0.8)
            logger.debug(f"Response for batch {i // batch + 1}:\n{response}\n")

            validated_response = self.validate_response(response)

            variations.extend(validated_response)   

        return variations

    def validate_response(self, response: str) -> list[dict]:
        logger.info("Validating response...")
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON")

        if not isinstance(data, list):
            raise ValueError("Response must be a list.")

        for i, item in enumerate(data):

            if not isinstance(item, dict):
                raise ValueError(f"Item {i} must be an object.")

            if "prompt" not in item:
                raise ValueError(f"Item {i} missing 'prompt'.")

            if not isinstance(item["prompt"], str):
                raise ValueError(f"Item {i} prompt must be a string.")

            if not item["prompt"].strip():
                raise ValueError(f"Item {i} prompt is empty.")

        return data
    
    def save_variations(self, behaviour: str, variations: list[dict]) -> None:
        logger.info(f"Saving variations for '{behaviour}'...")
        output_path = (
            self.root
            / "behaviours"
            / behaviour
            / "benchmark.json"
        )

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(variations, f, indent=2, ensure_ascii=False)

    def run(self, num_variations: int = 10, batch: int = 5, max_tokens: int = 1000) -> None:

        logger.info("Run Started - num_variations: %d, batch: %d, max_tokens: %d", num_variations, batch, max_tokens)

        behaviours_path = self.root / "behaviours"
        behaviours = [
            d.name
            for d in behaviours_path.iterdir()
            if d.is_dir()
        ]

        logger.info(
            f"Found behaviours: {behaviours}"
        )
        for behaviour_dir in behaviours_path.iterdir():

            if not behaviour_dir.is_dir():
                continue

            behaviour = behaviour_dir.name

            seeds = self.load_seeds(behaviour)

            logger.debug(f"Seeds for '{behaviour}': {seeds}")
            variations = self.generate_variations(
                behaviour=behaviour,
                seeds=seeds,
                num_variations=num_variations,
                batch=batch,
                max_tokens=max_tokens
            )

            self.save_variations(
                behaviour,
                variations
            )

            logger.info(f"✓ Saved {len(variations)} prompts for '{behaviour}'.")

