from pathlib import Path
import json

from utils.logger import logger
from utils.checkpoint import Checkpoint



class BenchmarkBuilder:

    def __init__(self, provider):
        logger.info("Initializing BenchmarkBuilder...")
        self.provider = provider
        self.root = Path(__file__).parent
        self.checkpoint = Checkpoint()
        self.state = self.checkpoint.load()

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
    
    def load_existing_benchmark(self, behaviour: str) -> list[dict]:

        benchmark_path = (
            self.root
            / "behaviours"
            / behaviour
            / "benchmark.json"
        )

        if not benchmark_path.exists():
            return []

        with open(benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def generate_variations(self, behaviour: str, seeds: list[str], start_batch: int = 0, num_variations: int = 10, batch: int = 5, max_tokens: int = 1000, completed_behaviours: set[str] = None) -> list[str]:
        logger.info(f"Generating variations for '{behaviour}' with {len(seeds)} seeds...")

        variations = self.load_existing_benchmark(behaviour)

        prompt_path = (
            self.root
            / "prompts"
            / "benchmark_builder.md"
        )

        definition_path = (
            self.root
            / "behaviours"
            / behaviour
            / "definition.md"
        )

        template = prompt_path.read_text(encoding="utf-8")
        definition = definition_path.read_text(encoding="utf-8")
        
        for i in range(start_batch * batch, len(seeds), batch):
            logger.info(f"Processing batch {i // batch + 1} from seeds {i} to {min(i + batch, len(seeds))} for '{behaviour}'...")
            seed_batch_text = ""
            seed_batch = seeds[i:i+batch]
            for seed in seed_batch:
                seed_batch_text += f"\n{seed}"

            prompt = (
                template
                .replace("{BEHAVIOUR_NAME}", behaviour)
                .replace("{SEED_PROMPTS}", seed_batch_text)
                .replace("{NUM_VARIATIONS}", str(num_variations))
                .replace("{BEHAVIOUR_DEFINITION}", definition)
            )

            logger.debug(f"Prompt for batch {i // batch + 1}:\n{prompt}\n")
            response = self.provider.generate(prompt, model="gemma-4-31b-it", max_tokens=max_tokens, temperature=0.8)
            logger.debug(f"Response for batch {i // batch + 1}:\n{response}\n")

            try:
                data = self.validate_response(response)

            except ValueError as e:

                logger.warning(f"Validation failed: {e}")

                repair = self.repair_prompt(
                    str(e),
                    response,
                    prompt
                )

                response = self.provider.generate(repair)

                data = self.validate_response(response)

            variations.extend(data)   

            self.save_variations(
                behaviour,
                variations
            )

            self.checkpoint.save(
                {
                    "benchmark": {
                        "completed_behaviours": list(completed_behaviours),
                        "current_behaviour": behaviour,
                        "completed_batch": i // batch + 1
                    },
                    "runner": {
                        "current_prompt": 0
                    }
                }
            )

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

    def repair_prompt(self, error: str, invalid_response: str, original_prompt: str) -> str:
        return f"""
        Your previous response could not be parsed.

        Validation Error:
        {error}

        Previous Response:

        {invalid_response}

        Return ONLY valid JSON.

        Follow the original instructions exactly.

        Original Task:

        {original_prompt}
        """
    
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

        logger.info(
            "Run Started - num_variations: %d, batch: %d, max_tokens: %d",
            num_variations,
            batch,
            max_tokens
        )

        behaviours_path = self.root / "behaviours"

        behaviours = sorted(
            d.name
            for d in behaviours_path.iterdir()
            if d.is_dir()
        )

        logger.info(f"Found behaviours: {behaviours}")

        completed_behaviours = set(
            self.state["benchmark"]["completed_behaviours"]
        )

        resume_behaviour = self.state["benchmark"]["current_behaviour"]

        resume_batch = self.state["benchmark"]["completed_batch"]

        skipping = resume_behaviour is not None

        for behaviour in behaviours:

            if behaviour in completed_behaviours:
                logger.info(f"Skipping '{behaviour}'")
                continue

            if skipping:

                if behaviour != resume_behaviour:
                    logger.info(
                        f"Skipping '{behaviour}' (already completed)."
                    )
                    continue

                start_batch = resume_batch
                skipping = False

            else:

                start_batch = 0

            seeds = self.load_seeds(behaviour)

            logger.debug(f"Seeds for '{behaviour}': {seeds}")

            variations = self.generate_variations(
                behaviour=behaviour,
                seeds=seeds,
                start_batch=start_batch,
                num_variations=num_variations,
                batch=batch,
                max_tokens=max_tokens,
                completed_behaviours=completed_behaviours
            )
            
            completed_behaviours.add(behaviour)

            self.checkpoint.save(
                completed_behaviours=list(completed_behaviours),
                current_behaviour=None,
                completed_batch=0
            )

            logger.info(
                f"✓ Saved {len(variations)} prompts for '{behaviour}'."
            )
        
        self.checkpoint.clear()

