from pathlib import Path
import json

from utils.logger import logger
from utils.checkpoint import Checkpoint


class BenchmarkRunner:

    def __init__(self, provider):

        logger.info("Initializing BenchmarkRunner...")

        self.provider = provider
        self.root = Path(__file__).parent.parent

        self.checkpoint = Checkpoint()
        self.state = self.checkpoint.load()

    def load_benchmark(self) -> list[dict]:

        logger.info("Loading evaluation benchmark...")

        benchmark_path = (
            self.root
            / "artifacts"
            / "evaluation_benchmark.json"
        )

        logger.debug(f"Benchmark path: {benchmark_path}")

        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

        logger.info(f"Loaded {len(benchmark)} prompts.")
        logger.debug(f"Benchmark contents:\n{benchmark}")

        return benchmark

    def load_existing_responses(self) -> list[dict]:

        responses_path = (
            self.root
            / "artifacts"
            / "responses.json"
        )

        if not responses_path.exists():
            return []

        logger.info("Loading existing responses...")

        with open(responses_path, "r", encoding="utf-8") as f:
            responses = json.load(f)

        logger.debug(f"Loaded responses:\n{responses}")

        return responses

    def run_benchmark(
        self,
        benchmark: list[dict],
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> list[dict]:

        start_prompt = self.state["runner"]["current_prompt"]

        logger.info(
            f"Running benchmark from prompt {start_prompt + 1}/{len(benchmark)}..."
        )

        responses = self.load_existing_responses()

        for i in range(start_prompt, len(benchmark)):

            item = benchmark[i]

            logger.info(
                f"Running prompt {i + 1}/{len(benchmark)}..."
            )

            logger.debug(f"Benchmark item:\n{item}")

            response = self.provider.generate(
                prompt=item["prompt"],
                max_tokens=max_tokens,
                temperature=temperature
            )

            logger.debug(f"Model response:\n{response}")

            result = {
                "id": item["id"],
                "behaviour": item["behaviour"],
                "prompt": item["prompt"],
                "response": response
            }

            logger.debug(f"Result:\n{result}")

            responses.append(result)

            if (i + 1) % 10 == 0 or i == len(benchmark) - 1:
                self.save_responses(responses)

                self.state["runner"]["current_prompt"] = i + 1
                self.checkpoint.save(self.state)

        logger.info(
            f"Finished benchmark. Generated {len(responses)} responses."
        )

        logger.debug(f"Responses:\n{responses}")

        self.state["runner"]["current_prompt"] = 0

        self.checkpoint.save(self.state)

        return responses

    def save_responses(
        self,
        responses: list[dict]
    ) -> None:

        logger.info("Saving benchmark responses...")

        output_path = (
            self.root
            / "artifacts"
            / "responses.json"
        )

        logger.debug(f"Output path: {output_path}")
        logger.debug(f"Responses:\n{responses}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                responses,
                f,
                indent=2,
                ensure_ascii=False
            )

        logger.info(
            f"Saved {len(responses)} benchmark responses."
        )

    def run(self, max_tokens: int = 128, temperature: float = 0.0) -> None:

        benchmark = self.load_benchmark()

        responses = self.run_benchmark(
            benchmark=benchmark,
            max_tokens=max_tokens,
            temperature=temperature
        )

        self.save_responses(responses)

        logger.info("Benchmark run complete.")