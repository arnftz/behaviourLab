from pathlib import Path
import json

from utils.logger import logger

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class Deduplicator:

    def __init__(self):
        logger.info("Initializing Deduplicator...")
        self.root = Path(__file__).parent.parent

        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_benchmarks(self) -> dict[str, list[dict]]:

        logger.info("Loading benchmarks...")
        behaviours_path = self.root / "behaviours"
        benchmarks = {}

        for behaviour_dir in behaviours_path.iterdir():
            if not behaviour_dir.is_dir():
                continue
            benchmark_path = behaviour_dir / "benchmark.json"
            if not benchmark_path.exists():
                logger.warning(
                    f"No benchmark found for '{behaviour_dir.name}'."
                )
                continue
            with open(benchmark_path, "r", encoding="utf-8") as f:
                benchmarks[behaviour_dir.name] = json.load(f)

        logger.info(
            f"Loaded {len(benchmarks)} benchmarks."
        )

        return benchmarks

    def normalize(self, prompt: str) -> str:
        return prompt.lower().strip().rstrip(",.!?")
    
    def remove_exact_duplicates(self, benchmarks: dict[str, list[dict]]) -> dict[str, list[dict]]:

        logger.info("Removing exact duplicates...")
        deduplicated = {}

        for behaviour, prompts in benchmarks.items():
            seen = set()
            unique = []
            for item in prompts:
                normalized = self.normalize(item["prompt"])
                if normalized in seen:
                    continue
                seen.add(normalized)
                unique.append(item)
            deduplicated[behaviour] = unique
            logger.info(
                f"{behaviour}: {len(prompts)} -> {len(unique)} prompts"
            )

        return deduplicated

    def remove_semantic_duplicates(self, benchmarks: dict[str, list[dict]], threshold: float = 0.93) -> dict[str, list[dict]]:

        logger.info("Removing semantic duplicates...")

        deduplicated = {}

        for behaviour, prompts in benchmarks.items():
            texts = [item["prompt"] for item in prompts]
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True
            )

            similarity = cos_sim(
                embeddings,
                embeddings
            )

            keep = []
            removed = set()

            for i in range(len(texts)):
                if i in removed:
                    continue
                keep.append(prompts[i])
                for j in range(i + 1, len(texts)):
                    if similarity[i][j] >= threshold:
                        removed.add(j)

            deduplicated[behaviour] = keep

            logger.info(
                f"{behaviour}: {len(prompts)} -> {len(keep)} prompts"
            )

        return deduplicated
        

    def build_evaluation_benchmark(self, output_file: str = "artifacts/evaluation_benchmark.json") -> list[dict]:

        logger.info("Building evaluation benchmark...")
        benchmarks = self.load_benchmarks()
        logger.debug(f"Loaded benchmarks: {benchmarks}")

        benchmarks = self.remove_exact_duplicates(benchmarks)
        logger.info(f"After exact deduplication: {sum(len(v) for v in benchmarks.values())} prompts")
        logger.debug(f"After exact deduplication: {benchmarks}")

        benchmarks = self.remove_semantic_duplicates(benchmarks)
        logger.info(f"After semantic deduplication: {sum(len(v) for v in benchmarks.values())} prompts")
        logger.debug(f"After semantic deduplication: {benchmarks}")

        evaluation_benchmark = []

        for behaviour, prompts in benchmarks.items():

            for index, item in enumerate(prompts, start=1):

                evaluation_benchmark.append({
                    "id": f"{behaviour}_{index:04d}",
                    "behaviour": behaviour,
                    "prompt": item["prompt"]
                })

        output_path = self.root / output_file

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                evaluation_benchmark,
                f,
                indent=2,
                ensure_ascii=False
            )

        logger.info(
            f"Saved {len(evaluation_benchmark)} evaluation prompts."
        )

        return evaluation_benchmark
