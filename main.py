from http.client import responses

from pipeline.benchmark import BenchmarkBuilder
from pipeline.deduplicator import Deduplicator
from pipeline.runner import BenchmarkRunner
from pipeline.evaluator import BatchEvaluator

from provider.gemini import GeminiProvider
from provider.huggingface import HuggingFaceProvider

from utils.logger import logger


def main():

    # builder = BenchmarkBuilder(
    #     provider=GeminiProvider()
    # )

    # builder.run()

    # deduplicator = Deduplicator()
    # deduplicator.build_evaluation_benchmark(
    #     output_file="artifacts/evaluation_benchmark.json"
    # )

    # runner = BenchmarkRunner(
    #     provider=HuggingFaceProvider(
    #         model_name="HuggingFaceTB/SmolLM2-135M-Instruct"
    #     )
    # )

    # runner.run(
    #     max_tokens=128,
    #     temperature=0.0
    # )

    evaluator = BatchEvaluator(
        provider=GeminiProvider()
    )
    evaluator.run(
        batch=20,
        max_tokens=2000
    )

if __name__ == "__main__":
    main()