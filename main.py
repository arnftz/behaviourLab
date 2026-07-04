from benchmark import BenchmarkBuilder
from provider.gemini import GeminiProvider

def main():

    builder = BenchmarkBuilder(
        provider=GeminiProvider()
    )

    builder.run()


if __name__ == "__main__":
    main()