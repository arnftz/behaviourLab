# Changelog

## 28-6-2026

worked on:

- created a gemma4 inference using google aistudio provider
- created a changelog
- created `behaviours/` folder and defined its basic structure

```
behaviours/
└── example/
    ├── benchmark.json
    ├── definition.md
    └── seeds.json
```

- created a `.env` file
- whole thing runs on `uv`
- created `benchmark.py` which builds the benchmark for every behaviour in the folder
- created 5 mvp behaviours (`acknowledge`, `clarify`, `close`, `continue`, `greet`) and filled it with seeds and definition, benchmark build is pending
- started with main.py

what i learned:

- learning to use `uv`
- learnt how to write and maintain changelogs
- inline suggestions are the better way of coding
- learnt about simple folder structuring

next steps:

- need to learn logging and implement it
- push to github and prepare it for kaggle runtime

## 4-7-2026

worked on:

- added logging to `benchmark.py` and `provider/gemini.py`
- created github repo and pushed changes
- created artifacts folder to stash the `evaluation_benchmark.json`
- this will also store the `responses.json`
- we ran behaviours using the `benchmark.py` and generated initial benchmark
- built `deduplicator` with normalization, exact duplicate removal and semantic duplicate removal
- generated final `evaluation_benchmark.json` using deduplicator
- added saved logs for future review
- moved `benchmark.py`, `deduplicator.py`, `runner.py` to a folder called `pipeline`
- build `provider/huggingface.py` to run the small parameter language models
- built `runner.py` to run the `evaluation_benchmark.json` on the small parameter language model to generate `responses.json`
- built a `utils` folder to stash utilities like `utils/checkpoint.py` to store checkpoints incase of interruptions, `utils/logger.py` to save logging functionality
- hardened the `provider/gemini.py` against random 500 errors by adding retries
- moved `checkpoint.json` to `artifacts/` and expanded capabilities to checkpoint the `runner.py` also

what i learned:

- learnt the use of logging
- learnt to use github

next steps:

- prepare readme.md
