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
- prepare readme.md

## 4-7-2026

worked on:

- added logging to `benchmark.py` and `provider/gemini.py`
- created github repo and pushed changes

what i learned:

- learnt the use of logging
- learnt to use github

next steps:
