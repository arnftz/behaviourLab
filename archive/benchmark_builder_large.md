You are helping build a benchmark for evaluating Small Language Models (SLMs).

## Task

You are given a **seed prompt** that tests a specific conversational behaviour.

**Behaviour:** {BEHAVIOUR_NAME}

**Seed Prompt:**

{SEED_PROMPT}

Generate **{NUM_VARIATIONS} new user prompts** that test the **same behaviour**.

## Requirements

- Preserve the same conversational behaviour being tested.
- Every generated prompt must require the same behaviour to respond correctly.
- Do **not** simply paraphrase the seed prompt.
- Change the wording, entities, context, and scenario.
- Include a mixture of everyday situations.
- Vary the ambiguity and difficulty.
- Keep prompts natural and realistic.
- Each prompt should contain only the user's message.
- Do not include assistant responses.
- Do not explain your reasoning.

## Output Format

Return ONLY a valid JSON array.

Do not wrap the output in Markdown.
Do not use ```json fences.
Do not include explanations, notes, or additional text.

The output must exactly follow this schema:

[
{
" prompt": "..."
},
{
"prompt": "..."
}
]

### Difficulty

Must be one of:

- easy
- medium
- hard

### Category

Choose the most appropriate category, for example:

- calendar
- messaging
- reminders
- shopping
- travel
- files
- devices
- media
- productivity
- navigation
- communication
- general

Do not output anything except the JSON array.
