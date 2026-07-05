from pathlib import Path
import json

from utils.logger import logger
from utils.checkpoint import Checkpoint


class BatchEvaluator:

    def __init__(self, provider):

        logger.info("Initializing Evaluator...")

        self.provider = provider
        self.root = Path(__file__).parent.parent
        self.checkpoint = Checkpoint()
        self.state = self.checkpoint.load()
        self.state.setdefault(
            "evaluator",
            {
                "completed_behaviours": [],
                "current_behaviour": None,
                "completed_batch": 0
            }
        )
        self.state["evaluator"].setdefault("completed_behaviours", [])
        self.state["evaluator"].setdefault("current_behaviour", None)
        self.state["evaluator"].setdefault("completed_batch", 0)

    def load_responses(self) -> list[dict]:

        logger.info("Loading benchmark responses...")
        responses_path = (self.root / "artifacts" / "responses.json")
        logger.debug(f"Responses path: {responses_path}")

        with open(responses_path, "r", encoding="utf-8") as f:
            responses = json.load(f)

        logger.info(f"Loaded {len(responses)} responses.")
        logger.debug(f"Responses:\n{responses}")

        return responses

    def load_definition(self, behaviour: str) -> str:

        logger.info(f"Loading definition for '{behaviour}'...")
        definition_path = (self.root / "behaviours" / behaviour / "definition.md")
        logger.debug(f"Definition path: {definition_path}")

        definition = definition_path.read_text(
            encoding="utf-8"
        )

        logger.debug(f"Definition:\n{definition}")

        return definition

    def validate_response(self, response: str) -> dict:

        logger.info("Validating response...")

        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON") from exc

        if not isinstance(data, dict):
            raise ValueError("Response must be an object.")

        logger.debug(f"Parsed JSON:\n{data}")

        return data

    def repair_prompt(self, error: str, invalid_response: str, original_prompt: str) -> str:

        return f"""
            Your previous response could not be parsed.

            Validation Error:
            {error}

            Previous Response:

            {invalid_response}

            Return ONLY valid JSON.

            Repair only the JSON and preserve the original intent.

            Original Task:

            {original_prompt}
            """

    def save_batch_critique(self, behaviour: str, batch_number: int, critique: dict) -> None:

        logger.info(f"Saving critique for '{behaviour}' batch {batch_number}...")

        output_path = (
            self.root
            / "artifacts"
            / "critiques"
            / behaviour
        )
        output_path.mkdir(parents=True, exist_ok=True)

        file = output_path / f"batch_{batch_number:04d}.json"

        logger.debug(f"Batch critique path: {file}")
        logger.debug(f"Batch critique:\n{critique}")

        with open(file, "w", encoding="utf-8") as f:
            json.dump(critique, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved critique for '{behaviour}' batch {batch_number}.")

    def load_batch_critiques(self, behaviour: str) -> list[dict]:

        logger.info(f"Loading critiques for '{behaviour}'...")

        critique_path = (
            self.root
            / "artifacts"
            / "critiques"
            / behaviour
        )

        if not critique_path.exists():

            logger.info(f"No saved critiques found for '{behaviour}'.")

            return []

        critiques = []

        for file in sorted(critique_path.glob("batch_*.json")):

            with open(file, "r", encoding="utf-8") as f:
                critiques.append(json.load(f))

        logger.debug(f"Loaded critiques for '{behaviour}':\n{critiques}")

        return critiques

    def evaluate_behaviour(self, behaviour: str, responses: list[dict], batch: int = 20, max_tokens: int = 2000, start_batch: int = 0) -> None:

        logger.info(f"Evaluating '{behaviour}'...")

        definition = self.load_definition(behaviour)

        for i in range(start_batch * batch, len(responses), batch):

            batch_responses = responses[i:i + batch]
            batch_number = i // batch + 1

            logger.info(f"Evaluating batch {batch_number}...")
            logger.debug(f"Batch responses:\n{batch_responses}")
            prompt = self.build_prompt(behaviour, definition, batch_responses)
            logger.debug(f"Evaluation prompt:\n{prompt}")

            critique = self.provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2
            )

            logger.debug(f"Batch critique:\n{critique}")

            try:
                data = self.validate_response(critique)

            except ValueError as e:

                logger.warning(f"Validation failed: {e}")

                repair = self.repair_prompt(
                    str(e),
                    critique,
                    prompt
                )

                logger.debug(f"Repair prompt:\n{repair}")

                critique = self.provider.generate(
                    prompt=repair,
                    max_tokens=max_tokens,
                    temperature=0.0
                )

                logger.debug(f"Repaired batch critique:\n{critique}")

                data = self.validate_response(critique)

            self.save_batch_critique(
                behaviour,
                batch_number,
                data
            )

            self.state["evaluator"]["current_behaviour"] = behaviour
            self.state["evaluator"]["completed_batch"] = batch_number

            self.checkpoint.save(self.state)

        logger.info(f"Generated critiques for '{behaviour}'.")

    def build_prompt(self, behaviour: str, definition: str, responses: list[dict]) -> str:

        prompt = f"""
            Behaviour

            {behaviour}

            Definition

            {definition}

            Below are responses from a Small Language Model.

            Read every prompt and response.

            Your task is to critique the model.

            Identify recurring strengths.

            Identify recurring weaknesses.

            Quote representative examples.

            Return ONLY valid JSON as an object with these keys:

            {{
                "behaviour": "...",
                "summary": "...",
                "strengths": [...],
                "weaknesses": [...],
                "examples": [...]
            }}

            Do not generate training data.

            Responses

            {json.dumps(responses, indent=2, ensure_ascii=False)}
            """

        return prompt

    def save_report(self, behaviour: str, report: dict) -> None:

        logger.info(f"Saving report for '{behaviour}'...")

        output_path = self.root / "artifacts" / "evaluation"
        output_path.mkdir(parents=True, exist_ok=True)

        file = output_path / f"{behaviour}.json"

        report = dict(report)
        report.setdefault("behaviour", behaviour)

        logger.debug(f"Output path: {file}")
        logger.debug(f"Report:\n{report}")

        with open(file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved report for '{behaviour}'.")

    def summarize_behaviour(self, behaviour: str, definition: str, critiques: list[dict], max_tokens: int = 2000) -> dict:

        logger.info(f"Summarizing '{behaviour}'...")

        prompt = f"""
            Behaviour

            {behaviour}

            Definition

            {definition}

            Below are critiques generated from multiple batches of responses for this behaviour.

            Merge them into a single report.

            Remove duplicate observations.

            You are coaching a conversational AI.

            Your objective is to identify the smallest number of behavioural improvements that would produce the largest increase in conversation quality.

            Prioritize the improvements.

            Be concrete.

            These recommendations will later be used to generate synthetic training conversations.

            Identify the most important strengths.

            Identify the most important weaknesses.

            Give representative examples where useful.

            Do NOT generate training data.

            Return ONLY valid JSON.

            Return an object with these top-level keys and no nested "report" object:

            {{
                "behaviour": "...",
                "summary": "...",
                "strengths": [...],
                "weaknesses": [...],
                "examples": [...],
                "training_priorities": [...]
            }}

            Critiques

            {json.dumps(critiques, indent=2, ensure_ascii=False)}
            """

        logger.debug(f"Summary prompt:\n{prompt}")

        summary = self.provider.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.2
        )

        logger.debug(f"Behaviour summary:\n{summary}")

        try:
            data = self.validate_response(summary)

        except ValueError as e:

            logger.warning(f"Validation failed: {e}")

            repair = self.repair_prompt(
                str(e),
                summary,
                prompt
            )

            logger.debug(f"Repair prompt:\n{repair}")

            summary = self.provider.generate(
                prompt=repair,
                max_tokens=max_tokens,
                temperature=0.0
            )

            logger.debug(f"Repaired behaviour summary:\n{summary}")

            data = self.validate_response(summary)

        return data

    def run(self, batch: int = 20, max_tokens: int = 2000) -> None:

        logger.info("Running evaluator...")

        responses = self.load_responses()

        grouped = {}

        for item in responses:

            grouped.setdefault(
                item["behaviour"],
                []
            ).append(item)

        logger.debug(f"Grouped responses:\n{grouped}")

        completed_behaviours = set(self.state["evaluator"]["completed_behaviours"])
        resume_behaviour = self.state["evaluator"]["current_behaviour"]
        resume_batch = self.state["evaluator"]["completed_batch"]

        for behaviour, behaviour_responses in grouped.items():

            if behaviour in completed_behaviours:

                logger.info(f"Skipping '{behaviour}' (already completed).")

                continue

            report_path = self.root / "artifacts" / "evaluation" / f"{behaviour}.json"

            if report_path.exists():

                logger.info(f"Skipping '{behaviour}' (already summarized).")

                completed_behaviours.add(behaviour)
                self.state["evaluator"]["completed_behaviours"] = sorted(completed_behaviours)
                self.checkpoint.save(self.state)

                if behaviour == resume_behaviour:
                    resume_behaviour = None
                    resume_batch = 0

                continue

            if resume_behaviour is not None and behaviour != resume_behaviour:

                logger.info(f"Skipping '{behaviour}' (already completed).")

                continue

            start_batch = 0

            if behaviour == resume_behaviour:
                start_batch = resume_batch
                resume_behaviour = None
                resume_batch = 0

            definition = self.load_definition(behaviour)

            self.evaluate_behaviour(
                behaviour,
                behaviour_responses,
                start_batch=start_batch,
                batch=batch,
                max_tokens=max_tokens
            )

            critiques = self.load_batch_critiques(behaviour)

            report = self.summarize_behaviour(
                behaviour,
                definition,
                critiques,
                max_tokens=max_tokens
            )

            self.save_report(
                behaviour,
                report
            )

            completed_behaviours.add(behaviour)
            self.state["evaluator"]["completed_behaviours"] = sorted(completed_behaviours)

            self.state["evaluator"]["current_behaviour"] = None
            self.state["evaluator"]["completed_batch"] = 0

            self.checkpoint.save(self.state)

        logger.info(
            "Evaluation complete."
        )