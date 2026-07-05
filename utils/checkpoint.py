from pathlib import Path
import json

from utils.logger import logger


class Checkpoint:

    def __init__(self):
        self.path = Path(__file__).parent.parent / "artifacts" / "checkpoint.json"

    def load(self) -> dict:

        logger.info("Loading checkpoint...")

        if not self.path.exists():

            logger.info("No checkpoint found.")

            return {
                "benchmark": {
                    "completed_behaviours": [],
                    "current_behaviour": None,
                    "completed_batch": 0
                },
                "runner": {
                    "current_prompt": 0
                }
            }

        with open(self.path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        logger.debug(f"Checkpoint:\n{checkpoint}")

        return checkpoint

    def save(self, checkpoint: dict) -> None:

        logger.info("Saving checkpoint...")

        logger.debug(f"Checkpoint:\n{checkpoint}")

        self.path.parent.mkdir(exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                checkpoint,
                f,
                indent=2,
                ensure_ascii=False
            )

    def clear(self) -> None:

        logger.info("Clearing checkpoint...")

        if self.path.exists():
            self.path.unlink()