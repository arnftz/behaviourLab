import json

data = """[
  {
    "prompt": "I need to book a flight for next Tuesday."
  },
  {
    "prompt": "Can you tell me if this is compatible with my current setup?"
  },
  {
    "prompt": "I'm having trouble with my account. Can you reset it for me?"
  },
  {
    "prompt": "Please write a formal email to my boss about the incident."
  },
  {
    "prompt": "What is the weather going to be like tomorrow?"
  },
  {
    "prompt": "I want to buy a gift for my friend; what do you suggest?"
  },
  {
    "prompt": "Can you summarize the latest report for me?"
  },
  {
    "prompt": "How long will it take to drive there?"
  },
  {
    "prompt": "I need a recipe that uses these ingredients."
  },
  {
    "prompt": "Could you help me fix the error I'm getting in my script?"
  }
]"""

def validate_response(response: str) -> list[dict]:

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

validated_data = validate_response(data)
print("Validated data:", validated_data)
var = []
var.extend(validated_data)
print("Extended list:", var)