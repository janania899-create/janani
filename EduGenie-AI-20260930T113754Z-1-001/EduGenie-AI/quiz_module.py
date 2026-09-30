import json
import re

from gemini_client import generate_text


# ---------------------------------------------------
# CLEAN GEMINI JSON
# ---------------------------------------------------

def clean_json_block(text: str) -> str:

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


# ---------------------------------------------------
# VALIDATE QUIZ
# ---------------------------------------------------

def validate_quiz(data):

    if not isinstance(data, list):

        raise ValueError(
            "Quiz must be a list."
        )

    if len(data) != 3:

        raise ValueError(
            "Quiz must contain exactly 3 questions."
        )

    cleaned = []


    for index, item in enumerate(
        data,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):

            raise ValueError(
                f"Question {index} is invalid."
            )


        question = str(
            item.get(
                "question",
                ""
            )
        ).strip()


        options = item.get(
            "options"
        )


        answer = str(
            item.get(
                "answer",
                ""
            )
        ).strip()


        if not question:

            raise ValueError(
                f"Question {index} is empty."
            )


        if (
            not isinstance(
                options,
                list
            )
            or len(options) != 4
        ):

            raise ValueError(
                f"Question {index} must have 4 options."
            )


        options = [
            str(option).strip()
            for option in options
        ]


        if answer not in options:

            raise ValueError(
                f"Question {index} has an invalid answer."
            )


        cleaned.append({

            "question": question,

            "options": options,

            "answer": answer

        })


    return cleaned


# ---------------------------------------------------
# GENERATE QUIZ
# ---------------------------------------------------

async def generate_quiz(text: str):

    prompt = f"""

Create exactly 3 multiple-choice questions
from the educational topic or passage below.

Return ONLY valid JSON.

Do NOT use Markdown.

The JSON must be:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Correct option"
  }}
]

Rules:

- Exactly 3 questions.
- Exactly 4 options per question.
- Only one correct answer.
- Questions must be relevant.
- Distractors should be plausible.
- Suitable for students.

Topic or passage:

{text}

"""

    try:

        raw = generate_text(

            prompt,

            temperature=0.2,

            max_output_tokens=1400
        )


        cleaned = clean_json_block(
            raw
        )


        data = json.loads(
            cleaned
        )


        quiz = validate_quiz(
            data
        )


        return {
            "success": True,
            "quiz": quiz
        }


    except json.JSONDecodeError:

        return {
            "success": False,
            "error":
                "Gemini returned invalid quiz JSON. "
                "Please try again."
        }


    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }