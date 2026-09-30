import os

from gemini_client import generate_text


# ============================================================
# OPTIONAL LOCAL MODEL
# ============================================================

_local_pipeline = None
_local_load_attempted = False


def _local_explain(topic: str):

    global _local_pipeline
    global _local_load_attempted

    enabled = (
        os.getenv(
            "USE_LOCAL_EXPLANATION",
            "false"
        ).lower()
        == "true"
    )

    if not enabled:
        return None

    if (
        _local_load_attempted
        and _local_pipeline is None
    ):
        return None

    _local_load_attempted = True

    try:

        from transformers import pipeline


        model_name = os.getenv(
            "LOCAL_EXPLANATION_MODEL",
            "MBZUAI/LaMini-Flan-T5-783M"
        )


        _local_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name
        )


        prompt = f"""
Explain the following topic completely
for a student.

Topic:
{topic}

Requirements:

1. Give a clear definition.
2. Explain the concept in detail.
3. Explain the important points.
4. Give suitable examples.
5. Use simple student-friendly language.
6. Organize the answer with headings.
7. Do not stop in the middle of the explanation.
8. Give a complete answer.
"""


        result = _local_pipeline(

            prompt,

            max_new_tokens=700,

            do_sample=False

        )


        if not result:
            return None


        answer = result[0].get(
            "generated_text",
            ""
        ).strip()


        if not answer:
            return None


        return answer


    except Exception as exc:

        print(
            "Local explanation model error:",
            exc
        )

        _local_pipeline = None

        return None


# ============================================================
# GEMINI EXPLANATION
# ============================================================

async def explain_topic(topic: str):

    try:

        # ----------------------------------------------------
        # Try local model if enabled
        # ----------------------------------------------------

        local_answer = _local_explain(
            topic
        )


        if local_answer:

            return {

                "success": True,

                "answer": local_answer,

                "model":
                    "LaMini-Flan-T5-783M"

            }


        # ----------------------------------------------------
        # Gemini prompt
        # ----------------------------------------------------

        prompt = f"""
You are EduGenie, an AI learning assistant.

Explain the following topic completely
for a student.

TOPIC:
{topic}

IMPORTANT INSTRUCTIONS:

1. Start with a clear and simple definition.

2. Explain the topic in detail using
   easy-to-understand student-friendly language.

3. Divide the explanation into clear headings
   and subheadings where appropriate.

4. Explain all important concepts related
   to the topic.

5. Give important points in bullet points
   when useful.

6. Give at least one simple example.

7. If the topic involves a process,
   explain the steps clearly.

8. If the topic involves advantages,
   disadvantages, features, types, uses,
   applications, or characteristics,
   explain the relevant points completely.

9. Do not give only a short summary.

10. Do not stop after the introduction.

11. Do not leave the explanation incomplete.

12. Make the answer long enough for a student
    to understand and study from it.

13. Use simple English.

14. Finish with a short conclusion or
    revision point.

15. Return ONLY the educational answer.
    Do not mention these instructions.

Give a COMPLETE answer.
"""


        # ----------------------------------------------------
        # Generate Gemini response
        # ----------------------------------------------------

        answer = generate_text(

            prompt,

            temperature=0.3,

            max_output_tokens=2048

        )


        # ----------------------------------------------------
        # Validate answer
        # ----------------------------------------------------

        if not answer:

            return {

                "success": False,

                "error":
                    "Gemini returned an empty answer."

            }


        return {

            "success": True,

            "answer": answer,

            "model": "Gemini"

        }


    except Exception as exc:

        print(
            "Gemini explanation error:",
            exc
        )


        return {

            "success": False,

            "error": str(exc)

        }