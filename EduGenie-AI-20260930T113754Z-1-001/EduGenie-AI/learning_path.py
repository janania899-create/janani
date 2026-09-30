from gemini_client import generate_text


async def get_learning_recommendations(
    topic: str,
    level: str = "beginner"
):

    prompt = f"""

Create a personalized learning path
for the topic:

{topic}

Learner level:

{level}

Include:

1. Learning goal
2. Prerequisites
3. Beginner topics
4. Intermediate topics
5. Advanced topics
6. Suggested timeline
7. Practice activities
8. Resource types such as:
   - Videos
   - Documentation
   - Articles
   - Books
9. Final project or self-test

Make the plan realistic
and easy for a student to follow.

Do not invent specific URLs.

"""

    try:

        plan = generate_text(

            prompt,

            temperature=0.4,

            max_output_tokens=1600
        )

        return {

            "success": True,

            "learning_path": plan,

            "level": level

        }

    except Exception as exc:

        return {

            "success": False,

            "error": str(exc)

        }