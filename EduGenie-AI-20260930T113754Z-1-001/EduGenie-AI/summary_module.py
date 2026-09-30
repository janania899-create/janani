from gemini_client import generate_text


async def summarize_text(text: str):

    prompt = f"""

Summarize the following educational text
for quick revision.

Requirements:

1. Keep the important facts.
2. Remove unnecessary repetition.
3. Use simple student-friendly language.
4. Use headings and bullet points where useful.
5. Do not add information that is not present.
6. Keep the original meaning.

Text:

{text}

"""

    try:

        summary = generate_text(

            prompt,

            temperature=0.2,

            max_output_tokens=1000
        )

        return {
            "success": True,
            "summary": summary
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }