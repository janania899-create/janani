from gemini_client import generate_text


async def ask_question(question: str):

    prompt = f"""
You are EduGenie, a student-friendly educational assistant.

Answer the student's question accurately and clearly.

Rules:

1. Start with the direct answer.
2. Use simple language.
3. Explain difficult terms.
4. Add an example when useful.
5. Keep the answer educational.
6. Do not invent sources.
7. If the question is ambiguous, clearly state the assumption.

Student question:

{question}
"""

    try:

        answer = generate_text(
            prompt,
            temperature=0.3,
            max_output_tokens=900
        )

        return {
            "success": True,
            "answer": answer
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }