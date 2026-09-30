import os

from dotenv import load_dotenv

from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# API KEY
# ============================================================

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is missing from the .env file."
    )


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(
    prompt: str,
    temperature: float = 0.3,
    max_output_tokens: int = 2048
):

    if not prompt.strip():

        raise ValueError(
            "Prompt cannot be empty."
        )


    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=prompt,

        config=types.GenerateContentConfig(

            temperature=temperature,

            max_output_tokens=max_output_tokens

        )

    )


    # --------------------------------------------------------
    # Get generated text
    # --------------------------------------------------------

    text = getattr(
        response,
        "text",
        None
    )


    if not text:

        raise RuntimeError(
            "Gemini did not return any text."
        )


    return text.strip()