from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from qna import ask_question
from explanation_module import explain_topic
from quiz_module import generate_quiz
from summary_module import summarize_text
from learning_path import get_learning_recommendations


app = FastAPI(
    title="EduGenie - Google Gemini Powered Learning Assistant",
    version="1.0.0",
    description="AI-powered educational assistant using FastAPI and Google Gemini."
)


# ---------------------------------------------------
# STATIC FILES
# ---------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ---------------------------------------------------
# HTML TEMPLATES
# ---------------------------------------------------

templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ---------------------------------------------------
# QUESTION ANSWERING
# ---------------------------------------------------

@app.post("/qa")
async def qa(payload: dict):

    question = str(
        payload.get("question", "")
    ).strip()

    if not question:

        return {
            "success": False,
            "error": "Please enter a question."
        }

    return await ask_question(question)


# ---------------------------------------------------
# EXPLANATION
# ---------------------------------------------------

@app.post("/explain")
async def explain(payload: dict):

    topic = str(
        payload.get("topic", "")
    ).strip()

    if not topic:

        return {
            "success": False,
            "error": "Please enter a topic to explain."
        }

    return await explain_topic(topic)


# ---------------------------------------------------
# QUIZ
# ---------------------------------------------------

@app.post("/quiz")
async def quiz(payload: dict):

    text = str(
        payload.get("text", "")
    ).strip()

    if not text:

        return {
            "success": False,
            "error": "Please enter a topic or passage."
        }

    return await generate_quiz(text)


# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------

@app.post("/summarize")
async def summarize(payload: dict):

    text = str(
        payload.get("text", "")
    ).strip()

    if not text:

        return {
            "success": False,
            "error": "Please enter text to summarize."
        }

    return await summarize_text(text)


# ---------------------------------------------------
# LEARNING RECOMMENDATIONS
# ---------------------------------------------------

@app.post("/learn/recommendations")
async def recommendations(payload: dict):

    topic = str(
        payload.get("topic", "")
    ).strip()

    level = str(
        payload.get(
            "level",
            "beginner"
        )
    ).strip().lower()

    if not topic:

        return {
            "success": False,
            "error": "Please enter a topic."
        }

    if level not in {
        "beginner",
        "intermediate",
        "advanced"
    }:

        level = "beginner"

    return await get_learning_recommendations(
        topic,
        level
    )