from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import os
from io import BytesIO

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if len(raw) < 2000:
            return {"text": "", "error": "音訊太短，無法辨識"}

        buffer = BytesIO(raw)
        buffer.name = file.filename or "audio.webm"

        transcript = openai.Audio.transcribe("whisper-1", buffer, language="en")
        return {"text": transcript["text"]}
    except Exception as e:
        return {"text": "", "error": str(e)}

@app.post("/translate")
async def translate_text(text: str = Form(...)):
    if not text or text.strip() == "":
        return {"translation": "[無法辨識語音內容]"}
    try:
        prompt = f"請將以下英文翻譯成自然流暢的中文：\n\n{text}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return {"translation": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"translation": f"[翻譯失敗：{str(e)}]"}
