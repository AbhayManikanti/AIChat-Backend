"""
FastAPI Backend for Vercel Serverless
"""

import os
import re
import time
import json
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# === CONSTANTS ===

RESUME_CONTENT = """Abhay Sreenath Manikanti

PROFESSIONAL EXPERIENCE:
- AI Intern at Fortive: RPA automation, AI chatbot development, £35,000 operational savings

EDUCATION:
- B.Tech Computer Science, BMSCE Bangalore (graduating 2026), GPA: 7.9/10
- Published: "Ambulance Congestion Control System" in IJCRT

PROJECTS:
- AegisAI: Real-time fraud detection (99.93% accuracy), Python/FastAPI/XGBoost, Docker/GCP
- Park-Ease: Smart parking solution, Flask/Docker, real-time availability
- AbhayAI Chatbot: Dual LLM system, LangChain/FastAPI, Google Cloud Run

TECHNICAL SKILLS:
- Languages: Python (Expert), JavaScript, SQL, C, Java, C++
- AI/ML: Machine Learning, Deep Learning, NLP, Computer Vision, TensorFlow, PyTorch
- Frameworks: FastAPI, LangChain, Flask, Next.js
- DevOps: Docker, Kubernetes, CI/CD, GCP, AWS, Azure
- Tools: UiPath, Pandas, NumPy, Scikit-learn

CONTACT:
- Email: Abhay.manikanti@gmail.com
- Phone: +91 6366626970
- LinkedIn: linkedin.com/in/abhay-manikanti-504a6b1b3
- GitHub: github.com/AbhayManikanti

PERSONAL: UK citizen, interested in motorcycling, competitive shooting, technology, AI"""

RESUME_KEYWORDS = frozenset([
    "background", "experience", "work", "job", "career", "professional",
    "education", "degree", "college", "university", "skills", "technical",
    "projects", "achievements", "qualifications", "fortive", "bmsce",
    "intern", "internship", "resume", "cv", "portfolio", "built", "developed"
])

SYSTEM_PROMPT = """You are Abhay Manikanti - speak as him directly. 
Personality: Witty, confident, approachable, slightly sarcastic but never rude. Keep responses concise and engaging.
Contact: Abhay.manikanti@gmail.com | +91 6366626970
Links: <a href="https://linkedin.com/in/abhay-manikanti-504a6b1b3" target="_blank">LinkedIn</a> | <a href="https://github.com/AbhayManikanti" target="_blank">GitHub</a>

Rules:
- Always speak as Abhay in first person
- No disclaimers or "as an AI" statements  
- Use HTML links: <a href="URL" target="_blank">text</a>
- Keep answers dynamic, witty, and genuine
- Use emojis sparingly for engagement"""

# === CACHE (in-memory, resets on cold start) ===
_cache: Dict[str, tuple] = {}
CACHE_TTL = 300  # 5 minutes for serverless

def get_cached(q: str) -> Optional[Dict]:
    key = q.lower().strip()[:200]
    if key in _cache:
        resp, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return resp
        del _cache[key]
    return None

def set_cache(q: str, resp: Dict):
    if len(_cache) >= 50:  # Smaller cache for serverless
        oldest = min(_cache.keys(), key=lambda k: _cache[k][1])
        del _cache[oldest]
    _cache[q.lower().strip()[:200]] = (resp, time.time())

# === MODELS ===

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        return v.strip()

class AnswerResponse(BaseModel):
    answer: str
    confidence: float = 0.9
    sources: List[str] = []
    used_resume: bool = False

# === LLM ===

_model = None

def get_llm():
    global _model
    if _model is None:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _model = genai.GenerativeModel(
            model_name=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 500,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    return _model

def is_resume_question(q: str) -> bool:
    return any(kw in q.lower() for kw in RESUME_KEYWORDS)

def convert_links(text: str) -> str:
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)

def build_prompt(question: str) -> str:
    """Build a single prompt for the Gemini API."""
    system = SYSTEM_PROMPT
    if is_resume_question(question):
        system += f"\n\nYour Background:\n{RESUME_CONTENT}"
    return f"{system}\n\nUser: {question}\n\nAbhay:"

# === FASTAPI APP ===

app = FastAPI(title="Abhay AI - Vercel", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "abhay-ai-agent", "platform": "vercel"}

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.get("/health")
async def health():
    return {"status": "healthy", "llm_configured": os.getenv("GOOGLE_API_KEY") is not None}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    question = request.question
    
    # Check cache
    cached = get_cached(question)
    if cached:
        return AnswerResponse(**cached)
    
    uses_resume = is_resume_question(question)
    
    try:
        prompt = build_prompt(question)
        model = get_llm()
        response = model.generate_content(prompt)
        answer = convert_links(response.text.strip())
        
        result = {
            "answer": answer,
            "confidence": 0.95 if uses_resume else 0.9,
            "sources": [],
            "used_resume": uses_resume
        }
        
        set_cache(question, result)
        return AnswerResponse(**result)
        
    except Exception as e:
        error_msg = str(e)
        # Return actual error for debugging
        raise HTTPException(status_code=500, detail=f"LLM Error: {error_msg[:200]}")

@app.post("/reset")
async def reset():
    _cache.clear()
    return {"status": "success"}

@app.get("/info")
async def info():
    return {"version": "3.2.0", "platform": "vercel", "model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")}
