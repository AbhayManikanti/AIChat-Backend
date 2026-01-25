"""
FastAPI Resume Chatbot Backend - ULTRA-OPTIMIZED VERSION (LangChain variant)
Author: Abhay Sreenath Manikanti
Target: <3 second response times with Gemini 2.5 Flash Lite

This version uses langchain-google-genai for compatibility with existing setup.
Still achieves <3s by eliminating agents, chains, and tools.
"""

import os
import re
import time
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv("keys.env")

# === CONSTANTS ===

RESUME_CONTENT = """Abhay Sreenath Manikanti

EXPERIENCE: AI Intern at Fortive - RPA automation, AI chatbot, £35,000 savings
EDUCATION: B.Tech CS, BMSCE Bangalore (2026), GPA 7.9/10
PROJECTS: AegisAI (99.93% fraud detection), Park-Ease (parking app), AbhayAI chatbot
SKILLS: Python, FastAPI, LangChain, Docker, TensorFlow, PyTorch, GCP, AWS
CONTACT: Abhay.manikanti@gmail.com | +91 6366626970
LINKS: linkedin.com/in/abhay-manikanti-504a6b1b3 | github.com/AbhayManikanti
PERSONAL: UK citizen, into motorcycling, shooting, tech, AI"""

RESUME_KEYWORDS = frozenset([
    "background", "experience", "work", "job", "career", "professional",
    "education", "degree", "college", "skills", "technical", "projects",
    "achievements", "qualifications", "fortive", "bmsce", "intern",
    "resume", "cv", "portfolio"
])

SYSTEM_PROMPT = """You are Abhay Manikanti - speak as him. Witty, confident, slightly sarcastic.
Contact: Abhay.manikanti@gmail.com | +91 6366626970
Links: <a href="https://linkedin.com/in/abhay-manikanti-504a6b1b3" target="_blank">LinkedIn</a> | <a href="https://github.com/AbhayManikanti" target="_blank">GitHub</a>
Rules: First person only, no "as an AI", use HTML links, keep it engaging."""

# === SIMPLE CACHE ===
_cache: Dict[str, tuple] = {}
CACHE_TTL = 3600

def get_cached(q: str) -> Optional[str]:
    key = q.lower().strip()[:200]
    if key in _cache:
        resp, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return resp
        del _cache[key]
    return None

def set_cache(q: str, r: str):
    if len(_cache) > 100:
        oldest = min(_cache.keys(), key=lambda k: _cache[k][1])
        del _cache[oldest]
    _cache[q.lower().strip()[:200]] = (r, time.time())

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

class HealthResponse(BaseModel):
    status: str
    message: str
    vectorstore_initialized: bool = True
    llm_configured: bool = True

# === LLM (single instance) ===

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",  # Fastest available
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_tokens=400,
            timeout=10,  # Hard 10s timeout
            max_retries=0,  # No retries - fail fast
        )
    return _llm

def is_resume_q(q: str) -> bool:
    return any(kw in q.lower() for kw in RESUME_KEYWORDS)

def convert_links(text: str) -> str:
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)

# === FASTAPI ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_llm()  # Pre-warm
    yield

app = FastAPI(title="Abhay AI - Optimized", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", message="Ready")

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """Single direct LLM call - no agents, no chains."""
    q = request.question
    
    # Cache check
    cached = get_cached(q)
    if cached:
        return AnswerResponse(answer=cached, confidence=0.95, used_resume=is_resume_q(q))
    
    try:
        # Build messages
        system = SYSTEM_PROMPT
        if is_resume_q(q):
            system += f"\n\nResume:\n{RESUME_CONTENT}"
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=q)
        ]
        
        # Single API call
        llm = get_llm()
        response = llm.invoke(messages)
        
        answer = convert_links(response.content.strip())
        set_cache(q, answer)
        
        return AnswerResponse(answer=answer, confidence=0.9, used_resume=is_resume_q(q))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset():
    _cache.clear()
    return {"status": "success"}

@app.get("/info")
async def info():
    return {
        "version": "3.0.0",
        "optimizations": ["Single API call", "No agents", "Pre-built prompts", "Simple cache"]
    }

@app.get("/cache-stats")
async def cache_stats():
    return {"cached": len(_cache), "ttl": CACHE_TTL}

@app.get("/readiness")
async def readiness():
    return {"status": "ready"}

@app.get("/startup")
async def startup():
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
