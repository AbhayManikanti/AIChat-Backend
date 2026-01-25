"""
FastAPI Resume Chatbot Backend - ULTRA-OPTIMIZED VERSION
Author: Abhay Sreenath Manikanti
Target: <3 second response times with Gemini 2.5 Flash Lite

OPTIMIZATION STRATEGY:
- Single direct API call (no agents, no chains, no tools)
- Pre-built prompts (no runtime concatenation)
- Minimal async overhead
- Simple in-memory caching
- Zero blocking operations
"""

import os
import re
import time
import hashlib
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

import google.generativeai as genai

# Load environment variables
load_dotenv("keys.env")

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === CONSTANTS (imported inline for speed) ===

RESUME_CONTENT = """
Abhay Sreenath Manikanti

PROFESSIONAL EXPERIENCE:
- AI Intern at Fortive: Developed RPA automation solutions and AI chatbot, achieving £35,000 in operational savings
- Implemented intelligent process automation using cutting-edge AI technologies

EDUCATION:
- Bachelors in Computer Science from BMSCE (Bangalore), graduating 2026
- GPA: 7.9/10
- Thesis: Ambulance Congestion Control System

PROJECTS:
1. Aegis AI: Real-time AI/ML fraud detection platform with 99.93% accuracy
2. Park-Ease: Smart parking solution with Flask and Docker
3. Resume Screening System: RPA-based automated HR solution using UiPath
4. AbhayAI Chatbot: AI powered chatbot with dual LLM switching, built on LangChain and FastAPI

TECHNICAL SKILLS:
- Languages: Python (Expert), JavaScript, SQL, C, Java, C++
- AI/ML: Machine Learning, Deep Learning, NLP, Computer Vision
- Frameworks: FastAPI, LangChain, Flask, Next.js
- DevOps: Docker, Kubernetes, CI/CD, Google Cloud Platform
- Tools: UiPath, TensorFlow, PyTorch, Pandas, NumPy

ACHIEVEMENTS:
- Published research paper on Ambulance Congestion Control in IJCRT
- £35,000 cost savings through RPA at Fortive

CONTACT:
- Email: Abhay.manikanti@gmail.com
- Phone: +91 6366626970
- LinkedIn: linkedin.com/in/abhay-manikanti-504a6b1b3
- GitHub: github.com/AbhayManikanti

PERSONAL:
- UK citizen, open to international opportunities
- Interests: Motorcycling, competitive shooting, technology, AI, cloud computing
"""

RESUME_KEYWORDS = frozenset([
    "background", "experience", "work", "job", "career", "professional",
    "education", "degree", "college", "skills", "technical", "projects",
    "achievements", "qualifications", "fortive", "bmsce", "intern"
])

# Pre-built system prompt (no runtime concatenation)
SYSTEM_PROMPT = """You are Abhay Manikanti's AI chatbot - his digital twin. Speak as him with wit, confidence, and a touch of sarcasm.

Personality: Witty, approachable, confident, slightly sarcastic but never rude. Keep responses concise and engaging.

Contact: Abhay.manikanti@gmail.com | +91 6366626970 | <a href="https://linkedin.com/in/abhay-manikanti-504a6b1b3" target="_blank">LinkedIn</a> | <a href="https://github.com/AbhayManikanti" target="_blank">GitHub</a>

Background: CS student at BMSCE Bangalore (2026), AI Intern at Fortive, skilled in Python, FastAPI, LangChain, Docker, AI/ML. Built AegisAI fraud detection, Park-Ease parking app, and this chatbot.

Rules:
- Always speak as Abhay, first person
- No disclaimers or "as an AI" statements
- Use HTML links: <a href="URL" target="_blank">text</a>
- Keep answers dynamic and genuine
- Use emojis sparingly for engagement"""

# === SIMPLE CACHE (no hashing overhead for common queries) ===
_cache: Dict[str, tuple] = {}  # {normalized_question: (response, timestamp)}
CACHE_TTL = 3600  # 1 hour

def _normalize(q: str) -> str:
    """Fast normalization"""
    return q.lower().strip()[:200]

def get_cached(question: str) -> Optional[str]:
    """Get cached response if valid"""
    key = _normalize(question)
    if key in _cache:
        resp, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return resp
        del _cache[key]
    return None

def set_cache(question: str, response: str):
    """Cache response"""
    if len(_cache) > 100:  # Simple size limit
        oldest = min(_cache.keys(), key=lambda k: _cache[k][1])
        del _cache[oldest]
    _cache[_normalize(question)] = (response, time.time())

# === REQUEST/RESPONSE MODELS ===

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

# === GEMINI CLIENT (single instance, reused) ===

_model = None

def get_model():
    """Get or create Gemini model (singleton)"""
    global _model
    if _model is None:
        _model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite",  # Fastest model
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=400,  # Keep responses concise
                top_p=0.8,
                top_k=40,
            ),
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )
    return _model

def is_resume_question(question: str) -> bool:
    """Fast keyword check"""
    q_lower = question.lower()
    return any(kw in q_lower for kw in RESUME_KEYWORDS)

def build_prompt(question: str) -> str:
    """Build minimal prompt based on question type"""
    if is_resume_question(question):
        return f"{SYSTEM_PROMPT}\n\nResume Context:\n{RESUME_CONTENT}\n\nUser: {question}\nAbhay:"
    else:
        return f"{SYSTEM_PROMPT}\n\nUser: {question}\nAbhay:"

def convert_links(text: str) -> str:
    """Convert Markdown links to HTML (if any slipped through)"""
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)

# === FASTAPI APP ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Minimal startup"""
    # Pre-warm the model
    get_model()
    yield

app = FastAPI(
    title="Abhay's AI Agent - Ultra Optimized",
    description="<3 second response times",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "abhay-ai-agent"}

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        message="Ultra-optimized backend ready"
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    Ultra-fast question answering.
    Single API call, no agents, no chains.
    """
    question = request.question
    
    # Check cache first
    cached = get_cached(question)
    if cached:
        return AnswerResponse(
            answer=cached,
            confidence=0.95,
            used_resume=is_resume_question(question)
        )
    
    try:
        # Build prompt
        prompt = build_prompt(question)
        
        # Single direct API call
        model = get_model()
        response = model.generate_content(prompt)
        
        # Extract text
        answer = response.text.strip()
        
        # Convert any markdown links
        answer = convert_links(answer)
        
        # Cache result
        set_cache(question, answer)
        
        return AnswerResponse(
            answer=answer,
            confidence=0.9,
            used_resume=is_resume_question(question)
        )
        
    except Exception as e:
        # Simple error handling - no retries, no fallback complexity
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

@app.post("/reset")
async def reset():
    """Clear cache"""
    _cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.get("/info")
async def info():
    return {
        "api_version": "3.0.0",
        "owner": "Abhay Sreenath Manikanti",
        "optimizations": [
            "Single direct API call",
            "No LangChain agents or chains",
            "Pre-built prompts",
            "Simple in-memory caching",
            "Gemini 2.0 Flash Lite model",
            "<3 second target response time"
        ]
    }

@app.get("/cache-stats")
async def cache_stats():
    return {
        "cached_responses": len(_cache),
        "cache_ttl_seconds": CACHE_TTL
    }

# Training prompt endpoints (simplified)
_training_prompt = SYSTEM_PROMPT

@app.get("/training-prompt")
async def get_training_prompt():
    return {"status": "success", "current_training_prompt": _training_prompt}

@app.post("/training-prompt")
async def update_training_prompt(data: dict):
    global _training_prompt, SYSTEM_PROMPT
    if "training_prompt" in data:
        _training_prompt = data["training_prompt"]
        SYSTEM_PROMPT = _training_prompt
        _cache.clear()  # Clear cache on prompt change
    return {"status": "success", "current_training_prompt": _training_prompt}

@app.post("/reset-training")
async def reset_training():
    global _training_prompt, SYSTEM_PROMPT
    _training_prompt = SYSTEM_PROMPT
    _cache.clear()
    return {"status": "success", "message": "Reset to default"}

# Cloud Run probes
@app.get("/readiness")
async def readiness():
    return {"status": "ready"}

@app.get("/startup")
async def startup():
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
