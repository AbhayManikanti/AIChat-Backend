"""
FastAPI Backend for Vercel Serverless
Dual LLM: Gemini (primary) + Perplexity (fallback)
"""

import os
import re
import time
import json
import httpx
from typing import Optional, List, Dict, Literal

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
    "intern", "internship", "resume", "cv", "portfolio", "built", "developed",
    "study", "studied", "school", "learning", "aegisai", "parkease", "fraud"
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

# Stronger prompt for Perplexity to prevent breaking character
PERPLEXITY_SYSTEM_PROMPT = """CRITICAL: You ARE Abhay Manikanti. Respond ONLY as Abhay in first person.
DO NOT cite sources in your response. DO NOT use [1], [2] citations. DO NOT be a research assistant.
You are a witty, confident tech enthusiast answering casually. Keep it personal and conversational.

Contact: Abhay.manikanti@gmail.com | +91 6366626970
Links: <a href="https://linkedin.com/in/abhay-manikanti-504a6b1b3" target="_blank">LinkedIn</a> | <a href="https://github.com/AbhayManikanti" target="_blank">GitHub</a>

Rules:
- Speak as Abhay, first person only ("I think...", "In my experience...")
- Be witty, slightly sarcastic, approachable
- Give your PERSONAL opinion, not research summaries
- NO citation numbers like [1] [2] in your text
- Use emojis sparingly"""

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
    provider: Optional[Literal["gemini", "perplexity", "auto"]] = "auto"
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        return v.strip()

class AnswerResponse(BaseModel):
    answer: str
    confidence: float = 0.9
    sources: List[str] = []
    used_resume: bool = False
    provider: str = "gemini"
    response_time: float = 0.0

# === LLM PROVIDERS ===

_gemini_model = None

def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _gemini_model = genai.GenerativeModel(
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
    return _gemini_model

def call_gemini(prompt: str) -> str:
    """Call Gemini API and return response text."""
    model = get_gemini_model()
    response = model.generate_content(prompt)
    return response.text.strip()

def call_perplexity(question: str, system_prompt: str) -> tuple[str, List[str]]:
    """Call Perplexity API and return (response text, citations)."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY not configured")
    
    # Use verify=False only for local testing (detects Vercel environment)
    is_vercel = os.getenv("VERCEL") is not None
    verify_ssl = True if is_vercel else False  # Disable SSL verify locally due to macOS cert issues
    
    with httpx.Client(timeout=25.0, verify=verify_ssl) as client:
        response = client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": os.getenv("PERPLEXITY_MODEL", "sonar"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        data = response.json()
        
        answer = data["choices"][0]["message"]["content"].strip()
        citations = data.get("citations", [])
        return answer, citations

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
    return {
        "status": "healthy", 
        "gemini_configured": os.getenv("GOOGLE_API_KEY") is not None,
        "perplexity_configured": os.getenv("PERPLEXITY_API_KEY") is not None
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    question = request.question
    provider_choice = request.provider or "auto"
    
    # Check cache
    cached = get_cached(question)
    if cached:
        return AnswerResponse(**cached)
    
    uses_resume = is_resume_question(question)
    system_prompt = SYSTEM_PROMPT
    if uses_resume:
        system_prompt += f"\n\nYour Background:\n{RESUME_CONTENT}"
    
    start_time = time.time()
    answer = None
    sources = []
    provider_used = "gemini"
    errors = []
    
    # Determine provider order
    if provider_choice == "gemini":
        providers = ["gemini"]
    elif provider_choice == "perplexity":
        providers = ["perplexity"]
    else:  # auto - try gemini first, fallback to perplexity
        providers = ["gemini", "perplexity"]
    
    for provider in providers:
        try:
            if provider == "gemini":
                prompt = build_prompt(question)
                answer = call_gemini(prompt)
                provider_used = "gemini"
                break
            elif provider == "perplexity":
                # Use stronger prompt for Perplexity to maintain character
                pplx_prompt = PERPLEXITY_SYSTEM_PROMPT
                if uses_resume:
                    pplx_prompt += f"\n\nYour Background:\n{RESUME_CONTENT}"
                answer, sources = call_perplexity(question, pplx_prompt)
                provider_used = "perplexity"
                break
        except Exception as e:
            errors.append(f"{provider}: {str(e)[:100]}")
            continue
    
    elapsed = time.time() - start_time
    
    if answer is None:
        raise HTTPException(
            status_code=500, 
            detail=f"All providers failed: {'; '.join(errors)}"
        )
    
    answer = convert_links(answer)
    
    result = {
        "answer": answer,
        "confidence": 0.95 if uses_resume else 0.9,
        "sources": sources[:5],  # Limit to 5 sources
        "used_resume": uses_resume,
        "provider": provider_used,
        "response_time": round(elapsed, 3)
    }
    
    set_cache(question, result)
    return AnswerResponse(**result)

@app.post("/reset")
async def reset():
    _cache.clear()
    return {"status": "success"}

@app.get("/info")
async def info():
    return {
        "version": "3.3.0", 
        "platform": "vercel", 
        "providers": {
            "gemini": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
            "perplexity": os.getenv("PERPLEXITY_MODEL", "sonar")
        }
    }
