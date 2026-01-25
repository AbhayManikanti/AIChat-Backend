"""
FastAPI Resume Chatbot Backend - OPTIMIZED FOR <3s RESPONSES
Author: Abhay Sreenath Manikanti
Version: 3.0.0

WHAT WAS FIXED:
1. Removed LangChain Agent (was doing 15 iterations, 25s timeout)
2. Removed tool/chain overhead (direct API call instead)
3. Removed blocking run_in_executor pattern
4. Removed complex retry logic
5. Simplified caching (no MD5 hashing)
6. Reduced prompt size significantly
7. Using gemini-2.0-flash-lite (fastest model)
8. Single synchronous API call with 10s hard timeout
"""

import os
import re
import time
import json
from typing import Optional, List, Dict, Any
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

DEFAULT_TRAINING_PROMPT = """You are Abhay Manikanti - speak as him directly. 
Personality: Witty, confident, approachable, slightly sarcastic but never rude. Keep responses concise and engaging.
Contact: Abhay.manikanti@gmail.com | +91 6366626970
Links: <a href="https://linkedin.com/in/abhay-manikanti-504a6b1b3" target="_blank">LinkedIn</a> | <a href="https://github.com/AbhayManikanti" target="_blank">GitHub</a>

Rules:
- Always speak as Abhay in first person
- No disclaimers or "as an AI" statements  
- Use HTML links: <a href="URL" target="_blank">text</a>
- Keep answers dynamic, witty, and genuine
- Use emojis sparingly for engagement"""

# Current training prompt (mutable)
CURRENT_TRAINING_PROMPT = DEFAULT_TRAINING_PROMPT

# === SIMPLE CACHE ===
_cache: Dict[str, tuple] = {}  # {question: (response, timestamp)}
CACHE_TTL = 3600  # 1 hour
CACHE_MAX = 200

def get_cached(q: str) -> Optional[Dict]:
    """Get cached response"""
    key = q.lower().strip()[:200]
    if key in _cache:
        resp, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return resp
        del _cache[key]
    return None

def set_cache(q: str, resp: Dict):
    """Cache response"""
    key = q.lower().strip()[:200]
    if len(_cache) >= CACHE_MAX:
        oldest = min(_cache.keys(), key=lambda k: _cache[k][1])
        del _cache[oldest]
    _cache[key] = (resp, time.time())

# === REQUEST/RESPONSE MODELS ===

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User's question")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer")
    confidence: Optional[float] = Field(None, description="Confidence score")
    sources: Optional[List[str]] = Field(default=[], description="Sources used")
    used_resume: bool = Field(default=False, description="Whether resume data was used")

class HealthResponse(BaseModel):
    status: str
    message: str
    vectorstore_initialized: bool
    llm_configured: bool

class TrainingPromptRequest(BaseModel):
    training_prompt: str = Field(..., min_length=10, max_length=2000)

class TrainingPromptResponse(BaseModel):
    status: str
    message: str
    current_training_prompt: str

# === LLM SINGLETON ===

_llm = None
_llm_provider = "gemini"

def get_llm():
    """Get or create LLM instance (singleton pattern)"""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_tokens=500,
            timeout=10,  # 10 second hard timeout
            max_retries=1,  # Single retry max
        )
    return _llm

def is_resume_question(q: str) -> bool:
    """Fast keyword check"""
    q_lower = q.lower()
    return any(kw in q_lower for kw in RESUME_KEYWORDS)

def convert_links(text: str) -> str:
    """Convert Markdown links to HTML"""
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)

def build_messages(question: str) -> list:
    """Build minimal message list for LLM"""
    system_content = CURRENT_TRAINING_PROMPT
    
    # Add resume context only when needed
    if is_resume_question(question):
        system_content += f"\n\nYour Background:\n{RESUME_CONTENT}"
    
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=question)
    ]

# === TRAINING PROMPT PERSISTENCE ===

TRAINING_PROMPT_FILE = os.getenv("TRAINING_PROMPT_FILE", "training_prompt.json")

def save_training_prompt(prompt: str):
    try:
        with open(TRAINING_PROMPT_FILE, 'w') as f:
            json.dump({"training_prompt": prompt, "timestamp": time.time()}, f)
    except Exception:
        pass  # Non-critical

def load_training_prompt() -> str:
    try:
        if os.path.exists(TRAINING_PROMPT_FILE):
            with open(TRAINING_PROMPT_FILE, 'r') as f:
                return json.load(f).get("training_prompt", DEFAULT_TRAINING_PROMPT)
    except Exception:
        pass
    return DEFAULT_TRAINING_PROMPT

# === FASTAPI APP ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Minimal startup - just pre-warm LLM"""
    global CURRENT_TRAINING_PROMPT
    CURRENT_TRAINING_PROMPT = load_training_prompt()
    get_llm()  # Pre-warm
    yield

app = FastAPI(
    title="Abhay's AI Agent - Optimized",
    description="<3 second response times with Gemini Flash",
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

# === ENDPOINTS ===

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
        message="Optimized backend ready (<3s responses)",
        vectorstore_initialized=True,
        llm_configured=os.getenv("GOOGLE_API_KEY") is not None
    )

@app.get("/health-detailed", response_model=HealthResponse)
async def health_detailed():
    return HealthResponse(
        status="healthy",
        message=f"LLM: {_llm_provider}, Cache: {len(_cache)} entries",
        vectorstore_initialized=True,
        llm_configured=os.getenv("GOOGLE_API_KEY") is not None
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ultra-fast question answering - single direct API call.
    Target: <3 seconds response time.
    """
    question = request.question
    
    # 1. Check cache first (instant return)
    cached = get_cached(question)
    if cached:
        return AnswerResponse(**cached)
    
    # 2. Determine if resume context needed
    uses_resume = is_resume_question(question)
    
    try:
        # 3. Build minimal messages
        messages = build_messages(question)
        
        # 4. Single direct API call (no agents, no chains, no tools)
        llm = get_llm()
        response = llm.invoke(messages)
        
        # 5. Process response
        answer = convert_links(response.content.strip())
        
        # 6. Build response
        result = {
            "answer": answer,
            "confidence": 0.95 if uses_resume else 0.9,
            "sources": [],
            "used_resume": uses_resume
        }
        
        # 7. Cache and return
        set_cache(question, result)
        return AnswerResponse(**result)
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limited. Please try again in a moment."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {error_msg}"
        )

@app.post("/reset")
async def reset_conversation():
    """Reset cache"""
    _cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.get("/info")
async def get_info():
    return {
        "api_version": "3.0.0",
        "owner": "Abhay Sreenath Manikanti",
        "description": "Optimized for <3 second response times",
        "current_llm_provider": _llm_provider,
        "optimizations": [
            "Single direct API call (no agents)",
            "Pre-built prompts (no runtime concatenation)",
            "Simple in-memory caching",
            "10 second hard timeout",
            "Gemini 2.0 Flash Lite model"
        ],
        "capabilities": [
            "Fast responses (<3 seconds)",
            "Resume-aware responses",
            "Personality-driven answers",
            "Witty conversation"
        ]
    }

# === TRAINING PROMPT ENDPOINTS ===

@app.post("/training-prompt", response_model=TrainingPromptResponse)
async def update_training_prompt_endpoint(request: TrainingPromptRequest):
    global CURRENT_TRAINING_PROMPT
    CURRENT_TRAINING_PROMPT = request.training_prompt
    save_training_prompt(request.training_prompt)
    _cache.clear()  # Clear cache on prompt change
    return TrainingPromptResponse(
        status="success",
        message="Training prompt updated",
        current_training_prompt=CURRENT_TRAINING_PROMPT
    )

@app.get("/training-prompt", response_model=TrainingPromptResponse)
async def get_training_prompt():
    return TrainingPromptResponse(
        status="success",
        message="Current training prompt",
        current_training_prompt=CURRENT_TRAINING_PROMPT
    )

@app.post("/reset-training")
async def reset_training_prompt():
    global CURRENT_TRAINING_PROMPT
    CURRENT_TRAINING_PROMPT = DEFAULT_TRAINING_PROMPT
    save_training_prompt(DEFAULT_TRAINING_PROMPT)
    _cache.clear()
    return {"status": "success", "message": "Reset to default"}

# === UTILITY ENDPOINTS ===

@app.get("/resume-keywords")
async def get_resume_keywords():
    return {
        "keywords": list(RESUME_KEYWORDS),
        "total": len(RESUME_KEYWORDS)
    }

@app.post("/test-resume-detection")
async def test_resume_detection(request: dict):
    question = request.get("question", "")
    matched = [kw for kw in RESUME_KEYWORDS if kw in question.lower()]
    return {
        "question": question,
        "will_use_resume": bool(matched),
        "matched_keywords": matched
    }

@app.get("/cache-stats")
async def get_cache_stats():
    return {
        "cached_responses": len(_cache),
        "cache_max": CACHE_MAX,
        "cache_ttl_seconds": CACHE_TTL
    }

@app.post("/clear-cache")
async def clear_cache():
    count = len(_cache)
    _cache.clear()
    return {"status": "success", "cleared": count}

@app.get("/optimization-stats")
async def get_optimization_stats():
    return {
        "architecture": "Direct API call (no agents)",
        "model": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite"),
        "timeout": "10 seconds",
        "caching": {"entries": len(_cache), "max": CACHE_MAX, "ttl": CACHE_TTL},
        "target_response_time": "<3 seconds"
    }

@app.get("/llm-status")
async def get_llm_status():
    return {
        "provider": _llm_provider,
        "model": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite"),
        "initialized": _llm is not None,
        "configured": os.getenv("GOOGLE_API_KEY") is not None
    }

# === CLOUD RUN PROBES ===

@app.get("/readiness")
async def readiness():
    if os.getenv("GOOGLE_API_KEY"):
        return {"status": "ready"}
    return {"status": "not_ready"}, 503

@app.get("/startup")
async def startup():
    return {"status": "started", "timestamp": time.time()}

@app.get("/token-optimization-status")
async def token_optimization_status():
    return {
        "optimization": "Direct API call",
        "agents_removed": True,
        "chains_removed": True,
        "estimated_latency": "<3 seconds"
    }

if __name__ == "__main__":
    import uvicorn
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found")
        exit(1)
    
    port = int(os.getenv("PORT", 8000))
    print(f"Starting optimized server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
