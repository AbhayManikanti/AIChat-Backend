"""
FastAPI Resume Chatbot Backend with LangChain - OPTIMIZED VERSION
Author: Abhay Sreenath Manikanti
Description: AI-powered resume chatbot using FastAPI, LangChain, and ChromaDB with API optimizations
"""

import os
import logging
import json
import time
import hashlib
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from functools import lru_cache

# SQLite3 compatibility fix for ChromaDB
try:
    # Try to use pysqlite3 if available (newer SQLite3 version)
    import pysqlite3.dbapi2 as sqlite3
except ImportError:
    # Fall back to system sqlite3
    import sqlite3

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import math
import operator

# Import constants
from constants import RESUME_CONTENT, RESUME_KEYWORDS, DEFAULT_TRAINING_PROMPT

# Load environment variables
load_dotenv("keys.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check SQLite3 compatibility for ChromaDB
def check_sqlite_compatibility():
    """Check SQLite3 version compatibility with ChromaDB - Cloud Run optimized"""
    try:
        # Detect if running in Google Cloud Run
        is_cloud_run = os.getenv("K_SERVICE") is not None
        
        if is_cloud_run:
            logger.info("Running in Google Cloud Run environment")
        
        # Check if we're using pysqlite3 or system sqlite3
        if 'pysqlite3' in sqlite3.__file__:
            logger.info("Using pysqlite3 for enhanced SQLite3 compatibility")
        else:
            logger.info(f"Using system SQLite3 version: {sqlite3.sqlite_version}")
            
            # Check version compatibility
            version_parts = [int(x) for x in sqlite3.sqlite_version.split('.')]
            if version_parts < [3, 35, 0]:
                logger.warning(f"SQLite3 version {sqlite3.sqlite_version} may be incompatible with ChromaDB (requires >= 3.35.0)")
                
                if is_cloud_run:
                    logger.info("In Cloud Run: Ensure your Dockerfile installs sqlite3 >= 3.35.0")
                    logger.info("Add to Dockerfile: RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev")
                else:
                    logger.info("Consider updating system SQLite3 or using Docker deployment")
                
                return False
        
        # Test ChromaDB initialization
        try:
            import chromadb
            test_client = chromadb.Client()
            logger.info("ChromaDB compatibility test passed")
            return True
        except Exception as chroma_error:
            logger.error(f"ChromaDB test failed: {chroma_error}")
            if "sqlite" in str(chroma_error).lower():
                logger.error("This appears to be a SQLite3 compatibility issue")
                if is_cloud_run:
                    logger.error("Rebuild your Cloud Run container with updated SQLite3")
            return False
            
    except Exception as e:
        logger.error(f"Error checking SQLite3 compatibility: {e}")
        return False

# Check compatibility at startup
check_sqlite_compatibility()

# === OPTIMIZATION 1: Response Caching ===
# Simple in-memory cache for responses
response_cache: Dict[str, Any] = {}
CACHE_SIZE = 100
CACHE_TTL = 3600  # 1 hour

def get_question_hash(question: str) -> str:
    """Generate hash for question caching"""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def is_cache_valid(cache_entry: Dict) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - cache_entry.get("timestamp", 0) < CACHE_TTL

def get_cached_response(question: str) -> Optional[Dict]:
    """Get cached response if available and valid"""
    question_hash = get_question_hash(question)
    if question_hash in response_cache:
        entry = response_cache[question_hash]
        if is_cache_valid(entry):
            logger.info(f"Cache hit for question: {question[:50]}...")
            return entry["response"]
        else:
            # Remove expired entry
            del response_cache[question_hash]
    return None

def cache_response(question: str, response: Dict):
    """Cache a response"""
    question_hash = get_question_hash(question)
    
    # Remove oldest entries if cache is full
    while len(response_cache) >= CACHE_SIZE:
        oldest_key = min(response_cache.keys(), 
                        key=lambda k: response_cache[k]["timestamp"])
        del response_cache[oldest_key]
    
    response_cache[question_hash] = {
        "response": response,
        "timestamp": time.time()
    }

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User's question")
    
    @validator('question')
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer")
    confidence: Optional[float] = Field(None, description="Confidence score of the answer")
    sources: Optional[List[str]] = Field(default=[], description="Relevant sources used")
    used_resume: bool = Field(default=False, description="Whether resume data was used")

class HealthResponse(BaseModel):
    status: str
    message: str
    vectorstore_initialized: bool
    llm_configured: bool

class TrainingPromptRequest(BaseModel):
    training_prompt: str = Field(..., min_length=10, max_length=2000, description="New training prompt for personality")
    
    @validator('training_prompt')
    def validate_training_prompt(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Training prompt cannot be empty")
        return v

class TrainingPromptResponse(BaseModel):
    status: str
    message: str
    current_training_prompt: str

# Global variables
vectorstore = None
agent_executor = None
memory = None
llm = None  # Direct LLM for responses
fallback_llm = None  # Perplexity LLM for rate limit fallback
current_llm_provider = "gemini"  # Track which LLM is currently active

# Lazy initialization flags
_llm_initialized = False
_fallback_llm_initialized = False
_agent_initialized = False

# Current training prompt (personality only)
CURRENT_TRAINING_PROMPT = DEFAULT_TRAINING_PROMPT

# File paths
TRAINING_PROMPT_FILE = os.getenv("TRAINING_PROMPT_FILE", "training_prompt.json")
VECTORSTORE_DIR = "./chroma_db"  # Persistent storage

# === OPTIMIZATION 2: Smart Question Routing ===

def should_use_agent(question: str) -> bool:
    """Determine if question needs agent - always use agent for consistent responses"""
    return True  # Always use agent to ensure comprehensive and consistent responses



def contains_resume_keywords(text: str) -> bool:
    """Check if the text contains keywords that should trigger resume search"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in RESUME_KEYWORDS)

def save_training_prompt_to_file(prompt: str):
    """Save training prompt to file for persistence"""
    try:
        data = {"training_prompt": prompt, "timestamp": str(time.time())}
        with open(TRAINING_PROMPT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Training prompt saved to {TRAINING_PROMPT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save training prompt: {str(e)}")

def load_training_prompt_from_file():
    """Load training prompt from file"""
    try:
        if os.path.exists(TRAINING_PROMPT_FILE):
            with open(TRAINING_PROMPT_FILE, 'r') as f:
                data = json.load(f)
                return data.get("training_prompt", DEFAULT_TRAINING_PROMPT)
        else:
            logger.info("No training prompt file found, using default")
            return DEFAULT_TRAINING_PROMPT
    except Exception as e:
        logger.error(f"Failed to load training prompt: {str(e)}")
        return DEFAULT_TRAINING_PROMPT

def update_training_prompt(new_training_prompt: str):
    """Update the training prompt and save to file"""
    global CURRENT_TRAINING_PROMPT
    CURRENT_TRAINING_PROMPT = new_training_prompt
    save_training_prompt_to_file(new_training_prompt)
    logger.info("Training prompt updated successfully")

# === OPTIMIZATION 3: Persistent Vector Storage ===

def initialize_vectorstore(resume_content: str) -> Chroma:
    """Initialize ChromaDB vectorstore with persistent storage and resume content"""
    try:
        # Check SQLite3 compatibility first
        if not check_sqlite_compatibility():
            logger.warning("Continuing despite SQLite3 version warnings...")
        
        # Check for GOOGLE API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Check if vectorstore already exists
        if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
            try:
                logger.info("Loading existing vectorstore from persistent storage...")
                vectorstore = Chroma(
                    persist_directory=VECTORSTORE_DIR,
                    embedding_function=embeddings,
                    collection_name="resume_collection"
                )
                
                # Verify it has documents
                if vectorstore._collection.count() > 0:
                    logger.info(f"Loaded existing vectorstore with {vectorstore._collection.count()} documents")
                    return vectorstore
                else:
                    logger.info("Existing vectorstore is empty, creating new one...")
            except Exception as e:
                logger.warning(f"Failed to load existing vectorstore: {str(e)}, creating new one...")
        
        # Create new vectorstore
        logger.info("Creating new vectorstore with persistent storage...")
        
        # Split the resume into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ",", " "],
            length_function=len
        )
        
        # Create documents from resume content
        chunks = text_splitter.split_text(resume_content)
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": "resume", "chunk_id": i}
            ) 
            for i, chunk in enumerate(chunks)
        ]
        
        # Create and populate vectorstore with persistence
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="resume_collection",
            persist_directory=VECTORSTORE_DIR  # Enable persistence
        )
        
        logger.info(f"Vectorstore created and persisted with {len(documents)} document chunks")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore: {str(e)}")
        raise

def create_calculator_tool():
    """Create a calculator tool for mathematical operations"""
    def calculate(expression: str) -> str:
        """Calculate mathematical expressions safely"""
        try:
            # Remove any potentially dangerous characters
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            # Evaluate the expression
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return Tool(
        name="Calculator",
        description="Useful for performing mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'.",
        func=calculate
    )

def create_resume_search_tool(vectorstore: Chroma):
    """Create a tool for searching resume information - only use when asked about background"""
    def search_resume(query: str) -> str:
        """Search for information in Abhay's resume - ONLY use when specifically asked about professional background, education, skills, projects, or achievements"""
        try:
            docs = vectorstore.similarity_search(query, k=3)
            if not docs:
                return "No relevant information found in the resume."
            
            results = []
            for doc in docs:
                results.append(doc.page_content)
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching resume: {str(e)}"
    
    return Tool(
        name="ResumeSearch",
        description="ONLY use this tool when the user specifically asks about Abhay's professional background, education, qualifications, skills, projects, work experience, or achievements. Do NOT use for general conversation, opinions, or non-professional topics.",
        func=search_resume
    )

def create_smart_system_prompt(user_question: str) -> str:
    """Create a context-aware system prompt based on the user's question"""
    base_prompt = f"""You are Abhay, an AI engineer with a unique personality. 

{CURRENT_TRAINING_PROMPT}

You have access to tools for calculations and resume information, but be smart about when to use them:

"""
    
    if contains_resume_keywords(user_question):
        base_prompt += """
IMPORTANT: The user is asking about your professional background, education, skills, or achievements. 
You should use the ResumeSearch tool to provide accurate information about your experience.
After getting the resume information, present it naturally as if you're talking about yourself.
"""
    else:
        base_prompt += """
IMPORTANT: The user is NOT asking about your professional background. 
Do NOT use the ResumeSearch tool unless they specifically ask about your work, education, skills, or achievements.
Just be yourself - witty, engaging, and helpful with whatever they're asking about.
"""
    
    return base_prompt

def initialize_gemini_llm():
    """Initialize Gemini LLM"""
    try:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
        raise

def initialize_perplexity_llm():
    """Initialize Perplexity LLM as fallback"""
    try:
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if not perplexity_api_key:
            logger.warning("PERPLEXITY_API_KEY not found - fallback LLM will not be available")
            return None
            
        return ChatOpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai",
            model=os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online"),
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Perplexity LLM: {str(e)}")
        return None

def initialize_llm():
    """Initialize the primary LLM for both direct use and agent use"""
    global current_llm_provider
    try:
        current_llm_provider = "gemini"
        return initialize_gemini_llm()
    except Exception as e:
        logger.error(f"Failed to initialize primary LLM: {str(e)}")
        raise

def ensure_llm_initialized():
    """Lazy initialization of primary LLM - only called when needed"""
    global llm, _llm_initialized
    if not _llm_initialized or llm is None:
        logger.info("Lazy initializing primary LLM (Gemini)...")
        llm = initialize_llm()
        _llm_initialized = True
        logger.info("Primary LLM initialized successfully")
    return llm

def ensure_fallback_llm_initialized():
    """Lazy initialization of fallback LLM - only called when needed"""
    global fallback_llm, _fallback_llm_initialized
    if not _fallback_llm_initialized:
        logger.info("Lazy initializing fallback LLM (Perplexity)...")
        fallback_llm = initialize_perplexity_llm()
        _fallback_llm_initialized = True
        if fallback_llm:
            logger.info("Fallback LLM initialized successfully")
        else:
            logger.warning("Fallback LLM not available")
    return fallback_llm

def ensure_agent_initialized():
    """Lazy initialization of agent - only called when needed"""
    global agent_executor, _agent_initialized
    if not _agent_initialized or agent_executor is None:
        logger.info("Lazy initializing agent...")
        # Ensure prerequisites are available
        if vectorstore is None:
            raise RuntimeError("Vectorstore must be initialized before agent")
        
        # Ensure LLM is initialized
        current_llm = ensure_llm_initialized()
        
        # Initialize agent
        agent_executor = initialize_agent(vectorstore, current_llm)
        _agent_initialized = True
        logger.info("Agent initialized successfully")
    return agent_executor

def is_rate_limit_error(error) -> bool:
    """Check if the error is a rate limit (429) or quota exceeded error"""
    error_str = str(error).lower()
    rate_limit_indicators = [
        "429", "rate limit", "quota exceeded", "too many requests", 
        "resource exhausted", "rate_limit_exceeded", "quota_exceeded"
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)

def switch_to_fallback_llm():
    """Switch to Perplexity LLM when Gemini hits rate limits"""
    global llm, agent_executor, current_llm_provider, _llm_initialized, _agent_initialized
    
    # Lazy initialize fallback LLM if not already done
    fallback = ensure_fallback_llm_initialized()
    if not fallback:
        logger.error("No fallback LLM available - cannot switch")
        return False
    
    try:
        logger.warning("Switching to Perplexity LLM due to Gemini rate limit")
        llm = fallback
        current_llm_provider = "perplexity"
        _llm_initialized = True
        
        # Re-initialize agent with new LLM
        if vectorstore:
            agent_executor = initialize_agent(vectorstore, llm)
            _agent_initialized = True
            logger.info("Successfully switched to Perplexity LLM and re-initialized agent")
            return True
        else:
            logger.error("Cannot re-initialize agent - vectorstore not available")
            return False
            
    except Exception as e:
        logger.error(f"Failed to switch to fallback LLM: {str(e)}")
        return False

def reset_to_primary_llm():
    """Try to reset back to Gemini LLM (call this periodically or on user request)"""
    global llm, agent_executor, current_llm_provider
    
    if current_llm_provider == "gemini":
        return True  # Already using primary
    
    try:
        logger.info("Attempting to reset to Gemini LLM")
        gemini_llm = initialize_gemini_llm()
        
        # Just initialize without testing (save tokens)
        llm = gemini_llm
        current_llm_provider = "gemini"
        _llm_initialized = True
        
        # Re-initialize agent
        if vectorstore:
            agent_executor = initialize_agent(vectorstore, llm)
            _agent_initialized = True
            logger.info("Successfully reset to Gemini LLM")
            return True
                
    except Exception as e:
        if is_rate_limit_error(e):
            logger.info("Gemini still rate limited, staying on Perplexity")
        else:
            logger.error(f"Error testing Gemini reset: {str(e)}")
        return False

def initialize_agent(vectorstore: Chroma, llm_instance) -> AgentExecutor:
    """Initialize the AI agent with tools and optimized settings"""
    try:
        # Create tools
        tools = [
            create_calculator_tool(),
            create_resume_search_tool(vectorstore)
        ]
        
        # Create the prompt template
        prompt = PromptTemplate.from_template("""
{system_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
        
        # Create the agent
        agent = create_react_agent(llm_instance, tools, prompt)
        
        # Create agent executor with optimized settings
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,  # Reduced from 5 to save API calls
            early_stopping_method="generate"  # Stop early when possible
        )
        
        logger.info("AI Agent initialized successfully with optimizations")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app - TOKEN OPTIMIZED"""
    # Startup
    global vectorstore, CURRENT_TRAINING_PROMPT
    global agent_executor, memory, llm, fallback_llm
    global _llm_initialized, _fallback_llm_initialized, _agent_initialized
    
    try:
        logger.info("Initializing AI Agent Backend with token optimization...")
        
        # Load training prompt from file (no tokens consumed)
        CURRENT_TRAINING_PROMPT = load_training_prompt_from_file()
        
        # Only initialize vectorstore (no tokens consumed if it already exists)
        vectorstore = initialize_vectorstore(RESUME_CONTENT)
        
        # LLM and agent will be lazy-loaded on first request
        logger.info("AI Agent Backend initialized successfully - LLMs will be loaded on first request")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Agent Backend...")
    
    # Reset all components
    vectorstore = None
    agent_executor = None
    memory = None
    llm = None
    fallback_llm = None
    
    # Reset initialization flags
    _llm_initialized = False
    _fallback_llm_initialized = False
    _agent_initialized = False

# Initialize FastAPI app
app = FastAPI(
    title="Abhay's AI Agent - Optimized",
    description="Intelligent AI agent with API optimizations for reduced costs and faster responses",
    version="2.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check (no token consumption)"""
    return HealthResponse(
        status="healthy",
        message="Abhay's AI Agent is running (Token Optimized)",
        vectorstore_initialized=vectorstore is not None,
        llm_configured=os.getenv("GOOGLE_API_KEY") is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint optimized for Cloud Run (no token consumption)"""
    # Detect Cloud Run environment
    is_cloud_run = os.getenv("K_SERVICE") is not None
    
    # Check system readiness without initializing LLMs
    vectorstore_ready = vectorstore is not None
    llm_configured = os.getenv("GOOGLE_API_KEY") is not None
    fallback_configured = os.getenv("PERPLEXITY_API_KEY") is not None
    
    # More detailed status for Cloud Run
    if vectorstore_ready and llm_configured:
        status_msg = "healthy"
        if _llm_initialized:
            message = f"All systems operational (LLM: {current_llm_provider})"
        else:
            message = "Ready for requests (LLMs will initialize on first use - token optimized)"
            if is_cloud_run:
                message += " | Cloud Run deployment"
    else:
        status_msg = "not_ready" 
        missing_items = []
        if not vectorstore_ready:
            missing_items.append("vectorstore")
        if not llm_configured:
            missing_items.append("GOOGLE_API_KEY")
        message = f"Missing: {', '.join(missing_items)}"
    
    return HealthResponse(
        status=status_msg,
        message=message,
        vectorstore_initialized=vectorstore_ready,
        llm_configured=llm_configured and (fallback_configured if is_cloud_run else True)
    )

@app.post("/ask", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for asking questions to Abhay's AI agent - WITH RATE LIMIT PROTECTION
    
    Features:
    - Automatic fallback from Gemini to Perplexity on rate limits (429 errors)
    - Response caching for identical questions
    - Consistent personality across LLM providers
    - Context-aware resume searching
    
    Args:
        request: QuestionRequest containing the user's question
        
    Returns:
        AnswerResponse with AI-generated answer and metadata
    """
    try:
        # Validate vectorstore is ready (this doesn't consume tokens)
        if not vectorstore:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized. Please try again later."
            )
            
        # Lazy initialize LLM and agent only when first request comes in
        try:
            ensure_agent_initialized()  # This will also ensure LLM is initialized
        except Exception as e:
            logger.error(f"Failed to initialize LLM/Agent on first request: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize AI components. Please try again later."
            )
        
        logger.info(f"Processing question: {request.question}")
        
        # === OPTIMIZATION: Check cache first ===
        cached_response = get_cached_response(request.question)
        if cached_response:
            return AnswerResponse(**cached_response)
        
        # === Always use agent for consistent responses ===
        logger.info(f"Using agent for question processing (Current LLM: {current_llm_provider})")
        
        # Check if question is about resume/background
        is_resume_question = contains_resume_keywords(request.question)
        
        # Create input for the agent with context-aware system prompt
        agent_input = {
            "input": request.question,
            "system_prompt": create_smart_system_prompt(request.question)
        }
        
        # Get answer from AI agent with rate limit handling
        result = None
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                result = agent_executor.invoke(agent_input)
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"Agent execution error (attempt {attempt + 1}): {str(e)}")
                
                # Check if it's a rate limit error
                if is_rate_limit_error(e):
                    logger.warning(f"Rate limit detected on attempt {attempt + 1}")
                    
                    # Try to switch to fallback LLM
                    if current_llm_provider == "gemini" and fallback_llm and switch_to_fallback_llm():
                        logger.info("Successfully switched to fallback LLM, retrying...")
                        continue  # Retry with fallback LLM
                    else:
                        logger.error("Cannot switch to fallback LLM or already using fallback")
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="AI service is temporarily rate limited. Please try again in a few moments."
                        )
                else:
                    # Non-rate limit error, don't retry
                    if attempt == max_retries - 1:  # Last attempt
                        raise
                    
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get response from AI agent after multiple attempts."
            )
        
        # Extract answer
        answer = result.get("output", "I couldn't process your question. Please try again.")
        
        # Check if resume search was actually used
        used_resume = False
        sources = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) >= 2:  # Check if there's an observation
                    action_name = step[0].tool if hasattr(step[0], 'tool') else str(step[0])
                    if "ResumeSearch" in action_name:
                        used_resume = True
                    if step[1]:
                        source_text = step[1][:100] + "..." if len(step[1]) > 100 else step[1]
                        sources.append(source_text)
        
        # Set confidence based on whether it's a resume question and if resume was used
        if is_resume_question and used_resume:
            confidence = 0.95  # High confidence for resume-based answers
        elif not is_resume_question:
            confidence = 0.9   # High confidence for general conversation
        else:
            confidence = 0.7   # Lower confidence if resume question but no resume data used
        
        response_data = {
            "answer": answer,
            "confidence": confidence,
            "sources": sources[:3],  # Limit to top 3 sources
            "used_resume": used_resume
        }
        
        # Cache the response
        cache_response(request.question, response_data)
        
        logger.info(f"Answer generated. Resume question: {is_resume_question}, Used resume: {used_resume}, Used agent: True")
        
        return AnswerResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your question: {str(e)}"
        )

@app.post("/reset")
async def reset_conversation():
    """Reset conversation memory and clear cache"""
    global memory, response_cache
    try:
        if memory:
            memory.clear()
        
        # Clear response cache
        response_cache.clear()
        
        logger.info("Conversation memory and cache reset successfully")
        return {"status": "success", "message": "Conversation and cache reset"}
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset conversation: {str(e)}"
        )

@app.get("/info")
async def get_info():
    """Get information about the API"""
    return {
        "api_version": "2.3.0",
        "owner": "Abhay Sreenath Manikanti",
        "description": "AI agent with token optimization and rate limit protection",
        "current_llm_provider": current_llm_provider if _llm_initialized else "none_initialized",
        "optimizations": [
            "Lazy LLM initialization (no tokens consumed until first request)",
            "Persistent vector storage (no re-embedding on restart)",
            "Response caching (identical questions cached for 1 hour)",
            "Reduced agent iterations (3 max instead of 5)",
            "Automatic LLM fallback on rate limits (Gemini → Perplexity)",
            "Rate limit error detection and recovery",
            "Zero token startup (LLMs only load when needed)"
        ],
        "capabilities": [
            "Witty and engaging conversation",
            "Mathematical calculations",
            "General knowledge and reasoning",
            "Professional background information (when asked)",
            "Personality-driven responses",
            "Context-aware resume searching",
            "Automatic rate limit handling"
        ],
        "personality_features": [
            "Human-like responses",
            "Witty and humorous",
            "Speaks as Abhay himself",
            "Context-aware behavior",
            "Consistent personality across LLM providers"
        ],
        "rate_limit_protection": {
            "enabled": os.getenv("PERPLEXITY_API_KEY") is not None,
            "primary_provider": "Google Gemini",
            "fallback_provider": "Perplexity AI",
            "automatic_switching": True,
            "lazy_initialization": True
        },
        "token_optimization": {
            "lazy_initialization": True,
            "zero_token_startup": True,
            "first_request_initializes": "LLM + Agent"
        },
        "powered_by": f"FastAPI + LangChain + Token Optimization + Rate Limit Protection"
    }

@app.post("/training-prompt", response_model=TrainingPromptResponse, status_code=status.HTTP_200_OK)
async def update_training_prompt_endpoint(request: TrainingPromptRequest):
    """Update the personality training prompt for the AI agent"""
    try:
        update_training_prompt(request.training_prompt)
        
        # Clear cache when training prompt changes
        global response_cache
        response_cache.clear()
        logger.info("Response cache cleared due to training prompt update")
        
        return TrainingPromptResponse(
            status="success",
            message="Personality training prompt updated successfully",
            current_training_prompt=CURRENT_TRAINING_PROMPT
        )
        
    except Exception as e:
        logger.error(f"Error updating training prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update training prompt: {str(e)}"
        )

@app.get("/training-prompt", response_model=TrainingPromptResponse)
async def get_training_prompt():
    """Get the current personality training prompt"""
    return TrainingPromptResponse(
        status="success",
        message="Current personality training prompt retrieved",
        current_training_prompt=CURRENT_TRAINING_PROMPT
    )

@app.post("/reset-training")
async def reset_training_prompt():
    """Reset the personality training prompt to default"""
    try:
        update_training_prompt(DEFAULT_TRAINING_PROMPT)
        
        # Clear cache when training prompt resets
        global response_cache
        response_cache.clear()
        
        logger.info("Training prompt reset to default and cache cleared")
        return {"status": "success", "message": "Personality training prompt reset to default"}
        
    except Exception as e:
        logger.error(f"Error resetting training prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset training prompt: {str(e)}"
        )

@app.get("/resume-keywords")
async def get_resume_keywords():
    """Get the list of keywords that trigger resume search"""
    return {
        "keywords": RESUME_KEYWORDS,
        "description": "These keywords in user questions will trigger resume-based responses",
        "total_keywords": len(RESUME_KEYWORDS)
    }

@app.post("/test-resume-detection")
async def test_resume_detection(request: dict):
    """Test if a question would trigger resume search"""
    question = request.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    will_use_resume = contains_resume_keywords(question)
    will_use_agent = should_use_agent(question)  # Always True now
    matched_keywords = [kw for kw in RESUME_KEYWORDS if kw in question.lower()]
    
    return {
        "question": question,
        "will_use_resume": will_use_resume,
        "will_use_agent": will_use_agent,
        "matched_keywords": matched_keywords,
        "routing": "Agent with resume search" if will_use_resume else "Agent without resume",
        "estimated_api_calls": "1-3 calls (always uses agent)",
        "llm_provider": current_llm_provider,
        "fallback_available": fallback_llm is not None
    }

# === NEW OPTIMIZATION ENDPOINTS ===

@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    total_entries = len(response_cache)
    valid_entries = sum(1 for entry in response_cache.values() if is_cache_valid(entry))
    
    return {
        "total_cached_responses": total_entries,
        "valid_cached_responses": valid_entries,
        "expired_entries": total_entries - valid_entries,
        "cache_size_limit": CACHE_SIZE,
        "cache_ttl_seconds": CACHE_TTL
    }

@app.post("/clear-cache")
async def clear_cache():
    """Manually clear the response cache"""
    global response_cache
    cache_size = len(response_cache)
    response_cache.clear()
    logger.info("Response cache cleared manually")
    return {
        "status": "success", 
        "message": f"Cleared {cache_size} cached responses"
    }

@app.get("/optimization-stats")
async def get_optimization_stats():
    """Get optimization statistics and settings"""
    vectorstore_exists = os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR)
    
    return {
        "persistent_storage": {
            "enabled": True,
            "directory": VECTORSTORE_DIR,
            "exists": vectorstore_exists
        },
        "caching": {
            "enabled": True,
            "current_entries": len(response_cache),
            "max_entries": CACHE_SIZE,
            "ttl_seconds": CACHE_TTL
        },
        "llm_management": {
            "current_provider": current_llm_provider,
            "fallback_available": fallback_llm is not None,
            "rate_limit_protection": "enabled" if fallback_llm else "disabled"
        },
        "agent_optimization": {
            "max_iterations": 3,
            "early_stopping": "generate"
        }
    }

@app.get("/llm-status")
async def get_llm_status():
    """Get current LLM provider status and availability (no token consumption)"""
    return {
        "current_provider": current_llm_provider if _llm_initialized else "none_initialized",
        "initialization_status": {
            "llm_initialized": _llm_initialized,
            "fallback_initialized": _fallback_llm_initialized,
            "agent_initialized": _agent_initialized
        },
        "primary_llm": {
            "provider": "gemini",
            "model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            "configured": os.getenv("GOOGLE_API_KEY") is not None,
            "initialized": _llm_initialized and current_llm_provider == "gemini"
        },
        "fallback_llm": {
            "provider": "perplexity",
            "model": os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online"),
            "configured": os.getenv("PERPLEXITY_API_KEY") is not None,
            "initialized": _fallback_llm_initialized
        },
        "rate_limit_protection": {
            "enabled": os.getenv("PERPLEXITY_API_KEY") is not None,
            "automatic_switching": True,
            "lazy_initialization": True
        }
    }

@app.post("/reset-to-primary-llm")
async def reset_to_primary_llm_endpoint():
    """Manually attempt to reset back to Gemini LLM"""
    try:
        if current_llm_provider == "gemini":
            return {
                "status": "info",
                "message": "Already using primary LLM (Gemini)",
                "current_provider": current_llm_provider
            }
        
        success = reset_to_primary_llm()
        if success:
            return {
                "status": "success",
                "message": "Successfully reset to primary LLM (Gemini)",
                "current_provider": current_llm_provider
            }
        else:
            return {
                "status": "failed",
                "message": "Could not reset to primary LLM - still rate limited or error occurred",
                "current_provider": current_llm_provider
            }
            
    except Exception as e:
        logger.error(f"Error in manual LLM reset: {str(e)}")
        return {
            "status": "error",
            "message": f"Error attempting to reset LLM: {str(e)}",
            "current_provider": current_llm_provider
        }

@app.get("/token-optimization-status")
async def get_token_optimization_status():
    """Get token optimization status and initialization state"""
    is_cloud_run = os.getenv("K_SERVICE") is not None
    
    return {
        "token_optimization": {
            "enabled": True,
            "zero_token_startup": True,
            "lazy_initialization": True,
            "cloud_run_optimized": is_cloud_run
        },
        "initialization_status": {
            "vectorstore_initialized": vectorstore is not None,
            "llm_initialized": _llm_initialized,
            "fallback_llm_initialized": _fallback_llm_initialized,
            "agent_initialized": _agent_initialized
        },
        "token_consumption": {
            "startup_tokens_used": 0,  # Always 0 with lazy initialization
            "tokens_used_only_on": "first_user_request",
            "current_provider": current_llm_provider if _llm_initialized else "none_initialized"
        },
        "environment": {
            "platform": "Google Cloud Run" if is_cloud_run else "Local/Other",
            "port": os.getenv("PORT", "8000"),
            "sqlite3_version": sqlite3.sqlite_version
        },
        "benefits": [
            "No tokens consumed during app startup",
            "LLM initialization only on first request",
            "Reduced unnecessary API calls", 
            "Faster startup time",
            "Cost optimization",
            "Cloud Run auto-scaling friendly" if is_cloud_run else "Deployment ready"
        ]
    }

@app.get("/readiness")
async def readiness_probe():
    """Cloud Run readiness probe - lightweight check"""
    # Quick readiness check for Cloud Run
    basic_ready = (
        vectorstore is not None and 
        os.getenv("GOOGLE_API_KEY") is not None
    )
    
    if basic_ready:
        return {"status": "ready", "message": "Service ready to accept requests"}
    else:
        return {"status": "not_ready", "message": "Service still initializing"}, 503

@app.get("/startup") 
async def startup_probe():
    """Cloud Run startup probe - check if app has started"""
    # Startup probe for Cloud Run
    try:
        # Just check if the app is responsive
        return {
            "status": "started",
            "message": "Application started successfully",
            "timestamp": time.time(),
            "sqlite3_compatible": check_sqlite_compatibility()
        }
    except Exception as e:
        return {"status": "failed", "message": f"Startup failed: {str(e)}"}, 503

if __name__ == "__main__":
    import uvicorn

    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found. Please set it in environment variables")
        exit(1)
    
    if not os.getenv("PERPLEXITY_API_KEY"):
        logger.warning("PERPLEXITY_API_KEY not found. Rate limit fallback will be disabled.")
        logger.info("Add PERPLEXITY_API_KEY to environment variables for automatic rate limit protection")
    
    # Google Cloud Run uses PORT environment variable
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting FastAPI server with rate limit protection on port {port}...")
    logger.info("Optimized for Google Cloud Run deployment")
    uvicorn.run(app, host="0.0.0.0", port=port)