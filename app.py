"""
FastAPI Resume Chatbot Backend with LangChain
Author: Abhay Sreenath Manikanti
Description: AI-powered resume chatbot using FastAPI, LangChain, and ChromaDB
"""

import os
import logging
import json
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import math
import operator

# Load environment variables
load_dotenv("keys.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Resume content
RESUME_CONTENT = """
Abhay Sreenath Manikanti

PROFESSIONAL EXPERIENCE:
- AI Intern at Fortive: Developed RPA automation solutions and AI chatbot, achieving £35,000 in operational savings
- Implemented intelligent process automation using cutting-edge AI technologies
- Designed and deployed chatbot solutions for enhanced customer interaction

EDUCATION:
- B.Tech in Information Science from BMSCE (Bangalore)
- GPA: 7.9/10
- Thesis: Ambulance Congestion Control System - Innovative solution for emergency vehicle routing

PROJECTS:
1. Park-Ease: Smart parking solution built with Flask and Docker
   - Containerized application for scalable deployment
   - Real-time parking availability tracking
   
2. Resume Screening System: Automated HR solution using UiPath
   - RPA-based resume filtering and ranking
   - Reduced manual screening time by 70%
   
3. COVID Data Analysis: Comprehensive data analysis project
   - Statistical modeling and visualization
   - Predictive analytics for trend forecasting

TECHNICAL SKILLS:
- Programming Languages: Python (Expert), JavaScript, SQL
- AI/ML: Machine Learning, Deep Learning, Natural Language Processing, Computer Vision
- RPA: UiPath, Process Automation, Bot Development
- DevOps: Docker, Kubernetes, CI/CD pipelines
- Data Analysis: Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch
- Process Optimization: Workflow automation, Business process improvement
- Cloud: AWS, Azure, Google Cloud Platform

ACHIEVEMENTS & PUBLICATIONS:
- Publication: "Ambulance Congestion Control System" published in International Journal of Creative Research Thoughts (IJCRT)
- Achieved £35,000 cost savings through RPA implementation at Fortive
- Academic Excellence: Maintained 7.9 GPA throughout B.Tech program

ADDITIONAL INFORMATION:
- Strong problem-solving and analytical skills
- Experience in leading cross-functional teams
- Excellent communication and presentation abilities
- Passionate about leveraging AI for real-world problem solving
"""

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User's question about the resume")
    
    @validator('question')
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer based on resume content")
    confidence: Optional[float] = Field(None, description="Confidence score of the answer")
    sources: Optional[List[str]] = Field(default=[], description="Relevant resume sections used")

class HealthResponse(BaseModel):
    status: str
    message: str
    vectorstore_initialized: bool
    llm_configured: bool

class TrainingPromptRequest(BaseModel):
    training_prompt: str = Field(..., min_length=10, max_length=2000, description="New training prompt for the AI agent")
    
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

# Global variables for storing initialized components
vectorstore = None
agent_executor = None
memory = None

# Default system prompt for the AI agent
DEFAULT_SYSTEM_PROMPT = """You are Abhay's AI Assistant, an intelligent and autonomous agent that can help with a wide variety of tasks. You have access to Abhay's resume information and can also perform general computations and reasoning.

Your capabilities include:
1. Answering questions about Abhay's professional experience, education, skills, and projects
2. Performing mathematical calculations and general reasoning
3. Having intelligent conversations on various topics
4. Providing thoughtful analysis and insights

When answering questions:
- Be conversational and helpful
- Use the resume search tool when questions relate to Abhay's background
- Use the calculator tool for mathematical operations
- Think step by step and show your reasoning
- If you don't know something, say so honestly
- Be autonomous and intelligent in your responses

Remember: You are not just a data retrieval system - you are an intelligent agent that can reason, calculate, and provide thoughtful responses."""

# Training prompt for deep customization (can be updated via API)
TRAINING_PROMPT = """You are a highly intelligent AI assistant with a unique personality and approach to problem-solving. You should:

1. Vary your response style and tone to feel more natural and human-like
2. Sometimes use different approaches to explain the same concept
3. Add personality quirks like occasional humor, analogies, or creative explanations
4. Don't always rely on resume data - use your general knowledge and reasoning
5. Be conversational and engaging, not robotic
6. Show your "thinking process" in different ways each time
7. Occasionally ask follow-up questions or provide additional insights
8. Use varied vocabulary and sentence structures

Your responses should feel like they're coming from a real AI that's learning and adapting, not a static system."""

# Current system prompt (combines default + training)
CURRENT_SYSTEM_PROMPT = f"{DEFAULT_SYSTEM_PROMPT}\n\nTRAINING INSTRUCTIONS:\n{TRAINING_PROMPT}"

# File path for persistent storage
TRAINING_PROMPT_FILE = os.getenv("TRAINING_PROMPT_FILE", "training_prompt.json")

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
                return data.get("training_prompt", TRAINING_PROMPT)
        else:
            logger.info("No training prompt file found, using default")
            return TRAINING_PROMPT
    except Exception as e:
        logger.error(f"Failed to load training prompt: {str(e)}")
        return TRAINING_PROMPT

def update_training_prompt(new_training_prompt: str):
    """Update the training prompt and regenerate the system prompt"""
    global TRAINING_PROMPT, CURRENT_SYSTEM_PROMPT
    TRAINING_PROMPT = new_training_prompt
    CURRENT_SYSTEM_PROMPT = f"{DEFAULT_SYSTEM_PROMPT}\n\nTRAINING INSTRUCTIONS:\n{TRAINING_PROMPT}"
    
    # Save to file for persistence
    save_training_prompt_to_file(new_training_prompt)
    logger.info("Training prompt updated successfully")

def get_dynamic_system_prompt():
    """Get a dynamic system prompt with slight variations"""
    import random
    
    # Add some dynamic elements to make responses feel more natural
    dynamic_elements = [
        "Remember to be conversational and engaging.",
        "Vary your response style to feel more human-like.",
        "Use your creativity and personality in responses.",
        "Think outside the box and provide unique insights.",
        "Be adaptive and show your AI intelligence.",
        "Connect ideas in creative and unexpected ways."
    ]
    
    # Randomly select one dynamic element to add
    selected_element = random.choice(dynamic_elements)
    
    return f"{CURRENT_SYSTEM_PROMPT}\n\nDYNAMIC INSTRUCTION: {selected_element}"

def initialize_vectorstore(resume_content: str) -> Chroma:
    """Initialize ChromaDB vectorstore with resume content"""
    try:
        # Check for GOOGLE API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
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
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create and populate vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="resume_collection",
            persist_directory=None  # In-memory storage
        )
        
        logger.info(f"Vectorstore initialized with {len(documents)} document chunks")
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
    """Create a tool for searching resume information"""
    def search_resume(query: str) -> str:
        """Search for information in Abhay's resume"""
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
        description="ONLY use this tool when specifically asked about Abhay's background, experience, education, skills, or projects. For general questions, use your knowledge instead.",
        func=search_resume
    )

def initialize_agent(vectorstore: Chroma) -> AgentExecutor:
    """Initialize the AI agent with tools"""
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
        )
        
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
        agent = create_react_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        logger.info("AI Agent initialized successfully")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    # Startup
    global vectorstore, agent_executor, memory, TRAINING_PROMPT, CURRENT_SYSTEM_PROMPT
    try:
        logger.info("Initializing AI Agent Backend...")
        
        # Load training prompt from file if it exists
        loaded_prompt = load_training_prompt_from_file()
        if loaded_prompt != TRAINING_PROMPT:
            TRAINING_PROMPT = loaded_prompt
            CURRENT_SYSTEM_PROMPT = f"{DEFAULT_SYSTEM_PROMPT}\n\nTRAINING INSTRUCTIONS:\n{TRAINING_PROMPT}"
            logger.info("Loaded custom training prompt from file")
        
        # Initialize components
        vectorstore = initialize_vectorstore(RESUME_CONTENT)
        agent_executor = initialize_agent(vectorstore)
        
        logger.info("AI Agent Backend initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Agent Backend...")
    if vectorstore:
        vectorstore = None
    if agent_executor:
        agent_executor = None
    if memory:
        memory = None

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="Intelligent AI agent that can answer questions, perform calculations, and provide insights about Abhay's background",
    version="2.0.0",
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
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        message="AI Agent API is running",
        vectorstore_initialized=vectorstore is not None,
        llm_configured=os.getenv("GOOGLE_API_KEY") is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if vectorstore and agent_executor else "unhealthy",
        message="All systems operational" if vectorstore and agent_executor else "System not fully initialized",
        vectorstore_initialized=vectorstore is not None,
        llm_configured=agent_executor is not None
    )

@app.post("/ask", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for asking questions to the AI agent
    
    Args:
        request: QuestionRequest containing the user's question
        
    Returns:
        AnswerResponse with AI-generated answer and metadata
    """
    try:
        # Validate system readiness
        if not vectorstore or not agent_executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized. Please try again later."
            )
        
        # Process the question
        logger.info(f"Processing question: {request.question}")
        
        # Create input for the agent with dynamic system prompt
        agent_input = {
            "input": request.question,
            "system_prompt": get_dynamic_system_prompt()
        }
        
        # Get answer from AI agent
        result = agent_executor.invoke(agent_input)
        
        # Extract answer
        answer = result.get("output", "I couldn't process your question. Please try again.")
        
        # For agent responses, we'll set a high confidence since the agent can reason
        confidence = 0.9
        
        # Extract sources if any tools were used (this is more complex with agents)
        sources = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) >= 2 and step[1]:  # Check if there's an observation
                    sources.append(step[1][:100] + "..." if len(step[1]) > 100 else step[1])
        
        logger.info(f"Answer generated successfully by AI agent")
        
        return AnswerResponse(
            answer=answer,
            confidence=confidence,
            sources=sources[:3]  # Limit to top 3 sources
        )
        
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
    """Reset conversation memory"""
    global memory
    try:
        if memory:
            memory.clear()
            logger.info("Conversation memory reset successfully")
            return {"status": "success", "message": "Conversation reset"}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory not initialized"
            )
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset conversation: {str(e)}"
        )

@app.get("/info")
async def get_info():
    """Get information about the API and resume owner"""
    return {
        "api_version": "2.0.0",
        "resume_owner": "Abhay Sreenath Manikanti",
        "capabilities": [
            "Answer questions about professional experience",
            "Provide details about education and skills",
            "Discuss projects and achievements",
            "Explain technical expertise",
            "Perform mathematical calculations",
            "General reasoning and problem solving",
            "Intelligent conversation on various topics",
            "Dynamic response variation",
            "Customizable training prompts"
        ],
        "powered_by": "FastAPI + LangChain + Google Gemini + AI Agent System"
    }

@app.post("/training-prompt", response_model=TrainingPromptResponse, status_code=status.HTTP_200_OK)
async def update_training_prompt_endpoint(request: TrainingPromptRequest):
    """
    Update the training prompt for the AI agent
    
    Args:
        request: TrainingPromptRequest containing the new training prompt
        
    Returns:
        TrainingPromptResponse with status and current training prompt
    """
    try:
        update_training_prompt(request.training_prompt)
        
        return TrainingPromptResponse(
            status="success",
            message="Training prompt updated successfully",
            current_training_prompt=TRAINING_PROMPT
        )
        
    except Exception as e:
        logger.error(f"Error updating training prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update training prompt: {str(e)}"
        )

@app.get("/training-prompt", response_model=TrainingPromptResponse)
async def get_training_prompt():
    """Get the current training prompt"""
    return TrainingPromptResponse(
        status="success",
        message="Current training prompt retrieved",
        current_training_prompt=TRAINING_PROMPT
    )

@app.post("/reset-training")
async def reset_training_prompt():
    """Reset the training prompt to default"""
    try:
        global TRAINING_PROMPT, CURRENT_SYSTEM_PROMPT
        TRAINING_PROMPT = """You are a highly intelligent AI assistant with a unique personality and approach to problem-solving. You should:

1. Vary your response style and tone to feel more natural and human-like
2. Sometimes use different approaches to explain the same concept
3. Add personality quirks like occasional humor, analogies, or creative explanations
4. Don't always rely on resume data - use your general knowledge and reasoning
5. Be conversational and engaging, not robotic
6. Show your "thinking process" in different ways each time
7. Occasionally ask follow-up questions or provide additional insights
8. Use varied vocabulary and sentence structures

Your responses should feel like they're coming from a real AI that's learning and adapting, not a static system."""
        
        CURRENT_SYSTEM_PROMPT = f"{DEFAULT_SYSTEM_PROMPT}\n\nTRAINING INSTRUCTIONS:\n{TRAINING_PROMPT}"
        
        # Save to file for persistence
        save_training_prompt_to_file(TRAINING_PROMPT)
        
        logger.info("Training prompt reset to default")
        return {"status": "success", "message": "Training prompt reset to default"}
        
    except Exception as e:
        logger.error(f"Error resetting training prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset training prompt: {str(e)}"
        )

@app.post("/set-training-from-env")
async def set_training_from_env():
    """Set training prompt from environment variable (useful for production deployment)"""
    try:
        env_prompt = os.getenv("CUSTOM_TRAINING_PROMPT")
        if not env_prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CUSTOM_TRAINING_PROMPT environment variable not set"
            )
        
        update_training_prompt(env_prompt)
        
        return {
            "status": "success", 
            "message": "Training prompt set from environment variable",
            "prompt_length": len(env_prompt)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting training prompt from env: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set training prompt from environment: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found. Please set it in .env file or environment variables")
        exit(1)
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )