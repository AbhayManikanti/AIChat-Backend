"""
FastAPI Resume Chatbot Backend with LangChain
Author: Abhay Sreenath Manikanti
Description: AI-powered resume chatbot using FastAPI, LangChain, and ChromaDB
"""

import os
import logging
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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

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

# Global variables for storing initialized components
vectorstore = None
qa_chain = None
memory = None

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

def initialize_qa_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """Initialize the QA chain with LangChain"""
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
    # max_tokens is not a standard parameter for ChatGoogleGenerativeAI
        )
        
        # Initialize memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create retrieval QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        logger.info("QA chain initialized successfully")
        return qa_chain, memory
        
    except Exception as e:
        logger.error(f"Failed to initialize QA chain: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    # Startup
    global vectorstore, qa_chain, memory
    try:
        logger.info("Initializing Resume Chatbot Backend...")
        
        # Initialize components
        vectorstore = initialize_vectorstore(RESUME_CONTENT)
        qa_chain, memory = initialize_qa_chain(vectorstore)
        
        logger.info("Resume Chatbot Backend initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Resume Chatbot Backend...")
    if vectorstore:
        vectorstore = None
    if qa_chain:
        qa_chain = None
    if memory:
        memory = None

# Initialize FastAPI app
app = FastAPI(
    title="AI Resume Chatbot API",
    description="AI-powered chatbot for answering questions about Abhay's resume",
    version="1.0.0",
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
        message="AI Resume Chatbot API is running",
        vectorstore_initialized=vectorstore is not None,
        llm_configured=os.getenv("GOOGLE_API_KEY") is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if vectorstore and qa_chain else "unhealthy",
        message="All systems operational" if vectorstore and qa_chain else "System not fully initialized",
        vectorstore_initialized=vectorstore is not None,
        llm_configured=qa_chain is not None
    )

@app.post("/ask", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for asking questions about the resume
    
    Args:
        request: QuestionRequest containing the user's question
        
    Returns:
        AnswerResponse with AI-generated answer and metadata
    """
    try:
        # Validate system readiness
        if not vectorstore or not qa_chain:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized. Please try again later."
            )
        
        # Process the question
        logger.info(f"Processing question: {request.question}")
        
        # Get answer from QA chain
        result = qa_chain({
            "question": request.question,
            "chat_history": []
        })
        
        # Extract answer and source documents
        answer = result.get("answer", "I couldn't find relevant information to answer your question.")
        source_docs = result.get("source_documents", [])
        
        # Extract unique source chunks for transparency
        sources = list(set([
            doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            for doc in source_docs
        ]))[:3]  # Limit to top 3 sources
        
        # Calculate a simple confidence score based on source relevance
        confidence = min(0.95, 0.3 + (len(source_docs) * 0.2)) if source_docs else 0.5
        
        logger.info(f"Answer generated successfully with {len(source_docs)} sources")
        
        return AnswerResponse(
            answer=answer,
            confidence=confidence,
            sources=sources
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
        "api_version": "1.0.0",
        "resume_owner": "Abhay Sreenath Manikanti",
        "capabilities": [
            "Answer questions about professional experience",
            "Provide details about education and skills",
            "Discuss projects and achievements",
            "Explain technical expertise"
        ],
        "powered_by": "FastAPI + LangChain + GOOGLE"
    }

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