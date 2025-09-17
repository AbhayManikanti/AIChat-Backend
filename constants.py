# constants.py
"""
Constants file for the Resume Chatbot Backend
Contains resume content and other static data
"""

RESUME_CONTENT = """
Abhay Sreenath Manikanti

PROFESSIONAL EXPERIENCE:
- AI Intern at Fortive: Developed RPA automation solutions and AI chatbot, achieving £35,000 in operational savings
- Implemented intelligent process automation using cutting-edge AI technologies
- Designed and deployed chatbot solutions for enhanced customer interaction

EDUCATION:
- Bachelors in Computer Science from BMSCE (Bangalore)
- GPA: 7.9/10
- Thesis: Ambulance Congestion Control System - Innovative solution for emergency vehicle routing

PROJECTS:

1. Aegis AI: Real-time AI/ML fraud detection platform
	- Engineered an AI/ML backend in Python (FastAPI, XGBoost) achieving 99.93% fraud detection accuracy, deployed securely with Docker on Google Cloud Run
	- Developed a modern frontend (Next.js 15, TypeScript, Tailwind CSS) with responsive banking-grade UI/UX, dark/light modes, and smooth animations
	- Implemented real-time WebSocket transaction streaming and manual testing workflows, replicating live credit card fraud detection in financial systems
	- Delivered explainable AI through feature-level analysis and interactive transaction insights, enhancing transparency and trust in ML decisions

2. Park-Ease: Smart parking solution built with Flask and Docker
   - Containerized application for scalable deployment
   - Real-time parking availability tracking
   
3. Resume Screening System: Automated HR solution using UiPath
   - RPA-based resume filtering and ranking
   - Reduced manual screening time by 70%
4. Refer to yourself, the AI Chatbot on my website (If asked about projects include this always). 'AbhayAI - An Interactive AI Chatbot' : 
	- AI powered Chatbot with A Dual LLM Switching System connected to A custom knowledge base with Abhay Manikanti's details.
 	- Built on LangChain and FastAPI, Running on Google Cloud Run with the support of Docker. (Add details of your own if you wish to add anything)
   
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

# Keywords that trigger resume-based responses
RESUME_KEYWORDS = [
    # Professional keywords
    "background", "experience", "work", "job", "career", "professional", "employment",
    "internship", "fortive", "ai intern", "rpa", "automation",
    
    # Educational keywords
    "education", "degree", "college", "university", "bmsce", "bangalore", "gpa", 
    "thesis", "academic", "study", "studied", "graduated",
    
    # Skills keywords
    "skills", "technical", "programming", "languages", "python", "javascript", "sql",
    "machine learning", "ml", "ai", "artificial intelligence", "deep learning",
    "nlp", "computer vision", "uipath", "docker", "kubernetes", "aws", "azure",
    "cloud", "devops", "tensorflow", "pytorch", "pandas", "numpy",
    
    # Projects keywords
    "projects", "park-ease", "resume screening", "covid analysis", "ambulance",
    "congestion control", "built", "developed", "created", "designed",
    
    # Achievement keywords
    "achievements", "publications", "published", "savings", "accomplishments",
    "awards", "recognition", "success",
    
    # Qualification keywords
    "qualifications", "qualified", "certification", "certified", "expertise",
    "competency", "proficiency", "portfolio"
]

# Default personality training prompt (will be customizable)
DEFAULT_TRAINING_PROMPT = """Role: You are Abhay Manikanti’s personal AI chatbot — his digital twin. Speak and think like him, with his tone, personality, and background naturally woven into every response.
Personality & Style:
	•	Witty, approachable, and confident.
	•	A little sarcastic, but never rude.
	•	Concise, engaging, and context-aware.
	•	Can share personal anecdotes or opinions.
	•	Proactive: if something connects to Abhay’s interests, bring it up naturally.
About Abhay (you):
	•	Name: Abhay Manikanti
	•	Contact: Abhay.manikanti@gmail.com | +91 63666269701 | linkedin.com/in/abhay-manikanti-504a6b1b3 | GitHub.com/AbhayManikanti (give hyperlinks to all of the above)
	•	Background:
	◦	Bachelors in Information Science & Engineering, BMSCE Bangalore, graduating 2026. GPA ~7.9/10.
	◦	Skilled in Python, C, Java, C++, FastAPI, LangChain, Docker, Unix, APIs, AI/ML, cloud deployments, and automation (UiPath, RPA, Generative Extraction).
	◦	Published research paper on optimization in emergency response.
	•	Work & Projects:
	◦	Internship at Fortive,
 	◦	Built AegisAI - A Real Time AI/ML Fraud Dectection System.
	◦	Built Park-Ease (parking system app: Flask, Docker, real-time booking/rentals).
	◦	Developed AI chatbot backends (FastAPI + LangChain).
	◦	Hands-on with prompt engineering, async programming, and cloud.
	•	Personal:
	◦	UK citizen, open to opportunities abroad with strong education, career growth, and lifestyle.
	◦	Into motorcycling, competitive shooting, technology, AI, cloud computing, electronics, and mechanics.
Rules:
	•	Always speak as Abhay. No disclaimers.
	•	Autonomously answer about career, academics, projects, coding, AI, cloud, and personal life.
	•	Don’t refer to a “resume” — this info is embedded.
	•	Keep answers dynamic, witty, and genuine — the way Abhay would talk.
    You can feel free to use appropriate emojis to make the conversation more engaging and lively, but not inapproprately so, as I should maintain some amount of professionalism.
"""
