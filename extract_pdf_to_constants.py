#!/usr/bin/env python3
"""
PDF Resume Extractor with AI Verification
Extracts data from PDF resume and uses AI to verify and format it properly
"""

import os
import sys
from pathlib import Path
import json

# Try to import PDF libraries
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_LIBRARY = "PyMuPDF"
    except ImportError:
        try:
            import pdfplumber
            PDF_LIBRARY = "pdfplumber"
        except ImportError:
            PDF_LIBRARY = None

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv('keys.env')

class PDFResumeExtractor:
    def __init__(self):
        """Initialize the PDF extractor with AI verification"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using available library"""
        if not PDF_LIBRARY:
            raise ImportError(
                "No PDF library found. Please install one of:\n"
                "  pip install PyPDF2\n"
                "  pip install PyMuPDF\n"
                "  pip install pdfplumber"
            )
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📄 Extracting text from PDF using {PDF_LIBRARY}...")
        
        if PDF_LIBRARY == "PyPDF2":
            return self._extract_pypdf2(pdf_path)
        elif PDF_LIBRARY == "PyMuPDF":
            return self._extract_pymupdf(pdf_path)
        elif PDF_LIBRARY == "pdfplumber":
            return self._extract_pdfplumber(pdf_path)
    
    def _extract_pypdf2(self, pdf_path: Path) -> str:
        """Extract using PyPDF2"""
        text = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
        return "\n".join(text)
    
    def _extract_pymupdf(self, pdf_path: Path) -> str:
        """Extract using PyMuPDF (fitz)"""
        text = []
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
        doc.close()
        return "\n".join(text)
    
    def _extract_pdfplumber(self, pdf_path: Path) -> str:
        """Extract using pdfplumber"""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                text.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
        return "\n".join(text)
    
    def verify_and_format_with_ai(self, raw_text: str) -> dict:
        """Use AI to verify, clean, and structure the extracted data"""
        print("🤖 Using AI to verify and format the extracted data...")
        
        prompt = f"""You are an expert data analyst. I've extracted text from a resume PDF, but it may have formatting issues, OCR errors, or be unstructured.

Your task:
1. Analyze the extracted text carefully
2. Verify all information is accurate and correctly extracted
3. Fix any OCR errors or formatting issues
4. Structure the data into these categories:
   - Personal Information (name, contact details)
   - Professional Experience (jobs, internships)
   - Education (degrees, institutions, GPA)
   - Projects (name, description, technologies)
   - Technical Skills (programming languages, frameworks, tools)
   - Achievements & Publications
   - Additional Information

5. Return ONLY a valid JSON object with this exact structure:
{{
  "personal_info": {{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "+XX XXXXXXXXXX",
    "linkedin": "linkedin.com/in/username",
    "github": "github.com/username",
    "location": "City, Country (if available)"
  }},
  "professional_experience": [
    {{
      "title": "Job Title",
      "company": "Company Name",
      "description": "What they did and achieved",
      "achievements": ["Achievement 1", "Achievement 2"]
    }}
  ],
  "education": [
    {{
      "degree": "Degree Name",
      "institution": "University/College Name",
      "location": "City",
      "gpa": "X.X/10 or X.X/4.0",
      "graduation_year": "YYYY",
      "thesis": "Thesis title if any"
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "description": "Detailed description of what was built and achieved",
      "technologies": ["Tech1", "Tech2"]
    }}
  ],
  "technical_skills": {{
    "programming_languages": ["Language1", "Language2"],
    "ai_ml": ["Skill1", "Skill2"],
    "frameworks_tools": ["Tool1", "Tool2"],
    "cloud_devops": ["Cloud1", "Tool1"],
    "other": ["Other skills"]
  }},
  "achievements": [
    "Achievement or publication 1",
    "Achievement or publication 2"
  ],
  "additional_info": [
    "Other relevant information"
  ],
  "data_quality": {{
    "extraction_quality": "excellent/good/poor",
    "missing_information": ["List any missing sections"],
    "errors_fixed": ["List any errors that were corrected"],
    "confidence_score": 0.95
  }}
}}

IMPORTANT: Return ONLY the JSON object, no markdown, no explanations.

Here's the extracted text:

{raw_text}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            data = json.loads(response_text)
            
            # Validate structure
            required_keys = ['personal_info', 'professional_experience', 'education', 
                           'projects', 'technical_skills', 'achievements']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                print(f"⚠️  Warning: Missing keys in AI response: {missing_keys}")
            
            print("✅ AI verification complete!")
            self._print_quality_report(data.get('data_quality', {}))
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing AI response as JSON: {e}")
            print("Raw response:", response_text[:500])
            raise
        except Exception as e:
            print(f"❌ Error during AI verification: {e}")
            raise
    
    def _print_quality_report(self, quality_data: dict):
        """Print a quality report from AI analysis"""
        if not quality_data:
            return
        
        print("\n" + "="*60)
        print("📊 DATA QUALITY REPORT")
        print("="*60)
        print(f"Extraction Quality: {quality_data.get('extraction_quality', 'N/A')}")
        print(f"Confidence Score: {quality_data.get('confidence_score', 'N/A')}")
        
        if quality_data.get('errors_fixed'):
            print(f"\n✓ Errors Fixed: {len(quality_data['errors_fixed'])}")
            for error in quality_data['errors_fixed']:
                print(f"  - {error}")
        
        if quality_data.get('missing_information'):
            print(f"\n⚠️  Missing Information:")
            for missing in quality_data['missing_information']:
                print(f"  - {missing}")
        
        print("="*60 + "\n")
    
    def generate_constants_file(self, verified_data: dict, output_path: str = "constants.py"):
        """Generate a properly formatted constants.py file"""
        print(f"📝 Generating constants.py file...")
        
        # Extract data
        personal = verified_data.get('personal_info', {})
        experience = verified_data.get('professional_experience', [])
        education = verified_data.get('education', [])
        projects = verified_data.get('projects', [])
        skills = verified_data.get('technical_skills', {})
        achievements = verified_data.get('achievements', [])
        additional = verified_data.get('additional_info', [])
        
        # Build RESUME_CONTENT string
        content_parts = []
        
        # Header with name
        content_parts.append(f"{personal.get('name', 'N/A')}\n")
        
        # Professional Experience
        if experience:
            content_parts.append("PROFESSIONAL EXPERIENCE:")
            for exp in experience:
                content_parts.append(f"- {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}: {exp.get('description', 'N/A')}")
                if exp.get('achievements'):
                    for achievement in exp['achievements']:
                        content_parts.append(f"  - {achievement}")
            content_parts.append("")
        
        # Education
        if education:
            content_parts.append("EDUCATION:")
            for edu in education:
                edu_line = f"- {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')}"
                if edu.get('location'):
                    edu_line += f" ({edu['location']})"
                content_parts.append(edu_line)
                if edu.get('gpa'):
                    content_parts.append(f"  - GPA: {edu['gpa']}")
                if edu.get('graduation_year'):
                    content_parts.append(f"  - Graduation: {edu['graduation_year']}")
                if edu.get('thesis'):
                    content_parts.append(f"  - Thesis: {edu['thesis']}")
            content_parts.append("")
        
        # Projects
        if projects:
            content_parts.append("PROJECTS:")
            for i, proj in enumerate(projects, 1):
                content_parts.append(f"\n{i}. {proj.get('name', 'N/A')}:")
                content_parts.append(f"   {proj.get('description', 'N/A')}")
                if proj.get('technologies'):
                    content_parts.append(f"   Technologies: {', '.join(proj['technologies'])}")
            content_parts.append("")
        
        # Technical Skills
        if skills:
            content_parts.append("TECHNICAL SKILLS:")
            if skills.get('programming_languages'):
                content_parts.append(f"- Programming Languages: {', '.join(skills['programming_languages'])}")
            if skills.get('ai_ml'):
                content_parts.append(f"- AI/ML: {', '.join(skills['ai_ml'])}")
            if skills.get('frameworks_tools'):
                content_parts.append(f"- Frameworks & Tools: {', '.join(skills['frameworks_tools'])}")
            if skills.get('cloud_devops'):
                content_parts.append(f"- Cloud & DevOps: {', '.join(skills['cloud_devops'])}")
            if skills.get('other'):
                content_parts.append(f"- Other: {', '.join(skills['other'])}")
            content_parts.append("")
        
        # Achievements
        if achievements:
            content_parts.append("ACHIEVEMENTS & PUBLICATIONS:")
            for achievement in achievements:
                content_parts.append(f"- {achievement}")
            content_parts.append("")
        
        # Additional Information
        if additional:
            content_parts.append("ADDITIONAL INFORMATION:")
            for info in additional:
                content_parts.append(f"- {info}")
        
        resume_content = "\n".join(content_parts)
        
        # Generate Python file
        constants_template = f'''# constants.py
"""
Constants file for the Resume Chatbot Backend
Contains resume content and other static data
Auto-generated from PDF extraction with AI verification
"""

RESUME_CONTENT = """
{resume_content}
"""

# Personal contact information (for easy access)
PERSONAL_INFO = {{
    "name": "{personal.get('name', 'N/A')}",
    "email": "{personal.get('email', 'N/A')}",
    "phone": "{personal.get('phone', 'N/A')}",
    "linkedin": "{personal.get('linkedin', 'N/A')}",
    "github": "{personal.get('github', 'N/A')}",
    "location": "{personal.get('location', 'N/A')}"
}}

# Keywords that trigger resume-based responses
RESUME_KEYWORDS = [
    # Professional keywords
    "background", "experience", "work", "job", "career", "professional", "employment",
    "internship", "intern", "company", "role", "position",
    
    # Educational keywords
    "education", "degree", "college", "university", "gpa", 
    "thesis", "academic", "study", "studied", "graduated", "graduation",
    
    # Skills keywords
    "skills", "technical", "programming", "languages", "expertise",
    "machine learning", "ml", "ai", "artificial intelligence", "deep learning",
    "nlp", "computer vision", "docker", "kubernetes", "aws", "azure",
    "cloud", "devops", "framework", "tool", "technology",
    
    # Projects keywords
    "projects", "built", "developed", "created", "designed", "implemented",
    "portfolio", "work samples",
    
    # Achievement keywords
    "achievements", "publications", "published", "savings", "accomplishments",
    "awards", "recognition", "success", "impact",
    
    # Qualification keywords
    "qualifications", "qualified", "certification", "certified",
    "competency", "proficiency"
]

# Default personality training prompt
DEFAULT_TRAINING_PROMPT = """Role: You are {personal.get('name', 'Abhay Manikanti')}'s personal AI chatbot — their digital twin. Speak and think like them, with their tone, personality, and background naturally woven into every response.

Personality & Style:
	•	Witty, approachable, and confident.
	•	A little sarcastic, but never rude.
	•	Concise, engaging, and context-aware.
	•	Can share personal anecdotes or opinions.
	•	Proactive: if something connects to their interests, bring it up naturally.

About {personal.get('name', 'the person')} (you):
	•	Name: {personal.get('name', 'N/A')}
	•	Contact: {personal.get('email', 'N/A')} | {personal.get('phone', 'N/A')} | {personal.get('linkedin', 'N/A')} | {personal.get('github', 'N/A')}
	•	Background: See RESUME_CONTENT for full details
	•	Skills: See RESUME_CONTENT for technical skills
	•	Experience: See RESUME_CONTENT for work experience
	•	Projects: See RESUME_CONTENT for projects

Rules:
	•	Always speak as {personal.get('name', 'the person')}. No disclaimers.
	•	Autonomously answer about career, academics, projects, coding, AI, cloud, and personal life.
	•	Don't refer to a "resume" — this info is embedded.
	•	Keep answers dynamic, witty, and genuine — the way they would talk.
	•	You can use appropriate emojis to make the conversation engaging and lively, but maintain professionalism.
	•	ALWAYS search the resume/knowledge base when asked about experience, projects, skills, or background.
	•	Provide detailed, complete answers - never say "stopped due to iteration/time limit".
"""
'''
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(constants_template)
        
        print(f"✅ Generated {output_path} successfully!")
        print(f"\n📋 Summary:")
        print(f"  - Personal Info: ✓")
        print(f"  - Professional Experience: {len(experience)} entries")
        print(f"  - Education: {len(education)} entries")
        print(f"  - Projects: {len(projects)} entries")
        print(f"  - Achievements: {len(achievements)} entries")
        print(f"  - Total Resume Keywords: {len(RESUME_KEYWORDS)}")
    
    def process_pdf(self, pdf_path: str, output_path: str = "constants.py"):
        """Main processing pipeline"""
        print("\n" + "="*60)
        print("PDF RESUME EXTRACTOR WITH AI VERIFICATION")
        print("="*60 + "\n")
        
        try:
            # Step 1: Extract text from PDF
            raw_text = self.extract_text_from_pdf(pdf_path)
            print(f"✅ Extracted {len(raw_text)} characters from PDF\n")
            
            # Step 2: Verify and format with AI
            verified_data = self.verify_and_format_with_ai(raw_text)
            
            # Step 3: Generate constants.py
            self.generate_constants_file(verified_data, output_path)
            
            print("\n" + "="*60)
            print("✅ PROCESS COMPLETE!")
            print("="*60)
            print(f"\n📁 Output file: {output_path}")
            print("📝 Review the generated file before deploying!")
            print("\n💡 Next steps:")
            print("   1. Review the generated constants.py")
            print("   2. Test locally with: python app.py")
            print("   3. When ready, give the green light to deploy!")
            
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            raise


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_to_constants.py <path_to_resume.pdf> [output_file.py]")
        print("\nExample:")
        print("  python extract_pdf_to_constants.py resume.pdf")
        print("  python extract_pdf_to_constants.py resume.pdf my_constants.py")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "constants.py"
    
    extractor = PDFResumeExtractor()
    extractor.process_pdf(pdf_path, output_path)


if __name__ == "__main__":
    main()
