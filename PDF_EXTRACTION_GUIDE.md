# PDF Resume Extraction Guide

## Overview
This guide explains how to use the AI-powered PDF extraction tool to automatically generate your `constants.py` file from your resume PDF.

## Why This Approach?

Instead of using RAG (Retrieval Augmented Generation) with embeddings, we're using **PDF extraction + AI verification** because:

1. ✅ **Perplexity fallback doesn't have embeddings module** - Direct text extraction works with all LLM providers
2. ✅ **Faster responses** - No vector database queries, just in-memory text
3. ✅ **Simpler deployment** - No need for embedding models or vector stores
4. ✅ **AI verification** - Gemini validates and formats the extracted data
5. ✅ **Better quality** - AI fixes OCR errors and ensures proper formatting

## How It Works

```
PDF Resume → Extract Text → AI Verification → Formatted constants.py
```

### Step 1: Extract Text
- Uses PyPDF2 to extract raw text from your resume PDF
- Handles multi-page documents
- Preserves structure and formatting

### Step 2: AI Verification
- Gemini AI analyzes the extracted text
- Fixes OCR errors and formatting issues
- Structures data into proper categories:
  - Personal Information
  - Professional Experience
  - Education
  - Projects
  - Technical Skills
  - Achievements & Publications
  - Additional Information
- Generates a quality report with confidence score

### Step 3: Generate Constants
- Creates a properly formatted `constants.py` file
- Includes RESUME_CONTENT, PERSONAL_INFO, RESUME_KEYWORDS, and DEFAULT_TRAINING_PROMPT
- Ready to use in your FastAPI backend

## Usage

### Prerequisites
```bash
# Install required libraries
pip install PyPDF2 google-generativeai python-dotenv

# Or use the virtual environment
.venv/bin/pip install PyPDF2
```

### Running the Extraction

```bash
# Basic usage - generates constants.py
python extract_pdf_to_constants.py your_resume.pdf

# Custom output file
python extract_pdf_to_constants.py your_resume.pdf my_constants.py

# With virtual environment
.venv/bin/python extract_pdf_to_constants.py resume.pdf
```

### Example Output

```
============================================================
PDF RESUME EXTRACTOR WITH AI VERIFICATION
============================================================

📄 Extracting text from PDF using PyPDF2...
✅ Extracted 2847 characters from PDF

🤖 Using AI to verify and format the extracted data...
✅ AI verification complete!

============================================================
📊 DATA QUALITY REPORT
============================================================
Extraction Quality: excellent
Confidence Score: 0.98

✓ Errors Fixed: 2
  - Fixed OCR error: 'Abhey' → 'Abhay'
  - Corrected phone format: '6366626970' → '+91 6366626970'

============================================================

📝 Generating constants.py file...
✅ Generated constants.py successfully!

📋 Summary:
  - Personal Info: ✓
  - Professional Experience: 1 entries
  - Education: 1 entries
  - Projects: 4 entries
  - Achievements: 3 entries
  - Total Resume Keywords: 45

============================================================
✅ PROCESS COMPLETE!
============================================================

📁 Output file: constants.py
📝 Review the generated file before deploying!

💡 Next steps:
   1. Review the generated constants.py
   2. Test locally with: python app.py
   3. When ready, give the green light to deploy!
```

## Output Structure

The generated `constants.py` includes:

### 1. RESUME_CONTENT
```python
RESUME_CONTENT = """
Abhay Manikanti

PROFESSIONAL EXPERIENCE:
- AI Intern at Fortive: ...

EDUCATION:
- Bachelors in Computer Science from BMSCE...

PROJECTS:
1. Aegis AI: Real-time fraud detection...
...
"""
```

### 2. PERSONAL_INFO
```python
PERSONAL_INFO = {
    "name": "Abhay Manikanti",
    "email": "Abhay.manikanti@gmail.com",
    "phone": "+91 6366626970",
    "linkedin": "linkedin.com/in/abhay-manikanti-504a6b1b3",
    "github": "github.com/AbhayManikanti",
    "location": "Bangalore, India"
}
```

### 3. RESUME_KEYWORDS
```python
RESUME_KEYWORDS = [
    "background", "experience", "work", "job", ...
]
```

### 4. DEFAULT_TRAINING_PROMPT
```python
DEFAULT_TRAINING_PROMPT = """
Role: You are Abhay Manikanti's personal AI chatbot...
"""
```

## Verification & Quality Control

The AI performs these checks:

1. ✅ **Accuracy**: Verifies all information is correctly extracted
2. ✅ **Formatting**: Fixes OCR errors and formatting issues
3. ✅ **Completeness**: Identifies missing information
4. ✅ **Structure**: Organizes data into proper categories
5. ✅ **Quality Score**: Provides confidence rating (0-1)

### Quality Report Fields

- **extraction_quality**: excellent/good/poor
- **confidence_score**: 0.0 to 1.0
- **errors_fixed**: List of corrections made
- **missing_information**: Sections that couldn't be extracted

## Testing After Extraction

### 1. Review the Generated File
```bash
# Open and review
cat constants.py
# or
code constants.py
```

### 2. Test Locally
```bash
# Kill existing server
pkill -f "python.*app.py"

# Start server with new constants
.venv/bin/python app.py

# Test in another terminal
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about your projects"}'
```

### 3. Verify Resume Detection
```bash
# This should return used_resume: true
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your work experience?"}'
```

## Troubleshooting

### PDF Not Found
```
❌ FileNotFoundError: PDF file not found: resume.pdf
```
**Solution**: Provide the full path to your PDF
```bash
python extract_pdf_to_constants.py /Users/you/Documents/resume.pdf
```

### API Key Missing
```
❌ GOOGLE_API_KEY not found in environment variables
```
**Solution**: Make sure `keys.env` exists with your API key
```bash
echo "GOOGLE_API_KEY=your_key_here" >> keys.env
```

### Poor Extraction Quality
```
⚠️ Extraction Quality: poor
```
**Solution**: 
1. Check if PDF is scanned (not text-based)
2. Try converting PDF to text-based format
3. Manually edit the generated constants.py

### Missing Information
```
⚠️ Missing Information:
  - Projects section incomplete
```
**Solution**: Manually add missing information to the generated file

## Best Practices

1. ✅ **Use text-based PDFs** - Not scanned images (better extraction)
2. ✅ **Review AI output** - Always verify the generated file before deploying
3. ✅ **Test locally first** - Ensure resume detection works properly
4. ✅ **Check quality report** - Look for confidence score > 0.9
5. ✅ **Update keywords** - Add domain-specific keywords if needed
6. ✅ **Backup original** - Keep a backup of your current constants.py

## Deployment Workflow

```bash
# 1. Extract and verify
python extract_pdf_to_constants.py resume.pdf

# 2. Review quality report
# Check confidence_score, errors_fixed, missing_information

# 3. Test locally
pkill -f "python.*app.py"
.venv/bin/python app.py &
sleep 3
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Tell me about your experience"}'

# 4. Check resume detection in logs
tail -20 test_server.log | grep "Resume search"

# 5. Get green light from user

# 6. Deploy to Cloud Run
source keys.env
gcloud run deploy fastapi-backend \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_API_KEY=$GOOGLE_API_KEY,PERPLEXITY_API_KEY=$PERPLEXITY_API_KEY" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 60 \
  --port 8000
```

## Benefits Over Manual Entry

| Aspect | Manual Entry | AI Extraction |
|--------|--------------|---------------|
| **Time** | 30-60 minutes | 2-3 minutes |
| **Accuracy** | Human error prone | AI verified |
| **Formatting** | Inconsistent | Standardized |
| **Updates** | Tedious | Re-run script |
| **Verification** | Manual review | Automated QA |
| **Scalability** | Not scalable | Scalable |

## Next Steps

1. Place your resume PDF in the project directory
2. Run the extraction script
3. Review the quality report and generated file
4. Test locally to ensure everything works
5. Give the green light for deployment! 🚀
