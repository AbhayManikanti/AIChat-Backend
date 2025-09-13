# Dependency Management

This project uses multiple requirements files for different environments:

## 📦 Requirements Files

### `requirements.txt` - Development (Flexible)
- Flexible version ranges for development
- Includes all necessary dependencies for local development
- Uses version ranges (e.g., `>=0.3.0,<0.4.0`) for flexibility

### `requirements-prod.txt` - Production (Pinned)  
- Exact version pinning for production stability
- All versions are locked to specific releases
- Ensures consistent deployments across environments

### `requirements-dev.txt` - Development Tools
- Includes testing, linting, and development tools
- Code quality tools (black, isort, flake8, mypy)
- Testing framework (pytest)
- Documentation tools (mkdocs)

## 🚀 Installation

### For Development
```bash
pip install -r requirements.txt
```

### For Production  
```bash
pip install -r requirements-prod.txt
```

### For Development with Tools
```bash
pip install -r requirements-dev.txt
```

## 🔧 Key Dependencies

### Core Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: Lightning-fast ASGI server
- **Pydantic**: Data validation using Python type annotations

### AI/ML Stack
- **LangChain**: Framework for developing LLM applications
- **langchain-google-genai**: Google Gemini integration (primary LLM)
- **langchain-openai**: OpenAI-compatible API client (Perplexity fallback)
- **ChromaDB**: Vector database for semantic search

### Token Optimization Features
- **Lazy initialization**: LLMs only load when needed
- **Rate limit protection**: Automatic Gemini → Perplexity fallback
- **Persistent vectorstore**: No re-embedding on restart
- **Response caching**: Avoid duplicate API calls

## ⚡ Performance Optimizations

1. **Zero Token Startup**: No LLM calls during app initialization
2. **Lazy Loading**: Components initialize only when first needed
3. **Persistent Storage**: ChromaDB persists between restarts
4. **Response Caching**: Identical questions cached for 1 hour
5. **Rate Limit Handling**: Automatic fallback prevents service interruption

## 🛡️ Production Considerations

- All production versions are pinned for stability
- Includes security-focused dependencies
- Optimized for minimal token consumption
- Built-in rate limit protection with fallback LLM
- Health checks don't consume tokens

## 🔄 Updating Dependencies

### Update Development Requirements
```bash
pip install -U -r requirements.txt
pip freeze > requirements-prod.txt  # Update production pins
```

### Security Updates
```bash
pip install -U pip
pip install -r requirements-dev.txt
bandit -r app.py  # Security scan
```