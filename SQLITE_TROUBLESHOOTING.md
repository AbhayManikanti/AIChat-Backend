# SQLite3 Compatibility Troubleshooting

## The Issue
ChromaDB requires SQLite3 version 3.35.0 or higher. Some systems may have older SQLite3 versions that cause compatibility issues.

## Error Messages You Might See
```
Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
```

## Check Your SQLite3 Version
```python
import sqlite3
print(f"SQLite3 version: {sqlite3.sqlite_version}")
```

## Solutions (in order of preference)

### 1. **System Update (Recommended)**
Update your system's SQLite3 if possible:

**macOS (Homebrew):**
```bash
brew install sqlite3
brew link sqlite3 --force
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install sqlite3 libsqlite3-dev
```

### 2. **Use Conda Environment**
Conda manages SQLite3 versions better:
```bash
conda create -n langback python=3.9
conda activate langback
conda install sqlite>=3.35.0
pip install -r requirements.txt
```

### 3. **Docker Solution (Production)**
Use Docker to ensure consistent SQLite3 version:

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Install system dependencies including newer SQLite3
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

# Copy application
COPY . /app
WORKDIR /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. **Alternative Vector Database**
If SQLite3 issues persist, consider using alternative vector stores:

**In-Memory ChromaDB (for development):**
```python
# Modify initialize_vectorstore function
vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="resume_collection"
    # Remove persist_directory for in-memory storage
)
```

**Pinecone (cloud-based):**
```bash
pip install pinecone-client langchain-pinecone
```

### 5. **Environment Variable Workaround**
Set environment variables to help ChromaDB:
```bash
export CHROMA_DB_IMPL="duckdb+parquet"
export SQLITE_THREADSAFE=1
```

## Current App Behavior
The app now includes:
- ✅ Automatic SQLite3 version checking
- ✅ Graceful degradation if version issues occur
- ✅ Detailed error logging
- ✅ Fallback strategies

## Testing SQLite3 Compatibility
Run this test to check compatibility:
```bash
cd /path/to/project
python -c "
import sqlite3
import chromadb
print(f'SQLite3 version: {sqlite3.sqlite_version}')
client = chromadb.Client()
print('ChromaDB works correctly!')
"
```

## Production Deployment
For production, use pinned requirements with Docker to ensure consistent SQLite3 versions across environments.

The application will continue to work even with SQLite3 warnings, but for best performance and reliability, ensure you have SQLite3 >= 3.35.0.