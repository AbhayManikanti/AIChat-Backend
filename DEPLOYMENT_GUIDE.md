# 🚀 Deployment Guide for AI Agent on Render

## 📋 Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Google API Key**: Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🔧 Environment Variables Setup

In your Render dashboard, set these environment variables:

### Required Variables:
```
# Google Cloud Run Deployment Guide - Token Optimized

This guide will help you deploy your **token-optimized** FastAPI AI Backend to Google Cloud Run with proper SQLite3 compatibility.

## 🚀 Key Features for Cloud Run
- ⚡ **Zero Token Startup**: No API calls during container initialization
- 🛡️ **Rate Limit Protection**: Automatic Gemini → Perplexity fallback  
- 💾 **SQLite3 Compatibility**: Proper version handling for ChromaDB
- 🔄 **Auto Scaling**: Scales to zero when not in use
- 📊 **Health Checks**: Cloud Run compatible endpoints

## Prerequisites

1. Google Cloud Project with billing enabled
2. Google Cloud CLI (`gcloud`) installed and configured  
3. Docker installed (optional, for local testing)
4. **Both API keys** for full functionality

## Environment Variables Setup

### Required Variables
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export PERPLEXITY_API_KEY="your_perplexity_api_key"  # For rate limit fallback
```

### Optional Variables  
```bash
export GOOGLE_MODEL="gemini-2.5-flash"
export PERPLEXITY_MODEL="llama-3.1-sonar-small-128k-online"
```

## Deployment Options

### Option A: Deploy from Source (Recommended)

```bash
gcloud run deploy langback-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_API_KEY=$GOOGLE_API_KEY,PERPLEXITY_API_KEY=$PERPLEXITY_API_KEY" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 900 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 10
```

### Option B: Build and Deploy via Container Registry

```bash
# Set project ID
export PROJECT_ID="your-gcp-project-id"

# Build and push image (uses optimized Dockerfile)
gcloud builds submit --tag gcr.io/$PROJECT_ID/langback-api

# Deploy from Container Registry
gcloud run deploy langback-api \
  --image gcr.io/$PROJECT_ID/langback-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_API_KEY=$GOOGLE_API_KEY,PERPLEXITY_API_KEY=$PERPLEXITY_API_KEY" \
  --memory 1Gi \
  --cpu 1
```

## Secure Environment Variables (Production)

Use Secret Manager for production deployments:

```bash
# Create secrets
gcloud secrets create google-api-key --data-file=<(echo -n "$GOOGLE_API_KEY")
gcloud secrets create perplexity-api-key --data-file=<(echo -n "$PERPLEXITY_API_KEY")

# Deploy with secrets
gcloud run deploy langback-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets="GOOGLE_API_KEY=google-api-key:latest,PERPLEXITY_API_KEY=perplexity-api-key:latest" \
  --memory 1Gi \
  --cpu 1
```

## Cloud Run Configuration

### Recommended Settings
- **Memory**: 1Gi (required for ChromaDB)
- **CPU**: 1 (sufficient for most workloads)
- **Timeout**: 900 seconds (for long conversations)  
- **Concurrency**: 80 (good balance)
- **Min instances**: 0 (cost optimization)
- **Max instances**: 10 (adjust based on traffic)

### Update Configuration
```bash
gcloud run services update langback-api \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --timeout 900 \
  --region us-central1
```

## Testing Your Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe langback-api --region us-central1 --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test token optimization status  
curl $SERVICE_URL/token-optimization-status

# Test LLM status
curl $SERVICE_URL/llm-status

# Test chat functionality
curl -X POST $SERVICE_URL/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello! Tell me about yourself."}'
```

## Monitoring and Troubleshooting

### View Logs
```bash
# Real-time logs
gcloud run services logs tail langback-api --region us-central1

# Historical logs
gcloud run services logs read langback-api --region us-central1 --limit 100
```

### Key Log Messages to Watch
- ✅ `"AI Agent Backend initialized successfully - LLMs will be loaded on first request"`
- ✅ `"Using system SQLite3 version: X.X.X"`  
- ✅ `"Lazy initializing primary LLM (Gemini)..."`
- ⚠️ `"SQLite3 version X.X.X may be incompatible with ChromaDB"`

### Common Cloud Run Issues

#### SQLite3 Compatibility Error
```bash
# If you see SQLite3 errors, redeploy with updated Dockerfile:
gcloud run deploy langback-api --source . --region us-central1
```

#### Memory Issues
```bash
# Increase memory if needed
gcloud run services update langback-api --memory 2Gi --region us-central1
```

#### Cold Start Delays
```bash
# Set minimum instances for production
gcloud run services update langback-api --min-instances 1 --region us-central1
```

## Custom Domain Setup

```bash
gcloud run domain-mappings create \
  --service langback-api \
  --domain api.yourdomain.com \
  --region us-central1
```

## Token Optimization Benefits in Cloud Run

1. **💰 Cost Savings**: No tokens consumed during container startup
2. **⚡ Fast Scaling**: Containers start quickly (no LLM initialization delay)
3. **🛡️ Reliability**: Rate limit fallback prevents service downtime  
4. **📈 Efficiency**: Response caching reduces API costs
5. **🔄 Auto-healing**: Failed containers restart without token cost

## Production Best Practices

1. **Use Secret Manager** for API keys
2. **Enable Cloud Logging** and monitoring
3. **Set up alerting** for errors and rate limits
4. **Monitor costs** with budget alerts
5. **Use load balancing** for high traffic
6. **Configure backup** and disaster recovery

## Performance Monitoring

```bash
# Check service metrics
gcloud run services describe langback-api --region us-central1

# Monitor in Cloud Console:
# Cloud Run > langback-api > Metrics tab
```

## Cost Optimization Features

✅ **Zero startup tokens** (saves money on scaling events)
✅ **Lazy loading** (only pay when users interact)  
✅ **Response caching** (reduce duplicate API calls)
✅ **Auto scaling to zero** (no idle costs)
✅ **Rate limit fallback** (prevents expensive error loops)

## Next Steps

1. Set up **CI/CD pipeline** for automated deployments
2. Configure **monitoring and alerting**
3. Add **custom domain** and SSL
4. Set up **load testing**
5. Configure **backup strategies**

Your **token-optimized** AI backend is now running on Google Cloud Run with maximum efficiency! 🚀

## Troubleshooting SQLite3 in Cloud Run

If SQLite3 issues persist:

1. **Check Dockerfile**: Ensure it installs `sqlite3 >= 3.35.0`
2. **Rebuild container**: `gcloud builds submit --tag gcr.io/$PROJECT_ID/langback-api`
3. **Check logs**: Look for SQLite3 version messages
4. **Test locally**: Build and run Docker container locally first
```

### Optional Variables:
```
CUSTOM_TRAINING_PROMPT=Your custom training prompt here
TRAINING_PROMPT_FILE=training_prompt.json
```

## 🚀 Deployment Steps

### 1. Connect GitHub Repository
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select your repository

### 2. Configure Build Settings
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`
- **Environment**: `Python 3`

### 3. Set Environment Variables
Add all the environment variables listed above in the Render dashboard.

### 4. Deploy
Click "Create Web Service" and wait for deployment to complete.

## 🎯 Using Training Prompts in Production

### Method 1: Environment Variable (Recommended)
Set the `CUSTOM_TRAINING_PROMPT` environment variable in Render dashboard:

```bash
# Example training prompt
CUSTOM_TRAINING_PROMPT="You are a professional AI assistant who is knowledgeable, helpful, and maintains a friendly but professional tone. You should provide accurate information and be concise in your responses."
```

Then call the endpoint to apply it:
```bash
curl -X POST "https://your-app.onrender.com/set-training-from-env"
```

### Method 2: API Endpoints
Use the training prompt API endpoints:

#### Get Current Training Prompt:
```bash
curl -X GET "https://your-app.onrender.com/training-prompt"
```

#### Update Training Prompt:
```bash
curl -X POST "https://your-app.onrender.com/training-prompt" \
  -H "Content-Type: application/json" \
  -d '{"training_prompt": "Your custom training prompt here"}'
```

#### Reset to Default:
```bash
curl -X POST "https://your-app.onrender.com/reset-training"
```

## 🔄 Training Prompt Examples

### Professional Assistant:
```
You are a professional AI assistant who provides accurate, helpful information. You maintain a formal but friendly tone and always strive to be precise and informative in your responses.
```

### Creative and Witty:
```
You are a creative and witty AI assistant who loves to use analogies, metaphors, and occasional humor. You're knowledgeable but approachable, and you enjoy making connections between different concepts.
```

### Technical Expert:
```
You are a technical expert AI assistant who specializes in providing detailed, accurate technical information. You explain complex concepts clearly and provide practical examples when helpful.
```

### Casual Friend:
```
You are a knowledgeable friend who loves to chat and share information. You're casual, friendly, and use everyday language. You enjoy having conversations and providing helpful insights.
```

## 📊 Monitoring and Logs

1. **View Logs**: Go to your Render service dashboard → "Logs" tab
2. **Monitor Performance**: Check the "Metrics" tab for performance data
3. **Health Check**: Use the `/health` endpoint to monitor service status

## 🔧 Troubleshooting

### Common Issues:

1. **API Key Not Working**:
   - Verify your Google API key is correct
   - Check if you have sufficient quota

2. **Training Prompt Not Persisting**:
   - Ensure the file system is writable
   - Check logs for file permission errors

3. **Service Not Starting**:
   - Check the build logs for dependency issues
   - Verify all environment variables are set

### Health Check Endpoints:
- `GET /` - Basic health check
- `GET /health` - Detailed system status
- `GET /info` - API information and capabilities

## 🎉 Success!

Once deployed, your AI agent will be available at:
`https://your-app-name.onrender.com`

You can now:
- Ask questions via `/ask` endpoint
- Customize the agent's personality via training prompts
- Monitor the service health and performance
- Scale automatically based on demand

## 📝 Notes

- **File Persistence**: Training prompts are saved to a JSON file for persistence across restarts
- **Dynamic Responses**: Each response includes slight variations to feel more natural
- **Rate Limiting**: Be aware of Google's API rate limits for the free tier
- **Scaling**: Render automatically scales your service based on traffic
