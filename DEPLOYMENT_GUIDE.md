# 🚀 Deployment Guide for AI Agent on Render

## 📋 Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Google API Key**: Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🔧 Environment Variables Setup

In your Render dashboard, set these environment variables:

### Required Variables:
```
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
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
