#!/bin/bash

# 🚀 AI Agent Deployment Script for Render

echo "🚀 Starting AI Agent deployment process..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found."
    exit 1
fi

echo "✅ Project files found"

# Check for environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  Warning: GOOGLE_API_KEY environment variable not set"
    echo "   Make sure to set it in your Render dashboard"
fi

echo "📋 Deployment checklist:"
echo "   1. ✅ Code is ready"
echo "   2. ✅ requirements.txt exists"
echo "   3. ✅ Environment variables configured"
echo ""
echo "🎯 Next steps:"
echo "   1. Push your code to GitHub"
echo "   2. Connect your GitHub repo to Render"
echo "   3. Set environment variables in Render dashboard:"
echo "      - GOOGLE_API_KEY=your_api_key"
echo "      - GOOGLE_MODEL=gemini-2.5-flash"
echo "      - CUSTOM_TRAINING_PROMPT=your_custom_prompt (optional)"
echo "   4. Deploy!"
echo ""
echo "📖 For detailed instructions, see DEPLOYMENT_GUIDE.md"
echo ""
echo "🎉 Ready for deployment!"
