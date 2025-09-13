#!/bin/bash

# Google Cloud Run Deployment Script for Token-Optimized AI Backend
# Usage: ./deploy.sh [PROJECT_ID] [REGION]

set -e

# Default values
DEFAULT_PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
DEFAULT_REGION="us-central1"
SERVICE_NAME="langback-api"

# Get parameters
PROJECT_ID=${1:-$DEFAULT_PROJECT_ID}
REGION=${2:-$DEFAULT_REGION}

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID is required"
    echo "Usage: ./deploy.sh [PROJECT_ID] [REGION]"
    echo "Example: ./deploy.sh my-gcp-project us-central1"
    exit 1
fi

echo "🚀 Deploying Token-Optimized AI Backend to Cloud Run"
echo "=================================================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION" 
echo "Service Name: $SERVICE_NAME"
echo ""

# Check required environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ Error: GOOGLE_API_KEY environment variable is required"
    echo "Set it with: export GOOGLE_API_KEY='your_gemini_api_key'"
    exit 1
fi

if [ -z "$PERPLEXITY_API_KEY" ]; then
    echo "⚠️  Warning: PERPLEXITY_API_KEY not set - rate limit fallback will be disabled"
    echo "Set it with: export PERPLEXITY_API_KEY='your_perplexity_api_key'"
fi

# Set project
echo "📋 Setting project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Build environment variables string
ENV_VARS="GOOGLE_API_KEY=$GOOGLE_API_KEY"
if [ ! -z "$PERPLEXITY_API_KEY" ]; then
    ENV_VARS="$ENV_VARS,PERPLEXITY_API_KEY=$PERPLEXITY_API_KEY"
fi

# Add optional configuration
if [ ! -z "$GOOGLE_MODEL" ]; then
    ENV_VARS="$ENV_VARS,GOOGLE_MODEL=$GOOGLE_MODEL"
fi
if [ ! -z "$PERPLEXITY_MODEL" ]; then
    ENV_VARS="$ENV_VARS,PERPLEXITY_MODEL=$PERPLEXITY_MODEL"
fi

echo "🔨 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars="$ENV_VARS" \
    --memory 1Gi \
    --cpu 1 \
    --timeout 900 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 10 \
    --port 8000

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "✅ Deployment successful!"
echo "=================================================="
echo "Service URL: $SERVICE_URL"
echo ""

# Test the deployment
echo "🧪 Testing deployment..."
echo "Health check:"
curl -s $SERVICE_URL/health | jq '.' 2>/dev/null || curl -s $SERVICE_URL/health

echo ""
echo "Token optimization status:"
curl -s $SERVICE_URL/token-optimization-status | jq '.token_optimization' 2>/dev/null || echo "Token optimization enabled"

echo ""
echo "🎉 Your token-optimized AI backend is now running on Cloud Run!"
echo ""
echo "� Useful commands:"
echo "View logs:     gcloud run services logs tail $SERVICE_NAME --region $REGION"
echo "Describe:      gcloud run services describe $SERVICE_NAME --region $REGION"
echo "Health check:  curl $SERVICE_URL/health"
echo "Chat test:     curl -X POST $SERVICE_URL/ask -H 'Content-Type: application/json' -d '{\"question\": \"Hello!\"}'"
echo ""
echo "🔗 Cloud Console: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics?project=$PROJECT_ID"
