# S3 ZIP Download Setup Guide

## Overview

The API now supports generating ZIP files on-demand by downloading transcript files from S3 using concurrent downloads for better performance. This replaces the previous local file-based ZIP generation.

## How It Works

1. **Lambda Processing**: Videos are processed by AWS Lambda functions that store transcripts in S3
2. **Lambda Callbacks**: Lambda calls back to your API to update job progress
3. **ZIP Generation**: When user requests download, API concurrently downloads files from S3 and creates ZIP
4. **Delivery**: ZIP is streamed directly to user without storing on disk

## Required Environment Variables

Add these to your `.env` file or environment:

```bash
# S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket-name

# Lambda Configuration (for dispatching jobs)
LAMBDA_FUNCTION_NAME=your-lambda-function-name
API_BASE_URL=https://your-api-domain.com  # For production
# API_BASE_URL=https://your-ngrok-url.ngrok.io  # For local testing with ngrok
```

## Local Testing Setup

### Option 1: Using ngrok (Recommended for Lambda testing)

1. **Install ngrok**: Download from https://ngrok.com/download
2. **Start your API**: `uvicorn src.transcript_api:app --reload --port 8000`
3. **Create ngrok tunnel**: `ngrok http 8000`
4. **Update Lambda environment**: Set `API_BASE_URL` to your ngrok URL
5. **Test end-to-end**: Lambda can now call back to your local API

### Option 2: Mock S3 files for testing

If you want to test ZIP generation without Lambda:

```python
# Create mock S3 files for testing
import boto3

s3_client = boto3.client('s3')
bucket_name = 'your-test-bucket'

# Upload test transcript files
test_files = [
    {'key': 'user123/job456/video1.txt', 'content': 'Test transcript 1'},
    {'key': 'user123/job456/video2.txt', 'content': 'Test transcript 2'},
]

for file_info in test_files:
    s3_client.put_object(
        Bucket=bucket_name,
        Key=file_info['key'],
        Body=file_info['content'].encode('utf-8'),
        ContentType='text/plain'
    )
```

## Testing the New Endpoints

### 1. Test Internal Callbacks

```bash
# Test video completion callback
curl -X POST "http://127.0.0.1:8000/internal/job/test-job-123/video-complete" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ",
    "s3_key": "user123/job456/dQw4w9WgXcQ.txt",
    "transcript_length": 1500,
    "metadata": {"language": "en"}
  }'

# Test video failure callback
curl -X POST "http://127.0.0.1:8000/internal/job/test-job-123/video-failed" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ",
    "error": "No transcript available",
    "error_type": "TranscriptNotFound"
  }'
```

### 2. Test ZIP Download

```bash
# Download ZIP file (requires authentication)
curl -X GET "http://127.0.0.1:8000/channel/download/results/your-job-id" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -o "test_download.zip"
```

### 3. Test Complete Workflow

1. **Create a job**: Use `/channel/download/selected` endpoint
2. **Simulate Lambda processing**: Use internal callback endpoints to mark videos complete
3. **Download results**: Use `/channel/download/results/{job_id}` endpoint

## Performance Comparison

| Method | 100 Videos (1MB each) | Pros | Cons |
|--------|----------------------|------|------|
| **Concurrent S3** | ~5-10 seconds | Fast, scalable | Requires S3 setup |
| **Sequential S3** | ~30-60 seconds | Simple, reliable | Slower for large jobs |
| **Local Files** | ~2-3 seconds | Fastest | Not scalable, memory issues |

## Monitoring & Debugging

### Check S3 Files

```bash
# List files in S3 bucket
aws s3 ls s3://your-bucket-name/user123/job456/ --recursive

# Download a specific file for inspection
aws s3 cp s3://your-bucket-name/user123/job456/video1.txt ./test_file.txt
```

### API Debug Endpoints

```bash
# Check memory usage
curl http://127.0.0.1:8000/debug/memory

# Check active jobs
curl http://127.0.0.1:8000/debug/jobs

# Force garbage collection
curl -X POST http://127.0.0.1:8000/debug/gc
```

### Error Handling

The new ZIP creation includes robust error handling:

- **S3 Connection Issues**: Falls back to sequential downloads
- **Missing Files**: Includes error placeholders in ZIP
- **Partial Failures**: Creates ZIP with available files + error reports
- **Authentication**: Verifies user owns the job before allowing download

## Cost Considerations

**S3 GET Requests**: $0.0004 per 1,000 requests
**Data Transfer**: $0.09 per GB (first 1GB free monthly)

**Example Cost (100 videos, 1MB each)**:
- GET requests: $0.00004 (practically free)
- Data transfer: $0.009 (less than 1 cent)
- **Total: ~$0.009 per download**

## Production Deployment

1. **Set production environment variables**
2. **Configure S3 bucket with proper permissions**
3. **Deploy Lambda function with callback URL**
4. **Set up CloudWatch monitoring**
5. **Configure S3 lifecycle policies for cleanup**

## Troubleshooting

### Common Issues

1. **"S3 bucket name not configured"**: Check `S3_BUCKET_NAME` environment variable
2. **"Access denied"**: Verify AWS credentials and S3 permissions
3. **"Job not found"**: Ensure job exists and user has access
4. **Lambda timeouts**: Check API_BASE_URL is accessible from Lambda

### Logs to Check

- FastAPI logs: Job creation and ZIP generation
- Lambda logs: Transcript processing and callbacks
- S3 access logs: File upload/download operations
- CloudWatch: Lambda execution metrics