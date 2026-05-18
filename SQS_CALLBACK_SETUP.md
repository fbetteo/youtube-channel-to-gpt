# SQS Callback Rollout

This repo now supports a callback-only SQS flow:

1. The API still dispatches Lambda the same way.
2. Lambda publishes `video_completed` / `video_failed` messages to SQS.
3. A separate Hetzner worker consumes the queue and updates Postgres locally.
4. If SQS is not configured, Lambda falls back to the existing HTTP callbacks.

## 1. Gunicorn Access Logs Into PM2

Update the PM2 command so Gunicorn emits access logs to stdout/stderr and PM2 captures them.

Use this command:

```bash
gunicorn transcript_api:app \
  --workers 3 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --capture-output \
  --log-level info
```

If you use PM2 directly:

```bash
pm2 start "gunicorn transcript_api:app --workers 3 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --access-logfile - --error-logfile - --capture-output --log-level info" --name transcript-api
```

If the process already exists:

```bash
pm2 restart transcript-api --update-env
pm2 logs transcript-api
pm2 monit
pm2 install pm2-logrotate
```

What you get immediately:

1. Per-request access logs in PM2.
2. App errors and Gunicorn errors in the same place.
3. CPU and memory visibility with `pm2 monit`.

## 2. Nginx Request Timing Logs

Add a timing-aware log format in your Nginx config, usually in `http {}` or the specific site config if you prefer.

```nginx
log_format timed_combined '$remote_addr - $remote_user [$time_local] '
                          '"$request" $status $body_bytes_sent '
                          '"$http_referer" "$http_user_agent" '
                          'rt=$request_time '
                          'uct=$upstream_connect_time '
                          'uht=$upstream_header_time '
                          'urt=$upstream_response_time '
                          'ua="$upstream_addr"';
```

Then use it on the API server block:

```nginx
access_log /var/log/nginx/transcript_api_access.log timed_combined;
error_log /var/log/nginx/transcript_api_error.log warn;
```

Reload Nginx:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

What to look for during an incident:

1. `rt=` total request time seen by Nginx.
2. `urt=` upstream time spent waiting for Gunicorn.
3. `499`, `502`, `504` spikes.

If `urt` spikes, the app layer is backing up.
If `rt` is high with low `urt`, the issue is probably before the app or client-side.

## 3. AWS SQS Setup

Create two queues in AWS SQS.

### Step 1: Create the dead-letter queue

In AWS Console:

1. Go to `SQS`.
2. Click `Create queue`.
3. Choose `Standard`.
4. Name it `youtube-transcript-results-dlq`.
5. Leave defaults for access policy, encryption, and delivery delay.
6. Optional but recommended: set message retention to `14 days` so failed messages are easy to inspect later.
7. Do not attach another DLQ to this queue.
8. Click `Create queue`.

What this queue is for:

1. It is a parking lot for bad messages.
2. If the consumer keeps failing the same message over and over, SQS moves it out of the main queue into the DLQ.
3. That prevents one poisoned message from being retried forever in the main flow.

### Step 2: Create the main queue

1. Click `Create queue` again.
2. Choose `Standard`.
3. Name it `youtube-transcript-results`.
4. Leave `Delivery delay` at `0`.
5. Set `Visibility timeout` to `120 seconds`.
6. Set `Receive message wait time` to `20 seconds`.
7. Expand the `Dead-letter queue` section.
8. Turn dead-letter queue handling `On` or choose `Enabled`.
9. Choose `Use existing queue`.
10. Select `youtube-transcript-results-dlq`.
11. Set `Maximum receives` to `5`.
12. Leave the rest at defaults unless you have a reason to change them.
13. Click `Create queue`.

What `Maximum receives = 5` means:

1. A consumer receives a message from the main queue.
2. If processing fails and the consumer does not delete it, the message becomes visible again after the visibility timeout.
3. After that happens 5 times, SQS automatically moves that message into the DLQ.

The DLQ is attached by the main queue's redrive policy, not the other way around.

What to check before attaching:

1. Both queues must be the same type.
2. In this setup, both must be `Standard`.
3. The DLQ dropdown will usually only show compatible queues in the same AWS account and region.

### Step 2b: If you already created the main queue without the DLQ

You do not need to recreate it.

1. Open the `youtube-transcript-results` queue.
2. Click `Edit`.
3. Find the `Dead-letter queue` section.
4. Turn it `On` / `Enabled`.
5. Choose `Use existing queue`.
6. Select `youtube-transcript-results-dlq`.
7. Set `Maximum receives` to `5`.
8. Click `Save`.

If AWS does not let you select the DLQ, check these first:

1. The main queue and DLQ are in the same region.
2. Both are `Standard`, not one `FIFO` and one `Standard`.
3. You are editing the main queue, not the DLQ.
4. The DLQ was created successfully and is visible in SQS.

Copy the queue URL. You will use it for `LAMBDA_RESULTS_QUEUE_URL`.

## 4. IAM Permissions

### Lambda role permissions

Add this policy to the Lambda execution role, replacing the region/account/queue ARN:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sqs:SendMessage"
      ],
      "Resource": "arn:aws:sqs:us-east-1:123456789012:youtube-transcript-results"
    }
  ]
}
```

### Hetzner consumer credentials

Your Hetzner app already uses AWS credentials for Lambda/S3. Extend that IAM user or IAM access key policy with:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:ChangeMessageVisibility",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:us-east-1:123456789012:youtube-transcript-results"
    }
  ]
}
```

## 5. Environment Variables

Set these on Hetzner for the API and the new consumer worker:

```bash
AWS_DEFAULT_REGION=us-east-1
LAMBDA_RESULTS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/youtube-transcript-results
SQS_CONSUMER_WAIT_TIME_SECONDS=20
SQS_CONSUMER_MAX_MESSAGES=10
SQS_CONSUMER_VISIBILITY_TIMEOUT=120
SQS_CONSUMER_CONCURRENCY=10
```

Set this on the Lambda function too:

```bash
LAMBDA_RESULTS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/youtube-transcript-results
```

With that env var present, Lambda will publish to SQS first and only fall back to HTTP callbacks if the SQS send fails.

## 6. Deploy Order

Do the rollout in this order:

1. Deploy the API changes from this repo.
2. Deploy the Lambda code changes.
3. Create the SQS queues and IAM permissions.
4. Set `LAMBDA_RESULTS_QUEUE_URL` on Hetzner and Lambda.
5. Start the SQS consumer worker.
6. Watch PM2 logs, Nginx timing logs, and queue depth during a large job.

## 7. Start the SQS Consumer On Hetzner

Run it as a separate PM2 process:

```bash
pm2 start "python3 src/sqs_result_consumer.py" --name transcript-sqs-consumer
pm2 save
pm2 logs transcript-sqs-consumer
```

This process should live separately from the Gunicorn API process.

## 8. Message Shape

Lambda now sends messages like:

```json
{
  "event_type": "video_completed",
  "job_id": "...",
  "video_id": "...",
  "user_id": "...",
  "lambda_request_id": "...",
  "sent_at": 1712345678,
  "s3_key": "user/job/video.txt",
  "transcript_length": 12345,
  "metadata": {
    "video_id": "...",
    "transcript_language": "en",
    "transcript_type": "manual"
  }
}
```

Failures use `event_type=video_failed` and include `error`, `error_type`, `stage`, `retriable`, and `attempts`.

## 9. Operational Notes

1. The queue is `Standard`, so duplicate delivery is possible.
2. The DB finalization path was updated to be idempotent so retries do not double-refund credits.
3. The legacy HTTP callback endpoints still exist and use the same processing logic as the SQS consumer.
4. You can disable direct HTTP callback usage later by removing `API_BASE_URL` from the Lambda if you want queue-only result delivery.

## 10. First Production Check

After rollout, run one large job and watch:

1. `pm2 logs transcript-api`
2. `pm2 logs transcript-sqs-consumer`
3. `/var/log/nginx/transcript_api_access.log`
4. SQS `ApproximateNumberOfMessagesVisible`
5. SQS DLQ depth

Healthy behavior looks like this:

1. Queue depth rises during bursts.
2. Consumer drains it steadily.
3. Dashboard status and credits endpoints remain responsive.
4. Nginx upstream timings stop spiking during completion bursts.
