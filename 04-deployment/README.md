# Module 4: Model Deployment

## What This Module Covers

- Deployment strategies: online vs. offline
- Web service deployment with Flask
- Streaming deployment with AWS Kinesis & Lambda
- Batch scoring for offline processing

## Structure

| Folder | Description |
|--------|-------------|
| `web-service/` | Flask REST API |
| `streaming/` | AWS Kinesis + Lambda |
| `batch/` | Offline batch scoring scripts |

## Deployment Modes

| Mode | Latency | Use Case | Tools |
|------|---------|----------|-------|
| Web service | Real-time | Single predictions on demand | Flask, FastAPI |
| Streaming | Near real-time | Event-driven predictions | AWS Kinesis, Lambda |
| Batch | Offline | Large-scale scoring jobs | Pandas, cron |

## Notes & Learnings

_Add your notes here as you work through the module._
