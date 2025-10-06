# Deployment Guide

**ML-AI Framework v0.1.0**

This guide provides step-by-step instructions for deploying the ML-AI Framework to production environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Deployment Options](#deployment-options)
6. [Monitoring & Logging](#monitoring--logging)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)
9. [Security Considerations](#security-considerations)
10. [Maintenance](#maintenance)

---

## Prerequisites

### System Requirements

**Minimum Requirements**:
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.10 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Storage**: 5GB free space (application + data + dependencies)
- **Network**: HTTPS outbound for OpenAI API access

**Recommended Production**:
- **OS**: Ubuntu 22.04 LTS or Amazon Linux 2023
- **RAM**: 16GB for concurrent workflows
- **CPU**: 4+ cores for parallel processing
- **Storage**: 20GB with SSD for performance

### Required Accounts & API Keys

1. **OpenAI API Key** (Required)
   - Sign up at: https://platform.openai.com/
   - Create API key in dashboard
   - Ensure billing is configured
   - Recommended: Set usage limits

2. **Optional Services**
   - AWS/GCP/Azure account for cloud deployment
   - GitHub account for CI/CD integration
   - Monitoring service (Datadog, New Relic, etc.)

### Software Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
# or
brew update && brew upgrade  # macOS

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip  # Ubuntu
# or
brew install python@3.10  # macOS

# Install git (if not present)
sudo apt install git  # Ubuntu
brew install git  # macOS

# Optional: Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

---

## Environment Setup

### 1. Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/ml-ai-framework.git
cd ml-ai-framework

# Or if deploying from tarball
tar -xzf ml-ai-framework-0.1.0.tar.gz
cd ml-ai-framework
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows PowerShell

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(crewai|langgraph|fastapi|pydantic)"

# Optional: Install development dependencies
pip install -r requirements-test.txt
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit environment file
nano .env  # or vim, code, etc.
```

**Required Environment Variables**:
```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_RELOAD=false  # Set to false in production

# Logging Configuration
LOG_LEVEL=INFO  # Use WARNING or ERROR in production
LOG_JSON=true
LOG_INCLUDE_CALLER=true
```

**Optional Configuration** (use defaults from .env.example):
- Workflow settings (MAX_AGENT_ITERATIONS, DEFAULT_WORKFLOW_TYPE)
- Error handling (RETRY_MAX_ATTEMPTS, CIRCUIT_BREAKER_THRESHOLD)
- Data processing (MISSING_VALUE_THRESHOLD, OUTLIER_METHOD)
- Model training (DEFAULT_TEST_SIZE, RANDOM_STATE)
- Performance (MAX_WORKERS, ASYNC_ENABLED)

### 5. Verify Installation

```bash
# Test imports
python -c "from src.workflows import crew_system, langgraph_system; print('✓ Imports successful')"

# Run tests
pytest tests/ --cov=src -v

# Check configuration
python -c "from config.settings import get_settings; s = get_settings(); print('✓ Configuration loaded')"
```

---

## Installation Methods

### Method 1: Manual Installation (Recommended for Development)

Already covered in [Environment Setup](#environment-setup) above.

### Method 2: Docker Deployment (Recommended for Production)

**Dockerfile** (create in project root):
```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "src.ag_ui_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml** (create in project root):
```yaml
version: '3.8'

services:
  ml-ai-framework:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - SERVER_RELOAD=false
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Build and Run**:
```bash
# Build Docker image
docker build -t ml-ai-framework:0.1.0 .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Method 3: Systemd Service (Linux Production)

**Create service file**: `/etc/systemd/system/ml-ai-framework.service`
```ini
[Unit]
Description=ML-AI Framework API Server
After=network.target

[Service]
Type=simple
User=appuser
Group=appuser
WorkingDirectory=/opt/ml-ai-framework
Environment="PATH=/opt/ml-ai-framework/venv/bin"
EnvironmentFile=/opt/ml-ai-framework/.env
ExecStart=/opt/ml-ai-framework/venv/bin/uvicorn src.ag_ui_server:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Enable and start service**:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable ml-ai-framework

# Start service
sudo systemctl start ml-ai-framework

# Check status
sudo systemctl status ml-ai-framework

# View logs
sudo journalctl -u ml-ai-framework -f
```

---

## Configuration

### Production Environment Variables

**Critical Settings for Production**:
```bash
# Security
OPENAI_API_KEY=sk-prod-key-here  # Use production key
SERVER_RELOAD=false              # Disable auto-reload

# Performance
LOG_LEVEL=WARNING                # Reduce log verbosity
MAX_WORKERS=8                    # Match CPU cores
ASYNC_ENABLED=true               # Enable async processing

# Reliability
RETRY_MAX_ATTEMPTS=5             # Increase retries
CIRCUIT_BREAKER_THRESHOLD=10     # More lenient in production
CIRCUIT_BREAKER_TIMEOUT=120      # Longer recovery period
```

### Configuration Best Practices

1. **Never commit .env files** - Use .env.example as template
2. **Use different keys for dev/staging/prod** - Isolate environments
3. **Rotate API keys regularly** - Every 90 days minimum
4. **Monitor API usage** - Set up billing alerts
5. **Use environment-specific values** - Different timeouts, thresholds per environment

### Configuration Validation

```bash
# Validate configuration
python -c "
from config.settings import get_settings
settings = get_settings()
print(f'Server: {settings.server_host}:{settings.server_port}')
print(f'Model: {settings.openai_model}')
print(f'Log Level: {settings.log_level}')
print('✓ Configuration valid')
"
```

---

## Deployment Options

### Option 1: AWS Deployment (EC2)

**1. Launch EC2 Instance**:
```bash
# Recommended instance: t3.medium (2 vCPU, 4GB RAM)
# AMI: Ubuntu Server 22.04 LTS
# Security Group: Allow inbound 8000, 22 (SSH)
```

**2. Configure Instance**:
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip git

# Clone and setup (see Environment Setup)
```

**3. Use Nginx as Reverse Proxy**:
```bash
# Install Nginx
sudo apt install nginx

# Create Nginx config: /etc/nginx/sites-available/ml-ai-framework
sudo nano /etc/nginx/sites-available/ml-ai-framework
```

**Nginx Configuration**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding on;
    }
}
```

**Enable and start Nginx**:
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/ml-ai-framework /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### Option 2: Google Cloud Platform (Cloud Run)

**1. Prepare for Cloud Run**:
```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

**2. Deploy to Cloud Run**:
```bash
# Build and deploy
gcloud run deploy ml-ai-framework \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10

# Get service URL
gcloud run services describe ml-ai-framework --region us-central1 --format 'value(status.url)'
```

### Option 3: Heroku Deployment

**1. Prepare Heroku Files**:

**Procfile**:
```
web: uvicorn src.ag_ui_server:app --host 0.0.0.0 --port $PORT
```

**runtime.txt**:
```
python-3.10.12
```

**2. Deploy to Heroku**:
```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key-here

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1

# View logs
heroku logs --tail
```

### Option 4: Azure Deployment (App Service)

**1. Install Azure CLI**:
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login
```

**2. Deploy to App Service**:
```bash
# Create resource group
az group create --name ml-ai-framework-rg --location eastus

# Create App Service plan
az appservice plan create --name ml-ai-framework-plan --resource-group ml-ai-framework-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group ml-ai-framework-rg --plan ml-ai-framework-plan --name your-app-name --runtime "PYTHON:3.10"

# Configure app settings
az webapp config appsettings set --resource-group ml-ai-framework-rg --name your-app-name --settings OPENAI_API_KEY=your-key

# Deploy
az webapp up --name your-app-name --resource-group ml-ai-framework-rg
```

---

## Monitoring & Logging

### Application Logging

The framework uses **structlog** for structured JSON logging:

```python
# Logs are automatically formatted as JSON
{
  "event": "workflow_started",
  "workflow_type": "crewai",
  "timestamp": "2025-10-05T10:30:45.123Z",
  "level": "info",
  "logger": "src.workflows.crew_system"
}
```

**Log Locations**:
- **Development**: Console output
- **Production**: `/var/log/ml-ai-framework/app.log` (if configured)
- **Docker**: `docker logs <container-id>`
- **Systemd**: `journalctl -u ml-ai-framework`

### Log Aggregation

**Option 1: CloudWatch (AWS)**:
```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure to collect application logs
# See: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/install-CloudWatch-Agent-on-EC2-Instance.html
```

**Option 2: ELK Stack**:
```yaml
# docker-compose.yml addition
  elasticsearch:
    image: elasticsearch:8.9.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.9.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.9.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

### Health Monitoring

**Add Health Check Endpoint** (recommended):

Edit `src/ag_ui_server.py`:
```python
from datetime import datetime

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "checks": {
            "api": "ok",
            "configuration": "ok"
        }
    }
```

**Monitor with External Services**:
- **UptimeRobot**: https://uptimerobot.com/
- **Pingdom**: https://www.pingdom.com/
- **Datadog**: https://www.datadoghq.com/

**Example Uptime Check**:
```bash
# Cron job for health check
*/5 * * * * curl -f http://localhost:8000/health || echo "Health check failed" | mail -s "ML-AI Framework Alert" admin@example.com
```

### Performance Monitoring

**Application Metrics**:
```python
# Add to ag_ui_server.py for Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

workflow_requests = Counter('workflow_requests_total', 'Total workflow requests', ['workflow_type'])
workflow_duration = Histogram('workflow_duration_seconds', 'Workflow execution time', ['workflow_type'])

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Scaling

### Horizontal Scaling

**Load Balancer Configuration** (Nginx):
```nginx
upstream ml_ai_backend {
    least_conn;  # Use least connections algorithm
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
}

server {
    listen 80;
    location / {
        proxy_pass http://ml_ai_backend;
        # ... other proxy settings
    }
}
```

**Run Multiple Instances**:
```bash
# Instance 1
SERVER_PORT=8001 uvicorn src.ag_ui_server:app --host 0.0.0.0 --port 8001 &

# Instance 2
SERVER_PORT=8002 uvicorn src.ag_ui_server:app --host 0.0.0.0 --port 8002 &

# Instance 3
SERVER_PORT=8003 uvicorn src.ag_ui_server:app --host 0.0.0.0 --port 8003 &
```

### Vertical Scaling

**Optimize Resource Usage**:
```bash
# Increase workers for CPU-bound tasks
MAX_WORKERS=16 uvicorn src.ag_ui_server:app --workers 4

# Allocate more memory (Docker)
docker run --memory=8g --cpus=4 ml-ai-framework:0.1.0
```

### Auto-Scaling (Kubernetes)

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-ai-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-ai-framework
  template:
    metadata:
      labels:
        app: ml-ai-framework
    spec:
      containers:
      - name: ml-ai-framework
        image: ml-ai-framework:0.1.0
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ml-ai-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-ai-framework-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-ai-framework
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Troubleshooting

### Common Issues

#### Issue 1: ModuleNotFoundError

**Symptom**: `ModuleNotFoundError: No module named 'structlog'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep structlog
```

#### Issue 2: OpenAI API Authentication Failed

**Symptom**: `openai.error.AuthenticationError: Invalid API key`

**Solution**:
```bash
# Check .env file exists
ls -la .env

# Verify API key format (should start with sk-)
grep OPENAI_API_KEY .env

# Test API key
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(f'API Key: {os.getenv(\"OPENAI_API_KEY\")[:10]}...')
"

# If key is valid, ensure it's loaded
python -c "from config.settings import get_settings; print(get_settings().openai_api_key[:10])"
```

#### Issue 3: Port Already in Use

**Symptom**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000
# or
netstat -tulpn | grep 8000

# Kill process
kill -9 <PID>

# Or use different port
SERVER_PORT=8001 uvicorn src.ag_ui_server:app --port 8001
```

#### Issue 4: High Memory Usage

**Symptom**: Application using excessive memory

**Solution**:
```bash
# Monitor memory usage
htop  # or top

# Reduce max workers
MAX_WORKERS=2 uvicorn src.ag_ui_server:app

# Implement request queuing
# Consider using Celery for async task processing
```

#### Issue 5: Slow Workflow Execution

**Symptom**: Workflows taking too long

**Solution**:
```bash
# Check OpenAI API latency
curl -w "@curl-format.txt" -o /dev/null -s "https://api.openai.com/v1/models"

# Enable async processing
ASYNC_ENABLED=true

# Reduce agent iterations
MAX_AGENT_ITERATIONS=5

# Use faster model
OPENAI_MODEL=gpt-3.5-turbo
```

### Debug Mode

**Enable Debug Logging**:
```bash
# Set debug log level
LOG_LEVEL=DEBUG uvicorn src.ag_ui_server:app --reload

# View detailed logs
tail -f /var/log/ml-ai-framework/app.log
```

**Python Debugger**:
```python
# Add to code for debugging
import pdb; pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()
```

### Log Analysis

```bash
# Search for errors
grep -i "error" /var/log/ml-ai-framework/app.log

# Count errors by type
grep "error" app.log | cut -d'"' -f4 | sort | uniq -c

# Find slow requests
grep "workflow_duration" app.log | awk '{print $NF}' | sort -n | tail -20
```

---

## Security Considerations

### API Key Security

1. **Never commit API keys to Git**
   ```bash
   # Ensure .env is in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables or secrets management**
   ```bash
   # AWS Secrets Manager
   aws secretsmanager get-secret-value --secret-id ml-ai/openai-key

   # HashiCorp Vault
   vault kv get secret/ml-ai/openai-key
   ```

3. **Rotate keys regularly**
   - Schedule: Every 90 days minimum
   - Use separate keys for dev/staging/prod
   - Implement key rotation procedures

### Network Security

**Firewall Configuration**:
```bash
# Ubuntu UFW
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8000/tcp   # Block direct access, use reverse proxy
sudo ufw enable
```

**HTTPS/TLS Configuration**:
```bash
# Install Certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Input Validation

**Already implemented via Pydantic**:
- All API inputs validated against schemas
- Type checking enforced
- Value bounds validated

### Rate Limiting

**Add rate limiting** (recommended for production):
```python
# Install slowapi
pip install slowapi

# Add to ag_ui_server.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/workflow/execute")
@limiter.limit("10/minute")
async def execute_workflow(request: Request, workflow_request: WorkflowRequest):
    # ... existing code
```

---

## Maintenance

### Regular Tasks

**Daily**:
- [ ] Monitor error logs
- [ ] Check API response times
- [ ] Verify health check endpoints

**Weekly**:
- [ ] Review resource usage (CPU, memory, disk)
- [ ] Check for dependency updates
- [ ] Review OpenAI API usage and costs
- [ ] Backup configuration files

**Monthly**:
- [ ] Update dependencies (`pip list --outdated`)
- [ ] Review and rotate logs
- [ ] Performance optimization review
- [ ] Security audit

**Quarterly**:
- [ ] Rotate API keys
- [ ] Review access controls
- [ ] Disaster recovery drill
- [ ] Capacity planning review

### Backup Procedures

**Configuration Backup**:
```bash
# Backup .env file (securely)
tar -czf config-backup-$(date +%Y%m%d).tar.gz .env config/

# Encrypt backup
gpg -c config-backup-$(date +%Y%m%d).tar.gz

# Store securely (S3, Vault, etc.)
aws s3 cp config-backup-$(date +%Y%m%d).tar.gz.gpg s3://your-backup-bucket/
```

**Data Backup**:
```bash
# Backup data directory
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/

# Automated daily backup (cron)
0 2 * * * /opt/ml-ai-framework/scripts/backup.sh
```

### Update Procedures

**Updating Dependencies**:
```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package-name

# Update all (use with caution)
pip install --upgrade -r requirements.txt

# Test after updates
pytest tests/ --cov=src
```

**Application Updates**:
```bash
# Pull latest code
git pull origin main

# Install new dependencies
pip install -r requirements.txt

# Run migrations (if any)
# python scripts/migrate.py

# Run tests
pytest tests/

# Restart service
sudo systemctl restart ml-ai-framework
```

### Disaster Recovery

**Recovery Plan**:
1. Keep backups in multiple locations
2. Document all configuration
3. Maintain runbook for common incidents
4. Test recovery procedures quarterly

**Recovery Steps**:
```bash
# 1. Restore from backup
aws s3 cp s3://your-backup-bucket/latest-backup.tar.gz.gpg .
gpg -d latest-backup.tar.gz.gpg | tar -xzf -

# 2. Reinstall application
git clone https://github.com/your-org/ml-ai-framework.git
cd ml-ai-framework
pip install -r requirements.txt

# 3. Restore configuration
cp backup/.env .env

# 4. Verify and start
pytest tests/
sudo systemctl start ml-ai-framework
```

---

## Additional Resources

### Documentation Links
- **Main README**: `/README.md`
- **Production Readiness**: `/docs/PRODUCTION_READINESS.md`
- **Quick Reference**: `/docs/QUICK_REFERENCE.md`
- **Contributing Guide**: `/docs/CONTRIBUTING.md`

### External Documentation
- **FastAPI**: https://fastapi.tiangolo.com/
- **CrewAI**: https://docs.crewai.com/
- **LangGraph**: https://python.langchain.com/docs/langgraph
- **Pydantic**: https://docs.pydantic.dev/

### Support
- **Issues**: https://github.com/your-org/ml-ai-framework/issues
- **Discussions**: https://github.com/your-org/ml-ai-framework/discussions

---

## Deployment Checklist

### Pre-Deployment
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with production values
- [ ] OpenAI API key set and validated
- [ ] All tests passing (`pytest tests/`)
- [ ] Code quality checks passed (black, ruff, mypy)
- [ ] Security scan completed

### Deployment
- [ ] Application deployed to chosen platform
- [ ] Environment variables configured on platform
- [ ] Health check endpoint responding
- [ ] API endpoints accessible
- [ ] HTTPS/TLS configured
- [ ] Firewall rules configured
- [ ] Reverse proxy configured (if applicable)
- [ ] Monitoring enabled
- [ ] Log aggregation configured

### Post-Deployment
- [ ] Smoke tests completed
- [ ] Performance baseline established
- [ ] Monitoring dashboards configured
- [ ] Alerts configured
- [ ] Backup procedures tested
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Runbook created

---

**Last Updated**: 2025-10-05
**Version**: 1.0
**Maintainer**: ML-AI Framework Team
