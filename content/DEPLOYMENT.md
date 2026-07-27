# Deployment — Meridian

This guide covers local Docker deployment, staging on GCP, and a production Cloud Run setup with Cloud SQL, Secret Manager, and Cloud Storage.

---

## 1. Local deployment (Docker Compose)

The fastest way to run Meridian end-to-end on a local machine.

### Prerequisites

- Docker Desktop 24.0+ with at least 4 GB RAM allocated to Docker
- 10 GB free disk space (for Docker images and the corpus)

### Steps

```bash
# Clone the repo
git clone https://github.com/yourhandle/meridian.git
cd meridian

# Copy and fill in environment variables
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY, HF_API_TOKEN, MERIDIAN_API_KEY

# Start all services (API, worker, PostgreSQL, Redis, MLflow)
docker compose up -d

# Wait for PostgreSQL to be ready (~10 seconds)
docker compose exec api python -c "from src.db import engine; engine.connect()"

# Run migrations
docker compose exec api alembic upgrade head

# Ingest the regulatory corpus (GDPR + SOC-2, ~10 min on first run)
docker compose exec api python scripts/ingest_corpus.py --source gdpr,soc2

# Verify the stack is healthy
curl http://localhost:8000/v1/health
```

### Service ports

| Service | Port | UI |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000/docs |
| Streamlit | 8501 | http://localhost:8501 |
| MLflow | 5000 | http://localhost:5000 |
| PostgreSQL | 5432 | — |
| Redis | 6379 | — |
| Celery Flower | 5555 | http://localhost:5555 |

### Stopping and resetting

```bash
# Stop all services (preserve data volumes)
docker compose down

# Stop and delete all data (full reset)
docker compose down -v
```

---

## 2. Building the production Docker image

The production image uses a multi-stage build to minimize the final image size.

```dockerfile
# docker/Dockerfile

# --- Build stage ---
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Runtime stage ---
FROM python:3.12-slim AS runtime
WORKDIR /app

# Install system dependencies required at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

# Non-root user
RUN useradd --create-home --shell /bin/bash meridian
USER meridian

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
```

### Build and push to GCP Artifact Registry

```bash
# Set your GCP project
export GCP_PROJECT=your-project-id
export GCP_REGION=us-central1
export REPO=meridian

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Build for linux/amd64 (Cloud Run requires x86_64)
docker build \
  --platform linux/amd64 \
  --tag ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO}/api:latest \
  --tag ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO}/api:$(git rev-parse --short HEAD) \
  -f docker/Dockerfile .

# Push both tags
docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO}/api:latest
docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO}/api:$(git rev-parse --short HEAD)
```

---

## 3. GCP infrastructure setup

### Required GCP services

- Cloud Run — API and worker services
- Cloud SQL (PostgreSQL 15) — primary database
- Cloud Storage — file uploads and reports
- Secret Manager — API keys
- Artifact Registry — Docker images
- VPC + Serverless VPC Connector — Cloud Run → Cloud SQL private networking

### One-time GCP setup

```bash
# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  vpcaccess.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create meridian \
  --repository-format=docker \
  --location=${GCP_REGION} \
  --description="Meridian container images"

# Create Cloud Storage buckets
gsutil mb -l ${GCP_REGION} gs://${GCP_PROJECT}-meridian-uploads
gsutil mb -l ${GCP_REGION} gs://${GCP_PROJECT}-meridian-reports

# Set lifecycle rules (auto-delete after 7 days)
gsutil lifecycle set docker/gcs-lifecycle-uploads.json \
  gs://${GCP_PROJECT}-meridian-uploads
gsutil lifecycle set docker/gcs-lifecycle-reports.json \
  gs://${GCP_PROJECT}-meridian-reports
```

### Cloud SQL setup

```bash
# Create PostgreSQL instance
gcloud sql instances create meridian-db \
  --database-version=POSTGRES_15 \
  --cpu=2 \
  --memory=4GB \
  --region=${GCP_REGION} \
  --storage-size=50GB \
  --storage-auto-increase \
  --availability-type=ZONAL \
  --no-assign-ip  # private IP only

# Create database and user
gcloud sql databases create meridian --instance=meridian-db
gcloud sql users create meridian \
  --instance=meridian-db \
  --password=$(openssl rand -base64 32)

# Install pgvector extension (connect via Cloud SQL Auth Proxy)
cloud-sql-proxy ${GCP_PROJECT}:${GCP_REGION}:meridian-db &
psql "host=127.0.0.1 port=5432 dbname=meridian user=meridian" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Secret Manager setup

```bash
# Store all secrets
echo -n "sk-ant-your-key" | \
  gcloud secrets create anthropic-api-key --data-file=-

echo -n "hf_your-token" | \
  gcloud secrets create hf-api-token --data-file=-

echo -n "mer_live_$(openssl rand -hex 32)" | \
  gcloud secrets create meridian-api-key --data-file=-

echo -n "postgresql://meridian:PASSWORD@/meridian?host=/cloudsql/PROJECT:REGION:meridian-db" | \
  gcloud secrets create database-url --data-file=-

echo -n "redis://10.0.0.x:6379/0" | \
  gcloud secrets create redis-url --data-file=-
```

---

## 4. Cloud Run deployment

### API service

```bash
gcloud run deploy meridian-api \
  --image=${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/meridian/api:latest \
  --region=${GCP_REGION} \
  --platform=managed \
  --min-instances=1 \
  --max-instances=5 \
  --concurrency=10 \
  --cpu=2 \
  --memory=4Gi \
  --timeout=900 \
  --no-allow-unauthenticated \
  --service-account=meridian-sa@${GCP_PROJECT}.iam.gserviceaccount.com \
  --add-cloudsql-instances=${GCP_PROJECT}:${GCP_REGION}:meridian-db \
  --vpc-connector=meridian-vpc-connector \
  --set-secrets=\
ANTHROPIC_API_KEY=anthropic-api-key:latest,\
HF_API_TOKEN=hf-api-token:latest,\
MERIDIAN_API_KEY=meridian-api-key:latest,\
DATABASE_URL=database-url:latest,\
REDIS_URL=redis-url:latest \
  --set-env-vars=\
ENVIRONMENT=production,\
LOG_LEVEL=INFO,\
LANGCHAIN_TRACING_V2=true,\
GCS_UPLOADS_BUCKET=${GCP_PROJECT}-meridian-uploads,\
GCS_REPORTS_BUCKET=${GCP_PROJECT}-meridian-reports
```

### Worker service (Celery)

The Celery worker runs as a separate Cloud Run job (not a service, since it doesn't serve HTTP traffic):

```bash
gcloud run jobs create meridian-worker \
  --image=${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/meridian/api:latest \
  --region=${GCP_REGION} \
  --cpu=2 \
  --memory=4Gi \
  --max-retries=3 \
  --service-account=meridian-sa@${GCP_PROJECT}.iam.gserviceaccount.com \
  --add-cloudsql-instances=${GCP_PROJECT}:${GCP_REGION}:meridian-db \
  --vpc-connector=meridian-vpc-connector \
  --set-secrets=[same as API] \
  --command="celery" \
  --args="-A,src.api.worker,worker,--loglevel=info,--concurrency=4"
```

**Note on Cloud Run for workers:** Cloud Run jobs have a maximum execution time of 24 hours, which is sufficient for any single job. For a true long-running worker, use GCP Cloud Run Services with `--command=celery` and `--min-instances=1`. The worker service has no `--allow-unauthenticated` flag and no exposed HTTP port.

### Run database migrations after deployment

```bash
gcloud run jobs create meridian-migrate \
  --image=${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/meridian/api:latest \
  --region=${GCP_REGION} \
  --command="alembic" \
  --args="upgrade,head" \
  [same secrets and cloudsql flags as API]

gcloud run jobs execute meridian-migrate --wait
```

---

## 5. IAM and service account

The Cloud Run services run under a dedicated service account `meridian-sa`:

```bash
# Create service account
gcloud iam service-accounts create meridian-sa \
  --display-name="Meridian service account"

SA=meridian-sa@${GCP_PROJECT}.iam.gserviceaccount.com

# Grant required roles
gcloud projects add-iam-policy-binding ${GCP_PROJECT} \
  --member=serviceAccount:${SA} \
  --role=roles/cloudsql.client

gcloud projects add-iam-policy-binding ${GCP_PROJECT} \
  --member=serviceAccount:${SA} \
  --role=roles/storage.objectAdmin

gcloud projects add-iam-policy-binding ${GCP_PROJECT} \
  --member=serviceAccount:${SA} \
  --role=roles/secretmanager.secretAccessor

gcloud projects add-iam-policy-binding ${GCP_PROJECT} \
  --member=serviceAccount:${SA} \
  --role=roles/cloudtrace.agent
```

---

## 6. Continuous deployment

The GitHub Actions workflow in `.github/workflows/deploy.yml` automatically builds and deploys on push to `main`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write     # Workload Identity Federation

    steps:
      - uses: actions/checkout@v4

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Build and push image
        run: |
          docker build --platform linux/amd64 \
            -t ${{ env.IMAGE }}:${{ github.sha }} \
            -t ${{ env.IMAGE }}:latest \
            -f docker/Dockerfile .
          docker push ${{ env.IMAGE }}:${{ github.sha }}
          docker push ${{ env.IMAGE }}:latest

      - name: Run migrations
        run: gcloud run jobs execute meridian-migrate --wait --region=${{ env.REGION }}

      - name: Deploy API
        run: |
          gcloud run deploy meridian-api \
            --image=${{ env.IMAGE }}:${{ github.sha }} \
            --region=${{ env.REGION }}

      - name: Deploy worker
        run: |
          gcloud run services update meridian-worker \
            --image=${{ env.IMAGE }}:${{ github.sha }} \
            --region=${{ env.REGION }}
```

---

## 7. Verifying the production deployment

```bash
# Get the Cloud Run service URL
SERVICE_URL=$(gcloud run services describe meridian-api \
  --region=${GCP_REGION} \
  --format="value(status.url)")

# Health check
curl -H "Authorization: Bearer $(gcloud secrets versions access latest \
  --secret=meridian-api-key)" \
  ${SERVICE_URL}/v1/health

# Submit a test job
curl -X POST ${SERVICE_URL}/v1/submit \
  -H "Authorization: Bearer <YOUR_KEY>" \
  -F "files=@data/sample_docs/sample_privacy_policy.pdf" \
  -F "regulation_scope=gdpr"
```

---

## 8. Rollback procedure

```bash
# List recent revisions
gcloud run revisions list --service=meridian-api --region=${GCP_REGION}

# Roll back to a specific revision (split traffic 100% to that revision)
gcloud run services update-traffic meridian-api \
  --region=${GCP_REGION} \
  --to-revisions=meridian-api-00042-abc=100
```

---

## 9. Cost estimate (portfolio/demo deployment)

Based on GCP pricing as of April 2026. Actual costs depend on usage.

| Resource | Spec | Estimated monthly cost |
|---|---|---|
| Cloud Run API (min 1 instance) | 2 vCPU, 4 GB, min 1 | ~$35 |
| Cloud Run worker | 2 vCPU, 4 GB, on demand | ~$10 |
| Cloud SQL (PostgreSQL) | 2 vCPU, 4 GB, 50 GB SSD, zonal | ~$70 |
| Cloud Storage (uploads + reports) | ~5 GB/month | ~$0.10 |
| Secret Manager | 6 secrets, ~100 accesses/day | ~$0.10 |
| Artifact Registry | ~2 GB image storage | ~$0.20 |
| VPC Connector | 1 instance | ~$15 |
| **Total** | | **~$130/month** |

Anthropic API and HuggingFace Inference API costs are additional and depend on job volume (~$0.05–0.20 per job depending on document length and modalities).
