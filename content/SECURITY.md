# Security policy — Meridian

---

## Supported versions

Security fixes are applied to the latest stable release only. Older versions do not receive patches.

| Version | Supported |
|---|---|
| 1.0.x (latest) | Yes |
| 0.2.x | No |
| 0.1.x | No |

---

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues via email to: `security@[your-domain].dev`

Include in your report:
- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept if available)
- Any suggested mitigations you have identified
- Your preferred contact for follow-up questions

You will receive an acknowledgment within 48 hours. If the issue is confirmed, a fix will be targeted within 14 days for critical issues and 30 days for moderate issues. You will be credited in the `CHANGELOG.md` security section unless you prefer to remain anonymous.

---

## Security model

### What Meridian processes

Meridian is designed to handle sensitive business documents including:
- Internal compliance policies
- Board meeting recordings
- Audit logs and financial compliance data
- Employee and customer-facing policy documents

These inputs may contain confidential business information and personally identifiable information (PII). The security controls below are designed accordingly.

### Authentication

The v1.0 API uses a single Bearer token for authentication (`MERIDIAN_API_KEY`). This is appropriate for a portfolio/demo deployment. A production multi-tenant deployment would require a proper OAuth2 implementation.

- API keys must be at least 32 characters of cryptographically random bytes
- Keys are stored in GCP Secret Manager in production, never in environment files committed to version control
- Keys are hashed before storage; only the first 8 characters (prefix) are stored in plaintext in the audit log for debugging purposes

### Data handling

**In transit:** All communication between the client and the API uses TLS 1.2+ enforced by GCP Cloud Run and the load balancer. Internal service-to-service communication within the Docker Compose network is unencrypted in development; in production, this is mitigated by network isolation.

**At rest:** Uploaded files and generated reports are stored in GCP Cloud Storage with server-side AES-256 encryption (Google-managed keys by default). The PostgreSQL database uses GCP Cloud SQL's disk encryption at rest.

**Data retention:** All uploaded files, extracted text, and generated reports are automatically deleted after 7 days. Audit log entries are retained for 90 days. Vector embeddings of the regulatory corpus (not user data) are retained indefinitely.

**Third-party transmission:** User-submitted content is sent to:
- Anthropic API (synthesis reasoning) — governed by Anthropic's data processing agreement
- HuggingFace Inference API (specialist model calls) — governed by HuggingFace's terms
- LangSmith (trace logging) — can be disabled by unsetting `LANGCHAIN_TRACING_V2`

If processing content under strict data residency requirements (GDPR Article 44+), review the data processing locations of each third-party service before deployment.

### File upload security

All uploaded files are validated before processing:

1. **Size check** — files over 500 MB are rejected before reading content
2. **MIME type validation** — the declared `Content-Type` is cross-checked against the actual file magic bytes using `python-magic`; mismatches are rejected
3. **Extension allowlist** — only explicitly permitted extensions are accepted
4. **Malware scanning** — not implemented in v1.0; recommended addition for production deployments handling untrusted uploads (ClamAV or GCP Security Command Center)
5. **Path traversal prevention** — storage keys are UUID-based with no user-controlled path components

### Webhook security

If `WEBHOOK_SECRET` is configured, outbound webhook payloads are signed with HMAC-SHA256. Receiving systems should:

1. Verify the `X-Meridian-Signature` header matches `sha256=HMAC(WEBHOOK_SECRET, raw_body)`
2. Verify `X-Meridian-Timestamp` is within 5 minutes of the current time (replay attack prevention)
3. Reject requests that fail either check with HTTP 400

### Dependency security

Dependencies are pinned to exact versions in `requirements.txt`. `pip-audit` runs in CI on every push to check for known CVEs in pinned dependencies. Dependabot is configured to open PRs for security updates.

### Secrets management

The following must never appear in:
- Source code
- Committed `.env` files
- Docker images
- Log output
- Error messages

Any of: `ANTHROPIC_API_KEY`, `HF_API_TOKEN`, `PINECONE_API_KEY`, `MERIDIAN_API_KEY`, `WEBHOOK_SECRET`, `DATABASE_URL` (contains password), `REDIS_URL` (may contain password).

The `.gitignore` excludes `.env`, `.env.*`, and `*.key`. A pre-commit hook (`detect-secrets`) is configured to scan for accidentally committed secrets.

---

## Known limitations (v1.0)

The following security properties are intentionally limited in the portfolio deployment and would need to be addressed before a production multi-tenant launch:

| Limitation | Impact | Production mitigation |
|---|---|---|
| Single shared API key | Any key compromise exposes all jobs | Per-user API key management |
| No rate limiting on status/report endpoints | Potential enumeration of job IDs | Require job ownership verification |
| SQLite checkpointer stores intermediate LLM outputs | Intermediate outputs not encrypted at rest | PostgreSQL checkpointer with encrypted disk |
| LangSmith traces contain submitted document excerpts | Submitted content visible to LangSmith | Use LangSmith's data masking features or self-host |
| No input sanitization beyond file type validation | Prompt injection via document content | Add prompt injection detection layer |

---

## Security changelog

**v1.0.1**
- Added HMAC-SHA256 webhook payload signing
- Added `python-magic` file type validation (magic byte inspection)
- Added path traversal prevention for storage keys
- Added `X-Meridian-Timestamp` replay attack prevention

**v1.0.0**
- Initial security model documented
- Bearer token authentication
- GCP Secret Manager integration for production deployments
- `detect-secrets` pre-commit hook
- `pip-audit` in CI
