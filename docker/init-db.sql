-- Meridian database initialization
-- Runs on first container start via docker-entrypoint-initdb.d

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Set default search path
ALTER DATABASE meridian SET search_path TO public;

-- Optimize PostgreSQL for vector workloads
-- These are advisory settings; Alembic handles the actual schema
ALTER SYSTEM SET shared_preload_libraries = 'vector';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

SELECT pg_reload_conf();
