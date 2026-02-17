#!/bin/sh
set -e

echo "Running database migrations..."
/app/gateway/.venv/bin/python -m alembic upgrade head

echo "Starting gateway server..."
exec /app/gateway/.venv/bin/python -m gateway.main
