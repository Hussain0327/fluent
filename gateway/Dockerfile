FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libopus-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/gateway
COPY . /app/gateway/

RUN uv venv .venv --python 3.12
RUN uv sync

RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
