FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=0 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY . /app

RUN uv sync

EXPOSE 8000

ENTRYPOINT ["openrsvp", "runserver"]
