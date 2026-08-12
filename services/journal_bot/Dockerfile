FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY . .

RUN uv sync --frozen

EXPOSE 8002

CMD uv run uvicorn services.journal_bot.main:app --host 0.0.0.0 --port 8002
