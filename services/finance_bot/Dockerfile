FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY . .

RUN uv sync --frozen

EXPOSE 8001

CMD uv run uvicorn services.finance_bot.main:app --host 0.0.0.0 --port 8001
