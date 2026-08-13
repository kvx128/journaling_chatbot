FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY . .

RUN uv sync --frozen --extra ml

COPY ml/train/adapters/ft-jrn /app/adapter

EXPOSE 8003

CMD ["uv", "run", "uvicorn", "services.model_server.main:app", "--host", "0.0.0.0", "--port", "8003"]
