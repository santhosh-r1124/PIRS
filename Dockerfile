# One image for both services (API and dashboard); docker-compose.yml picks the command.
FROM python:3.11-slim

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Same pinned requirements as a local install (CPU PyTorch index + PyG prebuilt wheels)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY run.py pyproject.toml ./

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
