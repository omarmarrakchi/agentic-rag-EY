FROM python:3.11-slim

WORKDIR /app

# System dependencies (OpenMP for torch/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first to avoid the 2 GB CUDA download
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend source code
COPY backend/ /app/backend/

EXPOSE 8000

WORKDIR /app/backend
CMD ["python", "run_api.py"]
