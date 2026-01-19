FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt* ./
RUN pip install --no-cache-dir torch torchvision numpy pillow scikit-image tensorboard

# Copy application code
COPY . .

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "-c", "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"]