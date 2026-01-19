FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    numpy==1.24.3 \
    pillow==10.0.0 \
    scikit-image==0.21.0 \
    tensorboard==2.13.0

# Copy application code
COPY . .

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "-c", "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"]