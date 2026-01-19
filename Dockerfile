FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ConvIR.ipynb ./
RUN pip install --no-cache-dir \
    torch>=1.8.0 \
    torchvision>=0.9.0 \
    numpy>=1.19.0 \
    pillow>=8.0.0 \
    scikit-image>=0.18.0 \
    tensorboard>=2.5.0 \
    jupyter \
    notebook

# Copy the rest of the application
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]