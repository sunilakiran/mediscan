FROM python:3.11-slim

WORKDIR /app

# All system dependencies ek saath
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Non root user
RUN useradd -m -u 1000 mediscan

# Install Python packages
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy app
COPY --chown=mediscan:mediscan . .

USER mediscan

EXPOSE 7860

CMD ["uvicorn", "mediscan.api:app", "--host", "0.0.0.0", "--port", "7860"]