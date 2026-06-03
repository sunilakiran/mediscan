FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 mediscan

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=mediscan:mediscan . .

# Download model at build time
RUN python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='sunilakiran56/mediscan-model', filename='mediscan_model.pt', repo_type='model')
shutil.copy(path, 'mediscan_model.pt')
print('Model downloaded!')
" || echo "Model download skipped"

USER mediscan

EXPOSE 7860

CMD ["uvicorn", "mediscan.api:app", "--host", "0.0.0.0", "--port", "7860"]