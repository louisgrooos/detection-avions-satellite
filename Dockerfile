# Image de démonstration (CPU) : interface Streamlit + modèles entraînés
FROM python:3.12-slim

WORKDIR /app

# Bibliothèques système nécessaires à OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch en version CPU (plus léger), puis le reste
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code et modèles entraînés
COPY app.py etape5_pipeline.py ./
COPY sauvegardes/best.pt sauvegardes/meilleur_modele.pth ./sauvegardes/

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
