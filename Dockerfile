# Usamos la imagen oficial de PyTorch
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Configurar directorio
WORKDIR /app

# 3. Clonar el repositorio
RUN git clone https://github.com/arqdariogomez/difflocks.git .

# 4. Instalar Dependencias de Python
RUN pip install natten==0.17.1+torch240cu121 \
    -f https://shi-labs.com/natten/wheels/ \
    --trusted-host shi-labs.com

RUN pip install \
    gradio==3.50.2 \
    opencv-python-headless \
    mediapipe==0.10.14 \
    packaging \
    ninja \
    plotly \
    numpy \
    jsonmerge \
    clean-fid \
    torchdiffeq \
    torchsde \
    einops \
    transformers \
    accelerate \
    scikit-image \
    trimesh \
    dctorch \
    libigl

# 5. Descargar Assets de Mediapipe
RUN mkdir -p inference/assets && \
    wget -q -O inference/assets/face_landmarker_v2_with_blendshapes.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Comando final
CMD ["python", "app.py"]