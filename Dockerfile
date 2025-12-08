# Usamos la imagen oficial de PyTorch
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git wget xz-utils \
    libgl1-mesa-glx libglib2.0-0 \
    libx11-6 libxrender1 libxxf86vm1 libxi6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Clonar Repositorio (PRIMERO, mientras la carpeta está vacía)
RUN git clone https://github.com/arqdariogomez/difflocks.git .

# 3. Instalar Blender 4.1 (DESPUÉS)
RUN mkdir -p /app/blender && \
    wget -q https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz && \
    tar -xf blender-4.1.1-linux-x64.tar.xz -C /app/blender --strip-components=1 && \
    rm blender-4.1.1-linux-x64.tar.xz

ENV PATH="/app/blender:$PATH"

# 4. Copiar Archivos Locales (Solo código, NO checkpoints)
COPY ./app.py /app/app.py
COPY ./converter.py /app/converter.py

# 5. Instalar Dependencias Python
RUN pip install natten==0.17.1+torch240cu121 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com
RUN pip install gradio==3.50.2 opencv-python-headless mediapipe==0.10.14 packaging ninja plotly numpy jsonmerge clean-fid torchdiffeq torchsde einops transformers accelerate scikit-image trimesh dctorch libigl

# 6. Assets
RUN mkdir -p inference/assets && \
    wget -q -O inference/assets/face_landmarker_v2_with_blendshapes.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

CMD ["python", "app.py"]