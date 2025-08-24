# Use a imagem base do Bitnami para PyTorch
FROM bitnami/pytorch:latest

# 1. Mude temporariamente para o usuário root para poder instalar pacotes do sistema
USER root

# 2. Atualize a lista de pacotes e instale as dependências do OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Volte para o usuário não-root padrão da imagem Bitnami para seguir as boas práticas
USER 1001


# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de requisitos e instala as bibliotecas Python.
# O --chown garante que o arquivo copiado pertença ao usuário não-root.
COPY --chown=1001:1001 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie os arquivos do seu projeto para o container
COPY . /app

# Apenas cria o diretório. Como o build já roda como não-root, ele será o dono.
RUN mkdir -p /app/logs

# Exponha a porta 3001 para acesso externo
EXPOSE 3001

# Define que o executável padrão do contêiner é o python
ENTRYPOINT ["python"]

# Define que o argumento padrão para o executável é o nosso script
CMD ["app.py"]