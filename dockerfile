# Use a imagem base do Bitnami para TensorFlow
FROM bitnami/tensorflow:latest

# As imagens da Bitnami utilizam o usuário 1001. Para garantir a permissão correta
# dos arquivos, definimos o usuário explicitamente.
USER 1001

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de requisitos e instala as bibliotecas Python.
# O --chown garante que o arquivo copiado pertença ao usuário não-root.
COPY --chown=1001:1001 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie os arquivos do seu projeto para o container
COPY . /app

# --- VERSÃO CORRETA E SIMPLIFICADA ---
# Apenas cria o diretório. Como o build já roda como não-root, ele será o dono.
RUN mkdir -p /app/logs

# Exponha a porta 3001 para acesso externo
EXPOSE 3001

# --- MODIFICAÇÃO IMPORTANTE ---
# Define que o executável padrão do contêiner é o python
ENTRYPOINT ["python"]

# Define que o argumento padrão para o executável é o nosso script
CMD ["app.py"]