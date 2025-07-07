# Use a imagem base do Bitnami para TensorFlow
FROM bitnami/tensorflow:latest

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie os arquivos do seu projeto para o container
COPY . /app

# Instale as dependências do seu projeto
# IMPORTANTE: Certifique-se de que o seu arquivo 'requirements.txt' está na raiz do projeto.
RUN pip install --no-cache-dir -r requirements.txt

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