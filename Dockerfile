# Use uma imagem base do Python
FROM python:3.9-slim

# Defina o diretório de trabalho na imagem
WORKDIR /app

# Copie o arquivo de requisitos para a imagem
COPY requirements.txt .

COPY setup.py .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação para a imagem
COPY ./app ./app

RUN pip install --no-cache-dir .

# Exponha a porta padrão do Streamlit
EXPOSE 8080

# Comando para executar a aplicação Streamlit
CMD ["streamlit", "run", "./app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]