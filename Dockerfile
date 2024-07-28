# Usar uma imagem oficial do Python como imagem pai
FROM python:3.11.7

# Definir o diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos de requisitos primeiro, para aproveitar o cache de camadas do Docker
COPY requirements.txt ./

# Instalar as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto do código fonte do projeto para o diretório de trabalho
COPY . .

# Define PYTHONPATH para incluir as subpastas necessárias
ENV PYTHONPATH=/app:/app/src:/app/notebooks

# Comando para rodar a aplicação
CMD ["python", "./notebooks/model_training_multiclassification.py"]