FROM python:3.10-slim as base

# Define as variáveis de ambiente
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PYTHONUNBUFFERED 1

# Define o diretório de trabalho
WORKDIR /app

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos necessários
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Add a 10-second delay before starting the application
ENTRYPOINT ["sh", "-c", "sleep 10 && python send_frames.py"]