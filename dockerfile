FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

COPY service/ /app/service/
COPY fashion_core/ /app/fashion_core/
COPY common/ /app/common/

# data 안에 있는 거는 볼륨
RUN mkdir -p /app/data /app/config

EXPOSE 3000

CMD ["bentoml", "serve", "service.service:FashionSearchService", "--host", "0.0.0.0", "--port", "3000"]
