FROM python:3.10-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/

RUN pip install --upgrade pip

RUN pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    -r requirements/docker.txt

COPY . .

CMD ["python", "main.py"]