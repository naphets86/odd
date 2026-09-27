FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tcpdump \
    libpcap-dev \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/     /app/
COPY static/  /app/static/
COPY templates/ /app/templates/

RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
