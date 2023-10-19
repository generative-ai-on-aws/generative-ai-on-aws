FROM python:3.9-slim

WORKDIR /app

COPY server.py /app
COPY requirements.txt /app

VOLUME /app/.env

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD uvicorn server:app --host 0.0.0.0
