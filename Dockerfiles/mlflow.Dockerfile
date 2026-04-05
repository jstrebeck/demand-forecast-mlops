FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.19.0 psycopg2-binary
