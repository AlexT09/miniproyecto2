FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias primero (aprovecha cache de Docker)
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
