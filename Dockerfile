FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install (keeps consistent environment)
COPY requirements-prod.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy models and app
COPY models/ /app/models/
COPY app/ /app/app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" ]
