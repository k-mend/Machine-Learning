FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install packages one by one with retries for large packages
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic && \
    pip install --no-cache-dir pandas numpy scikit-learn && \
    pip install --default-timeout=1000 --no-cache-dir xgboost

COPY . .

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
