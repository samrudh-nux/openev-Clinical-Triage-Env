FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir setuptools wheel

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "openenv-core>=0.2.0" || true

COPY . .

RUN pip install --no-cache-dir -e . || true

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
