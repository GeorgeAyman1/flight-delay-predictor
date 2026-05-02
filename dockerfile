FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/     ./backend/
COPY src/         ./src/
COPY models/      ./models/
COPY data/        ./data/
COPY frontend/    ./frontend/
COPY start.py     ./start.py

ENV PYTHONPATH=/app

CMD ["python", "start.py"]
