# Chua duoc build-test trong moi truong phat trien (khong co Docker san) - vui long
# chay `docker build` de xac nhan truoc khi dung trong production.
FROM python:3.11-slim

WORKDIR /app

# libGL/libglib can cho opencv-python doc/ghi anh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY api.py .
COPY configs/ configs/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
