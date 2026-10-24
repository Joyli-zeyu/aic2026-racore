FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py segmenter.py reportgen.py .
COPY static ./static

RUN curl -L --retry 3 -o brats20.zip "https://hf-mirror.com/datasets/Alanna-618/brats20-data/resolve/main/brats20.zip" \
    && python -c "import zipfile; zipfile.ZipFile('brats20.zip').testzip()" \
    && rm -rf /root/.cache

ENV BRATS_ZIP=/app/brats20.zip
ENV MPLCONFIGDIR=/tmp/mpl

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]