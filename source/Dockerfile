FROM python:3.11-slim

# 1) System deps for Chrome/Chromium in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libgbm1 \
    libxshmfence1 \
    wget curl unzip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit default command (adjust app path if different)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]