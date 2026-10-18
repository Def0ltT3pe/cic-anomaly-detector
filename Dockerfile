FROM python:3.12-slim

# Предсказуемый вывод и никакой телеметрии Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Зависимости отдельным слоем — пересобираются только при изменении requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Код, модели и normal_stats.json
COPY . .

# Запуск не от root; audit/ должен быть записываемым
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/audit \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

# Имя файла — App.py (регистр важен на Linux)
CMD ["streamlit", "run", "App.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]
