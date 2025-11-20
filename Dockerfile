# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: Dockerfile
# Descripción: Containerización completa con generación automática de datos
# ============================================================================

FROM python:3.10-slim

# Metadatos
LABEL maintainer="ARIMA-RL Project"
LABEL description="Agente RL-ARIMA para optimización de hiperparámetros"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app \
    PORT=8501

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR $APP_HOME

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear directorios necesarios ANTES de usar data/
RUN mkdir -p data models assets/logs

# Copiar código fuente y scripts
COPY src/ ./src/
COPY data/download_data.py ./data/download_data.py
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY assets/ ./assets/

RUN chmod +x scripts/*.sh data/download_data.py

# ============================================================================
# GENERACIÓN AUTOMÁTICA DE DATOS
# ============================================================================
# Este paso descarga/genera los 60 meses de datos de consumo eléctrico alemán
# y los divide automáticamente en conjuntos train/val/test

RUN echo "========================================" && \
    echo "📥 Generando datos automáticamente..." && \
    echo "========================================" && \
    python data/download_data.py && \
    echo "" && \
    echo "✅ Datos generados exitosamente" && \
    echo "   - 60 meses de consumo eléctrico alemán" && \
    echo "   - División: 48 train + 6 val + 6 test" && \
    echo "" && \
    ls -lh data/

# ============================================================================
# ENTRENAMIENTO DEL AGENTE RL (OPCIONAL - COMENTADO POR DEFECTO)
# ============================================================================
# El entrenamiento del agente RL tarda ~30-60 minutos y aumenta
# significativamente el tiempo de build. Se recomienda entrenar
# después de iniciar el contenedor usando:
#   docker exec -it arima-rl-container python -m src.rl_agent --train --timesteps 50000
#
# Para habilitar el entrenamiento automático durante el build, descomente las siguientes líneas:
#
# RUN echo "========================================" && \
#     echo "🎓 Entrenando agente RL..." && \
#     echo "========================================" && \
#     mkdir -p models && \
#     python -m src.rl_agent --train --timesteps 50000 --output-dir models && \
#     echo "" && \
#     echo "✅ Agente RL entrenado exitosamente" && \
#     ls -lh models/


# Exponer puerto de Streamlit
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

# Comando por defecto: ejecutar aplicación Streamlit
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
