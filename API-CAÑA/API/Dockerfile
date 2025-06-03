# Etapa de construcción
FROM python:3.9-slim as builder

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Crear y establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de requisitos
COPY requirements.txt .

# Instalar dependencias en un directorio virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Etapa final
FROM python:3.9-slim

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/model_epoch_18.pth \
    PATH="/opt/venv/bin:$PATH"

# Crear y establecer el directorio de trabajo
WORKDIR /app

# Instalar solo las dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el entorno virtual desde la etapa de construcción
COPY --from=builder /opt/venv /opt/venv

# Copiar el modelo pre-entrenado
COPY model_epoch_18.pth .

# Copiar el código de la aplicación
COPY API_CAÑA.py .

# Exponer el puerto
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "API_CAÑA.py"] 