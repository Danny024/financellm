FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY notebook.ipynb .
COPY data/ ./data/
COPY modules/ ./modules/
COPY vector_db/ ./vector_db/

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose ports for Streamlit and Jupyter
EXPOSE 8502 8888

# Environment variables
ENV OLLAMA_HOST=http://localhost:11434
ENV PYTHONUNBUFFERED=1

# Use entrypoint script to start Ollama and application
ENTRYPOINT ["./entrypoint.sh"]
CMD ["streamlit"]