FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/vector_store data/knowledge_graph data/chat_history data/cache data/logs

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py"]