FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl git

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start Ollama + API
CMD ["sh", "start.sh"]
