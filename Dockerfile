# Dockerfile for deployment alternative
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt || pip install streamlit crewai langsmith langchain-openai pandas python-dotenv tqdm kagglehub plotly psycopg2-binary sqlalchemy deepeval cohere langchain-anthropic langchain-cohere openai

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV STREAMLIT_SERVER_PORT=5000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]