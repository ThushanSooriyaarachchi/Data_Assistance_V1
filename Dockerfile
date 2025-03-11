FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Create data directory
RUN mkdir -p data

# Command to run the application
CMD ["python", "main.py"]

EXPOSE 8000