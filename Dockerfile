# Use official Python image
FROM python:3.11

# Set the working directory to the location of the main app
WORKDIR /langchain

# Copy the entire project into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory (if not already present)
RUN mkdir -p logs

# Expose port 8000 for the outside container
EXPOSE 8000

# Define environment variable for uvicorn
ENV UVICORN_CMD="uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"

# Run the FastAPI app using uvicorn when the container launches
CMD ["sh", "-c", "$UVICORN_CMD"]