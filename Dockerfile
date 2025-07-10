# Use a specific Python version as a base image
# Ensure this version (e.g., 3.13-slim-buster) is available on Docker Hub.
FROM python:3.13.5-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (including FastAPI, uvicorn, gunicorn)
# --no-cache-dir saves space by not caching pip packages
RUN pip install --no-cache-dir pandas nltk scikit-learn transformers torch fastapi uvicorn gunicorn

# Copy your entire application code
# The '.' copies everything from your project root into /app in the container
# This means your src/, data/, trained_models/ etc. will be inside /app
COPY . /app

# Expose the port your web application will listen on
EXPOSE 8000

# Define the command to run your application when the container starts
# This will run your FastAPI app using Gunicorn (for production-readiness)
# Assuming your main FastAPI app object is named 'app' in 'src/app/main.py'
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.app.main:app", "--bind", "0.0.0.0:8000"]