# DallasAI/src/app/main.py

from fastapi import FastAPI

# This is the FastAPI application instance that Gunicorn is looking for
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello from DallasAI FastAPI!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}