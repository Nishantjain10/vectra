"""
Vector Database Backend - FastAPI Application Entry Point
"""
from fastapi import FastAPI

app = FastAPI(
    title="Vector Database Backend",
    description="A REST API for indexing and querying vector embeddings",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Vector Database Backend is running"}

