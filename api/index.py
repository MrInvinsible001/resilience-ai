from fastapi import FastAPI

app = FastAPI(title="ResilienceAI API", version="1.0.0")


@app.get("/")
def root():
    return {
        "name": "ResilienceAI",
        "status": "running",
        "message": "Supply chain resilience API"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
