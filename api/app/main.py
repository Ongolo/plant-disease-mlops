from fastapi import FastAPI

app = FastAPI(title="Plant Disease API")


@app.get("/health")
def health():
    return {"status": "ok"}