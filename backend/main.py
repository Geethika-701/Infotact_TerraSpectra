from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_grid


app = FastAPI(
    title="TerraSpectra API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


@app.get("/")
def root():

    return {
        "project": "TerraSpectra",

        "status": "online",

        "model": "3D-CNN",

        "bands": 224,

        "classes": [
            "Healthy",
            "Chemically Stressed"
        ]
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/predictions")
def predictions():

    results = predict_grid()

    return {
        "count": len(results),

        "predictions": results
    }