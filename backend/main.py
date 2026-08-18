import os, json
import numpy as np
import torch, torch.nn as nn
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = FastAPI(title="Nepal Air Quality Prediction", version="3.0.0")


class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_out))

    def forward(self, x):
        return self.net(x)


class DNN(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.net(x)


def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)

    n_out = len(meta["classes"])
    mlp = MLP(3, n_out)
    mlp.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mlp_model.pt"), map_location="cpu"))
    mlp.eval()

    dnn = DNN(3, n_out, dropout=meta.get("dnn_dropout", 0.1))
    dnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dnn_model.pt"), map_location="cpu"))
    dnn.eval()

    with open(os.path.join(MODEL_DIR, "district_stats.json")) as f:
        district_stats = json.load(f)

    return scaler, le, mlp, dnn, meta, district_stats


scaler, le, mlp, dnn, META, DISTRICT_STATS = load_artifacts()

CATEGORY_ORDER = ["Good", "Moderate", "Unhealthy_SG", "Unhealthy", "Very_Unhealthy", "Hazardous"]
CATEGORY_RANGES = {
    "Good": "0 - 50", "Moderate": "51 - 100", "Unhealthy_SG": "101 - 150",
    "Unhealthy": "151 - 200", "Very_Unhealthy": "201 - 300", "Hazardous": "301 - 500",
}
TRAINED_CATEGORIES = set(le.classes_)


def run_models(pm25, pm10, no2):
    x = np.array([[pm25, pm10, no2]], dtype=np.float32)
    x_s = scaler.transform(x).astype(np.float32)
    x_t = torch.tensor(x_s)
    raw = {"PM2_5_ug_m3": pm25, "PM10_ug_m3": pm10, "NO2_ppb": no2}

    results = {}
    with torch.no_grad():
        for name, model in (("MLP", mlp), ("DNN", dnn)):
            probs = torch.softmax(model(x_t), dim=1).numpy()[0]
            pred_idx = int(probs.argmax())
            cat = le.inverse_transform([pred_idx])[0]
            raw_conf = {le.inverse_transform([i])[0]: round(float(p), 4) for i, p in enumerate(probs)}
            confidences = {c: raw_conf.get(c, 0.0) for c in CATEGORY_ORDER}
            top3 = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)[:3]
            results[name] = {
                "category": str(cat),
                "aqi_range": CATEGORY_RANGES[str(cat)],
                "confidence": round(float(probs[pred_idx]), 4),
                "confidences": confidences,
                "trained_categories": sorted(TRAINED_CATEGORIES),
                "top3": [{"category": c, "prob": p} for c, p in top3],
            }
    return {"features": raw, "feature_cols": META["feature_cols"], "models": results}


class PredictRequest(BaseModel):
    PM2_5_ug_m3: float
    PM10_ug_m3: float
    NO2_ppb: float


@app.get("/api/meta")
def get_meta():
    return {
        "feature_cols": META["feature_cols"],
        "category_order": CATEGORY_ORDER,
        "category_ranges": CATEGORY_RANGES,
        "trained_categories": sorted(TRAINED_CATEGORIES),
        "mlp_macro_f1": META.get("mlp_macro_f1"),
        "dnn_macro_f1": META.get("dnn_macro_f1"),
        "target": META.get("target"),
    }


@app.get("/api/districts")
def get_districts():
    return DISTRICT_STATS


@app.post("/api/predict")
def predict(payload: PredictRequest):
    try:
        return run_models(payload.PM2_5_ug_m3, payload.PM10_ug_m3, payload.NO2_ppb)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
