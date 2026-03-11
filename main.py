from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import numpy as np

app = FastAPI(title="Forecast API - Substituto TimeGPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SeriesData(BaseModel):
    y: List[float]
    sizes: List[int]

class ForecastRequest(BaseModel):
    series: SeriesData
    freq: Optional[str] = "MS"
    h: Optional[int] = 1
    model: Optional[str] = "timegpt-1"

class ForecastResponse(BaseModel):
    mean: List[float]

def holt_winters_forecast(values: List[float], h: int, season_length: int = 12) -> List[float]:
    """Holt-Winters exponential smoothing com sazonalidade aditiva."""
    y = np.array(values, dtype=float)
    n = len(y)

    y = np.where(y <= 0, 0.01, y)

    alpha = 0.3
    beta  = 0.1
    gamma = 0.3

    if n < season_length * 2:
        season_length = max(2, n // 2)

    level = np.mean(y[:season_length])

    if n >= season_length * 2:
        trend = (np.mean(y[season_length:season_length*2]) - np.mean(y[:season_length])) / season_length
    else:
        trend = 0.0

    season_avgs = [np.mean(y[i::season_length]) for i in range(season_length)]
    overall_avg = np.mean(y)
    if overall_avg == 0:
        overall_avg = 0.01
    seasonals = [s / overall_avg for s in season_avgs]

    levels   = [level]
    trends   = [trend]
    seasonal = list(seasonals) + [0.0] * n

    for t in range(n):
        s_idx = t % season_length
        old_level = levels[-1]
        old_trend = trends[-1]
        s = seasonal[s_idx] if s_idx < len(seasonals) else 1.0

        new_level = alpha * (y[t] / (s + 1e-10)) + (1 - alpha) * (old_level + old_trend)
        new_trend = beta  * (new_level - old_level) + (1 - beta) * old_trend
        new_s     = gamma * (y[t] / (new_level + 1e-10)) + (1 - gamma) * s

        levels.append(new_level)
        trends.append(new_trend)
        seasonal[s_idx] = new_s

    forecasts = []
    for i in range(1, h + 1):
        s_idx = (n + i - 1) % season_length
        s = seasonal[s_idx]
        forecast = (levels[-1] + i * trends[-1]) * s
        forecasts.append(max(0.0, round(float(forecast), 4)))

    return forecasts

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Forecast API rodando",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/v2/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    y_all = req.series.y
    sizes = req.series.sizes
    h = req.h

    if sum(sizes) != len(y_all):
        from fastapi import HTTPException
        raise HTTPException(400, f"Soma de sizes ({sum(sizes)}) != tamanho de y ({len(y_all)})")

    num_series = len(sizes)
    all_forecasts = []
    cursor = 0

    for size in sizes:
        series_values = y_all[cursor: cursor + size]
        cursor += size
        preds = holt_winters_forecast(series_values, h)
        all_forecasts.append(preds)

    result = []
    for step in range(h):
        for i in range(num_series):
            result.append(all_forecasts[i][step])

    return ForecastResponse(mean=result)
