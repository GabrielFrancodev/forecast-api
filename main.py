"""
Serviço de Forecast Gratuito - Substituto do TimeGPT
Imita exatamente o formato da API api.nixtla.io/v2/forecast
Usa StatsForecast (open-source da Nixtla, mesma empresa do TimeGPT)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

app = FastAPI(
    title="Forecast API - Substituto TimeGPT",
    description="API gratuita de previsão de séries temporais compatível com formato TimeGPT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos de dados (idênticos ao TimeGPT) ---
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

# --- Mapeamento de frequência ---
FREQ_MAP = {
    "MS": "MS",   # Mensal início
    "M":  "MS",   # Mensal
    "D":  "D",    # Diário
    "W":  "W",    # Semanal
    "Q":  "QS",   # Trimestral
    "Y":  "YS",   # Anual
    "H":  "h",    # Horário
}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Forecast API rodando com StatsForecast"}

@app.post("/v2/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Endpoint compatível com TimeGPT.
    Recebe múltiplas séries concatenadas via 'sizes' e retorna previsões no mesmo formato.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
        import pandas as pd
    except ImportError:
        raise HTTPException(500, "Dependências não instaladas. Execute: pip install statsforecast pandas")

    y_all = req.series.y
    sizes = req.series.sizes
    h = req.h
    freq = FREQ_MAP.get(req.freq, req.freq)

    # Valida tamanhos
    if sum(sizes) != len(y_all):
        raise HTTPException(400, f"Soma de sizes ({sum(sizes)}) != tamanho de y ({len(y_all)})")

    # Monta DataFrame no formato StatsForecast
    rows = []
    cursor = 0

    for i, size in enumerate(sizes):
        series_values = y_all[cursor: cursor + size]
        cursor += size

        # Gera datas retroativas a partir de hoje
        end_date = pd.Timestamp.now().normalize()
        if freq in ["MS", "M"]:
            end_date = end_date.replace(day=1)
            dates = pd.date_range(end=end_date, periods=size, freq="MS")
        else:
            dates = pd.date_range(end=end_date, periods=size, freq=freq)

        for date, val in zip(dates, series_values):
            rows.append({
                "unique_id": f"series_{i}",
                "ds": date,
                "y": float(val)
            })

    df = pd.DataFrame(rows)

    # Modelos (fallback automático: AutoARIMA → AutoETS → média)
    models = [
        AutoARIMA(season_length=12 if freq in ["MS", "M"] else 7),
        AutoETS(season_length=12 if freq in ["MS", "M"] else 7),
    ]

    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)

    try:
        forecast_df = sf.forecast(df=df, h=h)
    except Exception as e:
        # Fallback: retorna média histórica se o modelo falhar
        means = []
        cursor = 0
        for size in sizes:
            vals = [v for v in y_all[cursor:cursor+size] if v > 0]
            avg = float(np.mean(vals)) if vals else 0.0
            means.extend([avg] * h)
            cursor += size
        return ForecastResponse(mean=means)

    # Monta resposta no formato TimeGPT: [serie0_h1, serie1_h1, ..., serie0_h2, ...]
    num_series = len(sizes)
    result = []

    for step in range(h):
        for i in range(num_series):
            uid = f"series_{i}"
            row = forecast_df[forecast_df["unique_id"] == uid].iloc[step]
            # Pega AutoARIMA se disponível, senão AutoETS
            if "AutoARIMA" in row.index:
                val = float(row["AutoARIMA"])
            elif "AutoETS" in row.index:
                val = float(row["AutoETS"])
            else:
                val = 0.0
            result.append(max(0.0, round(val, 4)))  # Sem negativos

    return ForecastResponse(mean=result)


@app.post("/v2/forecast/simple")
def forecast_simple(req: ForecastRequest):
    """Alias para compatibilidade"""
    return forecast(req)
