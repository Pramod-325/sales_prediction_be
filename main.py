from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from prisma import Prisma
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware
from blob_storage import get_storage_backend

class SalesRecord(BaseModel):
    product_code: str
    ds: datetime
    y: float

class ForecastRecord(BaseModel):
    product_code: str
    forecast_date: datetime
    predicted_sales: float

class MetricsResponse(BaseModel):
    model_version: str
    wmape: Optional[float]
    accuracy: Optional[float]
    last_trained: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> 🟢 STARTUP: Initializing resources...")
    
    # 1. Initialize Storage
    try:
        app.state.storage = get_storage_backend()
    except Exception as e:
        print(f"   ❌ Storage Init Failed: {e}")
        app.state.storage = None

    # 2. Connect Database
    try:
        db = Prisma()
        await db.connect()
        app.state.db = db
        print("   ✅ Database connected")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")

    # 3. Load Models via Manifest Contract
    app.state.models = {}
    if app.state.storage:
        try:
            storage = app.state.storage
            
            # A. Load Manifest
            manifest = storage.load_manifest()
            app.state.manifest = manifest
            print(f"   📜 Manifest loaded (Version ID: {manifest.get('version_id', 'Unknown')})")

            # B. Load Global LGBM (Contract Checked inside storage)
            app.state.models['lgb'] = storage.load_global_lgbm(manifest)
            
            # C. Load Encoder Map (JSON Dict)
            app.state.models['encoder_map'] = storage.load_encoder(manifest)
            
            print("   ✅ ML Engine Ready (Pickle-Free Mode)")
            
        except Exception as e:
            print(f"   ⚠️ ML Engine Load Failed: {e}")
            print("      (Live inference endpoints will return 503)")
            app.state.models = None

    yield  # App runs here

    print(">>> 🔴 SHUTDOWN: Cleaning up...")
    if hasattr(app.state, 'db') and app.state.db.is_connected():
        await app.state.db.disconnect()

app = FastAPI(title="Sales Forecasting API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/sales/history/{product_id}", response_model=List[SalesRecord])
async def get_history(product_id: str):
    db = app.state.db
    records = await db.saleshistory.find_many(
        where={'product_code': product_id},
        order={'ds': 'asc'}
    )
    if not records:
        raise HTTPException(status_code=404, detail="Product history not found")
    return records

@app.get("/sales/forecast/{product_id}", response_model=List[ForecastRecord])
async def get_forecast(product_id: str):
    db = app.state.db
    records = await db.salesforecast.find_many(
        where={'product_code': product_id},
        order={'forecast_date': 'asc'}
    )
    if not records:
        raise HTTPException(status_code=404, detail="Forecasts not found")
    return records

@app.get("/metrics/model", response_model=MetricsResponse)
async def get_metrics():
    db = app.state.db
    metric = await db.modelmetric.find_first(order={'training_run_date': 'desc'})
    
    if not metric:
        return MetricsResponse(model_version="None", wmape=0, accuracy=0, last_trained=datetime.now())

    return MetricsResponse(
        model_version=metric.model_version,
        wmape=metric.wmape,
        accuracy=metric.accuracy,
        last_trained=metric.training_run_date
    )

@app.post("/sales/forecast/live/{product_id}")
async def generate_live_forecast(product_id: str, background_tasks: BackgroundTasks):
    """
    Safe Live Inference using Native Text/JSON models.
    """
    db = app.state.db
    storage = app.state.storage
    lgbm_model = app.state.models.get('lgb')
    encoder_map = app.state.models.get('encoder_map')
    manifest = app.state.manifest

    if not lgbm_model or not encoder_map:
        raise HTTPException(status_code=503, detail="ML Models are not loaded. Check server logs.")

    # 1. Fetch Latest History
    history_records = await db.saleshistory.find_many(
        where={'product_code': product_id},
        order={'ds': 'asc'}
    )
    
    if not history_records:
        raise HTTPException(status_code=404, detail="No sales history found.")

    df_history = pd.DataFrame([vars(r) for r in history_records])
    
    # Timezone clean up
    if pd.api.types.is_datetime64_any_dtype(df_history['ds']):
         df_history['ds'] = df_history['ds'].dt.tz_localize(None)
    else:
         df_history['ds'] = pd.to_datetime(df_history['ds']).dt.tz_localize(None)

    df_history = df_history.rename(columns={'ds': 'ds', 'y': 'y'})
    df_history['y_log'] = np.log1p(df_history['y'])

    # 2. Load or Create Prophet (JSON Mode)
    # Using 'storage.load_prophet_model' handles the JSON deserialization safely
    m = storage.load_prophet_model(product_id)
    
    seasonality = True if len(df_history) > 52 else False

    if m:
        try:
            m.fit(df_history) # Refit existing
        except Exception:
            # If old model state is corrupt or incompatible, reset
            m = Prophet(yearly_seasonality=seasonality)
            m.fit(df_history)
    else:
        m = Prophet(yearly_seasonality=seasonality)
        m.fit(df_history)

    # 3. Generate Features
    future = m.make_future_dataframe(periods=104, freq='W')
    forecast = m.predict(future)
    
    df_features = forecast[['ds', 'yhat']].rename(columns={'yhat': 'prophet_pred_log'})
    
    # --- Feature Engineering (MUST MATCH MANIFEST) ---
    df_features['month'] = df_features['ds'].dt.month
    df_features['week'] = df_features['ds'].dt.isocalendar().week.astype(int)
    df_features['year'] = df_features['ds'].dt.year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['week_sin'] = np.sin(2 * np.pi * df_features['week'] / 52)
    df_features['week_cos'] = np.cos(2 * np.pi * df_features['week'] / 52)

    # Encoding: Safe Dict Lookup
    # If key doesn't exist, return -1 (Unknown Product)
    encoded_val = encoder_map.get(product_id, -1)
    df_features['Product_Code_Encoded'] = int(encoded_val)

    # 4. Predict (LightGBM)
    # CRITICAL: Use the exact feature order from the manifest
    feature_order = manifest['global_model_config']['feature_names_ordered']
    
    # Ensure all columns exist
    for col in feature_order:
        if col not in df_features.columns:
            raise ValueError(f"Missing feature required by model: {col}")

    lgb_preds_log = lgbm_model.predict(df_features[feature_order])
    df_features['predicted_sales'] = np.expm1(lgb_preds_log).clip(min=0)

    # 5. Save & Return
    
    # Background: Save updated Prophet model as JSON
    background_tasks.add_task(storage.save_prophet_model, m, product_id)

    # Filter for future only
    last_history_date = df_history['ds'].max()
    df_final = df_features[df_features['ds'] > last_history_date]

    response_data = []
    db_batch = []
    
    for _, row in df_final.iterrows():
        response_data.append({
            'date': row['ds'].strftime('%Y-%m-%d'),
            'sales': round(float(row['predicted_sales']), 2)
        })
        db_batch.append({
            'product_code': product_id,
            'forecast_date': row['ds'],
            'predicted_sales': float(row['predicted_sales'])
        })

    # DB Transaction
    await db.salesforecast.delete_many(where={'product_code': product_id})
    if db_batch:
        await db.salesforecast.create_many(data=db_batch)

    return {
        "status": "success", 
        "message": f"Refreshed forecast for {product_id}",
        "forecast": response_data
    }
