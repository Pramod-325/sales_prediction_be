import os
import json
import io
import lightgbm as lgb
import tempfile
from abc import ABC, abstractmethod
from prophet.serialize import model_from_json, model_to_json
from prophet import Prophet

# --- Abstract Interface ---
class StorageBackend(ABC):
    @abstractmethod
    def load_manifest(self): pass
    @abstractmethod
    def load_global_lgbm(self, manifest): pass
    @abstractmethod
    def load_prophet_model(self, product_id: str): pass
    @abstractmethod
    def load_encoder(self, manifest): pass
    @abstractmethod
    def save_prophet_model(self, model, product_id: str): pass

# --- Option A: Local Disk (Windows/Mac) ---
class LocalStorage(StorageBackend):
    def __init__(self, base_dir="./ml_bin"):
        self.base_dir = base_dir
        self.manifest_path = os.path.join(base_dir, "global_lgbm", "model_manifest.json")
        print(f"📂 Storage initialized: Local Disk ({self.base_dir})")

    def load_manifest(self):
        """Step 1: Read the Source of Truth"""
        if not os.path.exists(self.manifest_path):
            # Fallback for flat directory structures during dev
            alt_path = os.path.join(self.base_dir, "model_manifest.json")
            if os.path.exists(alt_path):
                self.manifest_path = alt_path
            else:
                raise FileNotFoundError(f"CRITICAL: Manifest missing at {self.manifest_path}")
        
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def load_global_lgbm(self, manifest):
        """Step 2: Load Model & Verify Contract"""
        filename = manifest['active_global_model']
        # Try specific path first, then flat path
        path = os.path.join(self.base_dir, "global_lgbm", filename)
        if not os.path.exists(path):
            path = os.path.join(self.base_dir, filename)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"LGBM model file {filename} missing!")

        print(f"✅ Loading LGBM from: {path}")
        model = lgb.Booster(model_file=path)

        # --- THE CONTRACT CHECK ---
        expected_feats = manifest['global_model_config']['expected_features']
        actual_feats = model.num_feature()
        
        if actual_feats != expected_feats:
            raise ValueError(
                f"⛔ Model Contract Violation! Manifest expects {expected_feats} cols, "
                f"but loaded model has {actual_feats}. Update manifest or retrain model."
            )
        return model

    def load_encoder(self, manifest):
        """Load JSON Encoder Map (Dict lookup, not Class)"""
        filename = manifest['global_model_config']['encoder_file']
        path = os.path.join(self.base_dir, "global_lgbm", filename)
        if not os.path.exists(path):
             path = os.path.join(self.base_dir, filename)

        with open(path, 'r') as f:
            return json.load(f)

    def load_prophet_model(self, product_id: str):
        """Load Native JSON Prophet Model"""
        # Look in the folder defined in your manifest
        path = os.path.join(self.base_dir, "prophet_individual", f"{product_id}.json")
        
        if not os.path.exists(path):
            return None # Implies "New Model Needed"
            
        with open(path, 'r') as f:
            return model_from_json(f.read())

    def save_prophet_model(self, model, product_id: str):
        """Save Native JSON Prophet Model"""
        directory = os.path.join(self.base_dir, "prophet_individual")
        os.makedirs(directory, exist_ok=True)
        
        path = os.path.join(directory, f"{product_id}.json")
        with open(path, 'w') as f:
            f.write(model_to_json(model))
        print(f"✅ Saved Prophet model for {product_id}")

# --- Option B: Azure Blob (Production) ---
class AzureBlobStorage(StorageBackend):
    def __init__(self):
        try:
            from azure.storage.blob import BlobServiceClient
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            self.blob_service = BlobServiceClient.from_connection_string(conn_str)
            self.container = self.blob_service.get_container_client("models")
            print("☁️ Storage initialized: Azure Blob")
        except Exception as e:
            raise ImportError(f"Azure init failed: {e}")

    def load_manifest(self):
        # Always assume it's in the global_lgbm folder in cloud
        blob = self.container.get_blob_client("global_lgbm/model_manifest.json")
        stream = io.BytesIO()
        blob.download_blob().readinto(stream)
        stream.seek(0)
        return json.load(stream)

    def load_global_lgbm(self, manifest):
        filename = manifest['active_global_model']
        blob_path = f"global_lgbm/{filename}"
        
        # LGBM needs a physical file path (C++ binding requirement)
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, filename)
        
        blob = self.container.get_blob_client(blob_path)
        with open(local_path, "wb") as f:
            blob.download_blob().readinto(f)
            
        model = lgb.Booster(model_file=local_path)
        
        # Contract Check
        if model.num_feature() != manifest['global_model_config']['expected_features']:
             raise ValueError("Cloud Model Contract Violation!")
             
        return model

    def load_encoder(self, manifest):
        filename = manifest['global_model_config']['encoder_file']
        blob = self.container.get_blob_client(f"global_lgbm/{filename}")
        stream = io.BytesIO()
        blob.download_blob().readinto(stream)
        stream.seek(0)
        return json.load(stream)

    def load_prophet_model(self, product_id: str):
        try:
            blob = self.container.get_blob_client(f"prophet_individual/{product_id}.json")
            stream = io.BytesIO()
            blob.download_blob().readinto(stream)
            stream.seek(0)
            return model_from_json(stream.read().decode('utf-8'))
        except Exception:
            return None

    def save_prophet_model(self, model, product_id: str):
        data = model_to_json(model)
        blob = self.container.get_blob_client(f"prophet_individual/{product_id}.json")
        blob.upload_blob(data, overwrite=True)

# --- Factory ---
def get_storage_backend():
    if os.getenv("APP_ENV") == "production":
        print("Running in Prod Env")
        return AzureBlobStorage()
    print("Running in Dev Env")
    return LocalStorage()
