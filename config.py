from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "online_retail.csv"  # change to your file
MODEL_DIR = BASE_DIR / "models"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "recommendation-system"

APRIORI_MIN_SUPPORT = 0.01
APRIORI_METRIC = "lift"
APRIORI_MIN_THRESHOLD = 1.2

SVD_N_FACTORS = 50
