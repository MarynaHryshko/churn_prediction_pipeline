import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main folders
DATABASE_DIR = os.path.join(BASE_DIR, "database")
API_DIR = os.path.join(BASE_DIR, "api")
DATA_DIR = os.path.join(BASE_DIR, "data")
ETL_DIR = os.path.join(BASE_DIR, "spark_etl")
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# S3 keys
S3_PREFIX = "data/processed/"
S3_INPUT_KEY = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Model artifacts
MODEL_FILE = os.path.join(MODEL_DIR, "churn_model.pkl")
FEATURES_FILE = os.path.join(MODEL_DIR, "features.pkl")

# Log file
ETL_LOG_FILE = os.path.join(LOG_DIR, "churn_etl.log")
DB_LOG_FILE = os.path.join(LOG_DIR, "fill_db.log")

# Example: DB config file path (if needed)
DB_CONFIG_FILE = os.path.join(DATABASE_DIR, "db_config.py")



#S3_OUTPUT_KEY = "data/processed/"

# Optional: for local temporary work, if you want to structure it
#TMP_DIR = os.path.join(BASE_DIR, "tmp")
LOCAL_INPUT_CSV = "input.csv"
#LOCAL_PARQUET_OUTPUT = os.path.join(TMP_DIR, "output_parquet")
LOCAL_PARQUET_OUTPUT = "output_parquet"
