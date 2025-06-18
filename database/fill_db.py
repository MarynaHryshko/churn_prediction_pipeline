from dotenv import load_dotenv

load_dotenv()

import os
import logging
import pandas as pd
import joblib
import psycopg2
import pyarrow.parquet as pq
import pyarrow.fs as fs

from paths_config import (
    MODEL_FILE,
    FEATURES_FILE,
    DB_LOG_FILE,
    LOG_DIR,
    DATABASE_DIR
)



from db_config import DB_CONFIG

# --- Setup logging ---
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=DB_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Create table if not exists ---
try:
    logging.info("Connecting to PostgreSQL RDS...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    logging.info("Creating 'churn_predictions' table if it does not exist...")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS churn_predictions (
        id SERIAL PRIMARY KEY,
        customer_id TEXT,
        tenure INTEGER,
        monthly_charges FLOAT,
        gender TEXT,
        churn_predicted INTEGER,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    logging.info("✅ Table ready.")
    cur.close()
    conn.close()
except Exception as e:
    logging.error(f"❌ Failed to create table: {e}")
    raise

# --- Load data from S3 ---
try:
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_PREFIX = "data/processed/"
    s3 = fs.S3FileSystem(region=os.getenv("AWS_REGION"))
    full_path = f"{S3_BUCKET}/{S3_PREFIX}"

    dataset = pq.ParquetDataset(full_path, filesystem=s3)
    table = dataset.read()
    df = table.to_pandas()

#    print(df)

    logging.info(f"✅ Loaded {len(df)} rows from S3")
except Exception as e:
    logging.error(f"❌ Failed to load data from S3: {e}")
    raise

# --- Preprocessing ---
try:
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    X = pd.get_dummies(df.drop(["customerID", "Churn"], axis=1), drop_first=True)
    # Reindex to match training features

    # Get current script directory
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # # Build correct path to features.pkl (go one level up and into model/)
    # features_path = os.path.normpath(os.path.join(script_dir, "..", "model", "features.pkl"))
    # # Load columns
    # expected_columns = joblib.load(features_path)

    expected_columns = joblib.load(FEATURES_FILE)

    X = X.reindex(columns=expected_columns, fill_value=0)
    y = df["Churn"]
except Exception as e:
    logging.error(f"❌ Preprocessing failed: {e}")
    raise

# --- Load trained model ---
try:
    # # Get the absolute path to the current script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(script_dir, "..", "model", "churn_model.pkl")
    #
    # # Normalize the path and load the model
    # model_path = os.path.normpath(model_path)
    # model = joblib.load(model_path)

    model = joblib.load(MODEL_FILE)

    #model = joblib.load("model/churn_model.pkl")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Could not load model: {e}")
    raise

# --- Make predictions ---
try:
    predictions = model.predict(X)
    df["churn_predicted"] = predictions
    logging.info("✅ Predictions completed.")
except Exception as e:
    logging.error(f"❌ Prediction failed: {e}")
    raise

# --- Insert predictions into DB ---
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    inserted = 0
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO churn_predictions (customer_id, tenure, monthly_charges, gender, churn_predicted)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            row["customerID"],
            row["tenure"],
            row["MonthlyCharges"],
            row["gender"],
            int(row["churn_predicted"])
        ))
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"✅ Inserted {inserted} records into churn_predictions table.")
except Exception as e:
    logging.error(f"❌ Failed to insert data into database: {e}")
    raise
