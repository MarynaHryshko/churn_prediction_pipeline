from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

import pyarrow.parquet as pq
import pyarrow.fs as fs

from paths_config import (
    S3_PREFIX,
    MODEL_DIR,
    MODEL_FILE,
    FEATURES_FILE
)

# Define S3 path
S3_BUCKET = os.getenv("S3_BUCKET")
#S3_PREFIX = "data/processed/"  # folder path inside the bucket
AWS_REGION = os.getenv("AWS_REGION")
# Init S3 filesystem with correct region
s3 = fs.S3FileSystem(region=AWS_REGION)

# Combine bucket and prefix
full_path = f"{S3_BUCKET}/{S3_PREFIX}"  # NOT a full URI like s3://...

# Load Parquet dataset
dataset = pq.ParquetDataset(full_path, filesystem=s3)
table = dataset.read()
df = table.to_pandas()

# Prepare data
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(df.drop(["customerID", "Churn"], axis=1), drop_first=True)
y = df["Churn"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#joblib.dump(X_train.columns.tolist(), "features.pkl")
joblib.dump(X_train.columns.tolist(), FEATURES_FILE)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model locally
#os.makedirs("model", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_FILE)

print("✅ Model trained and saved locally.")
