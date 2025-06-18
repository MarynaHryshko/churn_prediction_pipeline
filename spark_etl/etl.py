from dotenv import load_dotenv

load_dotenv()
import os
import boto3
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim

from paths_config import (
    S3_PREFIX,
    S3_INPUT_KEY,
    LOCAL_INPUT_CSV,
    LOCAL_PARQUET_OUTPUT
)

# Load env vars
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# S3 paths
S3_OUTPUT_KEY = S3_PREFIX

# Create simple Spark session (no S3 dependencies needed)
spark = SparkSession.builder \
    .appName("ChurnETL") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Set log level to reduce noise
spark.sparkContext.setLogLevel("WARN")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

try:
    print("📥 Downloading CSV from S3...")

    # Create temporary directory for local processing
    with tempfile.TemporaryDirectory() as temp_dir:
        #local_input_path = os.path.join(temp_dir, "input.csv")
        #local_output_path = os.path.join(temp_dir, "output_parquet")
        local_input_path = os.path.join(temp_dir, LOCAL_INPUT_CSV)
        local_output_path = os.path.join(temp_dir, LOCAL_PARQUET_OUTPUT)

        # Download file from S3
        s3_client.download_file(S3_BUCKET, S3_INPUT_KEY, local_input_path)
        print(f"✅ Downloaded to {local_input_path}")

        print("📖 Reading CSV with Spark...")
        # Read CSV from local file
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(local_input_path)

        print(f"📊 Dataset shape: {df.count()} rows, {len(df.columns)} columns")

        # Show basic info about the dataset
        print("📋 Dataset schema:")
        df.printSchema()

        # Show sample data
        print("📝 Sample data:")
        df.show(5)

        # Basic cleanup: drop rows with missing customerID
        print("🧹 Cleaning data...")
        df_clean = df.dropna(subset=["customerID"])

        # Convert TotalCharges to numeric (has some non-numeric values)
        print("🔄 Processing TotalCharges column...")

        # First, replace empty strings and spaces with null
        df_clean = df_clean.withColumn("TotalCharges",
                                       col("TotalCharges").cast("string"))

        # Replace empty strings and whitespace with null, then cast to double
        df_clean = df_clean.withColumn("TotalCharges",
                                       when(trim(col("TotalCharges")) == "", None)
                                       .otherwise(col("TotalCharges")))

        df_clean = df_clean.withColumn("TotalCharges",
                                       col("TotalCharges").cast("double"))

        # Drop rows where TotalCharges couldn't be converted (will be null)
        df_clean = df_clean.dropna(subset=["TotalCharges"])

        print(f"🔄 After cleaning: {df_clean.count()} rows")

        # Show cleaned data sample
        print("📝 Cleaned data sample:")
        df_clean.select("customerID", "TotalCharges", "Churn").show(5)

        # Write to local Parquet first
        print("💾 Writing to local Parquet...")
        df_clean.write.mode("overwrite").parquet(local_output_path)

        # Upload Parquet files to S3
        print("📤 Uploading Parquet files to S3...")

        # Walk through the parquet directory and upload all files
        parquet_path = Path(local_output_path)
        uploaded_files = []

        for file_path in parquet_path.rglob("*"):
            if file_path.is_file():
                # Create S3 key maintaining directory structure
                relative_path = file_path.relative_to(parquet_path)
                s3_key = f"{S3_OUTPUT_KEY}{relative_path}"

                # Upload file
                s3_client.upload_file(str(file_path), S3_BUCKET, s3_key)
                uploaded_files.append(s3_key)
                print(f"  ✅ Uploaded: s3://{S3_BUCKET}/{s3_key}")

        print(f"📊 Summary:")
        print(f"  - Input records: {df.count()}")
        print(f"  - Cleaned records: {df_clean.count()}")
        print(f"  - Files uploaded: {len(uploaded_files)}")
        print(f"  - S3 output location: s3://{S3_BUCKET}/{S3_OUTPUT_KEY}")

        print("✅ ETL complete. Cleaned data saved to S3 as Parquet.")

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    import traceback

    traceback.print_exc()
    raise
finally:
    spark.stop()
    print("🔌 Spark session stopped.")