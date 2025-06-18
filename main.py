from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()
print("✅ Spark version:", spark.version)
spark.stop()

import os
for k, v in os.environ.items():
    if "60s" in v:
        print(f"{k}={v}")
