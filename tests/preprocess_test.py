import pandas as pd
df = pd.read_parquet("data/processed/train_preprocessed.parquet")
print(df.columns.tolist())
print(df.isnull().sum()[df.isnull().sum() > 0])