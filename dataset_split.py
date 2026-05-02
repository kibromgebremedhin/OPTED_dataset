import pandas as pd
from sklearn.model_selection import train_test_split
def stratified_split(csv_path, output_path):
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)
    train_df["split"] = "train"; val_df["split"] = "val"; test_df["split"] = "test"
    pd.concat([train_df, val_df, test_df]).to_csv(output_path, index=False)
