import pandas as pd
import sanalytics.algorithms.utils as sau
from sklearn.model_selection import train_test_split

## Read data
r10 = pd.read_parquet("datasets/model_selection_CV/X_train_r10.parquet")
r100 = pd.read_parquet("datasets/model_selection_CV/X_train_r100.parquet")

## Create stratification
r10["stratify"] = r10.source + "_" + r10.label
r100["stratify"] = r100.source + "_" + r100.label

## Split
train_r10, test_r10 = train_test_split(r10, train_size = 0.9, random_state=42, stratify=r10.stratify)
train_r100, test_r100 = train_test_split(r100, train_size = 0.9, random_state=42, stratify=r100.stratify)

## Save
train_r10.to_parquet("datasets/model_selection_CV/X10_train.parquet", compression=None, index=False)
test_r10.to_parquet("datasets/model_selection_CV/X10_test.parquet", compression=None, index=False)
train_r100.to_parquet("datasets/model_selection_CV/X100_train.parquet", compression=None, index=False)
test_r100 .to_parquet("datasets/model_selection_CV/X100_test.parquet", compression=None, index=False)