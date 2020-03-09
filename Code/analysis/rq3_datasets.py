import pandas as pd
from sklearn.model_selection import train_test_split

# Load the datasets
train = pd.read_parquet("../data/train.parquet")
train.drop(columns=["fold"])
test = pd.read_parquet("../data/test.parquet")

# Merge Data
df = pd.concat([train, test], sort=True)

# Extract unlabelled set (as this does not change)
unlabelled = df[df["label"] == "unlabelled"]

# Split positive data into sets
sec = df[df["label"] == "security"]
set1 = df[df["stratify"] == "set1_security"]
set2 = df[df["stratify"] == "set2_security"]
sse = df[df["stratify"] == "sse_security"]

# Save single set train/test
# Set 1 only (keyword matching)
set1 = df[(df["stratify"] == "set1_security") | (df["label"] == "unlabelled")]
train, test = train_test_split(set1, train_size = 0.9, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
test.to_parquet("../rq3_data/set1_test.parquet", compression=None, index=False)
train.to_parquet("../rq3_data/set1_train.parquet", compression=None, index=False)
# Set 2 only (tag matching)
set2 = df[(df["stratify"] == "set2_security") | (df["label"] == "unlabelled")]
train, test = train_test_split(set2, train_size = 0.9, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
test.to_parquet("../rq3_data/set2_test.parquet", compression=None, index=False)
train.to_parquet("../rq3_data/set2_train.parquet", compression=None, index=False)
# SO Data (set1 + set2)
so = df[df["source"] != "sse"]
train, test = train_test_split(so, train_size = 0.9, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
test.to_parquet("../rq3_data/so_test.parquet", compression=None, index=False)
train.to_parquet("../rq3_data/so_train.parquet", compression=None, index=False)
# SSE only
sse = df[(df["stratify"] == "sse_security") | (df["label"] == "unlabelled")]
train, test = train_test_split(sse, train_size = 0.9, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
test.to_parquet("../rq3_data/sse_test.parquet", compression=None, index=False)
train.to_parquet("../rq3_data/sse_train.parquet", compression=None, index=False)

# # Decrease security posts by 10% each time
fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in fracs:
	# Get a sample of security posts
	sample = sec.sample(frac=i, random_state=42)
	# Merge sample with unlabelled
	sample = pd.concat([sample, unlabelled], sort=True)
	# Train/Test split
	train, test = train_test_split(sample, train_size = 0.9, random_state=42)
	train = train.reset_index(drop=True)
	test = test.reset_index(drop=True)
	test.to_parquet("../rq3_data/sec"+str(i)+"_test.parquet", compression=None, index=False)
	train.to_parquet("../rq3_data/sec"+str(i)+"_train.parquet", compression=None, index=False)