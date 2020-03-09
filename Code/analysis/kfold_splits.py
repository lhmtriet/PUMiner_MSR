import os
import sanalytics.text_processing.utils as stu
import sanalytics.algorithms.utils as sau
from glob import glob
import dask.dataframe as dd
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import progressbar
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
from sklearn.model_selection import StratifiedKFold, train_test_split

## Number of folds
num_k = 10
nonsec_sec_ratio = 100

## Get all data: Positive labelled examples
set1, set2 = stu.data_prep("datasets/so_processed_new/*.parquet")
sse = dd.read_csv("datasets/sse_processed/security_new_*").fillna('').compute()
so = pd.concat(sau.p_map(lambda name : pd.read_parquet(name), glob("datasets/so_processed_new/*.parquet"), num_cpus=20)) # obtain from analysis/process_raw_so_data.py

## Combine data
set1["source"] = "set1"
set2["source"] = "set2"
sse["source"] = "sse"
set1["label"] = "security"
set2["label"] = "security"
sse["label"] = "security"

## Exclude selected postids from combined
blacklist_sse = set([str(i) for i in list(pd.read_csv("datasets/holdout_sse.csv").postid)])
holdout_sse = sse.loc[sse.postid.isin(blacklist_sse)]
holdout_sse.to_parquet("datasets/model_selection_CV/holdout_setA.parquet", compression=None, index=False)
sse = sse.loc[~sse.postid.isin(set(blacklist_sse))]

## Get holdout set and exclude from SO
holdout_so = set([str(i) for i in pd.read_csv("datasets/holdout_so.csv").postid])
ho_so = so.loc[so.postid.isin(holdout_so)]
ho_so.reset_index().to_parquet("datasets/model_selection_CV/holdout_setB.parquet", compression=None, index=False)
so = so.loc[~so.postid.isin(holdout_so)]

## Get unlabelled set
unlabel = so.sample(nonsec_sec_ratio*(len(set1)+len(set2))+len(sse), random_state=42)
unlabel["source"] = "so"
unlabel["label"] = "unlabelled"

# Combine datasets
combined = pd.concat([set1, set2, unlabel, sse]).fillna('')
combined.postid = combined.postid.astype(str)
combined.postid = combined.postid + "_" + combined.source
combined.to_parquet("datasets/model_selection_CV/X_train_r{}.parquet".format(nonsec_sec_ratio), compression=None, index=False)
combined["stratify"] = combined.source + "_" + combined.label

## Generate test set
train, test = train_test_split(combined, train_size = 0.9, random_state=42, stratify=combined.stratify)
train["fold"] = None
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

## Generate stratified kfolds
skf = StratifiedKFold(n_splits=num_k, random_state=42, shuffle=True)
indexes = [i for i in skf.split(train, train.stratify)]

## Set folds
for num, i in enumerate(indexes):
    train.loc[i[1], "fold"] = num

## Export test and train data
test.to_parquet("datasets/model_selection_CV/test.parquet", compression=None, index=False)
train.to_parquet("datasets/model_selection_CV/train.parquet", compression=None, index=False)

## Export IDs Only
def get_postid_info(postidlist, sourcelabels):
    return ["{}_{}".format(pid.split("_")[0], sla) for pid, sla in zip(postidlist, sourcelabels)]

def so_sse_sep(row):
    if row.source == "sse": return "{}_{}".format(row.postid, "sse")
    else: return "{}_{}".format(row.postid, "so")

## Organise IDs
fold_ids = pd.DataFrame([get_postid_info(train.loc[i[1]].postid, train.loc[i[1]].stratify) for i in indexes]).transpose()
test_ids = pd.DataFrame([get_postid_info(test.postid, test.stratify)]).transpose()
hosse_ids = pd.DataFrame(["{}_hosse_unlabelled_security".format(i) for i in pd.read_csv("datasets/holdout_sse.csv").postid])
hoso_ids = pd.DataFrame(["{}_hoso_unlabelled_nonsecurity".format(i) for i in pd.read_csv("datasets/holdout_so.csv").postid])
combined_ids = pd.concat([fold_ids, test_ids, hosse_ids, hoso_ids], axis=1)
combined_ids.columns = ["fold{}".format(i) for i in range(10)] + ["test", "hosse", "hoso"]

## Reshape final ID dataframe and save
melted_ids = pd.melt(combined_ids, var_name="set", value_name="postid").dropna()
melted_ids["source"] = melted_ids.apply(lambda row: row.postid.split("_")[1], axis=1)
melted_ids["label"] = melted_ids.apply(lambda row: row.postid.split("_")[2], axis=1)
melted_ids["postid"] = melted_ids.apply(lambda row: row.postid.split("_")[0], axis=1)
melted_ids["stxid"] = melted_ids.apply(lambda row: so_sse_sep(row), axis=1)
melted_ids.to_csv("datasets/model_selection_CV/fold_ids.csv", index=False)
