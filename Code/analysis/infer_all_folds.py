import os
import pandas as pd
import numpy as np
import sanalytics.algorithms.utils as sau
from gensim.models.doc2vec import Doc2Vec
import unittest

## Generate Train Set
for val_fold in range(0, 10):

    ## Read data and D2V Model
    X = pd.read_parquet("datasets/model_selection_CV/train.parquet")
    X_val = X.loc[X.fold == val_fold].copy()
    X_train = X.loc[X.fold != val_fold].copy()
    d2v = Doc2Vec.load("datasets/kfold_d2v/d2v_val_fold_{}.model".format(val_fold))

    ## Test that Doc2Vec is correct
    test = d2v[X_train.iloc[42].postid]
    unittest.TestCase().assertRaises(KeyError, lambda: d2v[X_val.iloc[42].postid])

    ## Infer DF
    infer_df = lambda df: df.apply(lambda row: d2v.infer_vector("{} {} {}".format(row.title, row.question, row.answers).split()), axis=1)
    X_train["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_train, 1000), num_cpus=30) for j in i]
    X_val["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_val, 200), num_cpus=30) for j in i]

    ## Test that the inferred vectors are correct for training set
    real_train = X_train.iloc[42]
    inferred_train = d2v.docvecs.most_similar([d2v.infer_vector("{} {} {}".format(real_train.title, real_train.question, real_train.answers).split())])
    assert(real_train.postid in dict(inferred_train))

    ## Test that the inferred vectors are correct for validation set
    real_val = X_val.iloc[42]
    inferred_val = d2v.docvecs.most_similar([d2v.infer_vector("{} {} {}".format(real_val.title, real_val.question, real_val.answers).split())])
    assert(real_train.postid not in dict(inferred_val))

    ## Save file
    X_train.to_parquet("datasets/folds/fold{}_train.parquet".format(val_fold), compression=None, index=False)
    X_val.to_parquet("datasets/folds/fold{}_val.parquet".format(val_fold), compression=None, index=False)

## Generate Test Set
X_train = pd.read_parquet("datasets/model_selection_CV/train.parquet")
X_test = pd.read_parquet("datasets/model_selection_CV/test.parquet")
d2v = Doc2Vec.load("datasets/kfold_d2v/d2v_val_fold_test.model")

## Test that Doc2Vec is correct
test = [d2v[X_train[X_train.fold==i].iloc[0].postid] for i in range(10)]
unittest.TestCase().assertRaises(KeyError, lambda: d2v[X_test.iloc[42].postid])

## Infer DF
infer_df = lambda df: df.apply(lambda row: d2v.infer_vector("{} {} {}".format(row.title, row.question, row.answers).split()), axis=1)
X_train["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_train, 1000), num_cpus=30) for j in i]
X_test["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_test, 200), num_cpus=30) for j in i]

## Test that the inferred vectors are correct for training set
real_train = X_train.iloc[42]
inferred_train = d2v.docvecs.most_similar([d2v.infer_vector("{} {} {}".format(real_train.title, real_train.question, real_train.answers).split())])
assert(real_train.postid in dict(inferred_train))

## Test that the inferred vectors are correct for validation set
real_val = X_test.iloc[42]
inferred_val = d2v.docvecs.most_similar([d2v.infer_vector("{} {} {}".format(real_val.title, real_val.question, real_val.answers).split())])
assert(real_train.postid not in dict(inferred_val))

## Save file
X_train.to_parquet("datasets/folds/fold_test_train.parquet".format(val_fold), compression=None, index=False)
X_test.to_parquet("datasets/folds/fold_test_val.parquet".format(val_fold), compression=None, index=False)

## --------------------------------- ##

## Infer folds for edge case testing
X_all = pd.concat([pd.read_parquet("datasets/model_selection_CV/train.parquet"), pd.read_parquet("datasets/model_selection_CV/test.parquet")], sort=False)
ho_setA = pd.read_parquet("datasets/model_selection_CV/holdout_setA.parquet")
ho_setB = pd.read_parquet("datasets/model_selection_CV/holdout_setB.parquet")
d2v = Doc2Vec.load("datasets/kfold_d2v/d2v_all.model")

## Infer DF
infer_df = lambda df: df.apply(lambda row: d2v.infer_vector("{} {} {}".format(row.title, row.question, row.answers).split()), axis=1)
X_all["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_all, 1000), num_cpus=30) for j in i]
ho_setA["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(ho_setA, 10), num_cpus=2) for j in i]
ho_setB["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(ho_setB, 10), num_cpus=2) for j in i]

## Test that the inferred vectors are correct for training set
real_train = X_all.iloc[42]
inferred_train = d2v.docvecs.most_similar([d2v.infer_vector("{} {} {}".format(real_train.title, real_train.question, real_train.answers).split())])
assert(real_train.postid in dict(inferred_train))

## Test that the inferred vectors are correct for validation set
real_val = ho_setA.iloc[20]
inferred_val = d2v.docvecs.most_similar([d2v.infer_vector("{} {} {}".format(real_val.title, real_val.question, real_val.answers).split())])
assert(real_train.postid not in dict(inferred_val))

## Test that the inferred vectors are correct for validation set
real_val = ho_setB.iloc[30]
inferred_val = d2v.docvecs.most_similar([d2v.infer_vector("{} {} {}".format(real_val.title, real_val.question, real_val.answers).split())])
assert(real_train.postid not in dict(inferred_val))

X_all.to_parquet("datasets/folds/train_all.parquet", compression=None, index=False)
ho_setA.to_parquet("datasets/folds/test_setA.parquet", compression=None, index=False)
ho_setB.to_parquet("datasets/folds/test_setB.parquet", compression=None, index=False)