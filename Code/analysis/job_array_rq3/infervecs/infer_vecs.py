import pandas as pd
import numpy as np
import sanalytics.algorithms.utils as sau
from gensim.models.doc2vec import Doc2Vec
from progressbar import progressbar
import re

## Get file from argument
filename = sys.argv[1]

## Regular test sets
if "all" not in filename:

    ## Read data
    X_train = pd.read_parquet("datasets/rq3_data/{}_train.parquet".format(filename))
    X_test = pd.read_parquet("datasets/rq3_data/{}_test.parquet".format(filename))

    ## Read D2V Model
    d2v = Doc2Vec.load("datasets/rq3_d2v/{}.model".format(filename))

    ## Infer DF
    infer_df = lambda df: df.apply(lambda row: d2v.infer_vector("{} {} {}".format(row.title, row.question, row.answers).split()), axis=1)
    X_train["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_train, 2000), num_cpus=30) for j in i]
    X_val["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_val, 2000), num_cpus=30) for j in i]
    X_test["d2v"] = [j for i in sau.p_map(infer_df, np.array_split(X_test, 100), num_cpus=30) for j in i]

    ## Save file
    X_train.to_pickle("datasets/rq3_vecdata_new/{}_train.parquet".format(filename))
    X_test.to_pickle("datasets/rq3_vecdata_new/{}_test.parquet".format(filename))

    X_train.to_parquet("datasets/rq3_vecdata_new/{}_train.parquet".format(filename), compression=None, index=False)
    X_test.to_parquet("datasets/rq3_vecdata_new/{}_test.parquet".format(filename), compression=None, index=False)

## Edge test sets
if "all" in filename:

    ## Read Data
    filename = re.sub("_all", "", filename)
    X_train = pd.read_parquet("datasets/rq3_data/{}_train.parquet".format(filename))
    X_test = pd.read_parquet("datasets/rq3_data/{}_test.parquet".format(filename))
    X_all = pd.concat([X_train, X_test], sort=False)
    X_edge = pd.read_parquet("datasets/model_selection_CV/X_edge.parquet")
    X_easy = pd.read_parquet("datasets/model_selection_CV/X_easy.parquet")

    ## Read D2V Model
    d2v = Doc2Vec.load("datasets/rq3_d2v/{}_all.model".format(filename))

    ## Infer DF
    X_all["d2v"] = [d2v.infer_vector("{} {} {}".format(i.title, i.question, i.answers).split()) for i in progressbar(X_all.itertuples())]
    X_edge["d2v"] = [d2v.infer_vector("{} {} {}".format(i.title, i.question, i.answers).split()) for i in progressbar(X_edge.itertuples())]
    X_easy["d2v"] = [d2v.infer_vector("{} {} {}".format(i.title, i.question, i.answers).split()) for i in progressbar(X_easy.itertuples())]

    ## Save file
    X_all.to_parquet("datasets/rq3_vecdata_new/{}_all_train.parquet".format(filename), compression=None, index=False)
    X_edge.to_parquet("datasets/rq3_vecdata_new/{}_all_test_edge.parquet".format(filename), compression=None, index=False)
    X_easy.to_parquet("datasets/rq3_vecdata_new/{}_all_test_easy.parquet".format(filename), compression=None, index=False)
