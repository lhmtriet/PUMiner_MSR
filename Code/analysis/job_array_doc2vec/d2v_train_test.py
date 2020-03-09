## Train a D2v Model on 100% of data
import pandas as pd
import sanalytics.estimators.d2vestimator as sed
import logging
logging.basicConfig(level=logging.DEBUG)

## Read all data
X = pd.read_parquet("datasets/model_selection_CV/train.parquet")
x = pd.read_parquet("datasets/model_selection_CV/test.parquet")
X = pd.concat([X,x], sort=False)

## Train Doc2Vec for each split
d2v = sed.D2VEstimator().fit(X)
model = d2v.model
model.save("datasets/kfold_d2v/d2v_all.model")