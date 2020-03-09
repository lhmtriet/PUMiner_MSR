import pandas as pd
import sanalytics.estimators.d2vestimator as sed
import logging
logging.basicConfig(level=logging.DEBUG)

## Validation fold
val_fold = sys.argv[1]
print(val_fold)

## Read test data and training folds
X = pd.read_parquet("datasets/model_selection_CV/train.parquet")
X = X[X.fold != int(val_fold)]
print(len(X))

## Train Doc2Vec for each split
d2v = sed.D2VEstimator().fit(X)
model = d2v.model
if val_fold == -1: val_fold = "test"
model.save("datasets/kfold_d2v/d2v_val_fold_{}.model".format(val_fold))