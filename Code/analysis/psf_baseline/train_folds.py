import pandas as pd
import sanalytics.estimators.d2vestimator as sed
import logging
logging.basicConfig(level=logging.DEBUG)

## Fold
fold = int(sys.argv[1])

## Read all data
X = pd.read_parquet("datasets/model_selection_CV/train.parquet")
X = X[X.label == "security"]
X = X[X.fold != fold]
X = X.sample(10)

## Train Doc2Vec for each split
d2v = sed.D2VEstimator().fit(X)
model = d2v.model
model.save("analysis/test_models/d2v_baseline/fold_d2v/d2v_val_fold_{}_onlypos.model".format(fold))
