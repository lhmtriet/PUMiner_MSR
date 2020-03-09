import pandas as pd
import sanalytics.estimators.d2vestimator as sed
import logging
logging.basicConfig(level=logging.DEBUG)
from time import time
from progressbar import progressbar
import sanalytics.evaluation.evaluation_metric as see
import sanalytics.algorithms.utils as sau

## Import Data
X_train = pd.read_parquet("datasets/rq3_data/sec1.0_train.parquet")
X_test = pd.read_parquet("datasets/rq3_data/sec1.0_test.parquet")
X_all = pd.concat([X_train, X_test], sort=False)
X_train_90 = X_train[X_train.label=="security"]
X_train_100 = X_all[X_all.label=="security"]

## Train D2V
d2v_90 = sed.D2VEstimator().fit(X_train_90)
d2v_100 = sed.D2VEstimator().fit(X_train_90)
d2v_90.model.save("datasets/rq3_d2v/sec1.0_posonly.model")
d2v_100.model.save("datasets/rq3_d2v/sec1.0_all_posonly.model")
