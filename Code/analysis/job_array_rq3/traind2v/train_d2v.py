import pandas as pd
import sanalytics.estimators.d2vestimator as sed
import logging
logging.basicConfig(level=logging.DEBUG)
from time import time

## Read test data and training folds
filename = sys.argv[1]
X = pd.read_parquet("datasets/rq3_data/{}_train.parquet".format(filename))
print(filename)
print(len(X))

## Train Doc2Vec for each split
start = time()
d2v = sed.D2VEstimator().fit(X)
end = time()
model = d2v.model
model.save("datasets/rq3_d2v/{}.model".format(filename))
pd.DataFrame([["{}".format(filename),end-start]], columns=["set", "training_time"]).to_csv("outputcsvs/d2v_training_times/{}.csv".format(filename), index=False)

## Read test data and training folds
X = pd.concat([pd.read_parquet("datasets/rq3_data/{}_train.parquet".format(filename)), pd.read_parquet("datasets/rq3_data/{}_test.parquet".format(filename))])
print(filename)
print(len(X))

## Train Doc2Vec for each split
start = time()
d2v = sed.D2VEstimator().fit(X)
end = time()
model = d2v.model
model.save("datasets/rq3_d2v/{}_all.model".format(filename))
pd.DataFrame([["{}_all".format(filename),end-start]], columns=["set", "training_time"]).to_csv("outputcsvs/d2v_training_times/{}_all.csv".format(filename), index=False)