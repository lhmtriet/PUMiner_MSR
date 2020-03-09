import pandas as pd
import numpy as np
from progressbar import progressbar
import sanalytics.algorithms.utils as sau
import sanalytics.estimators.pu_estimators as pu
import sanalytics.evaluation.utils as seu

X_train = pd.read_parquet("datasets/rq3_data/sec1.0R100_train.parquet")
X_train = np.array_split(X_train, 300)
for num, i in progressbar(enumerate(X_train)):
    i.to_parquet("datasets/rq3_dataR100/sec1.0R100_train.parquet.{}".format(num), compression=None, index=False)
    test = pd.read_parquet("datasets/rq3_dataR100/sec1.0R100_train.parquet.{}".format(num))

X_train = pd.read_parquet("datasets/rq3_data/sec1.0R100_test.parquet")
X_train = np.array_split(X_train, 50)
for num, i in progressbar(enumerate(X_train)):
    i.to_parquet("datasets/rq3_dataR100/sec1.0R100_test.parquet.{}".format(num), compression=None, index=False)
    test = pd.read_parquet("datasets/rq3_dataR100/sec1.0R100_test.parquet.{}".format(num))

X_train = pd.read_parquet("datasets/rq3_data/sec1.0R100_all_train.parquet")
X_train = np.array_split(X_train, 300)
for num, i in progressbar(enumerate(X_train)):
    i.to_parquet("datasets/rq3_dataR100/sec1.0R100_all_train.parquet.{}".format(num), compression=None, index=False)
    test = pd.read_parquet("datasets/rq3_dataR100/sec1.0R100_all_train.parquet.{}".format(num))

X_train = pd.read_parquet("datasets/rq3_data/sec1.0R100_all_test_edge.parquet")
X_train.to_parquet("datasets/rq3_dataR100/sec1.0R100_all_test_edge.parquet.0", compression=None, index=False)
X_train = pd.read_parquet("datasets/rq3_data/sec1.0R100_all_test_easy.parquet")
X_train.to_parquet("datasets/rq3_dataR100/sec1.0R100_all_test_easy.parquet.0", compression=None, index=False)