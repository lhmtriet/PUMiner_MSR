import dask.dataframe as dd
import sanalytics.estimators.d2vestimator as sed
import logging
logging.basicConfig(level=logging.DEBUG)
from time import time
import pandas as pd

if int(sys.argv[1]) == 1:
    filename = "X100_train.parquet+X100_test.parquet"
    files = ["datasets/model_selection_CV/{}".format(i) for i in filename.split("+")]
    df = dd.read_parquet(files).fillna('').compute() # generated using analysis/job_array_processing
    start = time()
    d2vest = sed.D2VEstimator().fit(df)
    end = time()
    d2vest.model.save("datasets/rq3_d2v/sec1.0R100_all.model")
    filename = "sec1.0R100_all"
    pd.DataFrame([["{}".format(filename),end-start]], columns=["set", "training_time"]).to_csv("outputcsvs/d2v_training_times/{}.csv".format(filename), index=False)

if int(sys.argv[1]) == 2:
    filename = "X100_train.parquet"
    files = ["datasets/model_selection_CV/{}".format(i) for i in filename.split("+")]
    df = dd.read_parquet(files).fillna('').compute() # generated using analysis/job_array_processing
    start = time()
    d2vest = sed.D2VEstimator().fit(df)
    end = time()
    d2vest.model.save("datasets/rq3_d2v/sec1.0R100.model")
    filename = "sec1.0R100"
    pd.DataFrame([["{}".format(filename),end-start]], columns=["set", "training_time"]).to_csv("outputcsvs/d2v_training_times/{}.csv".format(filename), index=False)

if int(sys.argv[1]) == 3:
    filename = "X10_train.parquet+X10_test.parquet"
    files = ["datasets/model_selection_CV/{}".format(i) for i in filename.split("+")]
    df = dd.read_parquet(files).fillna('').compute() # generated using analysis/job_array_processing
    start = time()
    d2vest = sed.D2VEstimator().fit(df)
    end = time()
    d2vest.model.save("datasets/rq3_d2v/sec1.0R100_all.model")
    filename = "sec1.0R10_all"
    pd.DataFrame([["{}".format(filename),end-start]], columns=["set", "training_time"]).to_csv("outputcsvs/d2v_training_times/{}.csv".format(filename), index=False)

if int(sys.argv[1]) == 4:
    filename = "X10_train.parquet"
    files = ["datasets/model_selection_CV/{}".format(i) for i in filename.split("+")]
    df = dd.read_parquet(files).fillna('').compute() # generated using analysis/job_array_processing
    start = time()
    d2vest = sed.D2VEstimator().fit(df)
    end = time()
    d2vest.model.save("datasets/rq3_d2v/sec1.0R100.model")
    filename = "sec1.0R10"
    pd.DataFrame([["{}".format(filename),end-start]], columns=["set", "training_time"]).to_csv("outputcsvs/d2v_training_times/{}.csv".format(filename), index=False)