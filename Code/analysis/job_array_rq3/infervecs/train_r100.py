import pandas as pd
import numpy as np
from progressbar import progressbar
import sanalytics.algorithms.utils as sau
import sanalytics.estimators.pu_estimators as pu
import sanalytics.evaluation.utils as seu
import logging
logging.basicConfig(level=logging.DEBUG)
from importlib import reload
reload(pu)
import joblib

id = int(sys.argv[1])

## Read classifiers
best_clfs = pd.read_pickle("datasets/best_clfs.pd.pkl")

## Set variation
variation = "sec1.0R100_all"

## 100%
x = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "all" in i and "test" not in i]
df = pd.concat([pd.read_parquet(i)[["postid","label","d2v"]] for i in progressbar(x)])

## XGB Naive
if id == 0:
    clf = pu.get_estimator("XGB", best_clfs.loc["XGB"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_XGB_naive.clf.pkl'.format(variation), compress = 1)
    print(time)

## LGBM Naive
if id == 1:
    clf = pu.get_estimator("LGBM", best_clfs.loc["LGBM"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_LGBM_naive.clf.pkl'.format(variation), compress = 1)
    print(time)

## XGB PU
if id == 2:
    df, step1_time = pu.step1(df, best_clfs.loc["XGB"].params_1)
    print(step1_time)
    clf = pu.get_estimator("XGB", best_clfs.loc["XGB"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_XGB_pu.clf.pkl'.format(variation), compress = 1)
    print(time)

## LGBM PU
if id == 3:
    df, step1_time = pu.step1(df, best_clfs.loc["LGBM"].params_1)
    print(step1_time)
    clf = pu.get_estimator("LGBM", best_clfs.loc["LGBM"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_LGBM_pu.clf.pkl'.format(variation), compress = 1)
    print(time)

## 90%
x = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "all" not in i and "test" not in i]
df = pd.concat([pd.read_parquet(i)[["postid","label","d2v"]] for i in progressbar(x)])

## XGB Naive
if id == 4:
    clf = pu.get_estimator("XGB", best_clfs.loc["XGB"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_XGB_naive.clf.pkl.2'.format(variation), compress = 1)
    print(time)

## LGBM Naive
if id == 5:
    clf = pu.get_estimator("LGBM", best_clfs.loc["LGBM"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_LGBM_naive.clf.pkl.2'.format(variation), compress = 1)
    print(time)

## XGB PU
if id == 6:
    df, step1_time = pu.step1(df, best_clfs.loc["XGB"].params_1)
    print(step1_time)
    clf = pu.get_estimator("XGB", best_clfs.loc["XGB"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_XGB_pu.clf.pkl.2'.format(variation), compress = 1)
    print(time)

## LGBM PU
if id == 7:
    df, step1_time = pu.step1(df, best_clfs.loc["LGBM"].params_1)
    print(step1_time)
    clf = pu.get_estimator("LGBM", best_clfs.loc["LGBM"].params_2)
    clf, time = clf.fit(df)
    joblib.dump(clf, r'analysis/job_array_rq3/testmodels/rq3_results/{}_LGBM_pu.clf.pkl.2'.format(variation), compress = 1)
    print(time)

