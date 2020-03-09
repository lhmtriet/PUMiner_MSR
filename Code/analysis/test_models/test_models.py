import dask.dataframe as dd
import pandas as pd
from collections import Counter
import plotly.express as px
import sanalytics.estimators.pu_estimators as pu
import sanalytics.evaluation.utils as seu
import joblib

## Get best classifiers
best_clfs = seu.get_best_classifiers("outputcsvs/validation/*.csv", "analysis/pu_learning/foldinfo.pkl").set_index("clf")

## Read CSV
X_train = pd.read_parquet("datasets/folds/fold_test_train.parquet")
X_test = pd.read_parquet("datasets/folds/fold_test_val.parquet")
combs = pd.read_csv("analysis/test_models/combs.csv", header=None).set_index(0)

## Loop
while True:
    final_test_results = []
    finished = set(['.'.join(i.split(".")[:-1]) for i in [i for i in os.walk("outputcsvs/test_naive_pu/")][0][2]])
    pick = combs.loc[~combs.index.isin(finished)].sample(1)
    id = pick.iloc[0].name
    clfid = id.split("_")[0]
    step1 = id.split("_")[1]
    clfinfo = best_clfs.loc[clfid]
    best = joblib.load("outputcsvs/validation/{}.pkl".format(clfinfo.id))
    print(clfid, step1)

    if step1 == "naive":
        step1_time = 0.0
        clf_naive, fit_time = best.fit(X_train)
        results_naive, eval_time = clf_naive.evaluate(X_test)
        final_test_results.append(["Best_{}".format(clfid)] + [step1] + list(results_naive) + [fit_time, step1_time, eval_time])
        joblib.dump(clf_naive, r'outputcsvs/test_naive_pu/{}_{}.pkl'.format(clfid, step1), compress = 1)

    if step1 == "pu":
        reliable_set, step1_time = pu.step1(X_train, clfinfo.params_1)
        clf_pu, fit_time = best.fit(reliable_set)
        results_pu, eval_time = clf_pu.evaluate(X_test)
        final_test_results.append(["Best_{}".format(clfid)] + [step1] + list(results_pu) + [fit_time, step1_time, eval_time])
        joblib.dump(clf_pu, r'outputcsvs/test_naive_pu/{}_{}.pkl'.format(clfid, step1), compress = 1)

    best_results_test = pd.DataFrame(final_test_results, columns=["clf","step1", "recall", "prec_lower", "prec_opt", "f1_lower", "f1_opt", "f_measure", "fit_time", "step1_time", "eval_time"])
    best_results_test.to_csv("outputcsvs/test_naive_pu/{}_{}.csv".format(clfid, step1), index=False)

