import pandas as pd
import numpy as np
from collections import Counter
import sanalytics.estimators.pu_estimators as pu
import sanalytics.evaluation.utils as seu
import sanalytics.algorithms.utils as sau
import joblib
import random
import itertools
from gensim.models.doc2vec import Doc2Vec
from collections import Counter
from progressbar import progressbar

## String parsing helper
def parse_path(i): return "_".join('.'.join (i.split(".")[:-1]).split("_")[:-1])

## Read data
best_clfs = seu.get_best_classifiers("outputcsvs/validation/*.csv", "analysis/pu_learning/foldinfo.pkl").set_index("clf")
c_label = ["variation", "classifier", "test_set", "recall", "precision", "f1_score", "gmean", "mcc", "fit_time", "step1_time", "eval_time"]
c_unlabel = ["variation", "classifier", "test_set", "recall", "prec_lower", "prec_opt", "f1_lower", "f1_opt", "f_measure", "fit_time", "step1_time", "eval_time"]

len(combs)
len([i for i in list(os.walk("analysis/job_array_rq3/testmodels/rq3_results"))[0][2] if "csv" in i])

## Loop
while True:
    ## Update combs
    s_vars = [parse_path(i) for i in sorted(list(os.walk("datasets/rq3_vecdata_new/"))[0][2]) if "train" in i]
    s_clfs = list(pd.read_csv("analysis/job_array_rq3/testmodels/classifiers.csv", header=None)[0])
    combs = list(itertools.product(s_vars, s_clfs))

    ## Get unperformed combination
    variation, classifier = random.sample(combs, 1)[0]
    if "{}_{}".format(variation, classifier) in ' '.join([i for i in list(os.walk("analysis/job_array_rq3/testmodels/rq3_results"))[0][2] if "csv" in i]):
        continue
    print("Starting {}_{}".format(variation, classifier))

    ## Load data
    X_train = pd.read_parquet("datasets/rq3_vecdata_new/{}_train.parquet".format(variation))

    ## Get classifier
    clfname, nvp = classifier.split("_")[0:2]
    clfinfo = best_clfs.loc[clfname]
    clf = pu.get_estimator(clfname, clfinfo.params_2)
    if clf==None:  clf = pu.NC(clfinfo.params_1)
    print("{}\n{}\n{}\n{}".format(variation, classifier, nvp, clf.model))
    final_test_results = []

    if nvp == "pu":
        X_train, step1_time = pu.step1(X_train, clfinfo.params_1)
        clf_fitted, fit_time = clf.fit(X_train)
        joblib.dump(clf_fitted, r'analysis/job_array_rq3/testmodels/rq3_results/{}_{}.clf.pkl'.format(variation, classifier), compress = 1)

    if nvp == "naive":
        step1_time = 0.0
        clf_fitted, fit_time = clf.fit(X_train)
        joblib.dump(clf_fitted, r'analysis/job_array_rq3/testmodels/rq3_results/{}_{}.clf.pkl'.format(variation, classifier), compress = 1)

    if "all" in variation:
        X_edge = pd.read_parquet("datasets/rq3_vecdata_new/{}_test_edge.parquet".format(variation))
        X_easy = pd.read_parquet("datasets/rq3_vecdata_new/{}_test_easy.parquet".format(variation))
        results_edge, eval_time_edge = clf_fitted.score(X_edge)
        results_easy, eval_time_easy = clf_fitted.score(X_easy)
        X_edge.to_parquet("analysis/job_array_rq3/testmodels/rq3_results/{}_{}.edge.preds.parquet".format(variation, classifier), compression=None, index=False)
        X_easy.to_parquet("analysis/job_array_rq3/testmodels/rq3_results/{}_{}.easy.preds.parquet".format(variation, classifier), compression=None, index=False)
        final_test_results.append([variation, classifier] + ["edge"] + list(results_edge) + [fit_time, step1_time, eval_time_edge])
        final_test_results.append([variation, classifier] + ["easy"] + list(results_easy) + [fit_time, step1_time, eval_time_easy])
        best_results_test = pd.DataFrame(final_test_results, columns=c_label)
        best_results_test.to_csv("analysis/job_array_rq3/testmodels/rq3_results/{}_{}.score.csv".format(variation, classifier), index=False)

    if "all" not in variation:
        X_test = pd.read_parquet("datasets/rq3_vecdata_new/{}_test.parquet".format(variation))
        results_test, eval_time_test = clf_fitted.evaluate(X_test)
        X_test.to_parquet("analysis/job_array_rq3/testmodels/rq3_results/{}_{}.test.preds.parquet".format(variation, classifier), compression=None, index=False)
        final_test_results.append([variation, classifier] + ["test"] + list(results_test) + [fit_time, step1_time, eval_time_test])
        best_results_test = pd.DataFrame(final_test_results, columns=c_unlabel)
        best_results_test.to_csv("analysis/job_array_rq3/testmodels/rq3_results/{}_{}.score.csv".format(variation, classifier), index=False)

    print("Finished {}_{}".format(variation, classifier))
