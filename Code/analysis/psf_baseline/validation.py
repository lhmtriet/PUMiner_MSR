import pandas as pd
import sanalytics.estimators.pu_estimators as pu
from gensim.models.doc2vec import Doc2Vec
import sanalytics.evaluation.evaluation_metric as see
from progressbar import progressbar
import sanalytics.algorithms.utils as sau
from time import time
import numpy as np

## Read threshold
arg = sys.argv[1].split("|")
t = float(arg[0])
name = arg[1]
fold = int(arg[1].split("_")[-2])

## Import Data
X_train = pd.read_parquet("datasets/rq3_data/sec1.0_train.parquet")
X_val = X_train[X_train.fold==fold]

## Import D2V
d2v = Doc2Vec.load("datasets/kfold_d2v/{}.model".format(name))

## In pos set
def pos_set(str):
    if "|" in str: return False
    if "sse" in str: return True
    if "set1" in str: return True
    if "set2" in str: return True

## Predict functions
def predict(post, thresh, d2v):
    vec = d2v.infer_vector("{} {} {}".format(post.title, post.question, post.answers).split())
    sims = d2v.docvecs.most_similar([vec], topn=1000)
    return min(len([i for i in sims if pos_set(i[0]) and i[1] > thresh]), 1)

## Columns
c_90 = ["variation", "classifier", "test_set", "recall", "prec_lower", "prec_opt", "f1_lower", "f1_opt", "f_measure", "eval_time"]

## Test set
results_90 = []
start_pred = time()
X_val["preds"] = [predict(i, t, d2v) for i in progressbar(X_val.itertuples())]
end_pred = time()
results = see.evaluate_metrics(X_val[X_val.label=="security"].preds, X_val.preds)
results_90.append(["sec1.0", "d2v_baseline_{}_{}_onlypos".format(t, name), "fold_{}".format(fold)] + list(results) + [end_pred - start_pred])
df_90 = pd.DataFrame(results_90, columns=c_90)
df_90.to_csv("analysis/test_models/d2v_baseline/preds_val/d2v_baseline_{}_{}_val_onlypos.csv".format(t, name), index=False)
