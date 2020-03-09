import pandas as pd
import sanalytics.estimators.pu_estimators as pu
from gensim.models.doc2vec import Doc2Vec
import sanalytics.evaluation.evaluation_metric as see
from progressbar import progressbar
import sanalytics.algorithms.utils as sau
from time import time

## Import Data
X_test = pd.read_parquet("datasets/rq3_data/sec1.0_test.parquet")
X_hard = pd.read_parquet("published/publicdata/test_edge.parquet")
X_norm = pd.read_parquet("published/publicdata/test_normal.parquet")

## Import D2V
d2v = Doc2Vec.load("datasets/rq3_d2v/sec1.0_all_posonly.model")

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
c_100 = ["variation", "classifier", "test_set", "recall", "precision", "f1_score", "gmean", "mcc", "eval_time"]

## Try all thresholds
ts = [0.5, 0.6, 0.7, 0.8, 0.9]
results_100 = []
for t in ts:
    start_pred = time()
    X_norm["preds"] = [predict(i, t, d2v) for i in progressbar(X_norm.itertuples())]
    end_pred = time()
    results = see.evaluate_metrics_known(X_norm.preds, sau.encode(X_norm.label))
    results_100.append(["sec1.0_all", "d2v_baseline_{}".format(t), "normal"] + list(results) + [end_pred - start_pred])

    start_pred = time()
    X_hard["preds"] = [predict(i, t, d2v) for i in progressbar(X_hard.itertuples())]
    end_pred = time()
    results = see.evaluate_metrics_known(X_hard.preds, sau.encode(X_hard.label))
    results_100.append(["sec1.0_all", "d2v_baseline_{}".format(t), "hard"] + list(results) + [end_pred - start_pred])
