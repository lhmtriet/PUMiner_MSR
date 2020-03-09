from glob import glob
import sanalytics.estimators.pu_estimators as pu
import joblib
from progressbar import progressbar
import pandas as pd

## Load models
models_90 = sorted([i for i in glob("analysis/job_array_rq3/testmodels/rq3_results/*R100*") if ".2" in i])
models_100 = sorted([i for i in glob("analysis/job_array_rq3/testmodels/rq3_results/*R100*") if ".2" not in i])

## Load times
times = pd.read_csv("outputcsvs/r100_manual_times.csv")
times["full"] = times["variation"] + "_" + times["classifier"]

## Load test sets
test_90 = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "all" not in i and "test" in i]
test_edge = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "edge" in i]
test_easy = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "easy" in i]
test_90 = pd.concat([pd.read_parquet(i) for i in progressbar(test_90)])
test_edge = pd.concat([pd.read_parquet(i) for i in progressbar(test_edge)])
test_easy = pd.concat([pd.read_parquet(i) for i in progressbar(test_easy)])

## Columns
c_100 = ["variation", "classifier", "test_set", "recall", "precision", "f1_score", "gmean", "mcc", "fit_time", "step1_time", "eval_time"]
c_90 = ["variation", "classifier", "test_set", "recall", "prec_lower", "prec_opt", "f1_lower", "f1_opt", "f_measure", "fit_time", "step1_time", "eval_time"]

## Get results for 90% train set
results_90 = []
for model in progressbar(models_90):
    info = [i for i in times.itertuples() if i.full in model][0]
    model = joblib.load(model)
    results, eval_time = model.evaluate(test_90)
    results_90.append([info.variation, info.classifier, info.test_set] + list(results) + [info.fit_time, info.step1_time] + [eval_time])
df_90 = pd.DataFrame(results_90, columns=c_90)

## Get results for 100% train set
results_100 = []
for model in progressbar(models_100):
    info = [i for i in times.itertuples() if i.full in model][0]
    model = joblib.load(model)
    results_edge, eval_time_ed = model.score(test_edge)
    results_easy, eval_time_ea = model.score(test_easy)
    results_100.append([info.variation, info.classifier, "edge"] + list(results_edge) + [info.fit_time, info.step1_time] + [eval_time_ed])
    results_100.append([info.variation, info.classifier, "easy"] + list(results_easy) + [info.fit_time, info.step1_time] + [eval_time_ea])
df_100 = pd.DataFrame(results_100, columns=c_100)

df_90.to_csv("outputcsvs/r100_results_90.csv", index=False)
df_100.to_csv("outputcsvs/r100_results_100.csv", index=False)
