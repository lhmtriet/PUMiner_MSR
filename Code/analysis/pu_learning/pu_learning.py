import csv
import pandas as pd
import sanalytics.algorithms.utils as sau
import sanalytics.estimators.pu_estimators as pu
from sanalytics.estimators.utils import diff_df, join_df
from gensim.models.doc2vec import Doc2Vec
import joblib
from time import time

## Read arguments
while True:
    finished = set(['.'.join(i.split(".")[:-1]) for i in [i for i in os.walk("outputcsvs/validation/")][0][2]])
    info = pd.read_pickle("analysis/pu_learning/foldinfo.pkl")
    info = info.loc[~info.index.isin(finished)].sample(1)
    id = info.index.item()
    val_fold = info.fold.item()
    pu_type = info.putype.item()
    classifier = info.clf.item()
    params_1 = info.params_1.item()
    params_2 = info.params_2.item()
    print(id)

    ## Read training/validation folds
    X_train = pd.read_parquet("datasets/folds/fold{}_train.parquet".format(val_fold))
    X_val = pd.read_parquet("datasets/folds/fold{}_val.parquet".format(val_fold))

    ## Step 1: Extract RN
    P = X_train.loc[X_train.label=="security"].copy()
    U = X_train.loc[X_train.label=="unlabelled"].copy()
    C1, fit_time = pu.NC(params_1).fit(X_train)
    UL1, pred_time = C1.predict(U)
    Q1 = UL1.loc[UL1.predicts == "nonsecurity"].copy()
    RN1 = Q1
    U1 = diff_df(U,Q1)

    ## Prepare CSV to write to
    df_rows = []
    columns = ["id", "recall", "prec_lower", "prec_opt", "f1_lower", "f1_opt", "f_measure", "fit_time", "predict_time", "eval_time"]
    filename = 'outputcsvs/validation/{}'.format(id)

    ## BLANK means Step 1 only
    if classifier=="BLANK":
        results, eval_time = C1.evaluate(X_val)
        row = [id] + list(results) + [fit_time, 0, eval_time]
        df_rows.append(row)
        pd.DataFrame(df_rows, columns=columns).to_csv("{}.csv".format(filename), index=False)
        joblib.dump(C1, r'{}.pkl'.format(filename), compress = 1)
        print(row)
        continue

    ## Get estimator
    pu_est = pu.get_estimator(classifier, params_2)
    print(pu_est.model)

    ## Step 2: Train Classifier Iteratively
    if (pu_type == '0'):
        C2, fit_time = pu_est.fit(pd.concat([P, RN1], sort=False))
        results, eval_time = C2.evaluate(X_val)
        row = [id] + list(results) + [fit_time, 0, eval_time]
        df_rows.append(row)
        pd.DataFrame(df_rows, columns=columns).to_csv("{}.csv".format(filename), index=False)
        joblib.dump(C2, r'{}.pkl'.format(filename), compress = 1)
        print(row)

if (pu_type == '1'):
    while True:
        C2, fit_time = pu_est.fit(pd.concat([P, RN1], sort=False))
        UL2, pred_time = C2.predict(U1)
        Q2 = UL2.loc[UL2.predicts=="nonsecurity"].copy()
        if len(Q2) == 0: break
        U2 = diff_df(U1, Q2)
        RN2 = join_df(RN1, Q2)
        RN1 = RN2
        U1 = U2
        C1 = C2
        results, eval_time = C2.evaluate(X_val)
        row = [id] + list(results) + [fit_time, pred_time, eval_time]
        df_rows.append(row)
        pd.DataFrame(df_rows, columns=columns).to_csv("{}.csv".format(filename), index=False)
        joblib.dump(C1, r'{}.pkl'.format(filename), compress = 1)
        print(row)

if (pu_type == '2'):
    while True:
        C2, fit_time = pu_est.fit(pd.concat([P, RN1], sort=False))
        RNL2, pred_time = C2.predict(RN1)
        Q2 = RNL2.loc[RNL2.predicts=="nonsecurity"].copy()
        # if len(Q2) >= len(Q1) and len(P) >= len(RN1):
        #     break
        RN2 = Q2
        RN1 = RN2
        C1 = C2
        results, eval_time = C2.evaluate(X_val)
        row = [id] + list(results) + [fit_time, pred_time, eval_time]
        df_rows.append(row)
        pd.DataFrame(df_rows, columns=columns).to_csv("{}.csv".format(filename), index=False)
        joblib.dump(C1, r'{}.pkl'.format(filename), compress = 1)
        print(row)