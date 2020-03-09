import dask.dataframe as dd
import pandas as pd
import plotly.express as px
import sanalytics.evaluation.utils as seu
from plotly.io import write_html
import joblib

## Read data
analysis = seu.get_results("outputcsvs/validation/*.csv", "analysis/pu_learning/foldinfo.pkl")

## Plot
id_vars = ['clf', "fold", "params_1", "params_2", "putype"]
value_vars = ['recall', 'prec_lower', 'prec_opt', 'f1_lower', 'f1_opt', 'f_measure', 'fit_time','eval_time']
melt_analysis = pd.melt(analysis, id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='value')
fig = px.box(melt_analysis, x="clf", y="value", color="metric", hover_data=["fold","params_1","params_2","putype"], points="all")
write_html(fig.update_layout(height=1000),file='outputcsvs/pu_learning_results.html')

## Read Test Data - Normal test set
test = dd.read_csv("outputcsvs/test_naive_pu/*.csv").compute()
test.to_csv("outputcsvs/test_naive_pu.csv", index=False)

## Read Test Data - Edge test set
test = dd.read_csv("outputcsvs/test_naive_pu_edge/*.csv").compute()
test = test[["clf","step1","recall","precision","f1_score","gmean","mcc","fit_time","eval_time","step1_time"]]
test.to_csv("outputcsvs/test_naive_pu_edge.csv", index=False)