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
import random
import sanalytics.evaluation.evaluation_metric as see
import sanalytics.visualisations.pca as pca
from collections import Counter
from sklearn.decomposition import PCA

## Read classifiers
best_clfs = pd.read_pickle("datasets/best_clfs.pd.pkl")

## 90%
x = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "all" not in i and "test" not in i]
x_test = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "all" not in i and "test" in i]
x_easy = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "test" in i and "easy" in i]
x_edge = ["datasets/rq3_vecdata_newR100/"+i for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2] if "test" in i and "edge" in i]
df = pd.concat([pd.read_parquet(i)[["label","d2v"]] for i in progressbar(x)])
df_test = pd.concat([pd.read_parquet(i)[["label","d2v"]] for i in progressbar(x_test)])
df_edge = pd.read_parquet(x_edge[0])
df_easy = pd.read_parquet(x_easy[0])

## Testing splits
pset = np.array([np.array(i) for i in df_test[df_test.label == "security"].d2v])
puset = np.array([np.array(i) for i in df_test.d2v])

## Sample training data
sampledf = pd.concat([df[df.label=="security"].sample(100000), df[df.label=="unlabelled"].sample(500000)])

## PCA Plot
pca = PCA(n_components=2).fit([np.array(i) for i in df.d2v], df.label)
pca_tf = pca.transform([np.array(i) for i in df.d2v])
plotdf = pd.DataFrame([(i[0], i[1][0], i[1][1]) for i in zip(df.label, pca_tf)])
plotdf.columns=["class","x","y"]
import seaborn as sns

# Use the 'hue' argument to provide a factor variable
plot = sns.lmplot( x="x", y="y", data=plotdf, hue="class")