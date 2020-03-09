import pandas as pd
import numpy as np
import sanalytics.algorithms.utils as sau
from gensim.models.doc2vec import Doc2Vec
from progressbar import progressbar
import re
import random
from glob import glob

print("READING D2V")

## Read D2V Model
d2v = Doc2Vec.load("datasets/rq3_d2v/sec1.0R100.model")
## Read D2V Model All
d2vall = Doc2Vec.load("datasets/rq3_d2v/sec1.0R100_all.model")

print("LOADED D2V")

## Read files
while True:
    files = list(os.walk("datasets/rq3_dataR100"))[0][2]
    filename = random.sample(files,1)[0]

    if filename in set([".".join(i.split(".")[:-1]) for i in list(os.walk("datasets/rq3_vecdata_newR100"))[0][2]]):
        continue

    print("start {}".format(filename))
    X = pd.read_parquet("datasets/rq3_dataR100/{}".format(filename))
    if "all" not in filename:
        X["d2v"] = [d2v.infer_vector("{} {} {}".format(i.title, i.question, i.answers).split()) for i in progressbar(X.itertuples())]
    if "all" in filename:
        X["d2v"] = [d2vall.infer_vector("{} {} {}".format(i.title, i.question, i.answers).split()) for i in progressbar(X.itertuples())]
    X.to_parquet("datasets/rq3_vecdata_newR100/{}.vecs".format(filename), compression=None, index=False)
    print("saved")