import itertools
import pandas as pd

classifiers = ["RF", "LR", "LGBM", "BLANK", "XGB", "SVM", "KNN"]
step1types = ["naive", "pu"]

allcombos = ["{}_all".format('_'.join(i)) for i in itertools.product(classifiers,step1types)]
allcombos = [i for i in allcombos if i != "BLANK_pu_all"]

pd.DataFrame(allcombos).to_csv("analysis/test_models/combs.csv", index=False, header=None)