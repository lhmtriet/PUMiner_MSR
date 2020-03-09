import pandas as pd
import itertools

step1 = { 'alpha': [0.8, 0.9, 1, 1.1, 1.2] }

## Classifiers
step2 = {
    'BLANK': {
        'clf': ['NC']
    },
    'KNN': {
        'n_neighbors': [11, 31, 51],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'LR': {
        'C': [0.01, 0.1, 1, 10, 100],
    },
    'SVM': {
        'C': [0.01, 0.1, 1, 10, 100],
    },
    'RF': {
        'n_estimators': [100, 300, 500],
        'max_depth': [None],
        'max_leaf_nodes': [100, 200, 300, None],
    },
    'XGB': {
        'n_estimators': [100, 300, 500],
        'max_depth': [0],
        'max_leaves': [100, 200, 300],
    },
    'LGBM': {
        'n_estimators': [100, 300, 500],
        'max_depth': [-1],
        'num_leaves': [100, 200, 300],
    },
}

def abbreviate(str):
    return ''.join([i[0].upper() for i in str.split("_")])

def params2str(d):
    ret = []
    for i in d.items():
        ret += [abbreviate(i[0])] + [str(i[1])]
    return '_'.join(ret)

def get_combinations(d):
    keys, values = zip(*d.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments

full = []
for fold in range(10):
    for putype in ["0"]:
        for p1 in get_combinations(step1):
            for clf in step2.items():
                for p2 in get_combinations(clf[1]):
                    full += [("F{}_{}_{}_{}_{}".format(fold, putype, clf[0], params2str(p1), params2str(p2)), fold, putype, clf[0], p1, p2)]

full = pd.DataFrame(full, columns=["id","fold","putype","clf","params_1","params_2"])
full.set_index("id").to_pickle("analysis/pu_learning/foldinfo.pkl")
full[["id"]].to_csv("analysis/pu_learning/folds.csv", index=False, header=None)