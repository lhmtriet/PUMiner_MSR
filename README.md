This is the README file for the reproduction package of the paper: "PUMiner: Mining Security Posts from Developer Question and Answer Websites with PU Learning", accepted for publication at the 17th [Mining Software Repositories](https://2020.msrconf.org/details/msr-2020-papers/4/PUMiner-Mining-Security-Posts-from-Developer-Question-and-Answer-Websites-with-PU-Le), 2020. Preprint: https://arxiv.org/abs/2003.03741

The package contains the following folders:
1. Data: contains the preprocessed dataset we used in our work.
2. Code: contains the source code we used in our work.

Due to the large size of the data, please refer to this link to download the data in the Data folder: https://figshare.com/s/fad6dd2d93dbed0d80d4

It is noted that R10 (meaning security, unlabelled ratio is 1 to 10) and R100 do not contain the raw text, and only contain the post id to save storage space.

Before running any code, please install all the required Python packages using the following command: pip install -r Code/requirements.txt

The details of data and source code used for each of the three Research Questions (RQs) are given as follows:

### Research Question 1: Is PUMiner effective for mining security posts on developer Q&A websites?

+ Data used: r1_train.parquet. More specifically, r1_train.parquet for selecting the optimal PU Model.

+ Code used:

1. `Code/analysis/job_array_doc2vec` trains a doc2vec model for each separate fold. This is submitted through a Slurm job array.
2. `Code/analysis/infer_all_folds.py` uses the corresponding doc2vec models to infer vectors for each of the folds, and saves them out into separate files.
3. `Code/analysis/pu_learning/create_folds.py` creates the plaintext combinations given the classifier hyperparameters. Classifiers are found in the module `sanalytics.estimators.pu_estimators.py`, including the code for step 1 of PU learning (nearest centroid + alpha tuning)
4. `Code/analysis/pu_learning/pu_learning.sh` is the job script for SLURM that submits jobs running `analysis/pu_learning/pu_learning.py` using the vectors inferred in step 2.

### Research Question 2: How does PUMiner perform compared to other learning approaches for mining security posts on developer Q&A websites?

+ Data used: r1_test.parquet, r1_train.parquet, test_hard.parquet, test_normal.parquet. More specifically, r1_train.parquet is used to train the optimal PUMiner model. Then, such model is evaluated on 10% of the development set (i.e., r1_test.parquet). test_normal.parquet and test_hard.parquet are used for evaluating PUMiner on normal and hard (keyword-matching method fails) test cases.

+ Code used:

1. `Code/analysis/test_models/oc_validation.py` is the code used to train, validate and test one class svm models. It has 4 modes of usage based on command line arguments: 
 - "val": Validation process. Run as a job array. Also requires the kernel ("linear" or "rbf"), value of nu, and the fold number (0-9) to be specified as additional command line arguments. 
 - "test": Testing process on a 90/10 train test split of the data. 
 - "train": Train the ocsvm model on 100% of the data. 
 - "predict": Make predictions on a holdout testset using a trained model. Specifiy the name of the testset ("easy" or "hard") as an additional commad line argument. 
2. `Code/analysis/psf_baseline/` is used to train, validate and test "positive-similarity filtering" baseline model using Doc2Vec.
 - `train_folds.sh` submits a job script using the corresponding python file to train Doc2Vec models for each of the 10 folds using only positive samples.
 - `train.py` trains Doc2Vec models for testing on all data using only positive samples.
 - `validation.sh` submits a job script using the corresponding python file to perform cross-validation. Tunes on thresholds 0.5, 0.6, 0.7, 0.8, 0.9.
 - `predict.sh` submits a job script using the corresponding python file to evaluate the regular holdout test set. Tunes on thresholds 0.5, 0.6, 0.7, 0.8, 0.9.
 - `test.py` evaluates the Doc2Vec model from `train.py` on the hard and normal test sets.
3. `Code/analysis/test_models/oc_jobscript.sh` is an example jobscript used to run the one-class svm model and contains the configuration parameters. 
4. `Code/analysis/test_models/create_combinations.py` is the code used to create a list of plaintext combinations of models to test. One-stage only PU learning is referred to as BLANK (i.e. blank classifier for step 2).
5. `Code/analysis/test_models/test_models.sh` is the job script for the corresponding python file. It uses the best classifiers found in RQ1 to obtain evaluation scores based on the test sets.
6. `Code/analysis/test_models/separate_test_models.sh` is the job script for the corresponding python file. It uses the best classifiers found in RQ1 to obtain evaluation scores based on the hard/normal sets.
7. `Code/analysis/test_models/results.py` is used to read to results.

### Research Question 3: How do the source and size of training data affect the performance of PUMiner?

+ Data used: r1_test.parquet, r1_train.parquet, r10_ids.parquet, r100_ids.parquet, test_hard.parquet, test_normal.parquet. r10_ids.parquet and r100_ids.parquet contain the ids of the posts selected in R10 and R100 test cases, respectively. The other ones have been explained in above in RQ1 and RQ2.

+ Code used:

1. `Code/analysis/rq3_datasets.py` creates subsets of the dataset to be used for the purposes of research question 3.
2. `Code/analysis/kfold_splits.py` is used to generate R10 and R100 data by changing the variable in the file, nonsec_sec_ratio to 10 and 100 respectively.
3. `Code/analysis/job_array_rq3/traind2v/traind2v.py` is used to train the doc2vec models for all the set variations (security ratio 0.1, 0.2, ..., 1, set1 only, set2 only, sse only, so only), and R10.
4. `Code/analysis/train_d2v_r100/temp_train_d2v.sh` is the job array script used to train doc2vec models for R100.
5. `Code/analysis/job_array_rq3/infervecs/infer_vecs.sh` and corresponding .py, .csv files are used to infer vectors for all set variations + R10, and saves them into a new folder, `rq3_vecdata_new`
6. `Code/analysis/job_array_rq3/infervecs/infer_split_large_r100.sh` is the same as `infer_vecs.sh` in step 5, but for R100 (large). This is done by operating on data that is split into multiple files using `analysis/job_array_rq3/infervecs/r100_split.py`
7. `Code/analysis/job_array_rq3/testmodels/test_models.py` is used to obtain train models and test results (hard and normal sets) for all set variations + R10 with all models.
8. `Code/analysis/job_array_rq3/infervecs/train_r100.py` is used to train PU models using R100 data
9. `Code/analysis/job_array_rq3/testmodels/testing_r100_vecdata.py` is used to obtain test results (hard, normal) using the R100-trained PU models.

Finally, the file `Data/SecPosts_PUMiner.parquet` contains 104,024 security posts identified by the optimal PUMiner model and our heuristics (kw_count >= 5 and kw_ratio >= 0.03) described in the Discussion section VII of the paper.
