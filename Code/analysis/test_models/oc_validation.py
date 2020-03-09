# Import libraries
import pandas as pd
import numpy as np
import math
import time
import sys
import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, matthews_corrcoef
from sklearn.svm import OneClassSVM

from evaluation_metric import evaluate_metrics

d2v_model = None

# Convert document to feature vector
def doc_to_vec(sen):
	words = sen.split()
	return d2v_model.infer_vector(words)

# Perform cross-validation for a given classifier
def validate_data(clf, fold):
	X_train = np.load("../../data/feature/X_train_"+str(fold)+".npy", allow_pickle=True)
	X_test_pos = np.load("../../data/feature/X_test_pos_"+str(fold)+".npy", allow_pickle=True)
	X_test_all = np.load("../../data/feature/X_test_all_"+str(fold)+".npy", allow_pickle=True)

	# Train
	t_start = time.time()
	clf.fit(X_train)
	train_time = time.time() - t_start

	# Evaluate 
	t_start = time.time()
	y_pred_pos = clf.predict(X_test_pos)
	y_pred_all = clf.predict(X_test_all)
	results = evaluate_metrics(y_pred_pos, y_pred_all)
	val_time = time.time() - t_start

	return "{:.3f}".format(results[0]) + "," + "{:.3f}".format(results[1]) + "," + \
			"{:.3f}".format(results[2]) + "," + "{:.3f}".format(results[3]) + "," + \
			"{:.3f}".format(results[4]) + "," + "{:.3f}".format(results[5]) + "," + \
			"{:.3f}".format(train_time) + "," + "{:.3f}".format(val_time)
 
def tune_OCSVM(k, n, fold):
	print("Classifier: One-Class SVM")
	print("kernel,nu,fold,Recall,Precision Lower,Precision Opt,F-Score Lower,F-Score Opt,F-Measure,Train Time,Val Time")
	clf = OneClassSVM(kernel=k, nu=n, max_iter=-1)
	results = validate_data(clf, fold)
	print(k+','+str(n)+','+str(fold)+','+results)

# Test the optimal model found from the validation process, on all the data, using a 90:10 split
def test():
	# Load the data
	X_train = np.load("../../data/feature/X_train_test.npy", allow_pickle=True)
	X_test_pos = np.load("../../data/feature/X_test_pos_test.npy", allow_pickle=True)
	X_test_all = np.load("../../data/feature/X_test_all_test.npy", allow_pickle=True)

	# Optimal classifier settings
	clf = OneClassSVM(kernel="linear", nu=0.01, max_iter=-1)

	# Train
	t_start = time.time()
	clf.fit(X_train)
	train_time = time.time() - t_start

	# Evaluate 
	t_start = time.time()
	y_pred_pos = clf.predict(X_test_pos)
	y_pred_all = clf.predict(X_test_all)
	results = evaluate_metrics(y_pred_pos, y_pred_all)
	val_time = time.time() - t_start

	outfile = open("result/oc_test_results.txt", 'w')
	outfile.write("Classifier: One-Class SVM\n")
	outfile.write("Recall,Precision Lower,Precision Opt,F-Score Lower,F-Score Opt,F-Measure,Train Time,Val Time\n")
	outfile.write("{:.3f}".format(results[0]) + "," + "{:.3f}".format(results[1]) + "," + \
			"{:.3f}".format(results[2]) + "," + "{:.3f}".format(results[3]) + "," + \
			"{:.3f}".format(results[4]) + "," + "{:.3f}".format(results[5]) + "," + \
			"{:.3f}".format(train_time) + "," + "{:.3f}".format(val_time) + '\n')
	# Save predictions
	test = pd.read_parquet("../../data/datasets/test.parquet")
	test["predict"] = y_pred_all
	test.to_parquet("result/test_predictions.parquet", compression=None, index=False)


# Train and save an optimal model on all the available data
def train():
	# Load the data
	X_train = np.load("../../data/feature/X_train_all.npy", allow_pickle=True)

	# Optimal classifier settings
	clf = OneClassSVM(kernel="linear", nu=0.01, max_iter=-1)
	t_start = time.time()
	clf.fit(X_train)
	train_time = time.time() - t_start
	print(train_time)

	# Save the model
	pickle.dump(clf, open("model/ocsvm_all.model", 'wb'))

# Use the trained classifier to make predictions on manual datasets
def predict_holdout(case):
	global d2v_model

	# Load the classifier
	clf = pickle.load(open("model/ocsvm_all.model", 'rb'))

	# Load the input
	X = pd.read_parquet("../../data/datasets/X_"+case+".parquet")
	X_text = X.title + ' ' + X.question + ' ' + X.answers
	# Extract features of input data
	d2v_model = Doc2Vec.load("../../data/d2v_models/d2v_all.model")
	X_transformed = np.asarray(X_text.map(doc_to_vec).values.tolist())

	# Make predictions
	t_start = time.time() 
	y_pred = clf.predict(X_transformed)
	pred_time = time.time() - t_start
	y_true = np.ones(int(len(y_pred)/2))
	y_trueB = np.negative(np.ones(int(len(y_pred)/2)))
	y_true = np.concatenate((y_true, y_trueB))
	
	# Save predictions
	X["pred"] = y_pred
	X.to_parquet("result/"+case+"_predictions.parquet", compression=None, index=False)

	# Evaluate
	outfile = open("result/"+case+"_test.txt", 'w')
	outfile.write("Recall,Precision,F-Score,G-Mean,MCC,Pred Time\n")
	recall = recall_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	metrics = str(recall) + ',' + str(precision) + ',' + str(f1_score(y_true, y_pred)) + ',' + \
		str(math.sqrt(recall*precision)) + ',' + str(matthews_corrcoef(y_true, y_pred)) + ',' + str(pred_time)
	outfile.write(metrics + '\n')

if __name__ == '__main__':
	if sys.argv[1] == "val":
		# Input "kernel" "nu" "fold"
		tune_OCSVM(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
	elif sys.argv[1] == "test":
		test()
	elif sys.argv[1] == "train":
		train()
	elif sys.argv[1] == "predict":
		# Input holdout case <easy|edge>
		predict_holdout(sys.argv[2])
	else:
		print("USAGE: python3 oc_validation.py <val|test|train|predict> *")
