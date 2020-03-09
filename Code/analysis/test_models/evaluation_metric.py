import numpy as np

# Calculated from unlabelled sample
security_ratio = 0.025	# 2.5%

# Calculate the recall of the positive set
# Input: A numpy array of the predictions on the positive set
def recall_score(y_pred_pos):
	return np.count_nonzero(y_pred_pos==1) / len(y_pred_pos)

# Calculate a lower bound approximation of the precision
# Input: A numpy array of the predictions on the positive and total set
def precision_lower_score(y_pred_pos, y_pred_all):
	return np.count_nonzero(y_pred_pos==1) / np.count_nonzero(y_pred_all==1)

# Calculate an optimal of the precision, using the ratio of security posts in the unlabelled data
# Input: A numpy array of the predictions on the positive and total set
def precision_opt_score(y_pred_pos, y_pred_all):
	size_unlabelled = len(y_pred_all) - len(y_pred_pos)
	tp_p = np.count_nonzero(y_pred_pos==1)
	p = np.count_nonzero(y_pred_all==1)
	return (tp_p+min(security_ratio*size_unlabelled, p-tp_p))/p

# Calculate an approximation of the f1-score from the estimated recall and precision
# Input: The estimated recall and precision
def f1_score(recall, precision):
	return (2*recall*precision) / (recall + precision)

# Calculate the "f-measure", an indicator of the f1-score
# Input: A numpy array of the predictions on the positive set
def f_measure(recall, y_pred_all):
	proba = np.count_nonzero(y_pred_all==1) / len(y_pred_all)
	return (recall**2)/proba

# Calculate recall, approx. precision, approx. f1, and "f-measure" for PU learning
# Input 1: an array-like object of the predicted values in the positive set. 
# 1 = postive, 0 = negative
# e.g. [1, 0, 1, 1]
# Input 2: an array-like object of the predicted values in the total set (positive and unlabelled).
# Output: Recall, Precision Lower Bound, Precision Optimal, F1 Lower Bound, F1 Optimal, F-Measure (true metric)
def evaluate_metrics(y_pred_pos, y_pred_all):
	# Ensure data type uniformity
	y_pred_pos = np.asarray(y_pred_pos)
	y_pred_all = np.asarray(y_pred_all)

	recall = recall_score(y_pred_pos)
	precision_lower = precision_lower_score(y_pred_pos, y_pred_all)
	precision_opt = precision_opt_score(y_pred_pos, y_pred_all)

	return recall, precision_lower, precision_opt, f1_score(recall, precision_lower), f1_score(recall, precision_opt), f_measure(recall, y_pred_all)