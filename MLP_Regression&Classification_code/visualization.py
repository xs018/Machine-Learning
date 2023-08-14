import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

res = pd.read_csv('result.csv').values
y_true = res[:, 1]
y_pred = res[:, 0]
y_prob = res[:, 2]

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel() # where 1 is positive, 0 is negative
print(f"True Negative: {tn}, False Positive: {fp}, False Negative: {fn}, True Postive: {tp}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['post-CHF(0)', 'pre-CHF(1)'])
disp.plot()
plt.savefig('res/hw1_2confusion_matrix.png')

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.savefig('res/hw1_2ROC.png')

prec, recall, _  = precision_recall_curve(y_true, y_prob)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
plt.savefig('res/hw1_2confusion_prediction.png')

print(f"Area Under Curve: {auc(fpr, tpr)}")
print(f"Accuracy: {(tp+tn) / (tn + fp + fn+ tp)}")
print(f"Precision: {(tp) / ( fp +  tp)}")
print(f"Recall: {(tp) / ( fn +  tp)}")
print(f"F1 Score: {tp / (tp + (fn + fp)/2)}")

