from sklearn import preprocessing
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from gensim.models import doc2vec
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix,  auc

from scipy.linalg import svd

from gensim.test.utils import common_texts
from gensim.sklearn_api import D2VTransformer

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# ------------------------ OUTPUT ------------------------ #
EvaluationMetric_10fold = pd.DataFrame(
    columns = [
        "SVM (BoW)",
        "RANDOM FOREST (BoW)",
        "SVM (SVD)",
        "RANDOM FOREST (SVD)",
        "SVM (W2V)",
        "RANDOM FOREST (W2V)",
        "MY METHOD"
    ],

    index = [
        "Accuracy",
        "Precision",
        "Recall",
        "F-Measure",
        "AUC"
    ]
)

testSet_categories = pd.DataFrame(
    columns=[
        "Test_Document_ID",
        "Predicted_Category"
    ]
)

# ------------------------ PREPARING THE 10-FOLD VALIDATION ------------------------ #
kfold = KFold(n_splits=10, random_state=42)
scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

test_df = pd.read_csv("./../data/test_set.csv", sep = "\t")
df = pd.read_csv("./../data/train_set.csv", sep = "\t")
df['Content'] = df.Content.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))
X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['Category'], random_state = 1)

# ------------------------ FEATURES ------------------------ #
count_vector = CountVectorizer(stop_words = 'english')
svd = TruncatedSVD(n_components = 90, algorithm = 'arpack')
doc2vec = D2VTransformer(min_count=1, size=5)

# ------------------------ CLASSIFIERS ------------------------ #
randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
svclassifier = SVC(kernel='linear')

training_data_bow = count_vector.fit_transform(X_train)
testing_data_bow = count_vector.fit_transform(X_test)
# randomForestClassifier.fit(training_data_bow, y_train)
# y_pred = randomForestClassifier.predict(testing_data_bow)

training_data_svd = svd.fit_transform(X_train)
testing_data_svd = svd.fit_transform(X_test)
# randomForestClassifier.fit(training_data_svd, y_train)
# y_pred = randomForestClassifier.predict(testing_data_svd)

training_data_d2v = doc2vec.fit_transform(X_train)
testing_data_d2v = doc2vec.fit_transform(X_test)
# randomForestClassifier.fit(training_data_d2v, y_train)
# y_pred = svclassifier.predict(testing_data_d2v)

data_mine = preprocessing.scale(X_train)
training_data_bow = count_vector.fit_transform(data_mine)
testing_data_bow = count_vector.transform(X_test)
randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
# randomForestClassifier.fit(training_data_bow, y_train)
# y_pred = randomForestClassifier.predict(testing_data_bow)

print(cross_val_score(randomForestClassifier, training_data_bow, y_train, cv=10))
print(cross_val_predict(randomForestClassifier, training_data_bow, y_train, cv=10))
print(cross_val_score(RandomForestClassifier, training_data_svd, y_train, cv=2))
print(cross_val_predict(RandomForestClassifier, training_data_svd, y_train, cv=10))
print(cross_val_score(RandomForestClassifier, training_data_d2v, y_train, cv=10))
print(cross_val_predict(RandomForestClassifier, testing_data_d2v, y_train, cv=10))

print(cross_val_score(svclassifier, training_data_bow, y_train, cv=10))
print(cross_val_predict(svclassifier, training_data_bow, y_train, cv=10))
print(cross_val_score(svclassifier, training_data_svd, y_train, cv=2))
print(cross_val_predict(svclassifier, training_data_svd, y_train, cv=10))
print(cross_val_score(svclassifier, training_data_d2v, y_train, cv=10))
print(cross_val_predict(svclassifier, testing_data_d2v, y_train, cv=10))

print(cross_val_score(randomForestClassifier, training_data_bow, y_train, cv=10))
print(cross_val_predict(randomForestClassifier, training_data_bow, y_train, cv=10))

# ------------------------ OUTPUTTING ------------------------ #
EvaluationMetric_10fold.to_csv("./../dist/EvaluationMetric_10fold.csv", sep='\t')
testSet_categories.to_csv("./../dist/testSet_categories.csv", sep='\t')

# PLOT
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()