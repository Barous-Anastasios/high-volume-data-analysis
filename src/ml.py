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
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix,  auc
from scipy.linalg import svd
from gensim.sklearn_api import D2VTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import scikitplot as skplt

# ------------------------------------------------ OUTPUT ------------------------------------------------ #
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

# ------------------------------------------------ PREPARING THE 10-FOLD VALIDATION ------------------------------------------------ #
kfold = KFold(n_splits=10, random_state=42)
scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

test_df = pd.read_csv("./../data/test_set.csv", sep = "\t")
df = pd.read_csv("./../data/train_set.csv", sep = "\t")
df['Content'] = df.Content.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))
X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['Category'], random_state = 1)

# ------------------------------------------------ FEATURES ------------------------------------------------ #
count_vector = CountVectorizer(stop_words = 'english')
# svd = TruncatedSVD(n_components = 90, algorithm = 'arpack')
doc2vec = D2VTransformer(min_count=1, size=5)

# ------------------------------------------------ CLASSIFIERS ------------------------------------------------ #
randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
svclassifier = SVC(kernel='linear')

# BAG OF WORDS
training_data_bow = count_vector.fit_transform(X_train)
testing_data_bow = count_vector.fit_transform(X_test)

## SVD
# training_data_svd = svd(X_train, full_matrices=False)
# testing_data_svd = svd(X_test, full_matrices=False)

## DOC 2 VECTOR
# training_data_d2v = doc2vec.fit_transform(X_train)
# testing_data_d2v = doc2vec.fit_transform(X_test)

## CUSTOM
# data_mine = preprocessing.scale(X_train)
# training_data_bow_mine = count_vector.fit_transform(data_mine)
# testing_data_bow_mine = count_vector.transform(X_test)
#
# print(cross_val_score(randomForestClassifier, training_data_bow, y_train, cv=2))
# print(cross_val_predict(randomForestClassifier, training_data_bow, y_train, cv=2))
# # print(cross_val_score(RandomForestClassifier, training_data_svd, y_train, cv=2))
# # print(cross_val_predict(RandomForestClassifier, training_data_svd, y_train, cv=10))
# # print(cross_val_score(RandomForestClassifier, training_data_d2v, y_train, cv=10))
# # print(cross_val_predict(RandomForestClassifier, testing_data_d2v, y_train, cv=10))
#
# print(cross_val_score(svclassifier, training_data_bow, y_train, cv=2))
# print(cross_val_predict(svclassifier, training_data_bow, y_train, cv=2))
# print(cross_val_score(svclassifier, training_data_svd, y_train, cv=2))
# print(cross_val_predict(svclassifier, training_data_svd, y_train, cv=10))
# print(cross_val_score(svclassifier, training_data_d2v, y_train, cv=10))
# print(cross_val_predict(svclassifier, testing_data_d2v, y_train, cv=10))

# print(cross_val_score(randomForestClassifier, training_data_bow_mine, y_train, cv=10))
# print(cross_val_predict(randomForestClassifier, training_data_bow, y_train, cv=10))

# ------------------------------------------------ OUTPUTTING ------------------------------------------------ #
for index, item in enumerate(cross_val_predict(randomForestClassifier, training_data_bow, y_train, cv=2)):
    testSet_categories.loc[-1] = [index, item]
    testSet_categories.index = testSet_categories.index + 1
    testSet_categories = testSet_categories.sort_index()
    
EvaluationMetric_10fold.to_csv("./../dist/EvaluationMetric_10fold.csv", sep='\t')
testSet_categories.to_csv("./../dist/testSet_categories.csv", sep='\t')

# ------------------------------------------------ PLOTTING ------------------------------------------------ #
# randomForestClassifier.fit(training_data_bow, y_train)
# predicted_probas = randomForestClassifier.predict_proba(testing_data_bow)
# skplt.metrics.plot_roc_curve(y_test, predicted_probas)
# plt.show()
