import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from gensim.models import doc2vec
from sklearn.decomposition import TruncatedSVD

# Not needed yet
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, auc

test_df = pd.read_csv("./../data/test_set.csv", sep = "\t")
df = pd.read_csv("./../data/train_set.csv", sep = "\t")
df['Content'] = df.Content.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))

X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['Category'], random_state = 1)
print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

# BAG OF WORDS
print('\n\nBAG OF WORDS \n')
count_vector = CountVectorizer(stop_words = 'english')
training_data_bow = count_vector.fit_transform(X_train)
testing_data_bow = count_vector.transform(X_test)
randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
randomForestClassifier.fit(training_data_bow, y_train)
y_pred = randomForestClassifier.predict(testing_data_bow)
print(cross_val_score(randomForestClassifier, training_data_bow, y_train, cv=10))
print(cross_val_predict(randomForestClassifier, training_data_bow, y_train, cv=10))

# SVD
print('\n\nSINGULAR VALUE DECOMPOSITION \n')
svd = TruncatedSVD(n_components = 90, algorithm = 'arpack')
training_data_svd = svd.fit_transform(X_train)
testing_data_svd = svd.transform(X_test)
randomForestClassifier = RandomForestClassifier(kernel='linear')
randomForestClassifier.fit(training_data_svd, y_train)
y_pred = randomForestClassifier.predict(testing_data_svd)
print(cross_val_score(RandomForestClassifier, training_data_svd, y_train, cv=10))
print(cross_val_predict(RandomForestClassifier, training_data_svd, y_train, cv=10))

# DOC TO VECTOR
print('\n\nDOC TO VEC \n')
training_data_dtv = doc2vec.Doc2Vec(X_train, size = 100, window = 300, min_count = 1, workers = 4)
testing_data_dtv = doc2vec.Doc2Vec(X_test, size = 100, window = 300, min_count = 1, workers = 4)
svclassifier = RandomForestClassifier(kernel='linear')
svclassifier.fit(training_data_dtv, y_train)
y_pred = svclassifier.predict(testing_data_dtv)
print(cross_val_score(RandomForestClassifier, training_data_dtv, y_train, cv=10))
print(cross_val_predict(RandomForestClassifier, training_data_dtv, y_train, cv=10))



