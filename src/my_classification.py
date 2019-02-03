from sklearn import preprocessing
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict

output = pd.DataFrame(
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

output.to_csv("./../dist/EvaluationMetric_10fold.csv", sep='\t')


test_df = pd.read_csv("./../data/test_set.csv", sep = "\t")
df = pd.read_csv("./../data/train_set.csv", sep = "\t")
df['Content'] = df.Content.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))

X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['Category'], random_state = 1)
print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

X_scaled = preprocessing.scale(X_train)

print('\n\nBAG OF WORDS \n')
count_vector = CountVectorizer(stop_words = 'english')
training_data_bow = count_vector.fit_transform(X_scaled)
testing_data_bow = count_vector.transform(X_test)
randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
randomForestClassifier.fit(training_data_bow, y_train)
y_pred = randomForestClassifier.predict(testing_data_bow)
print(cross_val_score(randomForestClassifier, training_data_bow, y_train, cv=10))
print(cross_val_predict(randomForestClassifier, training_data_bow, y_train, cv=10))


