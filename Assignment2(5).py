import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import string
import seaborn as sns

df = pd.read_csv("smsspamcollection/SMSSpamCollection", sep="\t", names=["label","message"])
df.head(2)


df = df.rename(columns={"v1":"label", "v2":"message"})
print(df.label.value_counts())

df['length'] = df['message'].apply(len)
print(df.head())

df['length'].plot(bins=50, kind='hist')
df.hist(column='length', by='label', bins=50,figsize=(10,4))
df.loc[:,'label'] = df.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df['message'], 
                                                    df['label'],test_size=0.20, 
                                                    random_state=1)

count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)

testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)

predictions = naive_bayes.predict(testing_data)
print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))
print('Precision score: {}'.format(precision_score(y_test, predictions)))
print('Recall score: {}'.format(recall_score(y_test, predictions)))
print('F1 score: {}'.format(f1_score(y_test, predictions)))