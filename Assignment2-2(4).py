
from sklearn.feature_extraction.text import TfidfVectorizer

d0 = 'I like dog as a pet'
d1 = 'Dog'
d2 = 'world'

string = [d0, d1, d2]
tfidf = TfidfVectorizer()

result = tfidf.fit_transform(string)


print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names(), tfidf.idf_):
	print(ele1, ':', ele2)


print('\nWord indexes:')
print(tfidf.vocabulary_)


print('\ntf-idf value:')
print(result)


print('\ntf-idf values in matrix form:')
print(result.toarray())
