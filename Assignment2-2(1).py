import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize


ps=PorterStemmer()
tokens=['consult','consulting','consulatation','consultative','consultant']

for w in tokens:
    print(w+'-->'+ps.stem(w))



