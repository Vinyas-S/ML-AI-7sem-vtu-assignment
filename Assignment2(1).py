import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize

ps=PorterStemmer()
tokens=['connect','connecting','connection','connections']

for w in tokens:
    print(w+'-->'+ps.stem(w))
    
    

