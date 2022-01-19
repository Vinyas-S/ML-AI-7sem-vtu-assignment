from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize

lemmatizer=WordNetLemmatizer()
input="there are several types of stemming algorithms"
input=word_tokenize(input)
for word in input:    
    print(lemmatizer.lemmatize(word))
