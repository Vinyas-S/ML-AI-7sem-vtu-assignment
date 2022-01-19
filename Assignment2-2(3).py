
import nltk
import re
import numpy as np
import heapq


text = "With the scale and scope we utilize on the internet, defining ownership of content is very difficult. Diagrams and sketches from research papers depict a lot of information about the work done by the author. The flow of working and logic is easily explained visually, rather than in text. Many engineering and technology problems are solved with the help of diagrams and tables. Due to vast widespread use of the internet worldwide plagiarism is seen everywhere. There are a few plagiarism checking systems for text, but plagiarism for images has not been explored much. Images such as tables of rows and columns, on comparison would say there is copyright infringement even if the data in it is completely different. Verifying and prevention of infringement is not something that has been attempted."
dataset = nltk.sent_tokenize(text)
for i in range(len(dataset)):
	dataset[i] = dataset[i].lower()
	dataset[i] = re.sub(r'\W', ' ', dataset[i])
	dataset[i] = re.sub(r'\s+', ' ', dataset[i])
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
freq_words = heapq.nlargest(50, word2count, key=word2count.get)
            
X = []
for data in dataset:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
X = np.asarray(X)