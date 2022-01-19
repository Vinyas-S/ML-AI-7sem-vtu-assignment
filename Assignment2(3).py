
import nltk
import re
import numpy as np
import heapq


text = "Intellectual property validation over the internet for new content especially in the domain of academia is very robust. But, they have limitations in forms of not being open source or openly available, being a too narrow domain or a too broad range problem, being restricted to only certain formats of data, or many other ethical concerns that throw out a wide range of exceptional cases that need to be handled outside the generic checking protocols. Although text based plagiarism checks available online are prominent and efficient, the modern output of data on research or anything in general is more dominantly media content like images, videos, audio or a combination of these formats of data. Multimedia alone has the potential to generate revenue. Duplication of content over the internet is something that is impossible to stop, but if the validation of the owner or creator of the content is properly recognised, then it will be much easier to recognise and support the appropriate content/creator. Such systems are already available as video copyright check on YouTube or stream copy-strike or copyright check on Twitch, but they are limited to only their domain of media i.e. video. To have a general or universal check of content, it is important to observe a given piece of data in all the formats and using different perspective learners. This project aims at creating a copyright infringement checker/ plagiarism tokenizer for multimedia content making use of, Natural Language, Machine Learning Processing, Deep Learning, Fuzzy Logic, Big Data and Analytics for checking the authenticity of the data and Blockchain Technologies, and Cryptography for assigning a unique identification token for such content"
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
            
freq_words = heapq.nlargest(100, word2count, key=word2count.get)
            
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