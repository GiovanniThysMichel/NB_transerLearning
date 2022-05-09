import os
import csv
import string
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
# A list of common english words which should not affect predictions
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone',
             'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount',
             'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around',
             'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
             'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both',
             'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
             'describe', 'detail', 'did', 'do', 'does', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg',
             'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone',
             'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for',
             'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
             'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
             'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less',
             'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
             'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine',
             'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
             'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming',
             'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 
             'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
             't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
             'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this',
             'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
             'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we',
             'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby',
             'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom',
             'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'you', 'your', 'yours', 'yourself',
             'yourselves']

X = [] # an element of X is represented as (filename, text)
Y = [] # an element of Y represents the category of the corresponding X element
#root_dir='text'
root_dir='email'
for category in os.listdir(root_dir):
    for document in os.listdir(root_dir+'/'+category):
        with open(root_dir+'/'+category+'/'+document, "r") as f:
            X.append((document,f.read()))
            Y.append(category)

print("there are %d messages/files\n %s " % (len(X),X[0:2]))
print("there are %d labels/files\n %s " % (len(Y),Y[0:1]))


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25)# Create vocabulary (using dictionary)

# Create vocabulary (using dictionary)
#topNum specify number of top words. If topNum=0, meaning selecting all words
def createVocabulary(inDataset, stopwordset,topNum):
    vocab = {}
    for i in range(len(inDataset)):
        word_list = []
        for word in inDataset[i][1].split():
            word_new  = word.strip(string.punctuation).lower()
            if (len(word_new)>2)  and (word_new not in stopwordset):
                if word_new in vocab:
                    vocab[word_new]+=1
                else:
                    vocab[word_new]=1
    # sort the dictionary to focus on most frequent words
    vocab_ordered=sorted(vocab.items(), key=lambda x: x[1],reverse=True)
    #import itertools
    if topNum==0:
        return(dict(vocab_ordered))
    elif topNum<len(vocab):
        V_cut=vocab_ordered[0:topNum]
        V_cut=dict(V_cut)
        return(V_cut)
    return(dict(vocab_ordered))

#stopwords=[]
V=createVocabulary(X_train,stopwords,2000)
len(V)
V

num_words = [0 for i in range(max(V.values())+1)]
freq = [i for i in range(max(V.values())+1)]
for key in V:
    num_words[V[key]]+=1
maxv=max(num_words)+10
plt.plot(freq,num_words)
plt.axis([1, 20, 0, maxv])
plt.xlabel("Frequency")
plt.ylabel("No of words")
plt.grid()
plt.show()

# To represent training data as bog of words vector representaton (including counts)
def BoWInstances(inDataset,features):
    inDataset_ = np.zeros((len(inDataset),len(features)))
    # This can take some time to complete
    for i in range(len(inDataset)):
        # print(i) # Uncomment to see progress
        word_list = [ word.strip(string.punctuation).lower() for word in inDataset[i][1].split()]
        for word in word_list:
            if word in features:
                inDataset_[i][features.index(word)] += 1
    return(inDataset_)

# To represent test data as bag of word vector counts
features = list(V.keys())
X_train_dataset=BoWInstances(X_train,features)
X_test_dataset = BoWInstances(X_test,features)

# Using sklearn's Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_dataset,Y_train)
Y_test_pred = clf.predict(X_test_dataset)
sklearn_score_train = clf.score(X_train_dataset,Y_train)
print("Sklearn's score on training data :",sklearn_score_train)
sklearn_score_test = clf.score(X_test_dataset,Y_test)
print("Sklearn's score on testing data :",sklearn_score_test)
print("Classification report for testing data :-")
print(classification_report(Y_test, Y_test_pred))


# Implementing Multinomial Naive Bayes from scratch
class MultinomialNaiveBayes:

    def __init__(self):
        # count is a dictionary which stores several dictionaries corresponding to each news category
        # each value in the subdictionary represents the freq of the key corresponding to that news category
        self.count = {}
        # classes represents the different news categories
        self.classes = None

    def fit(self, X_train, Y_train):
        # This can take some time to complete
        self.classes = set(Y_train)
        for class_ in self.classes:
            self.count[class_] = {}
            for i in range(len(X_train[0])):
                self.count[class_][i] = 0
            self.count[class_]['total'] = 0
            self.count[class_]['total_points'] = 0
        self.count['total_points'] = len(X_train)

        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                self.count[Y_train[i]][j] += X_train[i][j]
                self.count[Y_train[i]]['total'] += X_train[i][j]
            self.count[Y_train[i]]['total_points'] += 1

    def __probability(self, test_point, class_):

        log_prob = np.log(self.count[class_]['total_points']) - np.log(self.count['total_points'])
        total_words = len(test_point)
        for i in range(len(test_point)):
            current_word_prob = test_point[i] * (
                        np.log(self.count[class_][i] + 1) - np.log(self.count[class_]['total'] + total_words))
            log_prob += current_word_prob

        return log_prob

    def __predictSinglePoint(self, test_point):

        best_class = None
        best_prob = None
        first_run = True

        for class_ in self.classes:
            log_probability_current_class = self.__probability(test_point, class_)
            if (first_run) or (log_probability_current_class > best_prob):
                best_class = class_
                best_prob = log_probability_current_class
                first_run = False

        return best_class

    def predict(self, X_test):
        # This can take some time to complete
        Y_pred = []
        for i in range(len(X_test)):
            # print(i) # Uncomment to see progress
            Y_pred.append(self.__predictSinglePoint(X_test[i]))

        return Y_pred

    def score(self, Y_pred, Y_true):
        # returns the mean accuracy
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                count += 1
        return count / len(Y_pred)

clf2 = MultinomialNaiveBayes()
clf2.fit(X_train_dataset,Y_train)
Y_test_pred = clf2.predict(X_test_dataset)
our_score_test = clf2.score(Y_test_pred,Y_test)
print("Our score on testing data :",our_score_test)
print("Classification report for testing data :-")
print(classification_report(Y_test, Y_test_pred))


# now we are loading a target dataset which is Short messages (SMS). We will try to use previous trained email
# spam filter to classify these short messages.
import csv
X_target = []
Y_target=[]

with open('sms.csv') as f:
    X_target = [line.rstrip() for line in f]

with open('labels.csv') as f:
    Y_target = [line.rstrip() for line in f]

print(X_target[0:5],Y_target[0:5])

len(X_target),len(Y_target)
len(X_target),len(Y_target)

# To represent training data as bog of words vector representaton (including counts).
# short messages dataset has different format from the email. So the BoWInstances_ function needs to be revised.
def BoWInstances_s(inDataset,features):
    inDataset_ = np.zeros((len(inDataset),len(features)))
    # This can take some time to complete
    for i in range(len(inDataset)):
        # print(i) # Uncomment to see progress
        word_list = [ word.strip(string.punctuation).lower() for word in inDataset[i].split()]
        for word in word_list:
            if word in features:
                inDataset_[i][features.index(word)] += 1
    return(inDataset_)

X_target_dataset=BoWInstances_s(X_target,features)
# Using sklearn's Multinomial Naive Bayes to classify this SMS datasets
Y_test_pred = clf.predict(X_target_dataset)
sklearn_score_test = clf.score(X_target_dataset,Y_target)
print("Sklearn's score on testing data :",sklearn_score_test)
print("Classification report for testing data :-")
print(classification_report(Y_target, Y_test_pred))