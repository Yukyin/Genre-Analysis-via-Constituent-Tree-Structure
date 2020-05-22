#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:57:38 2020

@author: morischick
"""
import nltk
from nltk.tree import Tree
from nltk.corpus import ptb
import os, random
import pickle
import nltk.metrics
import collections


def with_internal_labels(parsed_sents):
    """
    input: tree.parsed_sents(sentence)
    returns: String
    """
    encoding = ""
    for node in parsed_sents.subtrees():
        #print(node.label(), node.leaves())
        # if it is a leaf node
        if len(node) == 1 and not isinstance(node[0], Tree):
            encoding += "1 "
            encoding += node.label()
            encoding += " "
        # if it is not a leaf node
        else:
            encoding += "0 "
    return encoding
    
def without_internal_labels(parsed_sents):
    """
    input: tree.parsed_sents(sentence)
    returns: String
    """
    encoding = ""
    for node in parsed_sents.subtrees():
        # if it is a leaf node
        if len(node) == 1 and not isinstance(node[0], Tree):
            encoding += "1 "
        # if it is not a leaf node
        else:
            encoding += "0 "
    return encoding

def get_productions(parsed_sent):
    d = {}
    for p in parsed_sent.productions():
        right_side = p.rhs()
        # if a production leads to more than 1 node, it must not be a terminal
        # if a production has 1 node it must be a terminal
        # we will only include non-terminal productions
        if len(right_side) > 1:
            #print(right_side, len(right_side))
            #parts = str(p).split('->')
            #print(parts)
            if str(p) in d:
                d[str(p)] += 1
            else:
                d[str(p)] = 1
    for key in d.keys():
        feature_dict[key] = d[key]

def get_sentence_length(parsed_sent):
    return len(parsed_sent.leaves())

def get_subtrees(parsed_sent):
    subtree_dict = {}
    for tree in parsed_sent.subtrees():
        encoding = without_internal_labels(tree)
        if encoding in subtree_dict:
            subtree_dict[encoding] += 1
        else:
            subtree_dict[encoding] = 1
    for key in subtree_dict.keys():
        feature_dict[key] = subtree_dict[key]

def get_height(t):
    return t.height()

# get genre of each file
print("Getting genre for each file...")
fin = open('/Users/morischick//nltk_data/corpora/ptb/allcats.txt').readlines()
all_genres = ['news', 'lore', 'belles_lettres', 'fiction', 'mystery', 'science_fiction', 'adventure', 'romance', 'humor']
genreCount = {'news': 0, 'lore': 0, 'belles_lettres': 0, 'fiction': 0, 'mystery': 0, 'science_fiction': 0, 'adventure': 0, 'romance': 0, 'humor': 0}   
# genreCount --> key: genre,        value: count of how many seen (should stop at 881)
genre_dict = {}   # key: file_name,    value: genre
feature_set = []  # list of tuples (feature_set, genre) where feature_set is the feature vector for each sentence
NUM_EXAMPLES = 881

for line in fin:
    line = line.replace("\n", "")
    line = line.split()
    file_name = line[0]
    genre = line[1]
    genre_dict[file_name] = genre

print("Genres loaded successfully.")
print("Adding WSJ corpus to feature set...")

try:
    pickled_feature_set = open('pickled_feature_set', 'rb')
    feature_set = pickle.load(pickled_feature_set)
    pickled_feature_set.close()
except Exception as e:
    print(e)

if len(feature_set) != 0:     # aka was previously created and then unpickled
    pass
else:                         # does not exist and must be created
    # add WSJ corpus to feature set
    # loop through each directory
    for i in range(0, 25):
        
        dir_num = str(i)
        if i < 10:
            dir_num = "0" + str(i)
    
        num_files_in_dir = len(os.listdir('/Users/morischick/nltk_data/corpora/ptb/WSJ/'+dir_num))
        #print(dir_num, num_files_in_dir)
        print("Beginning WSJ/", dir_num, "...")
            
        # loop through each file
        for j in range(0, num_files_in_dir):
            file_num = str(j)
            
            if j < 10:
                file_num = "0" + str(j)
            
                
            try:
                file_name = 'WSJ/' + dir_num + '/WSJ_' + dir_num + file_num + '.MRG'
                num_sentences = len(ptb.parsed_sents(file_name))
                genre = genre_dict[file_name]
                #print(file_name, i , j, num_sentences, genre)
                
            except:
                print("This file does not exist and a genre cannot be found for it")
            
            if genreCount[genre] < NUM_EXAMPLES:
            
                try:
                    # loop through each sentence
                    for x in range (0, num_sentences):
                        
                        if genreCount[genre] < NUM_EXAMPLES:
                            genreCount[genre] += 1
                            parsed_sent = ptb.parsed_sents(file_name)[x]
                        
                            feature_dict = {}
                        
                            no_internal_labels = without_internal_labels(parsed_sent)
                            height = get_height(parsed_sent)
                            get_productions(parsed_sent) # by running will add them to feature_dict
                            get_subtrees(parsed_sent) #same as get_productions()
                        
                            feature_dict['without_internal_labels'] = no_internal_labels
                            feature_dict['height'] = height
                        
                            feature_set += [(feature_dict, genre)]
                        if genreCount[genre] >= NUM_EXAMPLES:
                            break
                        
                except Exception as e:
                    print("Error adding ", file_name, " to feature vector")
                    print(e)
                    
            if genreCount[genre] >= NUM_EXAMPLES:
                break
                
    
    print("WSJ corpus loaded successfully.")
    print("Adding Brown corpus to feature set...")
    
    # add BROWN corpus to feature set
    brown_subdirs = ['CF', 'CG', 'CK', 'CL', 'CM', 'CN', 'CP', 'CR']
    
    # loop through each directory
    for i in range(len(brown_subdirs)):
        
        dir_name = brown_subdirs[i]
        print("Beginning Brown/", dir_name, "...")
        
        all_files_in_dir = os.listdir('/Users/morischick/nltk_data/corpora/ptb/BROWN/'+dir_name)
        needed_files_in_dir = [x for x in all_files_in_dir if x[:1] == 'C']
        #print(num_files_in_dir)
        
        
        # loop through each file
        # irregulars numbers because the lowest file name in any of the directories is 0004
        # and the highest file name is 998
        for file in needed_files_in_dir:
                
            file_name = 'BROWN/' + dir_name + '/' + file
    
            try:
                num_sentences = len(ptb.parsed_sents(file_name))
                genre = genre_dict[file_name]
                genre_dict[file_name] = genre
            
            except:
                print("This file does not exist and a genre cannot be found for it")
            
            if genreCount[genre] < NUM_EXAMPLES:
                
                try:
                    # loop through each sentence
                    for x in range (0, num_sentences):
                        
                        if genreCount[genre] < NUM_EXAMPLES:
                            genreCount[genre] += 1
                            parsed_sent = ptb.parsed_sents(file_name)[x]
                            
                            feature_dict = {}
                            
                            no_internal_labels = without_internal_labels(parsed_sent)
                            height = get_height(parsed_sent)
                            get_productions(parsed_sent)
                            get_subtrees(parsed_sent) #same as get_productions()
                            
                            feature_dict['without_internal_labels'] = no_internal_labels
                            feature_dict['height'] = height

                            feature_set += [(feature_dict, genre)]
                            
                        if genreCount[genre] >= NUM_EXAMPLES:
                            break
        
                        
                except Exception as e:
                    print("Error adding ", file_name, " to feature vector")
                    print(e)
            
            if genreCount[genre] >= NUM_EXAMPLES:
                break
    
    # was not previously pickled, so we will pickle it now
    # open a file, where you want to store the data
    pickled_feature_set = open('pickled_feature_set', 'wb')
    
    # dump information to that file
    pickle.dump(feature_set, pickled_feature_set)
    
    # close the file
    pickled_feature_set.close()
    
    
    print("Brown corpus loaded successfully.")

print("Genre Count Dictionary:", genreCount)


random.shuffle(feature_set)
size = len(feature_set)
train_set, test_set = feature_set[:int(size*0.9)], feature_set[int(size*0.9):]
practice_set = [vector[0] for vector in test_set] # no genre tags, just feature vectors

# get feature vectors by genre (100%)
d_feature_set = {}
for genre in all_genres:
    d_feature_set[genre] = [(feature, label) for feature, label in feature_set if label == genre]
    
size = len(d_feature_set['news']) # 881

# what we will use as the training set (90%)
training = []
for genre in all_genres:
    training.extend(d_feature_set[genre][:int(size * 0.9)])
    
random.shuffle(training)

# dictionary with remaining 10% within each genre stored by genre
d_testing = {}
testing = []
for genre in all_genres:
    d_testing[genre] = d_feature_set[genre][int(size * 0.9):]
    testing.extend(d_feature_set[genre][int(size * 0.9):])
    

all_testing             = []      # has feature and label
humor_testing           = []      # the rest just have the feature (know label from array name)
news_testing            = []
lore_testing            = []
belles_lettres_testing  = []
fiction_testing         = []
mystery_testing         = []
science_fiction_testing = []
adventure_testing       = []
romance_testing         = []

for genre in d_testing:
    for feature, label in d_testing[genre]:
        all_testing.append((feature, label))
        if genre == 'humor':
            humor_testing.append(feature)
        if genre == 'news':
            news_testing.append(feature)
        if genre == 'lore':
            lore_testing.append(feature)
        if genre == 'belles_lettres':
            belles_lettres_testing.append(feature)
        if genre == 'fiction':
            fiction_testing.append(feature)
        if genre == 'mystery':
            mystery_testing.append(feature)
        if genre == 'science_fiction':
            science_fiction_testing.append(feature)
        if genre == 'adventure':
            adventure_testing.append(feature)
        if genre == 'romance':
            romance_testing.append(feature)  

try:
    # if we have the pickled trained classifiers, open and deserialize them
    pickled_naiveBayes_classifier = open('pickled_naiveBayes_classifier', 'rb')
    naiveBayes_classifer = pickle.load(pickled_feature_set)
    pickled_naiveBayes_classifier.close()
    
    pickled_maxEnt_classifier = open('pickled_maxEnt_classifier', 'rb')
    maxEnt_classifer = pickle.load(pickled_maxEnt_classifier)
    pickled_maxEnt_classifier.close()
    
except:
    # if we didn't have the pickled trained classifiers, train and serialize them
    
    naiveBayes_classifier = nltk.NaiveBayesClassifier.train(training)
    maxEnt_classifier = nltk.MaxentClassifier.train(training)
    
    pickled_naiveBayes_classifier = open('pickled_naiveBayes_classifier', 'wb')
    pickle.dump(naiveBayes_classifier, pickled_naiveBayes_classifier)
    pickled_naiveBayes_classifier.close()

    pickled_maxEnt_classifier = open('pickled_maxEnt_classifier', 'wb')
    pickle.dump(maxEnt_classifier, pickled_maxEnt_classifier)
    pickled_maxEnt_classifier.close()

print("Beginning classification...")
sorted(naiveBayes_classifier.labels())      # could have used either classifier since both have the same labels


humor_pdist           = maxEnt_classifier.prob_classify_many(humor_testing)
news_pdist            = maxEnt_classifier.prob_classify_many(news_testing)
lore_pdist            = maxEnt_classifier.prob_classify_many(lore_testing)
belles_lettres_pdist  = maxEnt_classifier.prob_classify_many(belles_lettres_testing)
fiction_pdist         = maxEnt_classifier.prob_classify_many(fiction_testing)
mystery_pdist         = maxEnt_classifier.prob_classify_many(mystery_testing)
science_fiction_pdist = maxEnt_classifier.prob_classify_many(science_fiction_testing)
adventure_pdist       = maxEnt_classifier.prob_classify_many(adventure_testing)
romance_pdist         = maxEnt_classifier.prob_classify_many(romance_testing)

pdist_arr = [(humor_pdist, 'humor'), 
             (news_pdist, 'news'), 
             (lore_pdist, 'lore'),
             (belles_lettres_pdist, 'belles_lettres'),
             (fiction_pdist, 'fiction'),
             (mystery_pdist, 'mystery'),
             (science_fiction_pdist, 'science_fiction'),
             (adventure_pdist, 'adventure'),
             (romance_pdist, 'romance')]

for genre_pdist, genre in pdist_arr:
    print(genre)
    for i in range(0, len(genre_pdist), 5):
        # take 5 consecutive sentences and take product of their probabilities for each genre and
        # see if they match the assumed one (here its humor)
        try:
            first = humor_pdist[i]
            second = humor_pdist[i+1]
            third = humor_pdist[i+2]
            fourth = humor_pdist[i+3]
            fifth = humor_pdist[i+4]
        except Exception as e:
            print(e)
        
        try:
            # tp = total probability
            humor_tp = first.prob('humor') * second.prob('humor') * third.prob('humor') * fourth.prob('humor') * fifth.prob('humor')
            news_tp = first.prob('news') * second.prob('news') * third.prob('news') * fourth.prob('news') * fifth.prob('news')
            lore_tp = first.prob('lore') * second.prob('lore') * third.prob('lore') * fourth.prob('lore') * fifth.prob('lore')
            belles_lettres_tp = first.prob('belles_lettres') * second.prob('belles_lettres') * third.prob('belles_lettres') * fourth.prob('belles_lettres') * fifth.prob('belles_lettres')
            fiction_tp = first.prob('fiction') * second.prob('fiction') * third.prob('fiction') * fourth.prob('fiction') * fifth.prob('fiction')
            mystery_tp = first.prob('mystery') * second.prob('mystery') * third.prob('mystery') * fourth.prob('mystery') * fifth.prob('mystery')
            science_fiction_tp = first.prob('science_fiction') * second.prob('science_fiction') * third.prob('science_fiction') * fourth.prob('science_fiction') * fifth.prob('science_fiction')
            adventure_tp = first.prob('adventure') * second.prob('adventure') * third.prob('adventure') * fourth.prob('adventure') * fifth.prob('adventure')
            romance_tp = first.prob('romance') * second.prob('romance') * third.prob('romance') * fourth.prob('romance') * fifth.prob('romance')
            
            all_probs = {'humor_tp': humor_tp,
                         'news_tp': news_tp, 
                         'lore_tp': lore_tp, 
                         'belles_lettres_tp': belles_lettres_tp,
                         'fiction_tp': fiction_tp,
                         'mystery_tp': mystery_tp,
                         'science_fiction_tp': science_fiction_tp,
                         'adventure_tp': adventure_tp,
                         'romance_tp': romance_tp}
            max_prob = max(all_probs, key=all_probs.get)
            #print(max_prob)
            actual_genre_prob = genre + '_tp'
            print(actual_genre_prob, ": ", all_probs[actual_genre_prob], max_prob, ": ", all_probs[max_prob])
        except Exception as e:
            print(e)
            pass

print("Naive Bayes accuracy: ", nltk.classify.accuracy(naiveBayes_classifier, testing))
naiveBayes_classifier.show_most_informative_features(20)

print("Maxent accuracy: ", nltk.classify.accuracy(maxEnt_classifier, testing))
maxEnt_classifier.show_most_informative_features(20)


# metrics not working
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testing):
    refsets[label].add(i)
    observed = naiveBayes_classifier.classify(feats)
    testsets[observed].add(i)

 
print('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
print('pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
print('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg']))



"""
# how many unique productions are there?
# how many unique subtree shapes are there?

all_productions = list(all_productions)
​
pToi = {p:i for i, p in enumerate(all_productions)}
def feature_dict_to_vector(d):
    X = [None] * len(all_productions)
    for i, p in enumerate(all_productions):
        if p in d:
            X[i] = d[p]
        else:
            X[i] = 0
    return X
​
all_genres = [k for k in genreCount.keys()]
gToI = {g:i for i, g in enumerate(all_genres)}
​
def label_to_int(label):
    return gToI[label]
​
X = [feature_dict_to_vector(d) for d, label in train_set]
y = [label_to_int(label) for d, label in train_set]
X_test = [feature_dict_to_vector(d) for d, label in test_set]
y_test = [label_to_int(label) for d, label in test_set]
​
MNB_clf = MultinomialNB()
MNB_clf.fit(X, y)
y_pred = MNB_clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
​
regression_classifier = LogisticRegression(random_state=0)
regression_classifier.fit(X, y)
y_pred = regression_classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
​
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""
