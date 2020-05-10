#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:57:38 2020

@author: morischick
"""
import nltk
from nltk.tree import Tree
from nltk.corpus import treebank
from nltk.corpus import ptb
from typing import Tuple, List
import os, random
from nltk.classify import apply_features
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression, SGDClassifier


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
            #print("1", node.label(), end=" ")
        # if it is not a leaf node
        else:
            encoding += "0 "
            #print("0", end=" ")
    #print(encoding)
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

def get_productionsOLD(parsed_sent):
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
    productions = ", ".join(d)
    return productions

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

#with_internal_labels(parsed_sents)
#without_internal_labels(parsed_sents)


# below is from Professor Waxman
def get_tree_structure(tree):
    a = []
    
    def f(tree):
        nonlocal a
        
        # first check if this is a leaf
        # in wsj corpus, could either be (DT the)
        # or even just "in" as the single element
        if (len(tree) == 1 and isinstance(tree[0], str)) or \
                                isinstance(tree, str):
            a.append(0)
            a.append(tree.label())  #adding branching factor
            return
        
        # otherwise, add branching
        # and preorder traverse thru children
        a.append(len(tree))
        for child in tree:
            f(child)
            
    f(tree)
    return a

def get_tree_structure_and_labels(tree) -> Tuple[List[int], List[str]]:
    a = get_tree_structure(tree)
    labels = [x for x in a if isinstance(x, str)]
    structure = [x for x in a if isinstance(x, int)]
    return structure, labels


def print_subtree_structures(tree):
    # this is a demo of how we can get all
    # subtree structures to train on. you would
    # write similar code. Subtrees would first give
    # you the full tree, then the left subtree,
    # then the right subtree, etc. until we get to
    # the subtrees which are leaves
    x = list(tree.subtrees())
    for st in x:
        a = get_tree_structure(st)
        print(a)
    return x

def get_height(t):
    return t.height()

def get_tilt(tree):
    # we are defining the "tilt" of a tree
    # as how often the tree gets extended
    # towards the right (most often, in most
    # trees) and how often towards the left.
    # For an n-ary tree, not so helpful, but good
    # for binary trees
    
    # need to double check for wsj trees with period
    # at the end whether that is the issue
    
    left = 0
    right = 0
    
    # so long as we have not reached a leaf
    while not (len(tree) == 1 and isinstance(tree[0], str)):
        if len(tree) == 1:
            # handle unary branching
            tree = tree[0]
        else:
            # look through subtrees, find one with max depth
            maxpos = -1
            maxheight = -1
            for i, node in enumerate(tree):
                if node.height() > maxheight:
                    maxheight = node.height()
                    maxpos = i
                    
            if maxpos < len(tree) // 2:
                left += 1
            else:
                right += 1
                
        # follow the deeper path
        tree = tree[maxpos]
        
    return left, right

# get genre of each file
print("Getting genre for each file...")
fin = open('/Users/morischick//nltk_data/corpora/ptb/allcats.txt').readlines()
genreCount = {'news': 0, 'lore': 0, 'belles_lettres': 0, 'fiction': 0, 'mystery': 0, 'science_fiction': 0, 'adventure': 0, 'romance': 0, 'humor': 0}   
# genreCount --> key: genre,        value: count of how many seen (should stop at 881)
genre_dict = {}   # key: file_name,    value: genre
feature_set = []  # list of tuples (feature_set, genre) where feature_set is the feature vector for each sentence

for line in fin:
    line = line.replace("\n", "")
    line = line.split()
    file_name = line[0]
    genre = line[1]
    genre_dict[file_name] = genre

print("Genres loaded successfully.")
print("Adding WSJ corpus to feature set...")

# add WSJ corpus to feature set -- totally works
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
        
        if genreCount[genre] < 881:
        
            try:
                # loop through each sentence
                for x in range (0, num_sentences):
                    
                    if genreCount[genre] < 881:
                        genreCount[genre] += 1
                        parsed_sent = ptb.parsed_sents(file_name)[x]
                    
                        feature_dict = {}
                    
                        no_internal_labels = without_internal_labels(parsed_sent)
                        height = get_height(parsed_sent)
                        get_productions(parsed_sent) # by running will add them to feature_dict
                        get_subtrees(parsed_sent) #same as get_productions()
                        #length = get_sentence_length(parsed_sent)
                        #tilt = get_tilt(parsed_sent)
                    
                    
                        feature_dict['without_internal_labels'] = no_internal_labels
                        feature_dict['height'] = height
                        #feature_dict['tilt'] = (left, right)
                    
                        #yes_internal_labels = with_internal_labels(parsed_sent)
                    
                        #feature_set += [({'without_internal_labels': no_internal_labels, 'height': height, 'productions': productions, 'length': length}, genre)]
                        feature_set += [(feature_dict, genre)]
                    if genreCount[genre] >= 881:
                        break
                    
            except Exception as e:
                print("Error adding ", file_name, " to feature vector")
                print(e)
                
        if genreCount[genre] >= 881:
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
        #print(file_path)

        try:
            num_sentences = len(ptb.parsed_sents(file_name))
            genre = genre_dict[file_name]
            genre_dict[file_name] = genre
            #print(file_name, genre)
        
        except:
            print("This file does not exist and a genre cannot be found for it")
        
        if genreCount[genre] < 881:
            
            try:
                # loop through each sentence
                for x in range (0, num_sentences):
                    
                    if genreCount[genre] < 881:
                        genreCount[genre] += 1
                        parsed_sent = ptb.parsed_sents(file_name)[x]
                        
                        feature_dict = {}
                        
                        no_internal_labels = without_internal_labels(parsed_sent)
                        height = get_height(parsed_sent)
                        get_productions(parsed_sent)
                        get_subtrees(parsed_sent) #same as get_productions()
                        #left, right = get_tilt(parsed_sent)
                        
                        feature_dict['without_internal_labels'] = no_internal_labels
                        feature_dict['height'] = height
                        #feature_dict['tilt'] = (left, right)
                        
                        #yes_internal_labels = with_internal_labels(parsed_sent)
                        
                        feature_set += [(feature_dict, genre)]
                        
                    if genreCount[genre] >= 881:
                        break
    
                    
            except Exception as e:
                print("Error adding ", file_name, " to feature vector")
                print(e)
        
        if genreCount[genre] >= 881:
            break



print("Brown corpus loaded successfully.")

print("Genre Count Dictionary:", genreCount)


random.shuffle(feature_set)
train_set, test_set = feature_set[:3964], feature_set[3964:]

print("Beginning classification...")
naiveBayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
maxEnt_classifier = nltk.MaxentClassifier.train(train_set)

print("Naive Bayes accuracy: ", nltk.classify.accuracy(naiveBayes_classifier, test_set))
naiveBayes_classifier.show_most_informative_features(20)

print("Maxent accuracy: ", nltk.classify.accuracy(maxEnt_classifier, test_set))
maxEnt_classifier.show_most_informative_features(20)

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(train_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_clf, test_set))
