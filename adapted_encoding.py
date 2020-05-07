#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:57:38 2020

@author: morischick
"""
import nltk
from nltk.tree import Tree
from nltk.corpus import treebank
from nltk.corpus import ptb, brown
from typing import Tuple, List
import os, random
from nltk.classify import apply_features
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

# The Penn Treebank Corpus:
fileids = treebank.fileids()
words = treebank.words('wsj_0003.mrg')  # ends with ...
tagged_words = treebank.tagged_words('wsj_0003.mrg')
parsed_sents = treebank.parsed_sents('wsj_0003.mrg')[6]
productions = parsed_sents.productions()

"""
print("File IDs", fileids)
print("Words", words)
print("Tagged", tagged_words)   #[('A', 'DT'), ('form', 'NN'), ('of', 'IN'), ...]
"""

#print(parsed_sents)
"""
for node in parsed_sents.subtrees():
    print(node, end=" ")
    print("\t", len(node))
    print("--------------------")
"""
 
"""
print(type(tagged_words))
print(ptb.fileids())
print(ptb.tagged_words('WSJ/00/WSJ_0003.MRG'))
# issue with allcats.txt -- use the one in nltk_data
# /Users/morischick/nltk_data/corpora/ptb/WSJ <-- where NLTK is accessing it successfuly
"""

#print(parsed_sents.pos())

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

def get_productions(tree):
    return [str(p) for p in tree.productions()]

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
genre_dict = {}
feature_set = []

for line in fin:
    line = line.replace("\n", "")
    line = line.split()
    file_name = line[0]
    genre = line[1]
    genre_dict[file_name] = genre

print("Genres loaded successfully.")
print("Adding WSJ corpus to feature set...")

# add WSJ corpus to feature set -- totally works
for i in range(0, 25):
    
    dir_num = str(i)
    if i < 10:
        dir_num = "0" + str(i)

    num_files_in_dir = len(os.listdir('/Users/morischick/nltk_data/corpora/ptb/WSJ/'+dir_num))
    #print(dir_num, num_files_in_dir)
    print("Beginning WSJ/", dir_num, "...")

    for j in range(0, num_files_in_dir):
        file_num = str(j)
        
        if j < 10:
            file_num = "0" + str(j)
        
            
        try:
            file_name = 'WSJ/' + dir_num + '/WSJ_' + dir_num + file_num + '.MRG'
            #parsed_sents = ptb.parsed_sents(file_name)[0]
            num_sentences = len(ptb.parsed_sents(file_name))
            genre = genre_dict[file_name]
            #print(file_name, i , j, num_sentences, genre)
            
        except:
            print("This file does not exist and a genre cannot be found for it")
        
        try:
            for x in range (0, num_sentences):
                parsed_sent = ptb.parsed_sents(file_name)[x]
                
                #feature_dict = {}
                
                no_internal_labels = without_internal_labels(parsed_sent)
                height = get_height(parsed_sent)
                productions_list = get_productions(parsed_sent)
                productions_str = " | ".join(productions_list)
                #left, right = get_tilt(parsed_sent)
                #subtrees_list = parsed_sent.subtrees()
                #subtrees_str = " | ".join(subtrees_list)
                
                """
                feature_dict['without_internal_labels'] = no_internal_labels
                feature_dict['height'] = height
                feature_dict['productions'] = productions
                #feature_dict['tilt'] = (left, right)
                feature_dict['subtrees'] = subtrees
                """                
                
                #yes_internal_labels = with_internal_labels(parsed_sent)
                
                feature_set += [({'without_internal_labels': no_internal_labels, 'height': height, 'productions': productions_str}, genre)]
                #feature_set += [(feature_dict, genre)]
                
        except Exception as e:
            print("Error adding ", file_name, " to feature vector")
            print(e)

print("WSJ corpus loaded successfully.")
print("Adding Brown corpus to feature set...")

# add BROWN corpus to feature set
brown_subdirs = ['CF', 'CG', 'CK', 'CL', 'CM', 'CN', 'CP', 'CR']

for i in range(len(brown_subdirs)):
    
    dir_name = brown_subdirs[i]
    print("Beginning Brown/", dir_name, "...")
    
    all_files_in_dir = os.listdir('/Users/morischick/nltk_data/corpora/ptb/BROWN/'+dir_name)
    needed_files_in_dir = [x for x in all_files_in_dir if x[:1] == 'C']
    #print(num_files_in_dir)
    
    # irregulars numbers because the lowest file name in any of the directories is 0004
    # and the highest file name is 998
    for file in needed_files_in_dir:
    #for j in range(1, num_files_in_dir):
        
        #file_num = str(j)
        
        #if j < 10:
        #    file_num = "0" + str(j)
            
        file_name = 'BROWN/' + dir_name + '/' + file
        #print(file_path)

        try:
            num_sentences = len(ptb.parsed_sents(file_name))
            genre = genre_dict[file_name]
            genre_dict[file_name] = genre
            print(file_name, genre)
        
        except:
            print("This file does not exist and a genre cannot be found for it")
        
        try:
            for x in range (0, num_sentences):
                
                parsed_sent = ptb.parsed_sents(file_name)[x]
                no_internal_labels = without_internal_labels(parsed_sent)
                height = get_height(parsed_sent)
                productions_list = get_productions(parsed_sent)
                productions_str = " | ".join(productions_list)
                #left, right = get_tilt(parsed_sent)
                #subtrees_list = parsed_sent.subtrees()
                #subtrees_str = " | ".join(subtrees_list)
                #yes_internal_labels = with_internal_labels(parsed_sent)
        
                
                feature_set += [({'without_internal_labels': no_internal_labels, 'height': height, 'productions': productions_str}, genre)]
                #feature_set += [({'without_internal_labels': no_internal_labels, 'height': height}, genre)]
                
                # new way with more features
                """
                parsed_sent = ptb.parsed_sents(file_name)[x]
                
                feature_dict = {}
                
                no_internal_labels = without_internal_labels(parsed_sent)
                height = get_height(parsed_sent)
                productions = parsed_sent.productions()
                #left, right = get_tilt(parsed_sent)
                subtrees = parsed_sent.subtrees()
                
                feature_dict['without_internal_labels'] = no_internal_labels
                feature_dict['height'] = height
                feature_dict['productions'] = productions
                #feature_dict['tilt'] = (left, right)
                feature_dict['subtrees'] = subtrees
                
                
                #yes_internal_labels = with_internal_labels(parsed_sent)
                
                feature_set += [(feature_dict, genre)]
                """
                
        except Exception as e:
            print("Error adding ", file_name, " to feature vector")
            print(e)


print("Brown corpus loaded successfully.")

random.shuffle(feature_set)
train_set, test_set = feature_set[:12121], feature_set[12121:]

print("Beginning classification...")
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Naive Bayes accuracy: ", nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(20)
