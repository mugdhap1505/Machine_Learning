# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:14:26 2020

@author: mugdh
"""

import numpy as np
import math 

def cleantext(text):
    text = text.lower().strip()
    for letters in text:
        if letters in """~!@#$%^&*()_-+|{}[]<>?/\=''""":
            text.replace(letters, " ")
    return text

def countwords(words, is_spam, counted):    
    for each_word in words:
        if each_word in counted:
            if is_spam == 1:
                counted[each_word][1] = counted[each_word][1] + 1
            else:
                counted[each_word][0] = counted[each_word][0] + 1
        else:
            if is_spam == 1:
                counted[each_word] = [0,1]
            else:
                counted[each_word] = [1,0]
    return counted

def make_percent_list(k, theCount, spams, hams):
    for each_key in theCount:
        theCount[each_key][0] = (theCount[each_key][0] + k )/(2*k+hams)
        theCount[each_key][1] = (theCount[each_key][1] + k )/(2*k+spams)
    return theCount

def StopWords():
    stopwords_list = []
    for line in stopwords:
        stopwords_list.append(line.strip())
    common = set(stopwords_list)
    common = common.difference('')
    return common

#Import Training File
dataset = input("Enter the Train file name \n")
train = open(dataset, "r", encoding = 'unicode-escape')

#Import Stop Words File
stopwords_file = input("Enter the Stopwords file name \n")
stopwords = open(stopwords_file, "r", encoding = 'unicode-escape')

spam = 0
ham = 0
k = 1
counted = dict()
line = train.readline()
while line != "":
    k = k + 1
    is_spam = int(line[:1])
    if is_spam == 1:
        spam = spam + 1
    else:
        ham = ham + 1
    line = cleantext(line[2:])
    words = line.split()
    words = set(words)
    common = StopWords()
    words = words.difference(common)
    counted = countwords(words, is_spam, counted)
    line = train.readline()
vocab = (make_percent_list(1, counted, spam, ham))

prob_spam = spam/(spam+ham)
prob_ham = ham/(spam+ham)

#Import Test File
fname = input("Enter the Test file name  \n")
test = open(fname, "r", encoding = 'unicode-escape')

line = test.readline()
spam_count = 0
ham_count = 0
count = 1
correct = 0
tp = 0
tn = 0
fn = 0
fp = 0
while line != "":
    count = count + 1
    is_spam = int(line[:1])
    if is_spam == 1:
        spam_count += 1
    else:
        ham_count += 1
    spam_prob = 1
    ham_prob = 1
    line = cleantext(line[2:])
    words = line.split()
    words = set(words)
    words = words.difference(common)
    for w in counted:
        if w in words:
            spam_prob +=np.log(vocab[w][1])
            ham_prob += np.log(vocab[w][0])
        else:
            spam_prob += np.log((1 - vocab[w][1]))
            ham_prob += np.log((1 - vocab[w][0]))
    spam_prob = math.exp(spam_prob)
    ham_prob = math.exp(ham_prob)
    prob_main  = (spam_prob*prob_spam)/((spam_prob*prob_spam)+(ham_prob*prob_ham))
    if prob_main >= 0.5:
        pred = 1
    else:
        pred = 0
    if pred == 1 and is_spam == 1:
        tp = tp + 1
    elif is_spam == 1 and pred == 0:
        fn = fn + 1
    elif is_spam == 0 and pred == 0:
        tn = tn + 1
    else:
        fp = fp + 1
    try:
        accuracy = (tp+tn)/(tp+tn+fp+fn)
    except ZeroDivisionError:
        accuracy = 0
    try:
        precision = tp / (tp+fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp+fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2*(1/((1/precision)+(1/recall)))
    except ZeroDivisionError:
        f1 = 0
    line = test.readline()

#Results
print("\n Results \n")
print("Total Spam emails in Test set:", spam_count)
print("Total Ham emails in Test set: ", ham_count)
print("FP: ", fp)
print("TP: ",tp)
print("FN: ", fn)
print("TN: ", tn)
print("Accuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1: ",f1)
