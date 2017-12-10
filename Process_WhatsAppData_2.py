##############################################################################################################################################
# AUTHOR: KUNAL PALIWAL
# EMAIL ID: kupaliwa@syr.edu
# COURSE: ARTIFICAL NEURAL NETWORKS 
# This file is responsible for processing our dataset and building padded inputs and outputs for training our model
##############################################################################################################################################
import numpy as np
np.random.seed(0)
import pandas as pd
import os
from os import path
import csv
import nltk
import itertools
import operator
import pickle
from keras.preprocessing import sequence
from scipy import sparse, io
from numpy.random import permutation
import re
import tensorflow
print(tensorflow.__version__)

class NN:
    # --------------------------------< Initializing parameters (Constructor) >-------------------------------    
    def __init__(self):
        self.questions_file = 'questions'
        self.answers_file = 'answers'
        self.vocabulary_file = 'vocabulary_file'
        self.padded_questions_file = 'padded_questions'
        self.padded_answers_file = 'padded_answers'
        self.unknown_token = 'something'

        self.vocabulary_size = 7000
        self.max_features = self.vocabulary_size
        self.maxlen_input = 50
        self.maxlen_output = 50  # cut texts after this number of words  
    
    # --------------------------------< Extracting question and answers from our Whatsapp dataset >-------------------------------        
    def extract_question_answers(self):
        text = open('training_data','r')
        q = open('questions', 'w')
        a = open('answers', 'w')
        pre_pre_previous_raw=''
        pre_previous_raw=''
        previous_raw=''
        person = ' '
        previous_person=' '

        l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
        l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
        l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

        for i, raw_word in enumerate(text):
            pos = raw_word.find('+++$+++')

            if pos > -1:
                person = raw_word[pos+7:pos+10]
                raw_word = raw_word[pos+8:]
            while pos > -1:
                pos = raw_word.find('+++$+++')
                raw_word = raw_word[pos+2:]

            raw_word = raw_word.replace('$+++','')
            previous_person = person

            for j, term in enumerate(l1):
                raw_word = raw_word.replace(term,l2[j])

            for term in l3:
                raw_word = raw_word.replace(term,' ')

            raw_word = raw_word.lower()

            if i>0 :
                q.write(pre_previous_raw[:-1] + ' ' + previous_raw[:-1]+ '\n')  # python will convert \n to os.linese
                a.write(raw_word[:-1]+ '\n')

            pre_pre_previous_raw = pre_previous_raw
            pre_previous_raw = previous_raw
            previous_raw = raw_word

        q.close()
        a.close()

    
    # --------------------------------< Padding the question and anwer / input and output generated above >---------------- 
    def pad_question_answers(self):
        print ("Reading the context data...")
        q = open(self.questions_file, 'r')
        questions = q.read()
        print ("Reading the answer data...")
        a = open(self.answers_file, 'r')
        answers = a.read()
        all = answers + questions
        print ("Tokenazing the answers...")
        paragraphs_a = [p for p in answers.split('\n')]
        paragraphs_b = [p for p in all.split('\n')]
        paragraphs_a = ['BOS '+p+' EOS' for p in paragraphs_a]
        paragraphs_b = ['BOS '+p+' EOS' for p in paragraphs_b]
        paragraphs_b = ' '.join(paragraphs_b)
        tokenized_text = paragraphs_b.split()
        paragraphs_q = [p for p in questions.split('\n') ]
        tokenized_answers = [p.split() for p in paragraphs_a]
        tokenized_questions = [p.split() for p in paragraphs_q]

        vocab = pickle.load(open(self.vocabulary_file, 'rb'))


        index_to_word = [x[0] for x in vocab]
        index_to_word.append(self.unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        print ("Using vocabulary of size %d." % self.vocabulary_size)
        print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Replacing all words not in our vocabulary with the unknown token:
        for i, sent in enumerate(tokenized_answers):
            tokenized_answers[i] = [w if w in word_to_index else self.unknown_token for w in sent]
   
        for i, sent in enumerate(tokenized_questions):
            tokenized_questions[i] = [w if w in word_to_index else self.unknown_token for w in sent]

        # Creating the training data:
        X = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_questions])
        Y = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_answers])

        Q = sequence.pad_sequences(X, maxlen = self.maxlen_input)
        A = sequence.pad_sequences(Y, maxlen = self.maxlen_output, padding='post')

        with open(self.padded_questions_file, 'wb') as q:
            pickle.dump(Q, q)
    
        with open(self.padded_answers_file, 'wb') as a:
            pickle.dump(A, a)


# --------------------------------< Main method >---------------- 
if __name__ == "__main__":
    print('testing')
    c_processData = NN()
    c_processData.extract_question_answers()
    c_processData.pad_question_answers()
    # parse_whatsapp()    