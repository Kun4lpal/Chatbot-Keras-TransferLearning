##############################################################################################################################################
# AUTHOR: KUNAL PALIWAL
# SOURCES : 
# https://nlp.stanford.edu/projects/glove/
# https://github.com/nicolas-ivanov/debug_seq2seq
# https://github.com/farizrahman4u/seq2seq
# https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras
# https://github.com/codekansas/keras-language-modeling
#
# EMAIL ID: kupaliwa@syr.edu
# COURSE: ARTIFICAL NEURAL NETWORKS 
# This file is responsible for testing our chatbot
##############################################################################################################################################
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import SGD 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence

import keras.backend as K
import numpy as np
import pickle
import theano
import os.path
import sys
import nltk
import re
import time
np.random.seed(0)  

class NN:
        
    # --------------------------------<Initializing Keras model of the chatbot (Constructor)>-------------------------------
    def __init__(self):
        self.vocabulary_file = 'vocabulary_file'
        self.weights_file = 'model_weights.h5'
        self.unknown_token = 'something'        
        self.bot = 'RedEye'
        self.word_embedding_size = 100
        self.sentence_embedding_size = 300
        self.dictionary_size = 7000
        self.maxlen_input = 50
        self.vocabulary = pickle.load(open(self.vocabulary_file, 'rb'))

        gradDescent = SGD(lr=0.00004) 
        input_context = Input(shape=(self.maxlen_input,), dtype='int32', name='the_context_text')
        input_answer = Input(shape=(self.maxlen_input,), dtype='int32', name='the_answer_text')
        LSTM_encoder = LSTM(self.sentence_embedding_size, init= 'lecun_uniform', name='Encode_context')
        LSTM_decoder = LSTM(self.sentence_embedding_size, init= 'lecun_uniform', name='Encode_answer_up')
        if os.path.isfile(self.weights_file):
            Shared_Embedding = Embedding(output_dim=self.word_embedding_size, input_dim=self.dictionary_size, input_length=self.maxlen_input, name='Shared')

        word_embedding_context = Shared_Embedding(input_context)
        context_embedding = LSTM_encoder(word_embedding_context)
        word_embedding_answer = Shared_Embedding(input_answer)
        answer_embedding = LSTM_decoder(word_embedding_answer)
        merge_layer = merge([context_embedding, answer_embedding], mode='concat', concat_axis=1, name='concatenate_the_embeddings')

        out = Dense(int(self.dictionary_size/2), activation="relu", name='relu_activation')(merge_layer)
        out = Dense(int(self.dictionary_size), activation="softmax", name='likelihood_of_the')(out)

        self.model = Model(input=[input_context, input_answer], output = [out])
        self.model.compile(loss='categorical_crossentropy', optimizer=gradDescent)
        if os.path.isfile(self.weights_file):
            self.model.load_weights(self.weights_file)

    # --------------------------------< Tokenizing the input >-------------------------------    
    def tokenize(self,sentences):
            
        tokenized_sentences = nltk.word_tokenize(sentences)
        index_to_word = [x[0] for x in self.vocabulary]
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        tokenized_sentences = [w if w in word_to_index else self.unknown_token for w in tokenized_sentences]
        X = np.asarray([word_to_index[w] for w in tokenized_sentences])
        s = X.size
        q_sent = np.zeros((1,self.maxlen_input))
        if s < (self.maxlen_input + 1):
            q_sent[0,- s:] = X
        else:
            q_sent[0,:] = X[- self.maxlen_input:]
    
        return q_sent

    # --------------------------------< Testing our model using inbuilt predict function in keras >-------------------------------    
    def predict(self,input):
        flag = 0
        probability = 1
        ans_partial = np.zeros((1,self.maxlen_input))
        ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
        for k in range(self.maxlen_input - 1):
            ye = self.model.predict([input, ans_partial])
            yel = ye[0,:]
            p = np.max(yel)
            mp = np.argmax(ye)
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            ans_partial[0, -1] = mp
            if mp == 3:  #  he index of the symbol EOS (end of sentence)
                flag = 1
            if flag == 0:    
                probability = probability * p
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < (self.dictionary_size-2):
                w = self.vocabulary[k]
                text = text + w[0] + ' '
        return(text, probability)
    
    # --------------------------------<Preprocessing our output from the model before printing it>-------------------------------    
    def preprocess(self,raw_word, name):
        
        l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
        l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
        l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
        l4 = ['jeffrey','fred','benjamin','paula','walter','rachel','andy','helen','harrington','kathy','ronnie','carl','annie','cole','ike','milo','cole','rick','johnny','loretta','cornelius','claire','romeo','casey','johnson','rudy','stanzi','cosgrove','wolfi','kevin','paulie','cindy','paulie','enzo','mikey','i\97','davis','jeffrey','norman','johnson','dolores','tom','brian','bruce','john','laurie','stella','dignan','elaine','jack','christ','george','frank','mary','amon','david','tom','joe','paul','sam','charlie','bob','marry','walter','james','jimmy','michael','rose','jim','peter','nick','eddie','johnny','jake','ted','mike','billy','louis','ed','jerry','alex','charles','tommy','bobby','betty','sid','dave','jeffrey','jeff','marty','richard','otis','gale','fred','bill','jones','smith','mickey']    

        raw_word = raw_word.lower()
        raw_word = raw_word.replace(', ' + self.bot, '')
        raw_word = raw_word.replace(self.bot + ' ,', '')

        for j, term in enumerate(l1):
            raw_word = raw_word.replace(term,l2[j])
        
        for term in l3:
            raw_word = raw_word.replace(term,' ')
    
        for term in l4:
            raw_word = raw_word.replace(', ' + term, ', ' + name)
            raw_word = raw_word.replace(' ' + term + ' ,' ,' ' + name + ' ,')
            raw_word = raw_word.replace('i am ' + term, 'i am ' + self.bot)
            raw_word = raw_word.replace('my name is' + term, 'my name is ' + self.bot)
    
        for j in range(30):
            raw_word = raw_word.replace('. .', '')
            raw_word = raw_word.replace('.  .', '')
            raw_word = raw_word.replace('..', '')
       
        for j in range(5):
            raw_word = raw_word.replace('  ', ' ')
        
        if raw_word[-1] !=  '!' and raw_word[-1] != '?' and raw_word[-1] != '.' and raw_word[-2:] !=  '! ' and raw_word[-2:] != '? ' and raw_word[-2:] != '. ':
            raw_word = raw_word + ' .'
    
        if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
            raw_word = 'what ?'
    
        if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
            raw_word = 'i do not want to talk about it .'
      
        return raw_word


    # --------------------------------< Wrapper to help us test our model >-------------------------------
    def config(self):    
        print("\n \n \n \n")
        print("   Welcome to chatbot demo:     ")
        print("\n \n")
        # Processing the user query:
        probability = 0
        que = ''
        last_query  = ' '
        last_last_query = ''
        text = ' '
        last_text = ''
        name = 'Kunal'
        print('computer: hi , Kunal' +' ! My name is ' + self.bot + '.\n') 

        while que != 'exit .':    
            que = input('user: ')
            nameofc = self.bot
            que = self.preprocess(que, nameofc)
            # Collecting data for training:
            q = last_query + ' ' + text
            a = que
            
            # Composing the context:
            if probability > 0.2:
                query = text + ' ' + que
            else:    
                query = que

            last_text = text    
            Q = self.tokenize(query)

            # Using the trained model to predict the answer:        
            predout, probability = self.predict(Q[0:1])
            start_index = predout.find('EOS')
            # Process the output from the model
            text = self.preprocess(predout[0:start_index], name)
            print ('computer: ' + text + '    (with probability of %f)'%probability)
    
            last_last_query = last_query    
            last_query = que        


# --------------------------------< Main method >-------------------------------
if __name__ == "__main__":
    print('testing')
    c = NN()
    c.config()