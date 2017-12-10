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
# This file is responsible for training our chatbot
# Requirements: Need to build the padded input and output files first
##############################################################################################################################################
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.optimizers import SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence

import keras.backend as K
import numpy as np
np.random.seed(0)
import pickle
import theano.tensor as T
import os
import pandas as pd
import sys

class NN:
    # --------------------------------<Initializing parameters (Constructor) >-------------------------------
    def __init__(self):
        self.embeddings_index = {}
        self.word_embedding_size = 100
        self.sentence_embedding_size = 300
        self.dictionary_size = 7000
        self.maxlen_input = 50
        self.maxlen_output = 50
        self.num_subsets = 1
        self.Epochs = 100
        self.BatchSize = 128  #  Check the capacity of your GPU
        self.Patience = 0        
        self.n_test = 100

        self.vocabulary_file = 'vocabulary_file'
        self.questions_file = 'padded_questions'
        self.answers_file = 'padded_answers'
        self.weights_file = 'model_weights.h5'
        self.GLOVE_DIR = './glove.6B/'
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=self.Patience)
    

    # --------------------------------< print the resutls >-------------------------------
    def print_result(self,input):
        ans_partial = np.zeros((1,self.maxlen_input))
        ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
        for k in range(self.maxlen_input - 1):
            ye = model.predict([input, ans_partial])
            mp = np.argmax(ye)
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            ans_partial[0, -1] = mp
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < (self.dictionary_size-2):
                w = vocabulary[k]
                text = text + w[0] + ' '
        return(text)

    # --------------------------------<Apply transfer learning using glove directory >-------------------------------
    # source for glove dir :
    # https://nlp.stanford.edu/projects/glove/
    # glove.6b.zip
    def transfer_learning(self):        
        f = open(os.path.join(self.GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))
        embedding_matrix = np.zeros((self.dictionary_size, self.word_embedding_size))

        # Loading our vocabulary:
        vocabulary = pickle.load(open(self.vocabulary_file, 'rb'))

        # Using the Glove embedding:
        i = 0
        for word in vocabulary:
            embedding_vector = self.embeddings_index.get(word[0])
    
            if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            i += 1

        return embedding_matrix

    # --------------------------------<Build our model for training using SGD optimizer>-------------------------------
    def build_model(self, embedding_matrix):        
        gradDescent = SGD(lr=0.00005) 
        input_context = Input(shape=(self.maxlen_input,), dtype='int32', name='input_context')
        # input_context = tf.cast(input_context,tf.int32)
        input_answer = Input(shape=(self.maxlen_input,), dtype='int32', name='input_answer')
        # input_answer = tf.cast(input_answer,tf.int32)
        LSTM_encoder = LSTM(self.sentence_embedding_size, init= 'lecun_uniform')
        LSTM_decoder = LSTM(self.sentence_embedding_size, init= 'lecun_uniform')
        if os.path.isfile(self.weights_file):
            Shared_Embedding = Embedding(output_dim=self.word_embedding_size, input_dim=self.dictionary_size, input_length=self.maxlen_input) 
        else:
            Shared_Embedding = Embedding(output_dim=self.word_embedding_size, input_dim=self.dictionary_size, weights=[embedding_matrix], input_length=self.maxlen_input)
 

        word_embedding_context = Shared_Embedding(input_context)
        context_embedding = LSTM_encoder(word_embedding_context)
        word_embedding_answer = Shared_Embedding(input_answer)
        answer_embedding = LSTM_decoder(word_embedding_answer)
        merge_layer = merge([context_embedding, answer_embedding], mode='concat', concat_axis=1)
        print('dsize',self.dictionary_size/2)

        out = Dense(int(self.dictionary_size/2), activation="relu")(merge_layer)
        out = Dense(int(self.dictionary_size), activation="softmax")(out)

        model = Model(input=[input_context, input_answer], output = [out])
        model.compile(loss='categorical_crossentropy', optimizer=gradDescent)

        if os.path.isfile(self.weights_file):
            model.load_weights(self.weights_file)

        return model



    # --------------------------------<Train our model for specifiec hyper Parameters>-------------------------------
    def train(self,model):
        q = pickle.load(open(self.questions_file, 'rb'))
        a = pickle.load(open(self.answers_file, 'rb'))
        n_exem, n_words = a.shape

        qt = q[0:self.n_test,:]
        at = a[0:self.n_test,:]
        q = q[self.n_test + 1:,:]
        a = a[self.n_test + 1:,:]

        print('Number of exemples = %d'%(n_exem - self.n_test))
        step = np.around((n_exem - self.n_test)/self.num_subsets)
        round_exem = step * self.num_subsets
        round_exem = round_exem.astype('int64')
        step = step.astype('int64')

        x = range(0,self.Epochs) 
        valid_loss = np.zeros(self.Epochs)
        train_loss = np.zeros(self.Epochs)

        for m in range(self.Epochs):
            
            for n in range(0,round_exem,step):
            
                q2 = q[n:n+step]
                s = q2.shape
                count = 0
                for i, sent in enumerate(a[n:n+step]):
                    l = np.where(sent==3)  
                    limit = l[0][0]
                    count += limit + 1
            
                Q = np.zeros((count,self.maxlen_input))
                A = np.zeros((count,self.maxlen_input))
                Y = np.zeros((count,self.dictionary_size))
        
                # Loop over the training examples:
                count = 0
                for i, sent in enumerate(a[n:n+step]):
                    ans_partial = np.zeros((1,self.maxlen_input))
            
                # Loop over the positions of the current target output (the current output sequence):
                    l = np.where(sent==3)  #  the position of the symbol EOS
                    limit = l[0][0]

                    for k in range(1,limit+1):
                        # Mapping the target output (the next output word) for one-hot codding:
                        y = np.zeros((1, self.dictionary_size))
                        y[0, sent[k]] = 1

                        # preparing the partial answer to input:
                        ans_partial[0,-k:] = sent[0:k]

                        # training the model for one epoch using teacher forcing:                
                        Q[count, :] = q2[i:i+1] 
                        A[count, :] = ans_partial 
                        Y[count, :] = y
                        count += 1
                
                print('Training epoch: %d, training examples: %d - %d'%(m,n, n + step))
                model.fit([Q, A], Y, batch_size=self.BatchSize, epochs=1)
         
                test_input = qt[41:42]
                print(print_result(test_input))
                train_input = q[41:42]
                print(print_result(train_input))        
        
            model.save_weights(self.weights_file, overwrite=True)

# --------------------------------<Main method>-------------------------------
if __name__ == "__main__":
    print('testing')
    c = NN()
    tcf = c.transfer_learning()
    model = c.build_model(tcf)
    
    c.train(model)    