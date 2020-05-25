#===============================================================================================#

# Imports

#===============================================================================================#

import streamlit as sl

import pandas as pd
import numpy as np

import pickle

from tensorflow.keras.models import load_model
from keras.preprocessing import sequence

#===============================================================================================#

# Functions and Models Prepared

#===============================================================================================#

word_index_dict = pickle.load(open('Data/word_index_dict.pkl','rb'))

neural_net_model = load_model('Models/Neural_Network.h5')

def index_review_words(text):
    review_word_list = []
    for word in text.lower().split():
        if word in word_index_dict.keys():
            review_word_list.append(word_index_dict[word])
        else:
            review_word_list.append(word_index_dict['<UNK>'])

    return review_word_list 

def add_sum_suffix(text):
    
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        word = word + '_sum'
        new_text += word + ' '
        
    return new_text

def text_cleanup(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        new_text += word + ' '
        
    return new_text
#===============================================================================================#

# Streamlit

#===============================================================================================#

sl.title("Hotel Review Classifier Application")


review_summary_text = sl.text_input('Enter Your Review Summary Here')
review_text = sl.text_area('Enter Your Review Here')

if sl.button('Predict'):
    
    result_review_sum = review_summary_text.title()
    
    result_review = review_text.title()

    review_summary_text = add_sum_suffix(review_summary_text)

    review_text = text_cleanup(review_text)

    review_text = index_review_words(review_text)

    review_summary_text = index_review_words(review_summary_text)

    all_review_text = review_text + review_summary_text

    all_review_text = sequence.pad_sequences([all_review_text],value=word_index_dict['<PAD>'],padding='post',maxlen=250)

    prediction = neural_net_model.predict(all_review_text)
    
    prediction = np.argmax(prediction)
    
    sl.success(prediction+1)