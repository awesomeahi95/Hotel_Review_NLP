import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import confusion_matrix

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')


#===============================================================================================#

# Method to Add Suffix '_sum' to Review Summary

#===============================================================================================#


def add_sum_suffix(text):
    
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        word = word + '_sum'
        new_text += word + ' '
        
    return new_text


#===============================================================================================#

# Method to Tokenize Review Column

#===============================================================================================#


def text_cleanup(text):
    
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        new_text += word + ' '
        
    return new_text


#===============================================================================================#

# Method to Reverse Encode 'Score' Column

#===============================================================================================#


def reverse_encode(y_df):
    
    y_df['score'] = (y_df.iloc[:, 0:] == 1).idxmax(1)['score'] = (y_df.iloc[:, 0:] == 1).idxmax(1)
    for i in range(0,len(y_df)):
        if y_df.iloc[i,-1] == 'score_1':
            y_df.iloc[i,-1] = 1
        elif y_df.iloc[i,-1] == 'score_2':
            y_df.iloc[i,-1] = 2
        elif y_df.iloc[i,-1]== 'score_3':
            y_df.iloc[i,-1] = 3
        elif y_df.iloc[i,-1]== 'score_4':
            y_df.iloc[i,-1] = 4
        elif y_df.iloc[i,-1] == 'score_5':
            y_df.iloc[i,-1]= 5

            
#===============================================================================================#

# Method to Adjust Argmax of Score to a 1-5 Scale

#===============================================================================================#


def add_one_argmax_score(x):
    
    x = x+1
    
    return x


#===============================================================================================#

# Method to Create a Confusion Matriz

#===============================================================================================#


def conf_matrix(cm):
    
    plt.figure(figsize=(9,9))
    ax = sns.heatmap(cm,
                     annot= True, 
                     fmt = '.4g', 
                     cbar=0,
                     xticklabels=[1,2,3,4,5],
                     yticklabels=[1,2,3,4,5])
    ax.set(xlabel='Predicted', ylabel='True')
    plt.show()