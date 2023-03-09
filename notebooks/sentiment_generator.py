
import pickle
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#define a input function
def enter_text():
    text = input('Please write your comment about chatGPT : ')
    return text
def text_clean(text):
    text = str(text).lower()
    text = text.replace('\\n','')
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\W+',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = re.sub('\[.*?\]', '', text) 
    text = re.sub('<.*?>+', '', text)
    text = text.replace("'s", '')
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'[0-9]','',text)
    text = text.replace('chatgpt','')
    text = re.sub(r'\b\w*(\w)\1+\w*\b','',text) #replace words with repeated charatecers by ''
    text = re.sub(r'\d+','',text) # replace one or more digits by ''
    return text
def text_tokenization(text):
    tokens = text.split()
    return tokens
def remove_stop_words(text):
    tokenized_without_stop_words = []
    for word in text:
        if word not in stopwords.words('english'):
            tokenized_without_stop_words.append(word)
    return tokenized_without_stop_words
def remove_single_char_func(text,threshold=1):
    text_clean = []
    for word in text:
        if len(word) > threshold:
            text_clean.append(word)
    return text_clean
def text_lemmatize(text):
    lemm = WordNetLemmatizer()
    text_lemm = []
    for word in text:
        text_lemm.append(lemm.lemmatize(word))
    return text_lemm

def sentiment_generator():
    # enter text
    text = enter_text()
    
    # text cleaning
    text = text_clean(text)

    
    #text tokens
    tokens = text_tokenization(text)

    
    #remove stopwords
    text = remove_stop_words(tokens)

    
    # remove single character
    text = remove_single_char_func(text)

  
    # text lemmatization
    text_lemm = text_lemmatize(text)

    
    # remove single character again
    text_processed = remove_single_char_func(text_lemm)
    text_processed = pd.Series(' '.join(text_processed))

    
    #generate features
    filename = "../vectorizer/vectorizer.pickle" 
    with open(filename, "rb") as file:
        vectorizer = pickle.load(file)

    
    X_train = vectorizer.transform(text_processed)

   
    #load the model
    filename = '../models/logistic_regression_model.pickle'
    with open(filename,'rb') as file:
        model = pickle.load(file)
    
    #model predict
    sentiment = model.predict(X_train)
    
    return f"Your sentiment about ChatGPT is {sentiment[0]}"
