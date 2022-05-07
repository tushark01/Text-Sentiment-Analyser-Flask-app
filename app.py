from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
import pandas as pd
import numpy as np


import nltk

#!from nltk.corpus import punkt
from string import punctuation
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    #!convert to lowercase
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())
    
    #*remove punctuations
    #^text3 = ''.join(c for c in text2 if c not in punctuation)
        
    #?remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sanalyzer = SentimentIntensityAnalyzer()
    dd = sanalyzer.polarity_scores(text=processed_doc1)
    dd['text'] = processed_doc1

    #*create a dataframe
    df = pd.DataFrame(dd, index=[0])

    #!create a vectorizer
    vectorizer = TfidfVectorizer()

    #*create a matrix
    matrix = vectorizer.fit_transform(df['text'])
    #&create a similarity matrix

    sim_matrix = cosine_similarity(matrix)
    #?create a similarity matrix
    
    sim_matrix = sim_matrix.flatten()
    

    compound = round((1 + dd['compound'])/2, 2)

    return render_template('index.html', final=compound, text1=text_final,text2=dd['pos'],text5=dd['neg'],text4=compound,text3=dd['neu'], text6=sim_matrix)

if __name__ == "__main__":

    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True, processes=1)
