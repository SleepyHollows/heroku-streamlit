import streamlit as st
import pandas as pd
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import spacy
from nltk.corpus import stopwords
import numpy as np


st.set_option('deprecation.showPyplotGlobalUse', False)


try:
    
    url = "http://localhost:5000/api/1"
    overallurl = "http://localhost:5000/overallSentiment/1"
    yearlyurl = "http://localhost:5000/yearlySentiment/1"
    
except Exception as e:
    with open('ErrorsCaught.txt', 'w') as f:
        f.writelines(e)


Counter = {}

try:
    
    disneyData = pd.read_csv('Data/DisneylandReviews.csv')
    disneyData.head()
    
    universalData = pd.read_csv('Data/UniversalReviews.csv')
    universalData.head()
    
    
    st.title("Sentiment Analysis on Amusment Park Reviews")
    
    
    parkOptions = st.sidebar.selectbox("Select a park", ('Disney', 'Universal'))
    st.text(parkOptions + " Selected")
    
    
    yearOption = st.sidebar.selectbox("Select a year.",
                              ("2016",
                               "2017",
                               "2018",
                               "2019"))
    st.text(yearOption + " Selected.")
    
    
    monthOption = st.sidebar.selectbox("Select a month.",
                                  ('1',
                                   '2',
                                   '3',
                                   '4',
                                   '5',
                                   '6',
                                   '7',
                                   '8',
                                   '9',
                                   '10',
                                   '11',
                                   '12'))
    st.text(monthOption + " Selected")
    
except Exception as e:
    with open('ErrorsCaught.txt', 'w') as f:
        f.writelines(e)
    

def disney():
    try:
        selectedData = disneyData[disneyData['Date'] == yearOption + '-' + monthOption]['Review']
        st.text(selectedData)
     
        yearSelection = disneyData.loc[disneyData["Date"].between(yearOption + "-1", yearOption + '-12')]["Review"]
        
        selection = []
        for row in selectedData:
            selection.append(row)
        
        reviewOption = st.sidebar.selectbox("Select a review.", selection)
        st.markdown("Full Review : " + reviewOption)
    
        response = requests.post(url, {"Review": reviewOption})
        st.text(response.json())
        
        overallSentiment = requests.post(overallurl, {"Review": selectedData.to_json()})
        totalScores = overallSentiment.json()
        
        yearlySentiment = requests.post(yearlyurl, {"Review": yearSelection.to_json()})
        totalYearSentiment = yearlySentiment.json()
    
        
        pieChart_generator(totalScores, "Overall Sentiment for Date/Time selected")
        wordCloud_generator(selectedData, title  = "Top Words in Reviews")
        barChart_generator(totalYearSentiment, "Sentimnet for the year picked")
    except Exception as e:
        with open('ErrorsCaught.txt', 'w') as f:
            f.writelines(e)
            

def universal():
    try:
        
        selectedData = universalData[universalData['Date'] == yearOption + '-' + monthOption]['Review']
        st.text(selectedData)
     
        yearSelection = universalData.loc[universalData["Date"].between(yearOption + "-1", yearOption + '-12')]["Review"]
        st.text(type(yearSelection))
        
        selection = []
        for row in selectedData:
            selection.append(row)
        
        reviewOption = st.selectbox("Select a review.", selection)
        st.markdown("Full Review : " + reviewOption)
    
        response = requests.post(url, {"Review": reviewOption})
        st.text(response.json())
        
        overallSentiment = requests.post(overallurl, {"Review": selectedData.to_json()})
        totalScores = overallSentiment.json()
        
        yearlySentiment = requests.post(yearlyurl, {"Review": yearSelection.to_json()})
        totalYearSentiment = yearlySentiment.json()
    
        
        pieChart_generator(totalScores, "Overall Sentiment for Date/Time selected")
        wordCloud_generator(selectedData, title  = "Top Words in Reviews")
        barChart_generator(totalYearSentiment, "Sentimnet for the year picked")

    except Exception as e:
        with open('ErrorsCaught.txt', 'w') as f:
            f.writelines(e)
    
def wordCloud_generator(data, title = None):
    data_cleaned = cleaning(data)
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10
                         ).generate(" ".join(data_cleaned.values))                      
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=30)
    st.pyplot()
    
def pieChart_generator(scores, title = None):
    labels = ['negative', 'neutral', 'positive']
    sizes  = [scores['neg'], scores['neu'], scores['pos']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(title)
    st.pyplot()
    
def barChart_generator(data, title = None):
    fig = plt.figure()
    ax = fig.add_axes([0,1,2,3])
    x = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    sentimentsTitle = ['Positive', 'Negative']
    pos = data['positive']
    neg = data['negative']
    sentimentValues = [pos, neg]
    ax.bar(sentimentsTitle, sentimentValues)
    plt.yticks(np.arange(min(x), max(x), 5.0))
    plt.title(title)
    st.pyplot(use_container_width=True)


def cleaning(reviews):
    nlp = spacy.load('en_core_web_sm',disable=['parser','ner'])
    stop = stopwords.words('english')
    all_=[]
    for review in reviews:
        lower_case = review.lower() #lower case the text
        lower_case = lower_case.replace(" n't"," not") #correct n't as not
        lower_case = lower_case.replace("."," . ")
        lower_case = ' '.join(word.strip(string.punctuation) for word 
                              in lower_case.split()) #remove punctuation
        words = lower_case.split() #split into words
        words = [word for word in words if word.isalpha()] #remove numbers
        split = [word for word in words if word not in stop] #remove stop words
        reformed = " ".join(split) #join words back to the text
        doc = nlp(reformed)
        reformed = " ".join([token.lemma_ for token in doc]) #lemmatiztion
        all_.append(reformed)
    data_cleaned = pd.DataFrame()
    data_cleaned['clean_reviews'] = all_
    return data_cleaned['clean_reviews']

if parkOptions == 'Disney':
    disney()
    
else:
    universal()