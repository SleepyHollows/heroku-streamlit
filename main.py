from flask import Flask, request
from flask_restful import Api, Resource
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import json
from nltk.corpus import stopwords

sia = SentimentIntensityAnalyzer()

pos = "positive"
neg = "negative"

app = Flask(__name__)
api = Api(app)

nlp = spacy.load('en_core_web_sm',disable=['parser','ner'])
stop = stopwords.words('english')

nlp = spacy.load('en_core_web_sm')

def clean_string(text, stem="None"):
    final_string = ""
    # Make lower
    text = text.lower()
    # Remove line breaks
    text = re.sub(r'\n', '', text)
    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im', 'go', 'we', 'unless', 'back', 'lot']
    text_filtered = [word for word in text if not word in useless_words]
    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]
    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered
    final_string = ' '.join(text_stemmed)
    return final_string

class sentimentAnalysis(Resource):        
    def post(self, review):
        test = request.form["Review"]
        cleaned = clean_string(test)
        if (sia.polarity_scores(cleaned)['compound'] > 0):
            return "This review's sentiment is Positive"
        else:
            return "This review's sentiment is Negative"
        
class overallSentiment(Resource):        
    def post(self, review):
        test = request.form["Review"]
        cleaned = clean_string(test)
        return sia.polarity_scores(cleaned)
    
class yearlySentiment(Resource):
    def post(self, review):
        positiveReview = 0
        negativeReview = 0
        values = dict()
        test = request.form["Review"]
        Ratings = json.loads(test)
        for review in Ratings.values():
            if (sia.polarity_scores(review)['compound'] > 0):
                positiveReview += 1
            else:
                negativeReview += 1
        values['positive'] = positiveReview
        values['negative'] = negativeReview
        return values
    
api.add_resource(sentimentAnalysis, "/api/<review>")
api.add_resource(overallSentiment, "/overallSentiment/<review>")
api.add_resource(yearlySentiment, "/yearlySentiment/<review>")

if __name__ == "__main__":
    app.run(debug = False)    
    