import json
import plotly
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
nltk.download(['wordnet', 'punkt', 'stopwords'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re
from collections import Counter
import operator


app = Flask(__name__)

def tokenize(text):
    
    """
    INPUT - text: text to be tokenized  
    
    OUTPUT - list of tokenized words
    
    This function will first normalize input text,
    tokenize it into list of words, these words are 
    lemmatized as well
    
    """
    #Normalizing text
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = word_tokenize(text)
    
    #filter stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in words]
    

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #first graph variables
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Second graph vairables
    categories= ((df.iloc[:, 4:]).sum()/df.shape[0]).sort_values(ascending=False)
    cat_values = categories.values
    cat_names = categories.index

    # Third graph variables, most common words in random 1000 messages
    c = 0
    words = []
    selected = {}

    [words.extend(tokenize(text)) for text in df.sample(1000).message.values]
    words_count = Counter(words)      
    words_count_sorted = dict(sorted(words_count.items(),
                                 key=operator.itemgetter(1),
                                 reverse=True))
                                                    
    for k,v in words_count_sorted.items():
        selected[k]=v
        c+=1
        if c==10:
            break
        
    selected_words=list(selected.keys())
    selected_counts=list(selected.values())
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count",
                    'automargin': True
                },
                'xaxis': {
                    'title': "Genre",
                    'automargin': True
                    
                }
            }
        },
        {
           'data': [
                 Bar(
                     x=cat_names,
                     y=cat_values
                 )
             ],

             'layout': {
                 'title': 'Proportion of Messages by Category',
                 'yaxis': {
                     'title': "proportion"
                 },
                 'xaxis': {
                     'title': "Category",
                     'automargin': True,
                     'tickangle': -40
                 }
             }
         },
         {
             'data': [
                 Bar(
                     x=selected_words,
                     y=selected_counts
                 )
             ],

             'layout': {
                 'title': 'Most Repeated words in Messages (Sample of 1000 Messages)',
                 'yaxis': {
                     'title': "Count"
                 },
                 'xaxis': {
                     'title': "Word"
                 }
             }
         }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()