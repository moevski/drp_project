import sys
import numpy as np
import pandas as pd 
import sklearn
import nltk
import re
import pickle
import sqlite3

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
nltk.download(['wordnet', 'punkt', 'stopwords'])

from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    """
    load data from .db sql file into pandas DataFrame
    
    INPUTS:
    database_filepath (str): path to .db file
    
    
    OUTPUTS:
    X(DataFrame): features data frame
    Y(DataFrame): response dataframe
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    return X, Y
    
def tokenize(text):
    """
    tokenize input text
    
    INPUTS:
    text(str): text required to be tokenized
    
    OUTPUTS:
    clean_tokens(list): tokenized and lemmatized words
    """
    #Normalizing text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    
    #filter stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in words]
    
    return clean_tokens
    
    


def build_model():
    """
    build ML classifier model and select parameters based on grid search
    
    OUTPUTS:
    cv(list): classficiation model
    
    """
    # create pipleline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))    
    ])
    
    # Grid search parameters to select
    parameters = {
        'vect__max_df': (0.5, 1.0),
        'clf__estimator__n_estimators': [10,20]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    evaluate test message and print evaluation scores per each category and overall model         accuracy
    
    INPUTS:
    
    model: classification model
    X_test: test message
    Y_test: test response 
    
    """
    Y_pred = model.predict(X_test)
    
    #Display evaluation result, f1 scores
    n = 0
    for col in Y_test:
        print('Output Category {}: {}'.format(n+1, col))
        print(classification_report(Y_test[col], Y_pred[:,n], labels = np.unique(Y_pred)))
        n+=1
    accuracy =( Y_pred == Y_test.values).mean()
    print('Model Accuracy: {:.3f}',format(accuracy))


def save_model(model, model_filepath):
    
    """
    save the model as pickle file
    
    INPUTS:
    model: classification model
    model_filepath: file path to save to

    """
    with open (model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()