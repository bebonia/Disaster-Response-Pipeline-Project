import sys
# import libraries
import pandas as pd
import sqlalchemy as db
import pickle
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    
    engine = db.create_engine('sqlite:///'+database_filepath)
    metadata = db.MetaData()
    connection = engine.connect()

    emp = db.Table('Project2_ETL', metadata, autoload=True, autoload_with=engine)

    results = connection.execute(db.select([emp])).fetchall()
    df = pd.DataFrame(results)
    df.columns = results[0].keys()

    X = df['message'] 
    y = df.drop(['message','id','genre','original'],axis=1) 
    
    return X,y, y.columns.tolist()

def tokenize(text):
    
    #code based from solution of exercise
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #create lemmatizer
    lemmatizer = WordNetLemmatizer()

    #array to store tokens
    clean_tokens = []
    

    
    #for each token apply lemmatizer, remove white space/punctuation and convert to lower case
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        

    return clean_tokens

def build_model():
    
    new_pipeline =  Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier((AdaBoostClassifier())))
        ])


    parameters = {
                'vect__stop_words': ['english', None],
                'vect__max_features': (None, 5000)

            }


        #create grid search object
    new_grid_model = GridSearchCV(new_pipeline, parameters)

    return new_grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    for i in range(35):
        print(F"Precision, Recall and F1 Score for {Y_test.columns[i]}")
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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