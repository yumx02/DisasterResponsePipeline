# import libraries
import sys
import pandas as pd
import re
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download(['punkt'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sqlalchemy import create_engine

def load_data(database_filepath):
    '''load data
    INPUT:
    database_filepath - database with  id, message, original, genre,
                        and 36 categories(related, request...etc) columns.

    OUTPUT:
    X - pd.DataFrame of messages
    Y - pd.DataFrame of genre and 36 categories(related, request...etc)

    Description:
    load database which created by process_data.py
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('data_disaster', 'sqlite:///data/DisasterResponse.db')
    X = df.message
    Y = df.iloc[:,4:]
    return X, Y


def tokenize(text):
    '''tokenizer
    INPUT:
    text - text to be tokenized

    OUTPUT:
    text - tokenized text
    '''
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []

    for message in text:
        message_rem = re.sub(r"[^a-zA-Z0-9]", " ", message)  # Remove punctuation characters
        tokens = word_tokenize(message_rem)
        tokens_rem = [w for w in tokens if w not in stopwords.words("english")] # Remove stop worrem
        tokens_join = ' '.join(tokens_rem)  # join list to string
        clean_tokens.append(tokens_join)
    messages = np.array(clean_tokens)

    return text



def build_model(X_train, Y_train):
    '''build model
    INPUT:
    X_train, Y_train - train datasets

    OUTPUT:
    model - trained model

    Description:
    perform grid search 
    '''

    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)) ,
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

#additional searching grid
    parameters = {
    'clf__estimator__p': [1,2]}
    cv = GridSearchCV(model, param_grid=parameters, cv=2, n_jobs=-1)
    cv.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):

#nomal
    y_pred = model.predict(X_test)
    y_test_np = np.asarray(Y_test)
    y_pred_tr = y_pred.T
    y_test_np_tr = y_test_np.T
    for i in range(len(y_pred_tr)):
        print("---------------------------")
        print(Y_test.columns[i])
        print(classification_report(y_test_np_tr[i], y_pred_tr[i]))


def save_model(model, model_filepath):
    '''save model
    INPUT:
    model - model to be pickled
    model_filepath -path to save the model

    Description:
    save the model as a pickle at the specified path.
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
#        X, Y, category_names = load_data(database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(X_train, Y_train)

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
