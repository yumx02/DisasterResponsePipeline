# Disaster Response Pipeline Project
### Project Overview
- create a machine learning pipeline to categorize disaster messages so that you can send the messages to an appropriate disaster relief agency.
- An original data set, which is from Figure Eight(https://appen.com/), contains real messages that were sent during disaster events.
- project includes:
    - a web app where an emergency worker can input a new message and get classification results in several categories.
    - figures which display visualizations of the data.

### Project Components
1. ETL Pipeline
    - In process_data.py, write a data cleaning pipeline that:
        - Loads the messages and categories datasets
        - Merges the two datasets
        - Cleans the data
        - Stores it in a SQLite database

2. ML Pipeline
    - In train_classifier.py, write a machine learning pipeline that:
        - Loads data from the SQLite database
        - Splits the dataset into training and test sets
        - Builds a text processing and machine learning pipeline
        - Trains and tunes a model using GridSearchCV
        - Outputs results on the test set
        - Exports the final model as a pickle file

3. Flask Web App
    - Flask web app to visualize the results using Plotly.


### File Structure
  ┣━ app
  ┃    ┣━ run.py - Flask application
  ┃    ┗━ templates
  ┃        ┣━ go.html - web application of classification result.
  ┃        ┗━ master.html - web application of master page.
  ┣━ data
  ┃    ┣━ disaster_categories.csv - dataset of disaster categories.
  ┃    ┣━ disaster_messages.csv - dataset of disaster messages.
  ┃    ┣━ process_data.py - data processing script.
  ┃    ┗━ DisasterResponse.db - database of cleaned data.
  ┣━ models
  ┃    ┣━ train_classifier.py - NLP and ML pipeline script.
  ┃    ┗━ classifier.pkl - pickle file of classifier
  ┗━ README.md - this file


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
