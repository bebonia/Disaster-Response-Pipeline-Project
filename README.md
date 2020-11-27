# Disaster Response Pipeline Project

## Introduction
This project is project 2 - Disaster Reponse Pipeline for Udacity's Data Scientist Nanodegree. The project will allow users to classify disaster messages into different categories including medical, infrastructure and supply (full list of categories available in the application)

Three visualisations have been included:
1. Genre Counts - This is a count of messages split by genre used in the training set for the model.
2. Category Counts - Count of messages split by category
3. Message Length Distribution - Viz showing the distribution of message lengths. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
