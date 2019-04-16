# Disaster Response 

## Introduction:
In this project I analyzed disaster data from Figure Eight in order to build a model for an API that classifies disaster messages.
 
The data set contains real messages that were sent during disaster events. The goal is to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. 


## Table of Contents:

This project contains the following folders:

1. `data`
- Disaster_categories.csv: This file includes categories of messages received.
- Disaster_messages.csv: This file includes messages received.
- Process_data.py: This python file includes the ETL pipeline to extract, wrangle, clean and save data.
- DisasterResponse.db: This is the SQL database that contains the processed messages and categories data.

2. `models`
- train_classifier.py`: This includes the machine learning pipeline in order to train data and save the classifier.

- classifier.pkl: Output of the train_classifier.py. This is the trained classifier.

3. `app`
- Run.py: Contains code for web application.
- Templates contain files for web application.

4. ‘ETL Pipeline Preparation Python Notebook`:
This python notebook contains the ETL pipeline.

5. `ML Pipeline Preparation`:
This python notebook contains the ML pipeline.
  

## Summary of Results of Analysis:	

Created ETL pipeline to extract, transform and load the disaster data provided by Figure Eight. Created a machine learning pipeline to train the classifier. Developed a web application so that the emergency worker can input the message and get the message classification result.

## Software:

This project uses the following software and Python libraries:
-  [Python 2.7] (https://www.python.org/download/releases/2.7/)
-  [NumPy] (http://www.numpy.org/)
-  [Pandas] (http://pandas.pydata.org/)
-  [scikit-learn] (http://scikit-learn.org/stable/)
-  [matplotlib] (http://matplotlib.org/)
You will also need to have software installed to run and execute a [Jupyter Notebook] (http://ipython.org/notebook.html)
If you do not have Python installed yet, it is highly recommended that you install the [Anaconda] (http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk:
- Punkt
- Wordnet
- stopwords

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


