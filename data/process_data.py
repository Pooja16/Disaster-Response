import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages_filepath: File path of messages data
        categories_filepath: File path of categories data
    Output:
        df: Merged dataset from messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on =['id'])
    return df

def clean_data(df):
	'''
    Input:
        df: Merged dataset from messages and categories
    Output:
        df: Cleaned dataset
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x.str.split('-')[0][0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for columns in category_colnames:
        # set each value to be the last character of the string
        categories[columns] = categories[columns].str[-1]
        # convert column from string to numeric
        categories[columns] = pd.to_numeric(categories[columns])
        # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df=df.drop_duplicates(subset='id')
    print(df.dtypes)
    return df

def save_data(df, database_filename):
	'''
    Save df into sqlite db
    Input:
        df: cleaned dataset
        database_filename: database name, e.g. DisasterMessages.db
    Output: 
        A SQLite database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    conn = sqlite3.connect(database_filename)
    # get a cursor
    cur = conn.cursor()
   # drop the test table in case it already exists
    cur.execute("DROP TABLE IF EXISTS FigureEight")
    df.to_sql('FigureEight', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()