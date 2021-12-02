import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
      load data from messages and categories csv files
      
      INPUT:
      messages_filepath (str): path to messages csv file
      categories_filepath (str): path to categories csv file
      
      OUTPUT:
      df (DataFrame): merged data fram from messages and categories on 'id'
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, how ='left',on='id')
    
    return df


def clean_data(df):

    """
      Clean DataFrame and prepare it for ML pipleline analysis
      
      INPUT:
      df (DataFrame): df created from load_data() step
      
      
      OUTPUT:
      df (DataFrame): cleaned df ready to be saved
    """
    # Split text in categories column and expand the result into categorical new columns
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.split("-")[0]).tolist()
    categories.columns = category_colnames
    
    #remove text values leaving only 0,1 depending on category
    categories = categories.applymap(lambda x: x.split("-")[1]).astype('int32')
    
    # FIX1: some values under 'related' are 2, replacing with 1
    categories['related'] = categories['related'].replace(2,1)
    
    # drop original categories column, concat original dataframe with newly expanded columns
    # and remove duplicates
    df = df.drop(columns='categories')
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save the DataFrame into database file
    
    INPUTS:
    df(DataFrame): Df to be saved
    database_filename(str): file name to be saved to
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    #Fix2: adding if_exists to replace if file db file is already existing
    df.to_sql('messages', engine, index=False, if_exists = 'replace')
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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