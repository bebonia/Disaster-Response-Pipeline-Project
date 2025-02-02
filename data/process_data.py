import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    
    #read messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge two datasets together and return
    
    return messages.merge(categories,on='id')
    

def clean_data(df):
    
    categories = df['categories'].str.split(';',expand=True)
    categories.columns = categories.iloc[0,:]
    
    row = categories.iloc[0,:]
    category_colnames = [col.split('-')[0] for col in row]
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df = df.drop('categories',axis=1)
    df = pd.concat([categories,df],sort=False,axis=1).reindex(df.index)
    df = df.drop_duplicates()
    
    return df
    
    
    
def save_data(df, database_filename):
     
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Project2_ETL', engine, index=False)


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