import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''load data
    INPUT:
    messages_filepath - messages csv file path
    categories_filepath - categories csv file path

    OUTPUT:
    messages - pd.dataframe with id, message, original, genre columns
    categories - pd.dataframe with id, categories columns

    Description:
    load csv file and store it with pd dataframes
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how= 'outer', on =['id'])

    return df

def clean_data(df):
    '''clean data
    INPUT:
    df - pd.dataframe with id, message, original, genre, categories columns

    OUTPUT:
    df - pd.dataframe with id, message, original, genre,
         and 36 categories(related, request...etc) columns

    Description:
    create 36 individual category column with 0 or 1
    '''
    categories = df['categories'].str.split(';',expand=True)    # create 36 individual category column
    row = categories.loc[0]     # select the first row of the categories dataframe
    category_colnames = row.str[:-2] # extract a list of new column names for categories.
    categories.columns = category_colnames    # rename the columns
    for column in categories:
        categories[column] = categories[column].str[-1] # set each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric
    df = df.drop('categories',axis='columns')     # drop the original categories column
    df = pd.concat([df, categories],axis='columns')    # concatenate the original dataframe with the new `categories` dataframe
    df = df.drop_duplicates()  # drop duplicates
    return df


def save_data(df, database_filename):
    '''save data
    INPUT:
    df - pd.dataframe with id, message, original, genre,
         and 36 categories(related, request...etc) columns
    database_filename - database file path and name

    Description:
    save the data for database at the specified path.
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('data_disaster', engine, index=False)

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
