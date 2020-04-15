import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories

def clean_data(df):
    messages, categories = load_data(sys.argv[1],sys.argv[2])
    df = messages.merge(categories, how= 'outer', on =['id'])    # merge datasets
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
    df = clean_data(df)

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
