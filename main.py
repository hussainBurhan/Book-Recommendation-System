import numpy as np
import pandas as pd
import pickle

# Load data from CSV files
books = pd.read_csv('books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')

# Display first few rows of each dataset
print(books.head())
print(users.head())
print(ratings.head())

# Display the shape (number of rows and columns) of each dataset
print(books.shape)
print(users.shape)
print(ratings.shape)

# Check for missing values in each dataset
print(books.isnull().sum())
print(users.isnull().sum())
print(ratings.isnull().sum())

# Check for duplicated rows in each dataset
print(books.duplicated().sum())
print(users.duplicated().sum())
print(ratings.duplicated().sum())

# Popularity Based
ratings_with_name = ratings.merge(books, on='ISBN')
print(ratings_with_name.head())
print(ratings_with_name.shape)

# Calculate the number of ratings and average rating for each book
num_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].count().reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
print(num_rating_df)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index().round()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
print(avg_rating_df)

# Merge dataframes to create a popular books dataframe
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
print(popular_df.columns)

# Select top 50 popular books with at least 250 ratings
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
print(popular_df)

# Create a final popular books dataframe
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
    ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

print(popular_df.shape)

# Collaborative Filtering Based
# (remaining code for collaborative filtering)
# ...

# Save dataframes and models to pickle files
pickle.dump(popular_df, open('popular.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))
