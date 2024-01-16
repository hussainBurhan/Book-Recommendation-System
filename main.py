import numpy as np
import pandas as pd
import pickle

books = pd.read_csv('books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')

print(books.head())
print(users.head())
print(ratings.head())

print(books.shape)
print(users.shape)
print(ratings.shape)

print(books.isnull().sum())
print(users.isnull().sum())
print(ratings.isnull().sum())

print(books.duplicated().sum())
print(users.duplicated().sum())
print(ratings.duplicated().sum())

# Popularity Based
ratings_with_name = ratings.merge(books, on='ISBN')
print(ratings_with_name.head())
print(ratings_with_name.shape)

num_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].count().reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
print(num_rating_df)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index().round()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
print(avg_rating_df)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
print(popular_df.columns)

popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
print(popular_df)

popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
    ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

print(popular_df.shape)

# collaborative filtering based
users_with_200_ratings = ratings_with_name.groupby('User-ID')['Book-Rating'].count() > 200
users_with_200_ratings_index = users_with_200_ratings[users_with_200_ratings].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(users_with_200_ratings_index)]
y = filtered_rating.groupby('Book-Title')['Book-Rating'].count() >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(pt)
print(similarity_scores[0].shape)

def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x:x[1], reverse=True)[1:6]
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)
    return data



pickle.dump(popular_df, open('popular.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))