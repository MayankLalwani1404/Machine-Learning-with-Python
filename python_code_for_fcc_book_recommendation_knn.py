#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/MayankLalwani1404/Machine-Learning-with-Python/blob/main/fcc_book_recommendation_knn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[39]:


# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# In[ ]:


# get data files
get_ipython().system('wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip')

get_ipython().system('unzip book-crossings.zip')

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'


# In[ ]:


# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})


# In[49]:


# build book-user matrix and train KNN model
# 0) ensure dtypes and drop duplicate titles (keep first)
df_books['isbn'] = df_books['isbn'].astype(str)
df_books = df_books.drop_duplicates(subset=['title'], keep='first').copy()
# 1) use raw ratings INCLUDING zeros; filter users/books simultaneously
r = df_ratings.copy()
r['isbn'] = r['isbn'].astype(str)
if r['user'].dtype != np.int64 and r['user'].dtype != np.int32:
    r['user'] = pd.to_numeric(r['user'], errors='coerce').astype('Int64')
r = r.dropna(subset=['user','isbn','rating'])
user_counts = r['user'].value_counts()
isbn_counts = r['isbn'].value_counts()
eligible_users = set(user_counts[user_counts >= 200].index)
eligible_isbns = set(isbn_counts[isbn_counts >= 100].index)
r = r[r['user'].isin(eligible_users) & r['isbn'].isin(eligible_isbns)]
# 2) merge to books and keep unique title definitions
df = pd.merge(r, df_books[['isbn','title','author']], on='isbn', how='inner')
# 3) create ratings pivot (rows: title, cols: user), keep zeros as ratings
book_user_mat = df.pivot_table(index='title', columns='user', values='rating', aggfunc='mean').fillna(0)
# guard against empty matrix
if book_user_mat.shape[0] == 0 or book_user_mat.shape[1] == 0:
    raise ValueError('No data left after filtering. Verify the CSV parsing and thresholds.')
# convert to sparse matrix
book_user_sparse = csr_matrix(book_user_mat.values)
# fit NearestNeighbors model (cosine distance)
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_user_sparse)
# mapping for lookup
titles = book_user_mat.index
title_to_index = {t: i for i, t in enumerate(titles)}


# In[50]:


# function to return recommended books - this will be tested
def get_recommends(book = ""):
    # handle unknown book
    if book not in title_to_index:
        return [book, []]
    idx = title_to_index[book]
    distances, indices = model.kneighbors(book_user_mat.iloc[idx, :].values.reshape(1, -1), n_neighbors=6)
    recs = []
    for i in range(1, len(indices[0])):
        rec_title = titles[indices[0][i]]
        # return raw cosine distance to match the challenge's expected numbers
        rec_dist = float(distances[0][i])
        recs.append([rec_title, rec_dist])
    # reverse order to match the challenge's expected output
    recs.reverse()
    return [book, recs]


# In[51]:


books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()

