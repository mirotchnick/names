!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
print('unzipping...')
!unzip -o -j moviedataset.zip

import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
movies_df.head()
ratings_df.head()

#extract year from title if in brackets
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand = False)

#remove brackets from years
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand = False)

#remove years from titles
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

#strip extra spaces
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

#separate genres
movies_df['genres'] = movies_df.genres.str.split('|')

moviegenres_df = movies_df.copy()

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviegenres_df.at[index, genre] = 1

moviegenres_df = moviegenres_df.fillna(0)
moviegenres_df.head()

ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':'Pulp Fiction', 'rating':5},
            {'title':'Akira', 'rating':4.5}
]
inputMovies = pd.DataFrame(userInput)
inputMovies

inputID = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
