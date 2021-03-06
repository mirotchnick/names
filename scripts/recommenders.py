#content-based filtering


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
inputMovies = pd.merge(inputID, inputMovies)
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

userMovies = moviegenres_df[moviegenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies = userMovies.reset_index(drop=True)
usergenres = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userProfile = usergenres.transpose().dot(inputMovies['rating'])

genretable = moviegenres_df.set_index(moviegenres_df['movieId'])
genretable = genretable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

genretable.shape

recommendationTable_df = ((genretable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()

movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]



#collaborative filtering

movies_df.head()
movies_df = movies_df.drop('genres', 1)

ratings_df.head()

inputMovies

userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

userSubGroup = userSubset.groupby(['userId'])

userSubGroup.get_group(1130)

userSubGroup = sorted(userSubGroup, key=lambda x: len(x[1]), reverse=True)

userSubGroup[0:3]

userSubGroup = userSubGroup[1:100]

pearsonDict = {}

for name, group in userSubGroup:
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    nRatings = len(group)
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    tempRatings = temp_df['rating'].tolist()
    tempGroupList = group['rating'].tolist()
    Sxx = sum([i**2 for i in tempRatings]) - pow(sum(tempRatings), 2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList), 2)/float(nRatings)
    Sxy = sum(i*j for i,j in zip(tempRatings, tempGroupList)) - sum(tempRatings)*sum(tempGroupList)/float(nRatings)
    if Sxx != 0 and Syy != 0:
        pearsonDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonDict[name] = 0

pearsonDict.items()

pearsonDF = pd.DataFrame.from_dict(pearsonDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['UserId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

topUsersRating = topUsers.merge(ratings_df, left_on='UserId', right_on='userId', how='inner')
topUsersRating.head()

topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()

tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
tempTopUsersRating.head()

recommendation_df = pd.DataFrame()
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head()

movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]