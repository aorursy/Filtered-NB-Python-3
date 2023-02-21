#!/usr/bin/env python
# coding: utf-8



from IPython.display import HTML

# Youtube
HTML('<iframe width="976" height="600"  src="https://www.youtube.com/embed/793l7ZB7dkg" frameborder="0" allowfullscreen></iframe>')




get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from ast import literal_eval

import warnings
warnings.filterwarnings('ignore')

from wordcloud import WordCloud, STOPWORDS
import ast
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

sns.set_style('whitegrid')
sns.set(font_scale=1.25)
pd.set_option('display.max_colwidth', 50)




credits = pd.read_csv("../input/the-movies-dataset/credits.csv")
credits.head(5)




keywords = pd.read_csv("../input/the-movies-dataset/keywords.csv")
keywords.head(5)




movies = pd.read_csv("../input/the-movies-dataset/movies_metadata.csv")
movies.head(5)




indecies = movies[(movies.adult != 'True') & (movies.adult != 'False')].index
movies.drop(indecies, inplace = True)
print ("{} \n".format(movies['adult'].value_counts()))




credits['id'] = credits['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')
movies['id'] = movies['id'].astype('int')




movies = movies.merge(credits,on='id')
movies = movies.merge(keywords,on='id')
movies.head(5)




movies.columns




movies.shape




movies.info()




movies = movies.drop('original_title', axis=1)




movies[movies['revenue'] == 0].shape




movies['revenue'] = movies['revenue'].replace(0, np.nan)




movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
movies['budget'] = movies['budget'].replace(0, np.nan)
movies[movies['budget'].isnull()].shape




movies['adult'].value_counts()




movies = movies.drop('adult', axis=1)




base_poster_url = 'http://image.tmdb.org/t/p/w185/'
movies['poster_path'] = "<img src='" + base_poster_url + movies['poster_path'] + "' style='height:100px;'>"




movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)




movies['return'] = movies['revenue'] / movies['budget']
movies[movies['return'].isnull()].shape




movies['title'] = movies['title'].astype('str')
movies['overview'] = movies['overview'].astype('str')

title_corpus = ' '.join(movies['title'])
overview_corpus = ' '.join(movies['overview'])




title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()









overview_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(overview_corpus)
plt.figure(figsize=(16,8))
plt.imshow(overview_wordcloud)
plt.axis('off')
plt.show()









movies['production_countries'] = movies['production_countries'].fillna('[]').apply(ast.literal_eval)
movies['production_countries'] = movies['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = movies.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'countries'




con_df = movies.drop('production_countries', axis=1).join(s)
con_df = pd.DataFrame(con_df['countries'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_movies', 'country']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(20)




con_df = con_df[con_df['country'] != 'United States of America']




data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_movies'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(255, 0, 0)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]

layout = dict(
    title = 'Production Countries for the MovieLens Movies (Apart from US)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )




df_fran = movies[movies['belongs_to_collection'].notnull()]
df_fran['belongs_to_collection'] = df_fran['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)
df_fran = df_fran[df_fran['belongs_to_collection'].notnull()]




fran_pivot = df_fran.pivot_table(index='belongs_to_collection', values='revenue', aggfunc={'revenue': ['mean', 'sum', 'count']}).reset_index()




fran_pivot.sort_values('sum', ascending=False).head(10)




fran_pivot.sort_values('mean', ascending=False).head(10)




fran_pivot.sort_values('count', ascending=False).head(10)




movies['original_language'].drop_duplicates().shape[0]




lang_df = pd.DataFrame(movies['original_language'].value_counts())
lang_df['language'] = lang_df.index
lang_df.columns = ['number', 'language']
lang_df.head()




plt.figure(figsize=(12,5))
sns.barplot(x='language', y='number', data=lang_df.iloc[:11])
plt.show()




plt.figure(figsize=(12,5))
sns.barplot(x='language', y='number', data=lang_df.iloc[1:11])
plt.show()




def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan




movies['popularity'] = movies['popularity'].apply(clean_numeric).astype('float')
movies['vote_count'] = movies['vote_count'].apply(clean_numeric).astype('float')
movies['vote_average'] = movies['vote_average'].apply(clean_numeric).astype('float')




movies['popularity'].describe()




sns.distplot(movies['popularity'].fillna(movies['popularity'].median()))
plt.show()




movies[['title', 'popularity', 'year']].sort_values('popularity', ascending=False).head(10)




movies['vote_count'].describe()




movies[['title', 'vote_count', 'year']].sort_values('vote_count', ascending=False).head(10)




movies['vote_average'] = movies['vote_average'].replace(0, np.nan)
movies['vote_average'].describe()




sns.distplot(movies['vote_average'].fillna(movies['vote_average'].median()))




movies[movies['vote_count'] > 2000][['title', 'vote_average', 'vote_count' ,'year']].sort_values('vote_average', ascending=False).head(10)




sns.jointplot(x='vote_average', y='popularity', data=movies)




sns.jointplot(x='vote_average', y='vote_count', data=movies)




month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def get_month(x):
    try:
        return month_order[int(str(x).split('-')[1]) - 1]
    except:
        return np.nan
    
def get_day(x):
    try:
        year, month, day = (int(i) for i in x.split('-'))    
        answer = datetime.date(year, month, day).weekday()
        return day_order[answer]
    except:
        return np.nan




movies['day'] = movies['release_date'].apply(get_day)
movies['month'] = movies['release_date'].apply(get_month)




plt.figure(figsize=(12,6))
plt.title("Number of Movies released in a particular month.")
sns.countplot(x='month', data=movies, order=month_order)




month_mean = pd.DataFrame(movies[movies['revenue'] > 1e8].groupby('month')['revenue'].mean())
month_mean['mon'] = month_mean.index
plt.figure(figsize=(12,6))
plt.title("Average Gross by the Month for Blockbuster Movies")
sns.barplot(x='mon', y='revenue', data=month_mean, order=month_order)




plt.figure(figsize=(10,5))
plt.title("Number of Movies released on a particular day.")
sns.countplot(x='day', data=movies, order=day_order)




year_count = movies.groupby('year')['title'].count()
plt.figure(figsize=(18,5))
year_count.plot()




movies['budget'].describe()




sns.distplot(movies[movies['budget'].notnull()]['budget'])




movies['budget'].plot(logy=True, kind='hist')




movies['revenue'].describe()




sns.distplot(movies[movies['revenue'].notnull()]['revenue'])




sns.jointplot(x='budget', y='revenue', data=movies)




pd.set_option('display.max_colwidth', 50)




plt.figure(figsize=(18,5))
year_revenue = movies[(movies['revenue'].notnull()) & (movies['year'] != 'NaT')].groupby('year')['revenue'].max()
plt.plot(year_revenue.index, year_revenue)
plt.xticks(np.arange(1874, 2024, 1000.0))
plt.show()




vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_counts




# m is the 95% quantile of vote count
m= vote_counts.quantile(0.95)
m




vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float')
vote_averages




# C is the mean value for vote average
C= vote_averages.mean()
C




qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['title', 'imdb_id', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('float')
qualified['vote_average'] = qualified['vote_average'].astype('float')
qualified['popularity'] = qualified['popularity'].astype('float')
qualified.shape




def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)




qualified['score'] = qualified.apply(weighted_rating, axis=1)
qualified.head(10)




high_score = qualified.sort_values('score', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
plt.barh(high_score['title'].head(20),high_score['popularity'].head(20), align='center', color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Rating")
plt.title("Rating Movies")




high_popular = qualified.sort_values('popularity', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
plt.barh(high_popular['title'].head(20),high_popular['popularity'].head(20), align='center', color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")




qualified.to_csv(r'qualified.csv', header=True, index=True)




# Function that takes in movie count as input and outputs high rating movies
def get_high_rating_movies(movies_count):
    high_score = qualified.sort_values('score', ascending=False)
    return high_score[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)




get_high_rating_movies(10)




# generate to file High rating movies
get_high_rating_movies(30)[['imdb_id']].to_csv(r'HighRatingMovies.txt', header=None, index=None, sep=',', mode='w')




# Function that takes in movie count as input and outputs most popular movies
def get_popular_movies(movies_count):
    high_popular = qualified.sort_values('popularity', ascending=False)
    return high_popular[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)




get_popular_movies(10)




links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
links_small.head(10)




recommended_movies = movies[movies['id'].isin(links_small)]
recommended_movies.shape




recommended_movies['tagline'] = recommended_movies['tagline'].fillna('')
recommended_movies['description'] = recommended_movies['overview'] + recommended_movies['tagline']
recommended_movies['description'] = recommended_movies['description'].fillna('')




tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(recommended_movies['description'])




tfidf_matrix.shape




cosine_sim_word = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_word[0]




## put after cosine_sim_word
from numpy import save
# save to npy file
save('cosine_sim_word.npy', cosine_sim_word)




# Function that takes in movie imdb_id as input and outputs similar movies based on description
def get_movie_desc_recommendations(imdb_id, count):
    recommended = recommended_movies.reset_index()
    indices_mov_desc = pd.Series(recommended.index, index=recommended['imdb_id']).drop_duplicates()
    idx = indices_mov_desc[imdb_id]
    sim_scores = list(enumerate(cosine_sim_word[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:101]
    movie_indices = [i[0] for i in sim_scores]
    movies = recommended[['imdb_id','title', 'vote_count', 'vote_average', 'year']].iloc[movie_indices]
    return movies.head(count)




get_movie_desc_recommendations('tt1130884', 10)




# Function that takes in movie imdb_id as input and outputs similar high rated movies based on description
def get_best_movies_desc_recommendations(imdb_id, count):
    recommended = recommended_movies.reset_index()
    indices_mov_desc = pd.Series(recommended.index, index=recommended['imdb_id']).drop_duplicates()
    idx = indices_mov_desc[imdb_id]
    sim_scores = list(enumerate(cosine_sim_word[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:101]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = recommended.iloc[movie_indices][['imdb_id', 'title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.0)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    return qualified[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(count)




get_best_movies_desc_recommendations('tt0110357', 10)




features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    recommended_movies[feature] = recommended_movies[feature].fillna('[]').apply(literal_eval)




recommended_movies['genres'] = recommended_movies['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
print ("{} \n".format(recommended_movies['genres'].value_counts()))




recommended_movies['keywords'] = recommended_movies['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
print ("{} \n".format(recommended_movies['keywords'].value_counts()))




s = recommended_movies.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:5]




def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words




stemmer = SnowballStemmer('english')
recommended_movies['keywords'] = recommended_movies['keywords'].apply(filter_keywords)
recommended_movies['keywords'] = recommended_movies['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
print ("{} \n".format(recommended_movies['keywords'].value_counts()))




def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan




recommended_movies['director'] = recommended_movies['crew'].apply(get_director)
print ("{} \n".format(recommended_movies['director'].value_counts()))




recommended_movies['main_actors'] = recommended_movies['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
recommended_movies['main_actors'] = recommended_movies['main_actors'].apply(lambda x: x[:3] if len(x) >=3 else x)
print ("{} \n".format(recommended_movies['main_actors'].value_counts()))




# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''




features = ['main_actors', 'director', 'keywords', 'genres']

for feature in features:
    recommended_movies[feature] = recommended_movies[feature].apply(clean_data)




def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['main_actors']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

recommended_movies['soup'] = recommended_movies.apply(create_soup, axis=1)




recommended_movies[['imdb_id', 'title', 'genres', 'main_actors', 'director', 'keywords', 'soup']].head(5)




count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(recommended_movies['soup'])




# Compute the Cosine Similarity matrix based on the count_matrix
count_matrix = count_matrix.astype(np.float32)
cosine_sim_count = cosine_similarity(count_matrix, count_matrix)




## put after cosine_sim_word
from numpy import save
# save to npy file
save('cosine_sim_count.npy', cosine_sim_count)




recommended_movies.to_csv(r'recommended_movies.csv', header=True, index=True)




# Function that takes in movie imdb_id as input and outputs similar movies based on metadata
def get_movie_metadata_recommendations(imdb_id, count):
    recommended = recommended_movies.reset_index()
    indices_mov_mtdt = pd.Series(recommended.index, index=recommended['imdb_id'])
    idx = indices_mov_mtdt[imdb_id]
    sim_scores = list(enumerate(cosine_sim_count[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:101]
    movie_indices = [i[0] for i in sim_scores]
    movies = recommended[['imdb_id','title', 'vote_count', 'vote_average', 'year']].iloc[movie_indices]
    return movies.head(count)




get_movie_metadata_recommendations('tt0110357', 10) 




# Function that takes in movie imdb_id as input and outputs similar high rated movies based on metadata
def get_best_movies_metadata_recommendations(imdb_id, count):
    recommended = recommended_movies.reset_index()
    indices_mov_mtdt = pd.Series(recommended.index, index=recommended['imdb_id'])
    idx = indices_mov_mtdt[imdb_id]
    sim_scores = list(enumerate(cosine_sim_count[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:101]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = recommended.iloc[movie_indices][['imdb_id', 'title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.0)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    return qualified[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(count)




get_best_movies_metadata_recommendations('tt0110357', 10)




def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan




links_small_id = pd.read_csv("../input/the-movies-dataset/links_small.csv")[['movieId', 'tmdbId']]
links_small_id['tmdbId'] = links_small_id['tmdbId'].apply(convert_int)
links_small_id.columns = ['movieId', 'id']
links_small_id = links_small_id.merge(recommended_movies[['imdb_id', 'title', 'id']], on='id').set_index('imdb_id')
links_small_id.head(10)




links_small_id.to_csv(r'links_small_id.csv', header=True, index=True)




reader = Reader()
ratings = pd.read_csv("../input/the-movies-dataset/ratings_small.csv")
ratings.head()




data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)




svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)




trainset = data.build_full_trainset()
svd.fit(trainset)




ratings[ratings['userId'] == 1]




svd.predict(1, 302, 3)




# Function that takes in movie imdb_id as input and outputs similar movies based on user prediction
def get_user_recommendations(userId, imdb_id, movies_count):
    indices_user_id = links_small_id.set_index('id')
    links_small = links_small_id.reset_index()
    indices_mov_mtdt = pd.Series(links_small.index, index=links_small['imdb_id'])

    idx = indices_mov_mtdt[imdb_id]
    sim_scores = list(enumerate(cosine_sim_count[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = recommended_movies.iloc[movie_indices][['imdb_id','title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_user_id.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies[['imdb_id','title', 'vote_count', 'vote_average', 'year']].head(movies_count)




get_user_recommendations(1, 'tt0499549', 10) # Avatar




get_user_recommendations(500, 'tt0499549', 10) # Avatar




class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \.reset_index().rename(columns={user_id: 'genres'})
        
        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['imdb_id'].isin(items_to_ignore)] \.sort_values('genres', ascending = False) \.head(topn)
        
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
                
            recommendations_df = recommendations_df.merge(self.items_df, how = 'left',left_on = 'imdb_id',right_on = 'imdb_id')[['genres', 'imdb_id', 'title', 'keywords', 'soup']]


        return recommendations_df
    
    
    recommended_movies[['imdb_id', 'title', 'genres', 'main_actors', 'director', 'keywords', 'soup']].head(5)
cf_recommender_model = CFRecommender(cf_preds_df, articles_df)




print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)

