#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse, io
import gc
import time
from datetime import datetime




toy_dataset = False
# you able to use you own toy dataset to check how algorithm works
if toy_dataset:
    user = np.array([0,0,0,1, 1,1, 1,2,2,3,3]).astype(np.integer)
    item = np.array([2,1,3,0, 1,3, 5, 3,1,5,3]).astype(np.integer)
    rating = np.array([5,4,3,7, 8,9, 10, 5,2,4,7]).astype(np.float32)
    ts = [
        datetime.fromtimestamp(
            time.mktime(
                time.strptime(
                    "2017-09-{} 13:00".format(i), "%Y-%m-%d %H:%M"
        ))) 
        for i in range(1,12)]
    raw_data = pd.DataFrame({"userId": user,"movieId":item,"rating":rating, "timestamp": ts})
else:
    raw_data = pd.read_csv("../input/rating.csv")
raw_data.head(3)




movie_df = pd.read_csv("../input/movie.csv")
movie_df.head(1)
class Catalog():
    def __init__(self, movie_df):
        self.catalog = movie_df
        
    def get_titles(self, movie_list):
        """get content title by id
        
        return list of titles
        """
        return self.catalog[self.catalog.movieId.isin(movie_list)]["title"].values.tolist()

catalog = Catalog(movie_df)
catalog.get_titles([1,2,3])




raw_data.describe(percentiles=[])




raw_data.groupby("userId")["movieId"]    .agg(["count"])     .reset_index()     .groupby("count")["userId"]     .agg('count')     .sort_values(ascending=False)     .head(100)     .plot.hist(bins=10, title="film popularity distribution")




def random_index_sample(total_rows, row_sample):
    sample_capacity = np.ceil(
        total_rows * row_sample
    ).astype(int)
    row_num = range(total_rows)
    return np.random.choice(row_num, sample_capacity, replace=False)

def random_sampling_from_sparse(ui_matrix, row_sample):
    """Random sample of substring from ui_matrix
    
    preserve order of matrix rows
    """
    num_rows = ui_matrix.shape[0]
    random_index = random_index_sample(num_rows, row_sample)
    # preserve order of rows after sampling
    random_index = np.sort(random_index)
    return ui_matrix[random_index, :]

def user_sampling_from_df(ui_df, row_col_label, user_sample):
    """Random sample of users
    
    filter sourse dataframe rows to select random subsample of users
    """
    print("Rows in source df {}".format(ui_df.shape[0]))
    num_rows = ui_df[row_col_label].max()
    random_index = random_index_sample(num_rows, user_sample)
    # preserve order of rows after sampling
    ui_df = ui_df[ui_df[row_col_label].isin(random_index)]
    print("Rows in df after user sampling {}".format(ui_df.shape[0]))
    return ui_df

def df2matrix(df, row_label, col_label, feedback_label, shape=None):
    """Convert
    
    """
    row_index = df[row_label]
    col_index = df[col_label]
    feedback_values = df[feedback_label]
    if shape is None:
        shape = (row_index.max() + 1, col_index.max() + 1)
    ui_matrix = sparse.csr_matrix(( feedback_values,            (row_index, col_index)),            shape=shape
    ).astype(np.float32)
    return ui_matrix




class DataPreparator(object):
    def __init__(self, df, user_col, item_col, ts_col, feedback_col,
        item_threshold = 25, user_threshold = 10):
        self.data = df
        # explicit convert to timestamp
        self.data[ts_col] = pd.to_datetime(self.data[ts_col])
        self.user_col = user_col
        self.item_col = item_col
        self.ts_col = ts_col
        self.feedback_col = feedback_col
        self.item_threshold = item_threshold
        self.user_threshold = user_threshold
        self.user_encoder = None
        self.item_encoder = None
        self.ui_matrix = None
        self.train_set = None
        self.test_set = None
        self.shape = None
        
    def prepare_data(self, user_sample=0.6):
        """data preparation step
        
        """
        self.drop_duplicates()
        self.filter_data()
        self.data = user_sampling_from_df(
            self.data, self.user_col, user_sample
        )
        self.encode_labels()
        self.prepare_watch_history()
    
    def encode_labels(self):
        self.user_encoder = LabelEncoder()
        self.data[self.user_col] = self.user_encoder.fit_transform(self.data[self.user_col])
        self.item_encoder = LabelEncoder()
        self.data[self.item_col] = self.item_encoder.fit_transform(self.data[self.item_col])
    
    def prepare_watch_history(self):
        user_item_count, aggr_col = self.watch_history_agg('movieId')
        # add watch_history length
        self.data = self.data.merge(user_item_count[[self.user_col, aggr_col]], 
                          how='inner', on=self.user_col, copy=False)
        self.data.rename(columns={aggr_col:'content_max_rank'}, inplace=True)
        self._rank_watch_history()
    
    def watch_history_agg(self, target_col):
        # generate user-item matrix
        # TODO: assert if duplicates in data
        ui_matrix = self.data[[self.user_col, target_col]]                     .groupby(self.user_col)[target_col]                     .apply(np.array)                     .apply(len)
        aggr_col = '{}_agg'.format(target_col)
        ui_matrix = ui_matrix.to_frame(name=aggr_col)
        ui_matrix.reset_index(inplace=True)
        return ui_matrix, aggr_col
    
    def drop_duplicates(self):
        """ simple preprocess data
        
        in case user watch content multiply times
        """
        self.data =             self.data             .groupby([self.user_col,self.item_col])[[self.feedback_col,self.ts_col]]             .agg({self.feedback_col:'mean', self.ts_col:'max'}).reset_index()
    
    def filter_data(self):
        """ Filter data
        
        remove films with low popularity
        and users with few watches
        """
        # filtering users
        self.filter_by_threshold(self.user_col, self.item_col, self.item_threshold)
        # filtering items
        self.filter_by_threshold(self.item_col, self.user_col, self.user_threshold)
    
    def filter_by_threshold(self, group_col, filter_col, threshold):
        """ filter functions
        
        group over group_col then filter
        """
        entry_count = self.data.groupby(group_col)[filter_col]             .apply(np.array)             .apply(np.unique)             .apply(len)
        entry_count.sort_values(ascending = True, inplace=True)
        # determine which items to drop
        drop_entries = entry_count[entry_count > threshold]
        drop_entries = drop_entries.to_frame(name='entry_count').reset_index()             .drop('entry_count', axis=1)
        row_number_before = self.data.shape[0]
        self.data = self.data.merge(drop_entries, how='inner', on=group_col)
        print("Filter {}: shape before {}, shape after {}"               .format(filter_col, row_number_before, self.data.shape[0]))
    
    def _rank_watch_history(self):
        """ Adding content rank for users
        
        ranking over timestamp
        """
        # evaluate content_rank
        content_rank = self.data[[self.user_col, self.item_col, self.ts_col]]             .sort_values([self.user_col, self.ts_col]).groupby([self.user_col]).cumcount() + 1
        self.data['content_rank'] = content_rank
        self.data['content_order'] = self.data['content_rank'] / self.data['content_max_rank']
        # удаляем временные столбцы, content_max_rank оставляем для быстрого разбиения train-test
        self.data.drop(['content_rank'], inplace=True, axis=1)
    
    def train_test_split(self, split_rate):
        # for all users split watch_history by content_order field
        history_split_mask = self.data['content_order'] <= split_rate
        self.train_set = self.data[history_split_mask]
        self.test_set = self.data[~history_split_mask]
        assert self.train_set.shape[0] + self.test_set.shape[0] == self.data.shape[0]
        self.shape = (self.data[self.user_col].max()+1, self.data[self.item_col].max()+1)
        del self.data
        gc.collect()
    
    def ui2sparse(self):
        self.ui_matrix = df2matrix(
            self.train_set, 
            row_label = self.user_col, col_label = self.item_col, 
            feedback_label = self.feedback_col,
            shape = self.shape
        )
        # for future model evaluation
        self.ui_matrix_test = df2matrix(
            self.test_set, 
            row_label = self.user_col, col_label = self.item_col, 
            feedback_label = self.feedback_col, 
            shape = self.shape
        )




data_preparator = DataPreparator(
    raw_data, 'userId', 'movieId', 'timestamp', 'rating'                        
)
data_preparator.prepare_data(user_sample = 0.1)
data_preparator.train_test_split(0.65)
data_preparator.ui2sparse()




ui_matrix = data_preparator.ui_matrix
ui_matrix




#common functions for similarity matrix
def get_batches(num_rows, batch_size):
    return [[j for j in range(i, min(i+batch_size,num_rows))] for i in range(num_rows)[::batch_size]]

def normalize_ui_matrix(ui_matrix, sim_type, batch_size=1):
    # compute avg user rating
    if sim_type == 'user':
        ui_matrix = ui_matrix.tocsr()
    elif sim_type == 'item':
        ui_matrix = ui_matrix.T.tocsr()
    num_cols, num_rows = ui_matrix.shape[1], ui_matrix.shape[0]
    res = []
    start_time = time.clock()
    for i in range(num_rows):
        current_row = ui_matrix.getrow(i)
        nonzero_id = current_row.nonzero()
        # check if row has nonzero elements
        if nonzero_id[0].shape[0] > 0:
            nonzero_elem = current_row[nonzero_id].getA()[0]
            normed_vec = (nonzero_elem - nonzero_elem.mean()).astype(np.float32)
            result_row = sparse.csr_matrix(
                ( normed_vec, nonzero_id),
                shape=(1 , num_cols)
            ).astype(np.float32)
        #if only zeros - produce zero string
        else:
            normed_vec = np.zeros(1)
            result_row = sparse.csr_matrix(
                ( normed_vec, ([0],[0])),
                shape=(1 , num_cols)
            ).astype(np.float32)
        res.append(result_row)
    print ("Matrix normalization computed in {} seconds".format(time.clock() - start_time))
    res = sparse.vstack(res)
    # input and output must have the same shape
    res = res.T.tocsr() if sim_type == 'item' else res
    return res

def euclidean_norm(x):
    return np.sqrt( x.dot(x)  )
    
def nonzero_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def eval_norms(ui_matrix, sim_type):
    ui_matrix = ui_matrix.toarray() if sim_type == 'item' else ui_matrix.T.toarray()
    entries_norms = np.apply_along_axis(
        euclidean_norm, 0, ui_matrix
    )
    return entries_norms




class SimMatrix(object):
    def __init__(self, ui_matrix, sim_type, nn=5,
                 knn_matr=None, batch_size=None, sim_metric='cosine'
    ):
        self.ui_matrix = ui_matrix
        self.sim_matrix = None
        self.item_norms = None
        self.num_users = ui_matrix.shape[0]
        self.num_items = ui_matrix.shape[1]
        start_time = time.clock()
        if knn_matr is None:
            self.similarity_matrix(sim_type=sim_type, batch_size=batch_size, sim_metric=sim_metric, nn=nn)
        else:
            self.sim_matrix = knn_matr
        print ("KNN matrix computed in {} sec".format(time.clock() - start_time))
    
    def similarity_matrix(self, sim_metric, sim_type, nn, batch_size=None):
        """Compute similarity from ui matrix
        
        :param: sim_type - similarity of users or of items
        """
        # zero row for sparse matrix constructor
        nn_zero_row = np.zeros(nn)
        nn = nn + 1
        result_knn_columns = self.num_items if sim_type=='item' else self.num_users
        if result_knn_columns < 0:
            raise IndexError("Neighbors num exceeds dimensions")
        batch_size = result_knn_columns if batch_size is None else batch_size
        sim_matrix_row = []
        # if pearson corralation - need to norm matrix
        if sim_metric=='pearson':
            self.ui_matrix = normalize_ui_matrix(self.ui_matrix, sim_type)
        #evaluate norms for all users (items)
        start_time = time.clock()
        self.item_norms = eval_norms(self.ui_matrix, sim_type)
        print ("Norms computed in {} sec".format(time.clock() - start_time))
        # now need to multiply ui on ui.T - perform it row-wise
        # transposed matrix must have the same type of sparsity - csr or csc
        if sim_type =='item':
            self.ui_matrix = sparse.csr_matrix(self.ui_matrix.T)
            ui_matrix_T = sparse.csr_matrix(self.ui_matrix.T)
        elif sim_type =='user':
            ui_matrix_T = sparse.csr_matrix(self.ui_matrix.T)
        for i in range(batch_size):
            # get ratings for item i
            item_description = self.ui_matrix.getrow(i)
            # compute eucledian distance between other objects
            result_matrix = item_description.dot(ui_matrix_T)
            cosine_sim_vector = result_matrix.toarray()[0]
            # set self-similarity to zero
            cosine_sim_vector[i] = 0.0
            # divide on vectors norm
            norm_cosine_sim_vector =                 cosine_sim_vector / self.item_norms[i]                 if self.item_norms[i] > 0                 else np.zeros(cosine_sim_vector.shape[0])
            #compute cosine similarity norm
            final_vec = nonzero_divide(norm_cosine_sim_vector, self.item_norms)
            # replace negative similarities to zero
            final_vec = np.where(final_vec < 0, 0, final_vec)
            # compute top-k nearest neghbors
            nn_indexes = final_vec.argsort()[:-nn:-1]
            nonzero_values = final_vec[nn_indexes]
            knn_row = sparse.csr_matrix(
                    (nonzero_values, (nn_zero_row, nn_indexes)),
                    shape=(1 , result_knn_columns)
            ).astype(np.float32)
            sim_matrix_row.append(knn_row)
        # stack sparse rows of all items to result_matrix
        self.sim_matrix = sparse.vstack(sim_matrix_row)




# scipy.sparse.csr.csr_matrix
# отладка
sim_matrix = SimMatrix(data_preparator.ui_matrix, sim_type='user', sim_metric='pearson', nn=5)




io.mmwrite("user_similarity.mtx", sim_matrix.sim_matrix)
io.mmwrite("user_item_matrix.mtx", sim_matrix.ui_matrix)




sim_matrix.ui_matrix.shape, data_preparator.ui_matrix.shape




class KNNRecommender(object):
    """K Nearest Neighborhood Recommender
    
    compute recommendations based on nearest neighborhoods preferences
    """
    def __init__(self, sim_matrix, ui_matrix, ui_matrix_train, content_index, catalog):
        """Train matrix for model evaluation
        
        """
        self.sim_matrix = sim_matrix
        self.ui_matrix = ui_matrix
        self.index_to_id = content_index
        self.ui_matrix_train = ui_matrix_train
        self.catalog = catalog
        self.rec_list = None
        
    def get_recommendations(self, user_id=1, top_k=10, verbose=False):
        """ recommend items to user
        
        user index from sim_matrix
        """
        # get nearest neighborhoods
        user_nn_index = self.sim_matrix[user_id, :].nonzero()[1]
        # convert to dense array
        user_nn_weight = self.sim_matrix[user_id, user_nn_index].todense().getA()
        # get neighbors rating from ui matrix
        nn_rating_matrix = self.ui_matrix[user_nn_index,:]
        # cumpute weighted average of rating
        predicted_rating_matrix = user_nn_weight * nn_rating_matrix
        watch_history = self.ui_matrix_train.getrow(user_id).nonzero()[1]
        actual_watch = self.ui_matrix.getrow(user_id).nonzero()[1]
        # sort items by relevance
        recommended_items = np.argsort(predicted_rating_matrix)[0][::-1]
        # return top-k recommendations
        rec_list = recommended_items[:top_k]
        if verbose:
            print("nn_index", user_nn_index)
            print("weight shape", user_nn_weight.shape)
            nn_rating_matrix = self.ui_matrix[user_nn_index,:]
            print("rating matrix", nn_rating_matrix.shape)
            print("predicted matrix", predicted_rating_matrix.shape)
            intersection=list(set(actual_watch.tolist())&set(rec_list.tolist()))
            print(intersection)
            print("User watch history: {}\n User_recommendations {}\n User_actual watch: {}\n\n True prediction: {}".
                format(
                self.catalog.get_titles(self.index_to_id.inverse_transform(watch_history))[:10] ,
                self.catalog.get_titles(self.index_to_id.inverse_transform( rec_list ))[:10] ,
                self.catalog.get_titles(self.index_to_id.inverse_transform(actual_watch))[:10],
                self.catalog.get_titles(self.index_to_id.inverse_transform( intersection ))[:10]
                    if len(intersection)>0 else None,
                )
            )
        self.rec_list = rec_list




# select random user to widh nonzero history
random_user_id = np.random.choice(data_preparator.ui_matrix_test.nonzero()[0])




# pass source ui_matrix (not normalized)
knn_recommender = KNNRecommender(
    sim_matrix.sim_matrix, 
    data_preparator.ui_matrix_test,
    data_preparator.ui_matrix,
    data_preparator.item_encoder,
    catalog
)

knn_recommender.get_recommendations(user_id=random_user_id, top_k=50, verbose=True)




from scipy.sparse.linalg import svds

user_factors ,scale, item_factors = svds(sim_matrix.ui_matrix, k=40, return_singular_vectors=True)
#create square matrix
scale = np.diag(np.sqrt(scale))
user_factors = np.dot(user_factors, scale)
item_factors = np.dot(scale, item_factors)

print(user_factors.shape, scale.shape, item_factors.shape)




class SVDRecommender(object):
    """Support Vector Decomposition Recommender
    
    compute recommendations based on latent factors
    """
    def __init__(self, U, I, ui_matrix, ui_matrix_train, content_index, catalog):
        """Train matrix for model evaluation
        
        """
        self.users_factors = U
        self.items_factors = I
        self.ui_matrix = ui_matrix
        self.ui_matrix_train = ui_matrix_train
        self.index_to_id = content_index
        self.catalog = catalog
        self.rec_list = None
        
    def get_recommendations(self, user_id=1, top_k=10, verbose=False):
        """ recommend items to user
        
        eval recommened items from latent factors
        """
        user_factors = self.users_factors[user_id,:]
        # cumpute rating as dot product of latent factors
        predicted_rating_matrix = np.dot(user_factors.T, self.items_factors)
        actual_watch = self.ui_matrix.getrow(user_id).nonzero()[1]
        watch_history = self.ui_matrix_train.getrow(user_id).nonzero()[1]
        # sort items by relevance
        recommended_items = np.argsort(predicted_rating_matrix)
        # return top-k recommendations
        rec_list = recommended_items[:top_k]
        if verbose:
            intersection=list(set(actual_watch.tolist())&set(rec_list.tolist()))
            print("User watch history: {}\n\n User_recommendations {}\n User_actual watch: {}\n True prediction: {}".
                format(
                self.catalog.get_titles(self.index_to_id.inverse_transform(watch_history))[:20] ,
                self.catalog.get_titles(self.index_to_id.inverse_transform( rec_list ))[:20] ,
                self.catalog.get_titles(self.index_to_id.inverse_transform(actual_watch))[:20],
                self.catalog.get_titles(self.index_to_id.inverse_transform( intersection ))[:20]
                    if len(intersection)>0 else None,
                )
            )
        self.rec_list = rec_list




svd_recommender = SVDRecommender(
    user_factors, 
    item_factors,
    data_preparator.ui_matrix_test,
    data_preparator.ui_matrix,
    data_preparator.item_encoder,
    catalog
)

svd_recommender.get_recommendations(user_id=random_user_id, top_k=50, verbose=True)






