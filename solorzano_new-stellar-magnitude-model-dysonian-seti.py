#!/usr/bin/env python
# coding: utf-8



import pandas as pd

data_temp = pd.read_csv('../input/257k-gaiadr2-sources-with-photometry.csv', dtype={'source_id': str})




len(data_temp)




data_temp.columns




should_remove_set = set(pd.read_csv('../input/257k-gaiadr2-should-remove.csv', dtype={'source_id': str})['source_id'])




data_temp = data_temp[~data_temp['source_id'].isin(should_remove_set)]
data_temp.reset_index(inplace=True, drop=True)




len(data_temp)




assert len(data_temp) == len(set(data_temp['source_id']))




import numpy as np

np.random.seed(2018080028)

train_mask = np.random.rand(len(data_temp)) < 0.9
work_data = data_temp[train_mask]
work_data.reset_index(inplace=True, drop=True)
test_data = data_temp[~train_mask]
test_data.reset_index(inplace=True, drop=True)
data_temp = None # Get rid of big frame




len(work_data)




import inspect

pd_concat_argspec = inspect.getfullargspec(pd.concat)
pd_concat_has_sort = 'sort' in pd_concat_argspec.args

def pd_concat(frames):
    # Due to Pandas versioning issue
    new_frame = pd.concat(frames, sort=False) if pd_concat_has_sort else pd.concat(frames)
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame
    
def plt_hist(x, bins=30):
    # plt.hist() can be very slow.
    histo, edges = np.histogram(x, bins=bins)
    plt.bar(0.5 * edges[1:] + 0.5 * edges[:-1], histo, width=(edges[-1] - edges[0])/(len(edges) + 1))




import types
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 

def get_cv_model_transform(data_frame, label_extractor, var_extractor, trainer_factory, response_column='response', 
                           id_column='source_id', n_runs=2, n_splits=2, max_n_training=None, scale = False,
                           trim_fraction=None):
    '''
    Creates a transform function that results from training a regression model with cross-validation.
    The transform function takes a frame and adds a response column to it.
    '''
    default_model_list = []
    sum_series = pd.Series([0] * len(data_frame))
    for r in range(n_runs):
        shuffled_frame = data_frame.sample(frac=1)
        shuffled_frame.reset_index(inplace=True, drop=True)
        response_frame = pd.DataFrame(columns=[id_column, 'response'])
        kf = KFold(n_splits=n_splits)
        first_fold = True
        for train_idx, test_idx in kf.split(shuffled_frame):
            train_frame = shuffled_frame.iloc[train_idx]
            if trim_fraction is not None:
                helper_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor] 
                train_label_ordering = np.argsort(helper_labels)
                orig_train_len = len(train_label_ordering)
                head_tail_len_to_trim = int(round(orig_train_len * trim_fraction * 0.5))
                assert head_tail_len_to_trim > 0
                trimmed_ordering = train_label_ordering[head_tail_len_to_trim:-head_tail_len_to_trim]
                train_frame = train_frame.iloc[trimmed_ordering]
            if max_n_training is not None:
                train_frame = train_frame.sample(max_n_training)
            train_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor]
            test_frame = shuffled_frame.iloc[test_idx]
            train_vars = var_extractor(train_frame)
            test_vars = var_extractor(test_frame)
            scaler = None
            if scale:
                scaler = StandardScaler()  
                scaler.fit(train_vars)
                train_vars = scaler.transform(train_vars)  
                test_vars = scaler.transform(test_vars) 
            trainer = trainer_factory()
            fold_model = trainer.fit(train_vars, train_labels)
            test_responses = fold_model.predict(test_vars)
            test_id = test_frame[id_column]
            assert len(test_id) == len(test_responses)
            fold_frame = pd.DataFrame({id_column: test_id, 'response': test_responses})
            response_frame = pd_concat([response_frame, fold_frame])
            if first_fold:
                first_fold = False
                default_model_list.append((scaler, fold_model,))
        response_frame.sort_values(id_column, inplace=True)
        response_frame.reset_index(inplace=True, drop=True)
        assert len(response_frame) == len(data_frame), 'len(response_frame)=%d' % len(response_frame)
        sum_series += response_frame['response']
    cv_response = sum_series / n_runs
    assert len(cv_response) == len(data_frame)
    assert len(default_model_list) == n_runs
    response_map = dict()
    sorted_id = np.sort(data_frame[id_column].values) 
    for i in range(len(cv_response)):
        response_map[str(sorted_id[i])] = cv_response[i]
    response_id_set = set(response_map)
    
    def _transform(_frame):
        _in_trained_set = _frame[id_column].astype(str).isin(response_id_set)
        _trained_frame = _frame[_in_trained_set].copy()
        _trained_frame.reset_index(inplace=True, drop=True)
        if len(_trained_frame) > 0:
            _trained_id = _trained_frame[id_column]
            _tn = len(_trained_id)
            _response = pd.Series([None] * _tn)
            for i in range(_tn):
                _response[i] = response_map[str(_trained_id[i])]
            _trained_frame[response_column] = _response
        _remain_frame = _frame[~_in_trained_set].copy()
        _remain_frame.reset_index(inplace=True, drop=True)
        if len(_remain_frame) > 0:
            _unscaled_vars = var_extractor(_remain_frame)
            _response_sum = pd.Series([0] * len(_remain_frame))
            for _model_tuple in default_model_list:
                _scaler = _model_tuple[0]
                _model = _model_tuple[1]
                _vars = _unscaled_vars if _scaler is None else _scaler.transform(_unscaled_vars)
                _response = _model.predict(_vars)
                _response_sum += _response
            _remain_frame[response_column] = _response_sum / len(default_model_list)
        _frames_list = [_trained_frame, _remain_frame]
        return pd_concat(_frames_list)
    return _transform




import scipy.stats as stats

def print_evaluation(data_frame, label_column, response_column):
    response = response_column(data_frame) if isinstance(response_column, types.FunctionType) else data_frame[response_column]
    label = label_column(data_frame) if isinstance(label_column, types.FunctionType) else data_frame[label_column]
    residual = label - response
    rmse = np.sqrt(sum(residual ** 2) / len(data_frame))
    correl = stats.pearsonr(response, label)[0]
    print('RMSE: %.4f | Correlation: %.4f' % (rmse, correl,), flush=True)




def transform_init(data_frame):    
    new_frame = data_frame.copy()
    new_frame.reset_index(inplace=True, drop=True)
    distance = 1000.0 / new_frame['parallax']
    new_frame['distance'] = distance
    new_frame['abs_mag_ne'] = new_frame['phot_g_mean_mag'] - 5 * (np.log10(distance) - 1)
    new_frame['color_index'] = new_frame['phot_bp_mean_mag'] - new_frame['phot_rp_mean_mag']
    return new_frame




work_data = transform_init(work_data)




mag_column_groups = [
    ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'],
    ['gsc23_v_mag', 'gsc23_b_mag'],
    ['ppmxl_b1mag', 'ppmxl_b2mag', 'ppmxl_r1mag', 'ppmxl_imag'],
    ['tmass_j_m', 'tmass_h_m', 'tmass_ks_m'],
    ['tycho2_bt_mag', 'tycho2_vt_mag'],
]




def populate_mag_columns(data_frame, feature_list):
    for group in mag_column_groups:
        len_group = len(group)
        assert len_group >= 2
        for i in range(1, len_group):
            mag_diff = data_frame[group[i]] - data_frame[group[i - 1]]
            feature_list.append(mag_diff)




# Hyperparameters for gaussian transformations of distance-from-plane.
PLANE_DENSITY_2T_VAR1 = 2 * 15 ** 2
PLANE_DENSITY_2T_VAR2 = 2 * 50 ** 2




def extract_model_vars(data_frame):
    distance = data_frame['distance'].values
    log_distance = np.log(distance)
    latitude_rad = np.deg2rad(data_frame['b'].values)
    longitude_rad = np.deg2rad(data_frame['l'].values)
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_long = np.sin(longitude_rad)
    cos_long = np.cos(longitude_rad)
    
    distance_in_plane = np.abs(distance * cos_lat)
    distance_from_plane_sq = (distance * sin_lat) ** 2
    plane_density_feature1 = np.exp(-distance_from_plane_sq / PLANE_DENSITY_2T_VAR1)
    plane_density_feature2 = np.exp(-distance_from_plane_sq / PLANE_DENSITY_2T_VAR2)
    feature_list = [log_distance, distance, 
                    distance_in_plane, 
                    plane_density_feature1, plane_density_feature2,
                    sin_lat, cos_lat, sin_long, cos_long
                   ]
    
    populate_mag_columns(data_frame, feature_list)
    mag_g = data_frame['phot_g_mean_mag']
    mag_rp = data_frame['phot_rp_mean_mag']
    mag_bp = data_frame['phot_bp_mean_mag']
    feature_list.append(mag_g - data_frame['allwise_w2'])
    feature_list.append(mag_bp - data_frame['tmass_j_m'])
    feature_list.append(mag_bp - data_frame['gsc23_b_mag'])
    feature_list.append(mag_g - data_frame['ppmxl_r1mag'])
    feature_list.append(mag_rp - data_frame['ppmxl_imag'])
    feature_list.append(mag_rp - data_frame['tycho2_bt_mag'])
    feature_list.append(data_frame['tmass_j_m'] - data_frame['allwise_w2'])
    feature_list.append(data_frame['gsc23_b_mag'] - data_frame['ppmxl_imag'])
    
    return np.transpose(feature_list)    




LABEL_COLUMN = 'phot_g_mean_mag'




MAX_N_TRAINING = 50000




from sklearn.neural_network import MLPRegressor

def get_nn_trainer():
    return MLPRegressor(hidden_layer_sizes=(60, 60), max_iter=400, alpha=0.1, random_state=np.random.randint(1,10000))




def get_nn_transform(label_column):
    return get_cv_model_transform(work_data, label_column, extract_model_vars, get_nn_trainer, 
        n_runs=3, n_splits=2, max_n_training=MAX_N_TRAINING, response_column='nn_' + label_column, scale=True)




transform_nn = get_nn_transform(LABEL_COLUMN)
work_data = transform_nn(work_data)




print_evaluation(work_data, LABEL_COLUMN, 'nn_' + LABEL_COLUMN)




import lightgbm

def get_lgbm_trainer():
    return lightgbm.LGBMRegressor(num_leaves=80, max_depth=-1, learning_rate=0.1, n_estimators=1000, 
        subsample_for_bin=50000, reg_alpha=0.03, reg_lambda=0.0,
        random_state=np.random.randint(1,10000))




def get_lgbm_transform(label_column):
     return get_cv_model_transform(work_data, label_column, extract_model_vars, get_lgbm_trainer, 
        n_runs=2, n_splits=2, max_n_training=MAX_N_TRAINING, response_column='lgbm_' + label_column, scale=False)




transform_lgbm = get_lgbm_transform(LABEL_COLUMN)
work_data = transform_lgbm(work_data)




print_evaluation(work_data, LABEL_COLUMN, 'lgbm_' + LABEL_COLUMN)




def extract_blend_vars(data_frame):
    lgbm_responses = data_frame['lgbm_' + LABEL_COLUMN].values
    nn_responses = data_frame['nn_' + LABEL_COLUMN].values
    return np.transpose([lgbm_responses, nn_responses])




from sklearn import linear_model

def get_blend_trainer():
    return linear_model.LinearRegression()




def get_blend_transform(label_column):
    return get_cv_model_transform(work_data, label_column, extract_blend_vars, get_blend_trainer, 
        n_runs=3, n_splits=3, max_n_training=None, response_column='blend_' + label_column, scale=False)




transform_blend = get_blend_transform(LABEL_COLUMN)
work_data = transform_blend(work_data)




print_evaluation(work_data, LABEL_COLUMN, 'blend_' + LABEL_COLUMN)




def get_bres_label(data_frame):
    return data_frame[LABEL_COLUMN] - data_frame['blend_' + LABEL_COLUMN]




def extract_bres_vars(data_frame):
    distance = data_frame['distance'].values
    latitude = np.deg2rad(data_frame['b'].values)
    longitude = np.deg2rad(data_frame['l'].values)
    position_z = distance * np.sin(latitude)
    projection = distance * np.cos(latitude)
    position_x= projection * np.cos(longitude)
    position_y = projection * np.sin(longitude)
    color_index = data_frame['color_index']
    return np.transpose([position_x, position_y, position_z,
                        color_index])    




from sklearn.ensemble import RandomForestRegressor

def get_bres_trainer():
    return RandomForestRegressor(n_estimators=60, max_depth=18, min_samples_split=30, random_state=np.random.randint(1,10000))




transform_bres = get_cv_model_transform(work_data, get_bres_label, extract_bres_vars, get_bres_trainer, 
        n_runs=3, n_splits=2, max_n_training=MAX_N_TRAINING, response_column='modeled_bres', scale=False,
        trim_fraction=0.003)




work_data = transform_bres(work_data)
print_evaluation(work_data, get_bres_label, 'modeled_bres')




def transform_final_model(data_frame):
    new_frame = data_frame.copy()
    new_frame['model_response'] = new_frame['blend_' + LABEL_COLUMN] + new_frame['modeled_bres']
    return new_frame




work_data = transform_final_model(work_data)
print_evaluation(work_data, LABEL_COLUMN, 'model_response')




def color_index(data_frame):
    return data_frame['phot_bp_mean_mag'] - data_frame['phot_rp_mean_mag']




def giant_separation_y(x):
    return x * 40.0 - 25




import matplotlib.pyplot as plt




work_data_sample = work_data.sample(2000)
plt.rcParams['figure.figsize'] = (10, 5)
_color_index = color_index(work_data_sample)
plt.scatter(_color_index, work_data_sample['abs_mag_ne'] ** 2, s=2)
plt.plot(_color_index, giant_separation_y(_color_index), '--', color='orange')
plt.gca().invert_yaxis()
plt.title('Pseudo H-R diagram for giant removal')
plt.xlabel('BP - RP color index')
plt.ylabel('Absolute magnitude squared')
plt.show()




def transform_rm_giants(data_frame):
    new_frame = data_frame[data_frame['abs_mag_ne'] ** 2 >= giant_separation_y(color_index(data_frame))]
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame




work_data = transform_rm_giants(work_data)
len(work_data)




RESPONSE_COLUMN = 'model_response'




def transform_residual(data_frame):
    new_frame = data_frame.copy()
    new_frame['model_residual'] = data_frame[LABEL_COLUMN] - data_frame[RESPONSE_COLUMN]
    return new_frame




work_data = transform_residual(work_data)




mean_model_residual = np.mean(work_data['model_residual'].values)

def get_squared_res_label(data_frame):
    return (data_frame['model_residual'] - mean_model_residual) ** 2




def extract_residual_vars(data_frame):
    parallax = data_frame['parallax']
    parallax_error = data_frame['parallax_error']
    parallax_high = parallax + parallax_error
    parallax_low = parallax - parallax_error
    var_error_diff = np.log(parallax_high) - np.log(parallax_low)
    
    flux_error = data_frame['phot_g_mean_flux_error']
    
    latitude_rad = np.deg2rad(data_frame['b'].values)
    longitude_rad = np.deg2rad(data_frame['l'].values)
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_long = np.sin(longitude_rad)
    cos_long = np.cos(longitude_rad)

    distance = data_frame['distance']
    distance_in_plane = np.abs(distance * cos_lat)
    distance_from_plane_sq = (distance * sin_lat) ** 2
    plane_density_feature1 = np.exp(-distance_from_plane_sq / PLANE_DENSITY_2T_VAR1)
    plane_density_feature2 = np.exp(-distance_from_plane_sq / PLANE_DENSITY_2T_VAR2)
    return np.transpose([
        distance,
        sin_lat, cos_lat, sin_long, cos_long,
        plane_density_feature1, plane_density_feature2,
        var_error_diff,
        flux_error
    ])




def get_residual_trainer():
    return RandomForestRegressor(n_estimators=60, max_depth=9, min_samples_split=10, random_state=np.random.randint(1,10000))




transform_expected_res_sq = get_cv_model_transform(work_data, get_squared_res_label, extract_residual_vars, get_residual_trainer, 
        n_runs=4, n_splits=2, max_n_training=MAX_N_TRAINING, response_column='expected_res_sq', scale=False,
        trim_fraction=0.003)




work_data = transform_expected_res_sq(work_data)




print_evaluation(work_data, get_squared_res_label, 'expected_res_sq')




def transform_anomaly(data_frame):
    new_frame = data_frame.copy()
    new_frame_residual = new_frame['model_residual'].values
    new_frame['anomaly'] = (new_frame_residual - mean_model_residual) / np.sqrt(new_frame['expected_res_sq'].astype(float))
    return new_frame




work_data = transform_anomaly(work_data)




np.std(work_data['anomaly'])




transform_list = [transform_init,                          # extra info columns
                  transform_lgbm, transform_nn,            # individual models
                  transform_blend,                         # the blend
                  transform_bres, transform_final_model,   # position-based residual correction
                  transform_rm_giants,                     # removal of giants
                  transform_residual,                      # add the residual column
                  transform_expected_res_sq,               # regional residual variance
                  transform_anomaly                        # anomaly metric
                 ]




def combined_transform(data_frame):
    _frame = data_frame
    for t in transform_list:
        _frame = t(_frame)
    return _frame




test_data = combined_transform(test_data)




np.std(test_data['model_residual'])




np.std(work_data['model_residual'])




data = pd_concat([work_data, test_data])
work_data = None
test_data = None




len(data)




data[data['source_id'] == '2081900940499099136'][
    ['source_id', 'distance', 'abs_mag_ne', 'model_residual', 'anomaly']]




CAND_SD_THRESHOLD = 3.0




data_anomalies = data['anomaly']




anomaly_std = np.std(data_anomalies)




cand_threshold = anomaly_std * CAND_SD_THRESHOLD
candidates = data[data_anomalies >= cand_threshold]
len(candidates)




bright_control_group = data.sort_values('anomaly', ascending=True).head(len(candidates))




normal_control_group = data[(data_anomalies < anomaly_std) & (data_anomalies > -anomaly_std)].sample(len(candidates))




data_anomalies = None # Discard big array




def get_position_frame(data_frame):
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame['source_id'] = data_frame['source_id'].values
    distance = data_frame['distance'].values
    latitude = np.deg2rad(data_frame['b'].values)
    longitude = np.deg2rad(data_frame['l'].values)
    new_frame['z'] = distance * np.sin(latitude)
    projection = distance * np.cos(latitude)
    new_frame['x'] = projection * np.cos(longitude)
    new_frame['y'] = projection * np.sin(longitude)
    return new_frame




def get_sun():
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame.loc[0] = ['sun', 0.0, 0.0, 0.0]
    return new_frame




candidates_wbstar = pd_concat([candidates, data[data['source_id'] == '2081900940499099136']])
candidates_pos_frame = pd_concat([get_position_frame(candidates_wbstar), get_sun()])




import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=False)




def plot_pos_frame(pos_frame, star_color, sun_color = 'blue', bstar_color = 'black'):    
    star_color = [(bstar_color if row['source_id'] == '2081900940499099136' else (sun_color if row['source_id'] == 'sun' else star_color)) for _, row in pos_frame.iterrows()]
    trace1 = go.Scatter3d(
        x=pos_frame['x'],
        y=pos_frame['y'],
        z=pos_frame['z'],
        mode='markers',
        text=pos_frame['source_id'],
        marker=dict(
            size=3,
            color=star_color,
            opacity=0.67
        )
    )
    scatter_data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=scatter_data, layout=layout)
    py.iplot(fig)




get_ipython().run_cell_magic('html', '', '<!-- Allow bigger output cells -->\n<style>\n.output_wrapper, .output {\n    height:auto !important;\n    max-height: 1500px;\n}\n</style>')




normal_control_group_wbstar = pd_concat([normal_control_group, data[data['source_id'] == '2081900940499099136']])
normal_control_group_pos_frame = pd_concat([get_position_frame(normal_control_group_wbstar), get_sun()])
plot_pos_frame(normal_control_group_pos_frame, 'gray')




plot_pos_frame(candidates_pos_frame, 'green')




bright_control_group_wbstar = pd_concat([bright_control_group, data[data['source_id'] == '2081900940499099136']])
bright_control_group_pos_frame = pd_concat([get_position_frame(bright_control_group_wbstar), get_sun()])
plot_pos_frame(bright_control_group_pos_frame, 'red')




SAVED_COLUMNS = ['source_id', 'tycho2_id', 'ra', 'dec', 'pmra', 'pmdec', 'l', 'b', 'distance', 'color_index',
                 LABEL_COLUMN, 'blend_' + LABEL_COLUMN, 'model_residual', 'anomaly']




data[SAVED_COLUMNS].to_csv('mag-modeling-results.csv', index=False)

