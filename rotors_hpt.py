import pathlib
import datetime
import os
import functools
import math

import numpy
import pandas
import dask

import sklearn
import sklearn.preprocessing
import sklearn.model_selection

import tensorflow

import tensorflow.keras
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.metrics
import tensorflow.keras.layers
import tensorflow.keras.constraints

import intake

import ray.tune.integration.keras 
# import ray
# import ray.tune

temp_feature_names = [f'air_temp_{i1}' for i1 in range(1,23)]
humidity_feature_names = [f'sh_{i1}' for i1 in range(1,23)]
wind_direction_feature_names = [f'winddir_{i1}' for i1 in range(1,23)]
wind_speed_feature_names = [f'windspd_{i1}' for i1 in range(1,23)]

timestamp_template = '{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'
rse_run_name_template = 'rse_rotors_{network_name}_' + timestamp_template


def get_v_wind(wind_dir_name, wind_speed_name, row1):
    return math.cos(math.radians(row1[wind_dir_name])) * row1[wind_speed_name]

def get_u_wind(wind_dir_name, wind_speed_name, row1):
    return math.sin(math.radians(row1[wind_dir_name])) * row1[wind_speed_name]

def get_data():
    try:
        rse_root_data_dir = pathlib.Path(os.environ['RSE22_ROOT_DATA_DIR'])
    except KeyError as ke1:
        rse_root_data_dir = pathlib.Path(os.environ['HOME'])  / 'data' / 'ukrse2022'
    rotors_catalog = intake.open_catalog(rse_root_data_dir / 'rotors_catalog.yml')
    rotors_df = rotors_catalog['rotors'].read()
    rotors_df  = rotors_df [(rotors_df ['wind_speed_obs'] >= 0.0) &
                                (rotors_df ['air_temp_obs'] >= 0.0) &
                                (rotors_df ['wind_direction_obs'] >= 0.0) &
                                (rotors_df ['dewpoint_obs'] >= 0.0) 
                               ]
    rotors_df ['DTG'] = dask.dataframe.to_datetime(rotors_df['DTG'])
    rotors_df  = rotors_df .drop_duplicates(subset=['DTG'])
    rotors_df  = rotors_df [~rotors_df['DTG'].isnull()]
    
    temp_feature_names = [f'air_temp_{i1}' for i1 in range(1,23)]
    humidity_feature_names = [f'sh_{i1}' for i1 in range(1,23)]
    wind_direction_feature_names = [f'winddir_{i1}' for i1 in range(1,23)]
    wind_speed_feature_names = [f'windspd_{i1}' for i1 in range(1,23)] 
    
    u_feature_template = 'u_wind_{level_ix}'
    v_feature_template = 'v_wind_{level_ix}'
    u_wind_feature_names = []
    v_wind_features_names = []
    for wsn1, wdn1 in zip(wind_speed_feature_names, wind_direction_feature_names):
        level_ix = int( wsn1.split('_')[1])
        u_feature = u_feature_template.format(level_ix=level_ix)
        u_wind_feature_names += [u_feature]
        rotors_df[u_feature] = rotors_df.apply(functools.partial(get_u_wind, wdn1, wsn1), axis='columns')
        v_feature = v_feature_template.format(level_ix=level_ix)
        v_wind_features_names += [v_feature]
        rotors_df[v_feature] = rotors_df.apply(functools.partial(get_v_wind, wdn1, wsn1), axis='columns')
        
    feature_names_dict = {
        'u_wind': u_wind_feature_names,
        'v_wind': v_wind_features_names,
    }
    return rotors_df, feature_names_dict
    
    
def preproc_input(data_subset, pp_dict):
    return numpy.concatenate([scaler1.transform(data_subset[[if1]]) for if1,scaler1 in pp_dict.items()],axis=1)


def preproc_target(data_subset, enc1, feature_name):
     return enc1.transform(data_subset[[feature_name]])


def make_ml_ready_data(rotors_df, input_feature_names, target_feature_name):
    train_df = rotors_df[rotors_df['DTG'] < datetime.datetime(2020,1,1,0,0)]
    val_df = rotors_df[rotors_df['DTG'] > datetime.datetime(2020,1,1,0,0)]

    preproc_dict = {}
    for if1 in input_feature_names:
        scaler1 = sklearn.preprocessing.StandardScaler()
        scaler1.fit(train_df[[if1]])
        preproc_dict[if1] = scaler1

    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_encoder.fit(train_df[[target_feature_name]])    
    
    X_train = preproc_input(train_df, preproc_dict)
    y_train = numpy.concatenate(
        [preproc_target(train_df, target_encoder, target_feature_name).reshape((-1,1)),
        1.0 - (preproc_target(train_df, target_encoder, target_feature_name).reshape((-1,1))),],
        axis=1
    )    
    
    X_val = preproc_input(val_df, preproc_dict)
    y_val = numpy.concatenate(
        [preproc_target(val_df, target_encoder, target_feature_name).reshape((-1,1)),
        1.0 - (preproc_target(val_df, target_encoder, target_feature_name).reshape((-1,1))),],
        axis=1
    )
    return X_train, y_train, X_val, y_val
           
            
def build_ffnn_model(hyperparameters, input_shape):
    """
    Build a feed forward neural network model in tensorflow for predicting the occurence of turbulent orographically driven wind gusts called Rotors.
    """
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Dropout(hyperparameters['drop_out_rate'], 
                                              input_shape=input_shape))
    for i in numpy.arange(0,hyperparameters['n_layers']):
        model.add(tensorflow.keras.layers.Dense(hyperparameters['n_nodes'], 
                                                activation=hyperparameters['activation'], 
                                                kernel_constraint=tensorflow.keras.constraints.max_norm(3)))
        model.add(tensorflow.keras.layers.Dropout(hyperparameters['drop_out_rate']))
    model.add(tensorflow.keras.layers.Dense(2, activation='softmax'))             # This is the output layer
    return model


def do_training(hyperparameters_dict, X_train, y_train, X_val, y_val, input_shape):
    """
    """
    # current_run_name = rse_run_name_template.format(network_name='ffnn',
    #                                             dt=datetime.datetime.now()
    #                                            )
    # with mlflow.start_run(experiment_id=rse_rotors_exp.experiment_id, run_name=current_run_name) as current_run:
    rotors_ffnn_model = build_ffnn_model(hyperparameters=hyperparameters_dict,
                                         input_shape=input_shape,
                                        )
    rotors_ffnn_optimizer = tensorflow.optimizers.Adam(
        learning_rate=hyperparameters_dict['initial_learning_rate'])  
    
    rotors_ffnn_model.compile(optimizer=rotors_ffnn_optimizer, 
                          loss=hyperparameters_dict['loss'], 
                          metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])
    
    history=rotors_ffnn_model.fit(
        X_train, 
        y_train, 
        validation_data=(X_val, 
                          y_val), 
        epochs=hyperparameters_dict['n_epochs'], 
        batch_size=hyperparameters_dict['batch_size'], 
        shuffle=True,
        verbose=0,
        callbacks=[ray.tune.integration.keras.TuneReportCallback({'root_mean_squared_error': 'root_mean_squared_error'})],
    )    
    
def run_ml_pipeline(config):
    hyperparameters_dict = {
        'initial_learning_rate': config['initial_learning_rate'],
        'drop_out_rate': 0.2,
        'n_epochs': 100,
        'batch_size': 1000,
        'n_nodes': config['n_nodes'],
        'n_layers': config['n_layers'],
        'activation': 'relu',
        'loss': 'mse'
    }
    
    rotors_df, feature_names_dict = get_data()
    u_wind_feature_names = feature_names_dict['u_wind']
    v_wind_feature_names = feature_names_dict['v_wind']
    target_feature_name = 'rotors_present'
    
    input_feature_names = temp_feature_names + humidity_feature_names + u_wind_feature_names + v_wind_feature_names
    
    
    X_train, y_train, X_val, y_val = make_ml_ready_data(rotors_df, input_feature_names, target_feature_name)

    input_shape = (X_train.shape[1],)
    rotors_model = do_training(hyperparameters_dict, X_train, y_train, X_val, y_val, input_shape)
    
    
    

        