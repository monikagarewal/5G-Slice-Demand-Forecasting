#!/usr/bin/env python

#import nest_asyncio
#nest_asyncio.apply()

import os 
import re
import sys
import time
import collections
import itertools

import tensorflow as tf
import tensorflow_federated as tff

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp 

# creating regex for matching files in dataset
base_model_dir = '../Models/Mi-2013/Centralized/'
client_directory_path = '../Datasets/MI-2013/ClientData'

models = ['GRU','LSTM','CNN-LSTM']
scalers = ['Quantile','Robust','Manual']
optimization_algorithms = ['SGD','RMS-Prop']

config = [optimization_algorithms,scalers]
config_list = [item for item in itertools.product(*config)]

#global parameters
model = 'GRU'
scler = 'Quantile'
opalg = 'SGD'

#forecasting algorithm parameters
BATCH_SIZE = 32
NUM_EPOCHS = 200
NUM_CLIENTS = 100
SHUFFLE_BUFFER = 20
PREFETCH_BUFFER = 10

#time series parameters
NUM_FEATURE = 5
obsv_win_len = 0
pred_win_len = 0
PREDICTION_WINDOW_RANGE = [1,2,4,8,16]
OBSERVATION_WINDOW_RANGE = [6,12,24,36,48]

def preprocess(dataset, slice_id, ob_window_size, pred_window_size, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER, num_epochs=1):
	#Using quantile scaler - [As its more robust to outliers, which can be common in mobile n/w traffic]
	def standardize_ds(data):
		if scler == 'Manual':
			scaled_data = (data - data.mean()) / data.std()
			return scaled_data.values
		elif scler == 'Robust':
			datascaler = skp.RobustScaler()
		else:
			datascaler = skp.QuantileTransformer(n_quantiles=int(0.66*data.shape[0]),random_state=10)
		return datascaler.fit_transform(data.values)
	#reshape the tensor to add slice info to features 
	def reshape_sample(x,y):
		if model == 'CNN-LSTM':
			slice = tf.expand_dims(tf.cast(tf.convert_to_tensor(slice_id/10),x.dtype),axis=-1)
		else:
			slice = tf.expand_dims(tf.cast(tf.convert_to_tensor(slice_id),x.dtype),axis=-1)
		slice = slice[None,:]
		slice = tf.tile(slice, [tf.shape(x)[0], 1])
		x = tf.concat((slice,x),-1)
		return (x,y)
	#series = tf.expand_dims(dataset.values, axis=-1)
	ds = tf.data.Dataset.from_tensor_slices(standardize_ds(dataset))
	ds = ds.window(ob_window_size + pred_window_size, shift=pred_window_size, drop_remainder=True)
	ds = ds.flat_map(lambda w: w.batch(ob_window_size + pred_window_size))
	ds = ds.shuffle(shuffle_buffer)
	ds = ds.map(lambda w: (w[:-pred_window_size], w[-pred_window_size:]))
	ds = ds.map(reshape_sample)
	ds = ds.batch(batch_size).prefetch(PREFETCH_BUFFER)
	return ds.repeat(num_epochs)

#CNN-LSTM model
def create_cnnlstm_model(rnn_act_fn, dense_act_fn):
	return tf.keras.models.Sequential([
			tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[obsv_win_len, NUM_FEATURE+1]),
			tf.keras.layers.LSTM(32, activation=rnn_act_fn, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=True),
			tf.keras.layers.LSTM(32, activation=rnn_act_fn, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=False),
			tf.keras.layers.Dense(pred_win_len*NUM_FEATURE, activation=dense_act_fn),
			tf.keras.layers.Reshape([pred_win_len,NUM_FEATURE])])

#LSTM model
def create_lstm_model(rnn_act_fn, dense_act_fn):
	return tf.keras.models.Sequential([
			tf.keras.layers.LSTM(32, activation=rnn_act_fn, kernel_regularizer=tf.keras.regularizers.l2(0.001),return_sequences=True, input_shape=(obsv_win_len, NUM_FEATURE + 1)), 
			tf.keras.layers.LSTM(32, activation=rnn_act_fn, kernel_regularizer=tf.keras.regularizers.l2(0.001),return_sequences=False),
			tf.keras.layers.Dense(pred_win_len*NUM_FEATURE, activation=dense_act_fn),
			tf.keras.layers.Reshape([pred_win_len,NUM_FEATURE])])

#GRU model
def create_gru_model(rnn_act_fn, dense_act_fn):
	return tf.keras.models.Sequential([
			tf.keras.layers.GRU(32, activation=rnn_act_fn, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=True, input_shape=(obsv_win_len, NUM_FEATURE + 1)), 
			tf.keras.layers.GRU(32, activation=rnn_act_fn, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=False),
			tf.keras.layers.Dense(pred_win_len*NUM_FEATURE, activation=dense_act_fn),
			tf.keras.layers.Reshape([pred_win_len,NUM_FEATURE])])

def create_keras_model():
	if scler == 'Quantile':
		act_fn = 'sigmoid'
	else:
		act_fn = 'relu' 
	#This resets the global state and saves on memory - models apparently cannot be reset 
	tf.keras.backend.clear_session()
	if model == 'CNN-LSTM':
		return create_cnnlstm_model(rnn_act_fn='relu',dense_act_fn=act_fn)
	elif model == 'LSTM':
		return create_lstm_model(rnn_act_fn='relu',dense_act_fn=act_fn)
	else:
		return create_gru_model(rnn_act_fn='relu',dense_act_fn=act_fn)

#Generate processed training sample set for each slice for the given client-ID
#Returns concatenated dataset of invidiual slice training sets
def load_client_dataset(client_id):
	#print('------------------------ Loading client-'+ str(client_id) +'train dataset ------------------------')
	filename = 'client-' + str(client_id) + '.csv'
	client_data = pd.read_csv(os.path.join(client_directory_path, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	client_data = client_data.drop(['Date','StartTime'], axis=1)
	slice_set = set(client_data['NSSAI'])
	client_train_dataset = []
	for slice_id in slice_set:
		#print("Generating client_data for slice " + str(slice_id))
		slice_data = client_data.loc[client_data['NSSAI'] == slice_id]
		slice_data = slice_data.drop(['NSSAI'], axis=1)
		num_tr_samples = int(0.66*slice_data.shape[0])
		num_tt_samples = slice_data.shape[0] - num_tr_samples
		#retrieve train dataset and fill the list
		slice_tr_data = slice_data.head(num_tr_samples)
		slice_tr_data.reset_index(drop=True,inplace=True)
		slice_tr_dataset = preprocess(slice_tr_data, slice_id, obsv_win_len, pred_win_len)
		client_train_dataset.append(slice_tr_dataset)
	train_dataset = client_train_dataset[0]
	for i in range(1,len(client_train_dataset)):
		train_dataset.concatenate(client_train_dataset[i])
	return train_dataset

os.makedirs(base_model_dir, exist_ok=True)
sys.stdout = open(os.path.join(base_model_dir,'output.txt'), 'w')

#Load the training time log if already created, else use the empty dataframe
tr_time_log = pd.DataFrame(columns=['Approach','Combination','Optimizer','Scaler','Model','Elapsed-Time'])
if os.path.isfile('Tr-Time-Log.csv'):
	tr_time_log = pd.read_csv('Tr-Time-Log.csv', sep=',', header=0, low_memory=False, infer_datetime_format=False)

client_ids = list(range(0,NUM_CLIENTS))
client_lr = 0.05
client_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=client_lr,decay_steps=1000,decay_rate=0.9)

#Iterate over all combinations of obsv and pred window lengths
for obsv_win_len, pred_win_len in zip(OBSERVATION_WINDOW_RANGE, PREDICTION_WINDOW_RANGE):
	combo = 'Combo-' + str(obsv_win_len) +'-'+str(pred_win_len)
	combo_dir = base_model_dir + combo + '/' 
	os.makedirs(combo_dir, exist_ok=True)
	print('################################## '+ combo  +' ##################################')
	#Load all client datasets into one global dataset
	global_train_dataset = load_client_dataset(0)
	for client_id in client_ids:
		if client_id > 0:
			global_train_dataset.concatenate(load_client_dataset(client_id))
			print('################################## Loaded Client:'+ str(client_id) +' Data ##################################')
	#Iterate over all combinations of optimization and scaler config
	for configuration in config_list:
		opalg = configuration[0]
		scler = configuration[1]
		algo_dir = combo_dir + opalg + '/' + scler + '/'
		os.makedirs(algo_dir, exist_ok=True)
		print('################################## Optimizer:'+ opalg + ', Scaler:' + scler +' ##################################')
		if opalg == 'SGD':
			client_optimizer = tf.keras.optimizers.SGD(learning_rate=client_lr_schedule, momentum=0.9)
		else:
			client_optimizer = tf.keras.optimizers.RMSprop(learning_rate=client_lr_schedule)
		#Iterating and creating kears models
		#Improvement - Create model just once and reset to save runtime
		for model in models:
			model_dir = algo_dir + model + '/'
			os.makedirs(model_dir, exist_ok=True)
			#
			############################## Model-Creation ##############################
			start_time = time.time()
			keras_model = create_keras_model()
			print(keras_model.summary())
			keras_model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=client_optimizer,metrics=[tf.keras.metrics.MeanAbsoluteError()])
			############################## Training Procedure ##############################
			print('################################## Train using global train dataset ##################################')
			history = keras_model.fit(global_train_dataset,epochs=NUM_EPOCHS,verbose=1)
			finish_time = time.time()
			#save the global model in tensorflow SavedModel format instead of h5 format (h5 is lightweight but old, may have bugs)
			model_name = 'SDF_' + model 
			model_path = os.path.join(model_dir, model_name)
			keras_model.save(model_path)
			print('#################################### Training Complete - Saved model ####################################')
			tr_time = finish_time - start_time
			print('Time Elapsed - (Model Creation & Training) -' + str(tr_time))
			tr_time_log = tr_time_log.append({'Approach':'Centralized','Combination':combo,'Optimizer':opalg,'Scaler':scler,'Model':model,'Elapsed-Time':tr_time}, ignore_index=True)
#Write the time log to time log csv
tr_time_log.to_csv('Tr-Time-Log.csv', index=False, header=list(tr_time_log.columns))
sys.stdout.close()
