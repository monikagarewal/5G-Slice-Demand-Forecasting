#!/usr/bin/env python

#import nest_asyncio
#nest_asyncio.apply()

import os 
import re
import sys
import collections
import itertools
import time

import tensorflow as tf
import tensorflow_federated as tff

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp 

from matplotlib import pyplot

# creating regex for matching files in dataset
# client_file_pattern is 'client-.*\.csv$'
base_model_dir = '../Models/Mi-2013/Hybrid/'
client_directory = '../Datasets/MI-2013/ClientData'

models = ['LSTM','GRU','CNN-LSTM']
scalers = ['Quantile','Robust','Manual']
optimizationAlgorithms = ['SGD','RMS-Prop']

config = [optimizationAlgorithms,scalers]
config_list = [item for item in itertools.product(*config)]

#global parameters
model = 'GRU'
scler = 'Quantile'
opalg = 'RMS-Prop'

#forecasting algorithm parameters
NUM_EPOCHS = 25
BATCH_SIZE = 32
SHUFFLE_BUFFER = 20
PREFETCH_BUFFER = 10

#time series parameters
NUM_FEATURE = 5
PREDICTION_WINDOW_RANGE = [1,2,4,8,16]
OBSERVATION_WINDOW_RANGE = [6,12,24,36,48]

obsv_win_len = 0
pred_win_len = 0

#Federated parameters
NUM_ROUNDS = 20
NUM_CLIENTS = 100
server_learning_rate = 1.0
client_learning_rate = 0.05

#Learning rate schedule for the worker/client nodes
client_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate=client_learning_rate,
	decay_steps=1000,
	decay_rate=0.9)

def preprocess(dataset, sliceId, ob_window_size, pred_window_size, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER, num_epochs=NUM_EPOCHS):
	#Using quantile scaler - [As its more robust to outliers, which can be common in mobile n/w traffic]
	def standardize_ds(data):
		if scler == 'Manual':
			scaledData = (data - data.mean()) / data.std()
			return scaledData.values
		elif scler == 'Robust':
			datascaler = skp.RobustScaler()
		else:
			datascaler = skp.QuantileTransformer(n_quantiles=int(0.66*data.shape[0]),random_state=10)
		return datascaler.fit_transform(data.values)
	#reshape the tensor to add slice info to features 
	def reshape_sample(x,y):
		if model == 'CNN-LSTM':
			slice = tf.expand_dims(tf.cast(tf.convert_to_tensor(sliceId/10),x.dtype),axis=-1)
		else:
			slice = tf.expand_dims(tf.cast(tf.convert_to_tensor(sliceId),x.dtype),axis=-1)
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
	#This is being done due to the difference in range of scaling transformations. The other two scalers have low values and relu can help
	if scler == 'Quantile':
		act_fn = 'sigmoid'
	else:
		act_fn = 'relu' 
	#This resets the global state and saves on memory - models apparently cannot be reset
	#tf.keras.backend.clear_session()
	if model == 'CNN-LSTM':
		return create_cnnlstm_model(rnn_act_fn='relu',dense_act_fn=act_fn)
	elif model == 'LSTM':
		return create_lstm_model(rnn_act_fn='relu',dense_act_fn=act_fn)
	else:
		return create_gru_model(rnn_act_fn='relu',dense_act_fn=act_fn)

#creating the model factory for tff framework
def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
	keras_model = create_keras_model() 
	print(keras_model.summary())
	return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError()])

#Generate processed training sample set for each slice for the given client-ID
#Returns concatenated dataset of invidiual slice training sets
def load_client_dataset(clientId, num_epochs=NUM_EPOCHS):
	#print('------------------------ Loading client-'+ str(clientId) +'train dataset ------------------------')
	filename = 'client-' + str(clientId) + '.csv'
	clientData = pd.read_csv(os.path.join(client_directory, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	clientData = clientData.drop(['Date','StartTime'], axis=1)
	slice_set = set(clientData['NSSAI'])
	slice_datasets = []
	for sliceId in slice_set:
		#print("Generating client_data for slice " + str(sliceId))
		sliceData = clientData.loc[clientData['NSSAI'] == sliceId]
		sliceData = sliceData.drop(['NSSAI'], axis=1)
		numTrainSamples = int(0.66*sliceData.shape[0])
		slice_TrainData = sliceData.head(numTrainSamples)
		slice_TrainData.reset_index(drop=True,inplace=True)
		slice_dataset = preprocess(slice_TrainData, sliceId, obsv_win_len, pred_win_len, num_epochs)
		slice_datasets.append(slice_dataset)
	client_dataset = slice_datasets[0]
	for i in range(1,len(slice_datasets)):
		client_dataset.concatenate(slice_datasets[i])
	return client_dataset

#Helper function passed to FL API to retrieve federated datasets
def create_train_dataset_for_client_fn(clientId):
	#print("Creating client training dataset for client " + str(clientId))
	return load_client_dataset(clientId)

#create test data in a similar fashion to the train dataset
def load_client_testDataset(clientId):
	#print('------------------------ Loading client-'+ str(clientId) +'test dataset ------------------------')
	filename = 'client-' + str(clientId) + '.csv'
	clientData = pd.read_csv(os.path.join(client_directory, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	clientData = clientData.drop(['Date','StartTime'], axis=1)
	slice_set = set(clientData['NSSAI'])
	#loading each slice dataset separately to ensure they are standardized fairly and there are sufficient number of examples
	slice_datasets = []
	for sliceId in slice_set:
		sliceData = clientData.loc[clientData['NSSAI'] == sliceId]
		sliceData = sliceData.drop(['NSSAI'], axis=1)
		numTestSamples = sliceData.shape[0] - (int(0.66*sliceData.shape[0]))
		slice_TestData = sliceData.tail(numTestSamples)
		slice_TestData.reset_index(drop=True, inplace=True)
		#num_epochs is explicitly set to 1 as this is test data we do not want to repeat the samples
		slice_datasets.append(preprocess(slice_TestData, sliceId, obsv_win_len, pred_win_len, num_epochs=1))
	#now concatenate all the individual slice dataset
	client_dataset = slice_datasets[0]
	for i in range(1,len(slice_datasets)):
		client_dataset.concatenate(slice_datasets[i])
	return client_dataset

#Helper function passed to FL API to retrieve federated datasets
def create_test_dataset_for_client_fn(clientId):
   return load_client_testDataset(clientId)

#this just creates a map of clientIds and their datasets
def make_federated_data(client_data, client_ids):
	return [
      client_data.create_tf_dataset_for_client(x)
      for x in client_ids
  ]

#Internal clients - can aggregate data | external - cannot aggregate data
internal_clientIds = list(range(0,30))
external_clientIds = list(range(30,NUM_CLIENTS))

os.makedirs(base_model_dir, exist_ok=True) 
sys.stdout = open(os.path.join(base_model_dir,'output.txt'), 'w')

#Load the training time log if already created, else use the empty dataframe
tr_time_log = pd.DataFrame(columns=['Approach','Combination','Optimizer','Scaler','Model','Elapsed-Time'])
if os.path.isfile('Tr-Time-Log.csv'):
	tr_time_log = pd.read_csv('Tr-Time-Log.csv', sep=',', header=0, low_memory=False, infer_datetime_format=False)

#Iterate over all combinations of obsv and pred window lengths
for obsv_win_len, pred_win_len in zip(OBSERVATION_WINDOW_RANGE, PREDICTION_WINDOW_RANGE):
	combo = 'Combo-' + str(obsv_win_len) + '-' + str(pred_win_len)
	combo_dir_path = base_model_dir + combo + '/'
	os.makedirs(combo_dir_path, exist_ok=True)
	print('######################## ' + combo + ' ########################')
	#Load all internal client datasets into one global internal dataset
	global_internal_dataset = load_client_dataset(internal_clientIds[0], num_epochs=1)
	for client_id in internal_clientIds:
		if client_id != internal_clientIds[0]:
			global_internal_dataset.concatenate(load_client_dataset(client_id, num_epochs=1))
			print('######################## Loaded Client:'+ str(client_id) +' Data ########################')
	#creating sample_batch for the element spec (doing it here because it depends on window lengths)
	clientData = pd.read_csv(os.path.join(client_directory, 'client-0.csv'), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	sliceSet = set(clientData['NSSAI'])
	slice_Data = (clientData.loc[clientData['NSSAI'] == 39]).reset_index(drop=True)
	slice_Data = slice_Data.drop(['Date','StartTime','NSSAI'], axis=1)
	preprocessed_example_dataset = preprocess(slice_Data, 39, obsv_win_len, pred_win_len)
	#sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))
	#print('########################################################')
	#print(sample_batch)
	#print(sample_batch[0].shape)
	#print(sample_batch[1].shape)
	#print('########################################################')
	#
	#Iterate over the combinations of scaling and optimization algorithms
	for configuration in config_list:
		opalg = configuration[0]
		scler = configuration[1]
		algorithm =  opalg + '/' + scler 
		algo_dir_path = combo_dir_path + algorithm + '/'
		os.makedirs(algo_dir_path, exist_ok=True)
		print('######################## '+ algorithm +' ########################')
		#Improvement - Create model just once and reset to save runtime
		#Iterating and creating kears models
		for model in models:
			print('######################## '+ model +' ########################')
			model_dir_path = algo_dir_path + model + '/'
			os.makedirs(algo_dir_path, exist_ok=True)
			#Single optimizer object isn't working, need to find why?
			if opalg == 'SGD':
				client_optimizer = tf.keras.optimizers.SGD(learning_rate=client_lr_schedule, momentum=0.9)
			else:
				client_optimizer = tf.keras.optimizers.RMSprop(learning_rate=client_lr_schedule)
			start_time = time.time()
			##############################  Federated Model Creation (Clients: 30-99) ##############################
			print('######################## Creating federated processes ########################')
			train_data = tff.simulation.ClientData.from_clients_and_fn(external_clientIds, create_tf_dataset_for_client_fn=create_train_dataset_for_client_fn)
			#Selecting all clients - can be modified to consider cases with only a subset of clients
			selected_clients = train_data.client_ids
			federated_train_data = make_federated_data(train_data, selected_clients)
			#FL-API to build an iterative process to perform federated averaging 
			iterative_process = tff.learning.build_federated_averaging_process(
			    model_fn,
			    client_optimizer_fn=lambda: client_optimizer,
			    server_optimizer_fn=lambda: tf.keras.optimizers.RMSprop(learning_rate=server_learning_rate))
			############################## Centralized Training Procedure (Clients: 0-29) ##############################
			global_model = create_keras_model()
			print(global_model.summary())
			global_model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=client_optimizer,metrics=[tf.keras.metrics.MeanAbsoluteError()])
			print('######################## Train using internal training dataset ########################')
			history = global_model.fit(global_internal_dataset,epochs=NUM_EPOCHS*2,verbose=1)
			print(history)
			##############################  Federated Training Procedure (Clients: 30-99) ##############################
			print('######################## Train using external federated dataset ########################')
			#Initializes server state i.e. model weights and optimizer variables
			state = iterative_process.initialize()
			print(type(state.model))
			# Load our pre-trained Keras model weights into the global model state.
			state = tff.learning.state_with_new_model_weights(
							state,
							trainable_weights=[v.numpy() for v in global_model.trainable_weights], 
							non_trainable_weights=[v.numpy() for v in global_model.non_trainable_weights])
			#Executes one iteration of the federated averaging process i.e. we perform 1 round of FL training
			state, metrics = iterative_process.next(state, federated_train_data)
			print('round  1, metrics={}'.format(metrics))
			#Repeat the process over multiple rounds
			for round_num in range(2, NUM_ROUNDS):
			  state, metrics = iterative_process.next(state, federated_train_data)
			  print('round {:2d}, metrics={}'.format(round_num, metrics))
			#
			############################# Saving Model ##############################
			#create keras model using the final global model to generate local metrics per slice and per client
			state.model.assign_weights_to(global_model)
			print(global_model.summary())
			model_file = 'SDF_' + model
			model_path = os.path.join(model_dir_path,model_file)
			global_model.save(model_path)
			#Training is complete, now we build the evaluation process to calculate final training metrics 
			evaluation = tff.learning.build_federated_evaluation(model_fn)
			train_metrics = evaluation(state.model, federated_train_data)
			print(train_metrics)
			finish_time = time.time()
			elapsed_time = finish_time - start_time
			tr_time_log = tr_time_log.append({'Approach':'Hybrid','Combination':combo,'Optimizer':opalg,'Scaler':scler,'Model':model,'Elapsed-Time':elapsed_time}, ignore_index=True)
tr_time_log.to_csv('Tr-Time-Log.csv', index=False, header=list(tr_time_log.columns))
sys.stdout.close()
