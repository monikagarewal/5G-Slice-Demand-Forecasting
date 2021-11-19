#!/usr/bin/env python

#import nest_asyncio
#nest_asyncio.apply()

import os 
import re
import sys
import collections
import itertools

import tensorflow as tf
import tensorflow_federated as tff

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp 

from matplotlib import pyplot

# creating regex for matching files in dataset
clientFilePattern = re.compile('client-.*\.csv$')
clientDirectoryPath = '/home/akhil/SliceCapacityForecasting/Datasets/SDF_Datasets/MI-2013/ClientData'

baseResultsDir = './Results/Milano/Federated/'  

models = ['GRU','LSTM','CNN-LSTM']
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
numFeature = 5
OBSERVATION_WINDOW_RANGE = [6,12,24,36,48]
PREDICTION_WINDOW_RANGE = [1,2,4,8,16]

#Federated parameters
NUM_ROUNDS = 20
NUM_CLIENTS = 100
server_learning_rate = 1.0
client_learning_rate = 0.05

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
def create_cnnlstm_model():
	return tf.keras.models.Sequential([
			tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[obsvWinLen, numFeature+1]),
			tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.0001), return_sequences=True),
			tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.0001), return_sequences=False),
			tf.keras.layers.Dense(predWinLen*numFeature),
			tf.keras.layers.Reshape([predWinLen,numFeature])])

#LSTM model
def create_lstm_model():
	return tf.keras.models.Sequential([
			tf.keras.layers.LSTM(32,kernel_regularizer=tf.keras.regularizers.l2(0.0001),return_sequences=True, input_shape=(obsvWinLen, numFeature + 1)), 
			tf.keras.layers.LSTM(32,kernel_regularizer=tf.keras.regularizers.l2(0.0001),return_sequences=False),
			tf.keras.layers.Dense(predWinLen*numFeature),
			tf.keras.layers.Reshape([predWinLen,numFeature])])

#GRU model
def create_gru_model():
	return tf.keras.models.Sequential([
			tf.keras.layers.GRU(32,kernel_regularizer=tf.keras.regularizers.l2(0.0001),return_sequences=True, input_shape=(obsvWinLen, numFeature + 1)), 
			tf.keras.layers.GRU(32,kernel_regularizer=tf.keras.regularizers.l2(0.0001),return_sequences=False),
			tf.keras.layers.Dense(predWinLen*numFeature),
			tf.keras.layers.Reshape([predWinLen,numFeature])])

def create_keras_model():
	if model == 'CNN-LSTM':
		return create_cnnlstm_model()
	elif model == 'LSTM':
		return create_lstm_model()
	else:
		return create_gru_model()

#creating the model factory for tff framework
def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
	keras_model = create_keras_model()
	return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError()])

#this just creates a map of clientIds and their datasets
def make_federated_data(client_data, client_ids):
	return [
      client_data.create_tf_dataset_for_client(x)
      for x in client_ids
  ]

#Generate processed training sample set for each slice for the given client-ID
#Returns concatenated dataset of invidiual slice training sets
def load_client_trainDataset(clientId):
	#print('------------------------ Loading client-'+ str(clientId) +'train dataset ------------------------')
	filename = 'client-' + str(clientId) + '.csv'
	clientData = pd.read_csv(os.path.join(clientDirectoryPath, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
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
		slice_dataset = preprocess(slice_TrainData, sliceId, obsvWinLen, predWinLen)
		slice_datasets.append(slice_dataset)
	client_dataset = slice_datasets[0]
	for i in range(1,len(slice_datasets)):
		client_dataset.concatenate(slice_datasets[i])
	return client_dataset
 
def create_train_dataset_for_client_fn(clientId):
	#print("Creating client training dataset for client " + str(clientId))
	return load_client_trainDataset(clientId)

#create test data in a similar fashion to the train dataset
#load dataset
def load_client_testDataset(clientId):
	#print('------------------------ Loading client-'+ str(clientId) +'test dataset ------------------------')
	filename = 'client-' + str(clientId) + '.csv'
	clientData = pd.read_csv(os.path.join(clientDirectoryPath, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
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
		slice_datasets.append(preprocess(slice_TestData, sliceId, obsvWinLen, predWinLen, num_epochs=1))
	#now concatenate all the individual slice dataset
	client_dataset = slice_datasets[0]
	for i in range(1,len(slice_datasets)):
		client_dataset.concatenate(slice_datasets[i])
	return client_dataset
# 
def create_test_dataset_for_client_fn(clientId):
   return load_client_testDataset(clientId)

##create test data in a similar fashion to the train dataset
##load dataset
def load_testDataset(clientId, sliceSet):
	filename = 'client-' + str(clientId) + '.csv'
	clientData = pd.read_csv(os.path.join(clientDirectoryPath, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	clientData = clientData.drop(['Date','StartTime'], axis=1)
	#Create empty dataset and dictionary
	slice_data = []
	slice_datasets = {}
	for sliceId in sliceSet:
		sliceData = clientData.loc[clientData['NSSAI'] == sliceId]
		sliceData = sliceData.drop(['NSSAI'], axis=1)
		numTestSamples = sliceData.shape[0] - (int(0.66*sliceData.shape[0]))
		slice_TestData = sliceData.tail(numTestSamples)
		slice_TestData.reset_index(drop=True, inplace=True)
		slice_dataset = preprocess(slice_TestData, sliceId, obsvWinLen, predWinLen, num_epochs=1)
		slice_datasets[sliceId] = slice_dataset
		#num_epochs is explicitly set to 1 as this is test data we do not want to repeat the samples
		slice_data.append(slice_dataset)
	#now concatenate all the individual slice dataset
	client_dataset = slice_data[0]
	for i in range(1,len(slice_data)):
		client_dataset.concatenate(slice_data[i])
	return client_dataset, slice_datasets

##create test data in a similar fashion to the train dataset
##load dataset
def load_slice_dataset(clientId, sliceId):
	filename = 'client-' + str(clientId) + '.csv'
	print('------------------------ Loading client-'+ str(clientId) +' and slice-'+ str(sliceId) +' test dataset ------------------------')
	clientData = pd.read_csv(os.path.join(clientDirectoryPath, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	clientData = clientData.drop(['Date','StartTime'], axis=1)
	sliceData = clientData.loc[clientData['NSSAI'] == sliceId]
	sliceData = sliceData.drop(['NSSAI'], axis=1)
	numTestSamples = sliceData.shape[0] - (int(0.66*sliceData.shape[0]))
	slice_TestData = sliceData.tail(numTestSamples)
	slice_TestData.reset_index(drop=True, inplace=True)
	return preprocess(slice_TestData, sliceId, obsvWinLen, predWinLen, num_epochs=1)

def extract_qci_values(dataset,qci_index):
	series = np.empty((0,numFeature),dtype=np.float64)
	for x,y in dataset:
		y_reshaped = (y.numpy()).reshape(y.shape[0]*y.shape[1],y.shape[2])
		series = np.append(series,y_reshaped,axis=0)
	if (qci_index >= 0) and (qci_index < series.shape[1]):
		return series[:,qci_index]
	return series[:,0]

def plot_series(axes,time, series, format="-", start=0, end=None, label_plot='Values', x_label='Time', y_label='Normalized Class Activity'):
	axes.plot(time[start:end], series[start:end], format, label=label_plot)
	axes.set(xlabel=x_label, ylabel=y_label)
	axes.grid(False)

def plot_contour(testMetrics, plotPath):
	fig, axes = pyplot.subplots(2, sharex=True, sharey=True, figsize=(6,6))
	keys = np.array(list(testMetrics.keys()))
	clients = np.array(list(set(keys[:,0])))
	snssais = np.array(list(range(len(set(keys[:,1])))))
	x, y = np.meshgrid(clients,snssais)
	metrics = np.array(list(testMetrics.values()))
	mse_values = metrics[:,0].reshape(snssais.shape[0],clients.shape[0])
	mse_contour = axes[0].contourf(x,y,mse_values)
	fig.colorbar(mse_contour, ax=axes[0])
	axes[0].set_title('MSE-Distribution')
	mae_values = metrics[:,1].reshape(snssais.shape[0],clients.shape[0])
	mae_contour = axes[1].contourf(x,y,mae_values)
	fig.colorbar(mae_contour, ax=axes[1])
	axes[1].set_title('MAE-Distribution')
	fig.text(0.5, 0.04, 'Client-ID', ha='center')
	fig.text(0.04, 0.5, 'S-NSSAI', va='center', rotation='vertical')
	fig.suptitle('Per-Client-Slice-Test-Metrics')
	fig.savefig(os.path.join(plotPath,'Per-Client-Slice-Test-Metrics.pdf'))
	pyplot.close(fig)
	return

def plot_client_metrics(client_mae, client_mse, plotDir):
	fig, axes = pyplot.subplots(2,sharey=True,tight_layout=True)
	#plot mae values
	axes[0].hist(client_mae.values(), bins=10)
	axes[0].set_title('MAE-Distribution')
	#plot mse values
	axes[1].hist(client_mse.values(), bins=10)
	axes[1].set_title('MSE-Distribution')
	#pyplot.bar(range(len(client_metrics.values())), client_metrics.values())
	#pyplot.xticks(range(len(client_metrics.keys())), client_metrics.keys())
	fig.suptitle('Test-Metrics-Client-Distribution')
	fig.savefig(os.path.join(plotDir,'Test-Metrics-Client-Distribution.pdf'))
	pyplot.close(fig)
	return

def plot_bargraph_metrics(mae_metrics, mse_metrics, plot_title, plot_path):
	fig, axes = pyplot.subplots(2,tight_layout=True)
	#plot mae values
	axes[0].bar(range(len(mae_metrics.values())),mae_metrics.values())
	axes[0].set_title('MAE-Distribution')
	axes[0].set_xticks(range(len(mae_metrics.keys())))
	axes[0].set_xticklabels(mae_metrics.keys())
	#plot mse values
	axes[1].bar(range(len(mse_metrics.values())),mse_metrics.values())
	axes[1].set_title('MSE-Distribution')
	axes[1].set_xticks(range(len(mse_metrics.keys())))
	axes[1].set_xticklabels(mse_metrics.keys())
	fig.suptitle(plot_title)
	fig.savefig(plot_path)
	pyplot.close(fig)
	return

def plot_activity(client_id, plot_title, plot_path):
	ix = 0
	slice_set = [0,39]
	qci_set = ['Q1','Q2','Q3','Q4','Q5']
	fig, axes = pyplot.subplots(nrows=2,ncols=5,sharex=True, sharey=True, figsize=(6,6))
	for sliceId in slice_set:
		testDataset = load_slice_dataset(client_id, slice_id)
		print('###################### Predictions for Slice-'+ str(sliceId)+' ##########################')
		test_Output = keras_model.predict(testDataset, verbose=1)
		test_Output = test_Output.reshape(test_Output.shape[0]*predWinLen,numFeature)
		for index in range(test_Output.shape[1]):
			qci_real_activity = extract_qci_values(testDataset, index)
			qci_pred_activity = test_Output[:,index]
			time = list(range(test_Output.shape[0]))
			plot_series(axes[ix][index], time, qci_real_activity, label_plot = 'True-Values')
			plot_series(axes[ix][index], time, qci_pred_activity, label_plot = model)
			axes[ix][index].set_title(qci_set[index])
		ix = ix + 1
	handles, labels = axes[0][0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right')
	fig.suptitle(plot_title)
	fig.savefig(plot_path)
	pyplot.close(fig)
	return

def calculate_new_avg(new_value, cur_avg, num_samples):
	new_mean = (new_value + (cur_avg * (num_samples-1))) / num_samples
	return new_mean

clientIds = list(range(0,NUM_CLIENTS))

for model in models:
	algorithm = model + '/'
	modelDirPath = baseResultsDir + algorithm
	os.makedirs(modelDirPath, exist_ok=True) 
	sys.stdout = open(os.path.join(modelDirPath,'output.txt'), 'w')
	for configuration in config_list:
		opalg = configuration[0]
		scler = configuration[1]
		algorithm = algorithm + opalg + '/' + scler + '/'
		print('################################## Model:'+ model +', Optimization:'+ opalg +', Scaler:'+ scler +'##################################')
		if opalg == 'SGD':
			client_optimizer = tf.keras.optimizers.SGD(learning_rate=client_lr_schedule, momentum=0.9)
		else:
			client_optimizer = tf.keras.optimizers.RMSprop(learning_rate=client_lr_schedule)
		#window length metrics
		win_count = {}
		win_mse_metrics = {}
		win_mae_metrics = {}
		#Iterate over all combinations of obsv and pred window lengths
		for obsvWinLen, predWinLen in zip(OBSERVATION_WINDOW_RANGE, PREDICTION_WINDOW_RANGE):
			win_len = 'O-' + str(obsvWinLen) + '-P-' + str(predWinLen)
			resultDir = algorithm + 'Combo-' + str(obsvWinLen) +'-'+str(predWinLen)
			resultDirectoryPath = baseResultsDir + resultDir 
			os.makedirs(resultDirectoryPath, exist_ok=True)
			print('################################## ObsLen:' + str(obsvWinLen) + ', PredLen:'+ str(predWinLen)+ ' ##################################')
			#
			#creating sample_batch for trial-run
			clientData = pd.read_csv(os.path.join(clientDirectoryPath, 'client-0.csv'), sep=',', header=0, low_memory=False, infer_datetime_format=True)
			sliceSet = set(clientData['NSSAI'])
			slice_Data = (clientData.loc[clientData['NSSAI'] == 39]).reset_index(drop=True)
			slice_Data = slice_Data.drop(['Date','StartTime','NSSAI'], axis=1)
			preprocessed_example_dataset = preprocess(slice_Data, 39, obsvWinLen, predWinLen)
			#sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))
			#print('########################################################')
			#print(sample_batch)
			#print(sample_batch[0].shape)
			#print(sample_batch[1].shape)
			#print('########################################################')
			#
			############################## Federated Training Procedure ##############################
			#print(clientIds)
			train_data = tff.simulation.ClientData.from_clients_and_fn(clientIds, create_tf_dataset_for_client_fn=create_train_dataset_for_client_fn)
			#print(train_data)
			#print(train_data._element_type_structure)
			selected_clients = train_data.client_ids[0:NUM_CLIENTS]
			federated_train_data = make_federated_data(train_data, selected_clients)
			#print(preprocessed_example_dataset.element_spec)
			iterative_process = tff.learning.build_federated_averaging_process(
			    model_fn,
			    client_optimizer_fn=lambda: client_optimizer,
			    server_optimizer_fn=lambda: tf.keras.optimizers.RMSprop(learning_rate=server_learning_rate))
			
			state = iterative_process.initialize()
			#print(type(state.model))
			
			state, metrics = iterative_process.next(state, federated_train_data)
			print('round  1, metrics={}'.format(metrics))
			
			for round_num in range(2, NUM_ROUNDS):
			  state, metrics = iterative_process.next(state, federated_train_data)
			  print('round {:2d}, metrics={}'.format(round_num, metrics))
			
			evaluation = tff.learning.build_federated_evaluation(model_fn)
			train_metrics = evaluation(state.model, federated_train_data)
			print('###################### Training metrics - Start #########################')
			print(train_metrics)
			print('###################### Training metrics - Stop #########################')
			#
			############################## Federated Test Procedure ##############################
			#generate evalution metrics for test data of all clients
			test_data = tff.simulation.ClientData.from_clients_and_fn(clientIds, create_tf_dataset_for_client_fn=create_test_dataset_for_client_fn)
			federated_test_data = make_federated_data(test_data, selected_clients)
			#print(len(federated_test_data))
			#print(federated_test_data[0])
			test_metrics = evaluation(state.model, federated_test_data)
			if win_len in win_count:
				num_samples = win_count[win_len] + 1
				win_count[win_len] = num_samples
				win_mse_metrics[win_len] = calculate_new_avg(test_metrics['loss'], win_mae_metrics[win_len], num_samples)
				win_mae_metrics[win_len] = calculate_new_avg(test_metrics['mean_absolute_error'], win_mae_metrics[win_len], num_samples)
			else:
				win_count[win_len] = 1
				win_mse_metrics[win_len] = test_metrics['loss'] 
				win_mae_metrics[win_len] = test_metrics['mean_absolute_error']
			print('###################### Test metrics - Stop #########################')
			print(test_metrics)
			print('###################### Test metrics - Stop #########################')
			#
			############################# Saving Model ##############################
			#create keras model using the final global model to generate local metrics per slice and per client
			keras_model = create_keras_model()
			keras_model.compile(loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.MeanAbsoluteError()])
			state.model.assign_weights_to(keras_model)
			print(keras_model.summary())
			
			modelPath = os.path.join(resultDirectoryPath,'model.h5')
			keras_model.save(modelPath)
			#keras_model = tf.keras.models.load_model(modelPath)
			#
			############################# Client & Slice - Individual Metrics ##############################
			#Plot per slice and client MAE
			slice_count = {}
			slice_mae_metrics = {}
			slice_mse_metrics = {}
			slice_client_metrics = {}
			client_mae = {}
			client_mse = {}
			#Run forward pass on all client test datasets
			for clientId in selected_clients:
				clientTestDataset, slice_datasets = load_testDataset(clientId)
				print('###################### Client ' + str(clientId) + '- Predictions & Metrics #########################')
				client_output = keras_model.evaluate(clientTestDataset, verbose=1)
				client_mae[clientId] = client_output[1]
				client_mse[clientId] = client_output[0]
				for slice_id in slice_datasets:
					print('###################### Slice-'+ str(slice_id)+' ##########################')
					test_dataset = slice_datasets[slice_id]
					test_output = keras_model.evaluate(test_dataset, verbose=1)
					slice_client_metrics[(client_id,slice_id)] = test_output
					if slice_id in slice_count:
						num_slice_samples = slice_count[slice_id] + 1
						slice_count[slice_id] = num_slice_samples
						slice_mae_metrics[slice_id] = calculate_new_avg(test_output[1], slice_mae_metrics[slice_id], num_slice_samples)
						slice_mse_metrics[slice_id] = calculate_new_avg(test_output[0], slice_mse_metrics[slice_id], num_slice_samples)
					else:
						slice_count[slice_id] = 1
						slice_mae_metrics[slice_id] = test_output[1]
						slice_mse_metrics[slice_id] = test_output[0]
			max_mae_client_id = max(client_mae, key=client_mae.get) 
			min_mae_client_id = min(client_mae, key=client_mae.get) 
			plot_client_metrics(client_mae, client_mse, resultDirectoryPath)
			plot_contour(slice_client_metrics, resultDirectoryPath)
			plot_bargraph_metrics(slice_mae_metrics, slice_mse_metrics,'Slice-Metrics-Distribution', os.path.join(resultDirectoryPath,'Slice-Metrics-Distribution.pdf'))
			#
			############################## ClientWise - Test Activity Procedure ##############################
			#
			#Plotting time series for visual analysis
			plot_activity(max_mae_client_id, 'Max-class-activity-predictions', os.path.join(resultDirectoryPath,'Max-Class-activity-predicitions.pdf')) 
			plot_activity(min_mae_client_id, 'Min-class-activity-predictions', os.path.join(resultDirectoryPath,'Min-Class-activity-predicitions.pdf'))
		title = algorithm + 'Window-Combination-Metrics' 
		plot_bargraph_metrics(win_mae_metrics, win_mse_metrics, title, os.path.join(modelDirPath, 'Window-Combination-Metrics.pdf'))
	sys.stdout.close()
