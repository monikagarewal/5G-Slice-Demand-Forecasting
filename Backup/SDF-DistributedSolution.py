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
base_results_dir = '/home/akhil/SliceCapacityForecasting/Results/Milano/Centralized/'
home_dataset_dir = '/home/akhil/SliceCapacityForecasting/Datasets/SDF_Datasets/'
client_directory_path = home_dataset_dir + 'MI-2013/ClientData'

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
NUM_EPOCHS = 500
NUM_CLIENTS = 30
SHUFFLE_BUFFER = 20
PREFETCH_BUFFER = 10

#time series parameters
NUM_FEATURE = 5
obsv_win_len = 0
pred_win_len = 0
OBSERVATION_WINDOW_RANGE = [6,12,24,36,48]
PREDICTION_WINDOW_RANGE = [1,2,4,8,16]

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
def create_cnnlstm_model(activation_function):
	return tf.keras.models.Sequential([
			tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[obsv_win_len, NUM_FEATURE+1]),
			tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=True),
			tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=False),
			tf.keras.layers.Dense(pred_win_len*NUM_FEATURE, activation=activation_function),
			tf.keras.layers.Reshape([pred_win_len,NUM_FEATURE])])

#LSTM model
def create_lstm_model(activation_function):
	return tf.keras.models.Sequential([
			tf.keras.layers.LSTM(32,kernel_regularizer=tf.keras.regularizers.l2(0.001),return_sequences=True, input_shape=(obsv_win_len, NUM_FEATURE + 1)), 
			tf.keras.layers.LSTM(32,kernel_regularizer=tf.keras.regularizers.l2(0.001),return_sequences=False),
			tf.keras.layers.Dense(pred_win_len*NUM_FEATURE, activation=activation_function),
			tf.keras.layers.Reshape([pred_win_len,NUM_FEATURE])])

#GRU model
def create_gru_model(activation_function):
	return tf.keras.models.Sequential([
			tf.keras.layers.GRU(32,kernel_regularizer=tf.keras.regularizers.l2(0.001),return_sequences=True, input_shape=(obsv_win_len, NUM_FEATURE + 1)), 
			tf.keras.layers.GRU(32,kernel_regularizer=tf.keras.regularizers.l2(0.001),return_sequences=False),
			tf.keras.layers.Dense(pred_win_len*NUM_FEATURE, activation=activation_function),
			tf.keras.layers.Reshape([pred_win_len,NUM_FEATURE])])

def create_keras_model():
	if scaler == 'Quantile':
		activation_function = 'sigmoid'
	else:
		activation_function = 'relu' 
	if model == 'CNN-LSTM':
		return create_cnnlstm_model(activation_function)
	elif model == 'LSTM':
		return create_lstm_model(activation_function)
	else:
		return create_gru_model(activation_function)

#Generate processed training sample set for each slice for the given client-ID
#Returns concatenated dataset of invidiual slice training sets
def load_client_dataset(client_id):
	#print('------------------------ Loading client-'+ str(client_id) +'train dataset ------------------------')
	filename = 'client-' + str(client_id) + '.csv'
	client_data = pd.read_csv(os.path.join(client_directory_path, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	client_data = client_data.drop(['Date','StartTime'], axis=1)
	slice_set = set(client_data['NSSAI'])
	client_test_dataset = []
	client_train_dataset = []
	slice_test_datasets = {}
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
		#retrieve test dataset and fill both list and map
		slice_tt_data = slice_data.tail(num_tt_samples)
		slice_tt_data.reset_index(drop=True, inplace=True)
		##num_epochs is explicitly set to 1 as this is test data we do not want to repeat the samples
		slice_tt_dataset = preprocess(slice_tt_data, slice_id, obsv_win_len, pred_win_len)
		client_test_dataset.append(slice_tt_dataset)
		slice_test_datasets[slice_id] = slice_tt_dataset
	train_dataset = client_train_dataset[0]
	for i in range(1,len(client_train_dataset)):
		train_dataset.concatenate(client_train_dataset[i])
	test_dataset = client_test_dataset[0]
	for i in range(1,len(client_test_dataset)):
		test_dataset.concatenate(client_test_dataset[i])
	return (train_dataset, test_dataset, slice_test_datasets)

##create the slice and client dataset
def load_slice_dataset(client_id, slice_id):
	filename = 'client-' + str(client_id) + '.csv'
	print('------------------------ Loading client-'+ str(client_id) +' and slice-'+ str(slice_id) +' test dataset ------------------------')
	client_data = pd.read_csv(os.path.join(client_directory_path, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	client_data = client_data.drop(['Date','StartTime'], axis=1)
	slice_data = client_data.loc[client_data['NSSAI'] == slice_id]
	slice_data = slice_data.drop(['NSSAI'], axis=1)
	numTestSamples = slice_data.shape[0] - (int(0.66*slice_data.shape[0]))
	slice_test_data = slice_data.tail(numTestSamples)
	slice_test_data.reset_index(drop=True, inplace=True)
	return preprocess(slice_test_data, slice_id, obsv_win_len, pred_win_len)

def calculate_new_avg(new_value, cur_avg, num_samples):
	new_mean = (new_value + (cur_avg * (num_samples-1))) / num_samples
	return new_mean

def transform_dataset(dataset):
	series = np.empty((0,NUM_FEATURE),dtype=np.float64)
	for x,y in dataset:
		y_reshaped = (y.numpy()).reshape(y.shape[0]*y.shape[1],y.shape[2])
		series = np.append(series,y_reshaped,axis=0)
	return series

def extract_qci_values(dataset,qci_index):
	transform_dataset(dataset)
	if (qci_index >= 0) and (qci_index < series.shape[1]):
		return series[:,qci_index]
	return series[:,0]

def plot_metrics_contour(test_metrics, y_label, x_label, plot_name, plot_path):
	fig, axes = pyplot.subplots(2, figsize=(6,6))
	keys = np.array(list(test_metrics.keys()))
	clients = np.array(list(set(keys[:,0])))
	snssais = np.array(list(range(len(set(keys[:,1])))))
	x, y = np.meshgrid(clients,snssais)
	metrics = np.array(list(test_metrics.values()))
	mse_values = metrics[:,0].reshape(snssais.shape[0],clients.shape[0])
	mse_contour = axes[0].contourf(x,y,mse_values)
	fig.colorbar(mse_contour, ax=axes[0])
	axes[0].set_title('MSE-Distribution')
	mae_values = metrics[:,1].reshape(snssais.shape[0],clients.shape[0])
	mae_contour = axes[1].contourf(x,y,mae_values)
	fig.colorbar(mae_contour, ax=axes[1])
	axes[1].set_title('MAE-Distribution')
	fig.text(0.5, 0.04, x_label, ha='center')
	fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
	fig.suptitle(plot_name)
	fig.savefig(plot_path)
	pyplot.close(fig)
	return

def plot_mape_contour(test_metrics, y_label, x_label, plot_name, plot_path):
	x_range = np.array(list(test_metrics.keys()), dtype=np.int)
	y_range = np.array(list(range(1,6)), dtype=np.int)
	x, y = np.meshgrid(x_range,y_range)
	mape_metrics = np.array(list(test_metrics.values()))
	mape_values = mape_metrics.reshape(y_range.shape[0],x_range.shape[0])
	pyplot.contourf(x,y,mape_values)
	pyplot.colorbar()
	pyplot.xlabel(x_label)
	pyplot.ylabel(y_label)
	pyplot.title(plot_name)
	pyplot.savefig(plot_path)
	pyplot.close()
	return

def plot_mape_histogram(test_metrics, numBins, y_label, plot_name, plot_path):
	qci_set = ['Q1','Q2','Q3','Q4','Q5']
	#qci_mape.shape ~ num_client * num_features | num_slices * num_features 
	mape_values = np.array(list(test_metrics.values()))
	#Each subplot plots a frequency histogram of QCI MAPE values
	fig, axes = pyplot.subplots(NUM_FEATURE,sharey=True,figsize=(6,6))
	for ix in list(range(NUM_FEATURE)):
		axes[ix].hist(mape_values[:,ix], bins=numBins)
		axes[ix].set_title(qci_set[ix])
	fig.text(0.5, 0.04, 'Average MAPE', ha='center')
	fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
	fig.suptitle(plot_name)
	fig.savefig(plot_path)
	pyplot.close(fig)
	return

def plot_models_mape(test_metrics, plot_name, plot_path):
	qci_set = ['Q1','Q2','Q3','Q4','Q5']
	models = ['GRU','LSTM','CNN-LSTM']
	bar_locations = np.arange(len(qci_set))  
	bar_width = 0.30  
	fig, ax = pyplot.subplots(tight_layout=True)
	model_mape_values = test_metrics['GRU']
	ax.bar(bar_locations - bar_width/3, model_mape_values, bar_width, label='GRU')
	model_mape_values = test_metrics['LSTM']
	ax.bar(bar_locations, model_mape_values, bar_width, label='LSTM')
	model_mape_values = test_metrics['CNN-LSTM']
	ax.bar(bar_locations + bar_width/3, model_mape_values, bar_width, label='CNN-LSTM')    
	ax.set_ylabel('MAPE')
	ax.set_title(plot_name)
	ax.set_xticks(bar_locations)
	ax.set_xticklabels(qci_set)
	ax.legend()
	fig.savefig(plot_path)
	pyplot.close(fig)
	return

def plot_series(axes,time, series, format="-", start=0, end=None, label_plot='Values', x_label='Time', y_label='Normalized Class Activity'):
	axes.plot(time[start:end], series[start:end], format, label=label_plot)
	axes.set(xlabel=x_label, ylabel=y_label)
	axes.grid(False)

def plot_test_metrics(mae_metrics, mse_metrics, mape_metrics, plotName, plotPath, numBins=10):
	fig, axes = pyplot.subplots(2,sharey=True,tight_layout=True)
	#plot mae values
	axes[0].hist(mae_metrics.values(), bins=numBins)
	axes[0].set_title('MAE-Distribution')
	#plot mse values
	axes[1].hist(mse_metrics.values(), bins=numBins)
	axes[1].set_title('MSE-Distribution')
	#plot mape_metrics
	fig.suptitle(plotName)
	fig.savefig(plotPath)
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

def plot_activity(client_model,client_id, plot_title, plot_path):
	ix = 0
	slice_set = [0,39]
	qci_set = ['Q1','Q2','Q3','Q4','Q5']
	fig, axes = pyplot.subplots(nrows=2, ncols=5, figsize=(6,6))
	for slice_id in slice_set:
		testDataset = load_slice_dataset(client_id, slice_id)
		print('###################### Predictions for Slice-'+ str(slice_id)+' ##########################')
		test_Output = client_model.predict(testDataset, verbose=1)
		test_Output = test_Output.reshape(test_Output.shape[0]*pred_win_len,NUM_FEATURE)
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

sys.stdout = open(os.path.join(base_results_dir,'output.txt'), 'w')

client_ids = list(range(0,NUM_CLIENTS))
client_lr = 0.05
client_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=client_lr,decay_steps=1000,decay_rate=0.9)

#Iterate over all combinations of obsv and pred window lengths
for obsv_win_len, pred_win_len in zip(OBSERVATION_WINDOW_RANGE, PREDICTION_WINDOW_RANGE):
	win_len = 'O-' + str(obsv_win_len) + '-P-' + str(pred_win_len)
	combo_dir = 'Combo-' + str(obsv_win_len) +'-'+str(pred_win_len) + '/'
	config_dir = base_results_dir + combo_dir 
	os.makedirs(config_dir, exist_ok=True)
	print('################################## '+ win_len  +' ##################################')
	client_datasets = {}
	#Load all datasets 
	for client_id in client_ids:
		client_datasets[client_id] = load_client_dataset(client_id)
		print('################################## Loaded Client:'+ str(client_id) +' Data ##################################')
	#Iterate over all combinations of optimization and scaler config
	for configuration in config_list:
		opalg = configuration[0]
		scler = configuration[1]
		algorithm = opalg + '/' + scler + '/'
		algo_dir = config_dir + algorithm
		os.makedirs(algo_dir, exist_ok=True)
		print('##################################'+ algorithm +'##################################')
		if opalg == 'SGD':
			client_optimizer = tf.keras.optimizers.SGD(learning_rate=client_lr_schedule, momentum=0.9)
		else:
			client_optimizer = tf.keras.optimizers.RMSprop(learning_rate=client_lr_schedule)i
		#Storage collections to maintain the client and slice level metrics
		mod_clnt_mae = {}
		mod_clnt_mse = {}
		mod_clnt_mape = {}
		mod_clnt_count = {}
		mod_slce_count = {}
		mod_slce_mae = {}
		mod_slce_mse = {}
		mod_slce_mape = {}
		mod_slce_clnt_metrics = {}
		#kearas_model dictionary for each client 
		keras_models = {}
		for model in models:
			keras_models[model] = {}
			#Initialize client metrics dictionary for current model
			mod_clnt_mae[model] = {}
			mod_clnt_mse[model] = {}
			mod_clnt_mape[model] = {}
			mod_clnt_count[model] = {}
			#Initialize slice metrics dictionary for current model
			mod_slce_mae[model] = {}
			mod_slce_mse[model] = {}
			mod_slce_mape[model] = {}
			mod_slce_count[model] = {}
			#Initialize slice client metrics for current model
			mod_slce_clnt_metrics[model] = {}
		#model metrics averaged over all clients 
		model_mse = {}
		model_mae = {}
		model_mape = {}
		#Train and test process for each client
		for client_id in client_ids:
			#client_datasets = dic{'client_id':(train_dataset, test_dataset, slice_test_datasets)}
			#client_data = (train_dataset, test_dataset, slice_test_datasets)
			client_data = client_datasets[client_id] 
			for model in models:
				#Retrieve client metrics dictionary for current model
				client_mse = mod_clnt_mae[model]
				client_mae = mod_clnt_mse[model]
				client_mape = mod_clnt_mape[model]
				client_count = mod_clnt_count[model]
				#Retrieve slice metrics dictionary for current model
				slice_mae = mod_slce_mae[model]
				slice_mse = mod_slce_mse[model]
				slice_mape = mod_slce_mape[model]
				slice_count = mod_slce_count[model]
				#Retrieve slice client metrics for current model
				slice_client_metrics = mod_slce_clnt_metrics[model]
				#Create the result directory of this model if it doesn't exist
				model_dir = algo_dir + model + '/'
				os.makedirs(model_dir, exist_ok=True)
				#
				############################## Model-Creation ##############################
				keras_model = create_keras_model()
				print(keras_model.summary())
				keras_model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=client_optimizer,metrics=[tf.keras.metrics.MeanAbsoluteError()])
				############################## Client - Local Training Procedure ##############################
				print('################################## Training using data of client:'+ str(client_id) +'##################################')
				#Train a model at each client using the local dataset
				history = keras_model.fit(client_data[0],epochs=NUM_EPOCHS,verbose=1)
				#save the global model
				model_path = os.path.join(model_dir,'model.h5')
				keras_model.save(model_path)
				keras_models[model][client_id] = keras_model
				print('#################################### Saving model ####################################')
				############################# Client - Local Test Procedure ##############################
				#Test the locally trained model using the client's local test data
				test_dataset = client_data[1]
				client_output = keras_model.evaluate(test_dataset, verbose=1)
				client_predictions = keras_model.predict(test_dataset, verbose=1)
				client_predictions = client_predictions.reshape(client_predictions.shape[0]*pred_win_len, NUM_FEATURE)
				client_true_values = transform_dataset(test_dataset)
				client_output_mape = tf.keras.losses.mean_absolute_percentage_error(client_true_values.T, client_predictions.T).numpy()
				############################# Client Metrics - MAE, MSE & MAPE ##############################
				if client in client_count:
					num_client_samples = client_count[client_id] + 1 
					client_count[client_id] = num_client_samples 
					client_mse[client_id] = calculate_new_avg(client_output[0], client_mse[client_id], num_client_samples) 
					client_mae[client_id] = calculate_new_avg(client_output[1], client_mae[client_id], num_client_samples)
					client_mape[client_id] = calculate_new_avg(client_output_mape, client_mape[client_id], num_client_samples) 
				else:
					client_count[client_id] = 1
					client_mse[client_id] = client_output[0]
					client_mae[client_id] = client_output[1]
					client_mape[client_id] = client_output_mape
				############################# Slice Metrics - MAPE, MAE & MSE ##############################
				slice_datasets = client_data[2]
				for slice_id in slice_datasets:
					print('###################### Slice-'+ str(slice_id)+' ##########################')
					slice_dataset = slice_datasets[slice_id]
					slice_output = keras_model.evaluate(slice_dataset, verbose=1)
					#generate the predictions and calculate percentage match for each QCI
					slice_predictions = keras_model.predict(slice_dataset, verbose=1)
					slice_predictions = slice_predictions.reshape(slice_predictions.shape[0]*pred_win_len, NUM_FEATURE)
					slice_true_values = transform_dataset(slice_dataset)
					slice_output_mape = tf.keras.losses.mean_absolute_percentage_error(slice_true_values.T, slice_predictions.T).numpy()
					#update slice metrics and loss statistics
					slice_client_metrics[(client_id,slice_id)] = slice_output
					if slice_id in slice_count:
						num_slice_samples = slice_count[slice_id] + 1
						slice_count[slice_id] = num_slice_samples
						slice_mae[slice_id] = calculate_new_avg(slice_output[1], slice_mae[slice_id], num_slice_samples)
						slice_mse[slice_id] = calculate_new_avg(slice_output[0], slice_mse[slice_id], num_slice_samples)
						slice_mape[slice_id] = calculate_new_avg(slice_output_mape, slice_mape[slice_id], num_slice_samples)
					else:
						slice_count[slice_id] = 1
						slice_mae[slice_id] = slice_output[1]
						slice_mse[slice_id] = slice_output[0]
						slice_mape[slice_id] = slice_output_mape
				#model-metrics
				if model in model_count:
					model_num_samples = model_count[model]
					model_count[model] = model_num_samples + 1
					model_mse[model] = calculate_new_avg(client_output[0], model_mse[model], model_num_samples) 
					model_mae[model] = calculate_new_avg(client_output[1], model_mae[model], model_num_samples)
					model_mape[model] = calculate_new_avg(client_output_mape, model_mape[model], model_num_samples)
				else:
					model_count[model] = 1
					model_mse[model] = client_output[0]
					model_mae[model] = client_output[1]
					model_mape[model] = client_output_mape
		######################### Plot Graphs for each model ###############################
		for model in models:
			#Retrieve client metrics dictionary for current model
			client_mse = mod_clnt_mae[model]
			client_mae = mod_clnt_mse[model]
			client_mape = mod_clnt_mape[model]
			client_count = mod_clnt_count[model]
			#Retrieve slice metrics dictionary for current model
			slice_mae = mod_slce_mae[model]
			slice_mse = mod_slce_mse[model]
			slice_mape = mod_slce_mape[model]
			slice_count = mod_slce_count[model]
			#Retrieve slice client metrics for current model
			slice_client_metrics = mod_slce_clnt_metrics[model]
			#set the result directory path for the plots 
			model_dir = algo_dir + model + '/'
			#plot client level metrics
			max_mae_client_id = max(client_mae, key=client_mae.get) 
			min_mae_client_id = min(client_mae, key=client_mae.get) 
			#plot client & slice level metrics - bargraph and contour
			plot_test_metrics(client_mae, client_mse, client_mape,'Test-Metrics-Client-Distribution', os.path.join(model_dir,'Test-Metrics-Client-Distribution.pdf'))
			plot_bargraph_metrics(slice_mae, slice_mse, 'Slice-Metrics-Distribution', os.path.join(model_dir,'Slice-Metrics-Distribution.pdf'))
			plot_metrics_contour(slice_client_metrics, 'S-NSSAI', 'Client-Id','Client-Slice-Test-Metrics', os.path.join(model_dir,'Client-Slice-Test-Metrics.pdf'))
			#plot client & slice level MAPE - contours
			plot_mape_contour(client_mape, 'QCI', 'Client-Id','Client-MAPE-Distribution', os.path.join(model_dir,'Client-MAPE-Distribution.pdf'))
			plot_mape_contour(slice_mape, 'QCI', 'Slice-Id','Slice-QCI-MAPE-Distribution', os.path.join(model_dir,'Slice-QCI-MAPE-Distribution.pdf'))
			#plot client & slice level MAPE - histograms
			plot_mape_histogram(slice_mape, numBins=5, 'Number of slices','Slice-MAPE-Distribution', os.path.join(model_dir,'Slice-MAPE-Histogram.pdf'))
			plot_mape_histogram(client_mape, numBins=10, 'Number of clients','Client-MAPE-Distribution', os.path.join(model_dir,'Client-MAPE-Histogram.pdf'))
			#
			############################## Time Series Plots & Predictions ##############################
			#Plotting time series for visual analysis for the min and max clients
			keras_model = keras_models[model][max_mae_client_id]
			plot_activity(keras_model,max_mae_client_id, 'Max-class-activity-predictions', os.path.join(model_dir,'Max-Class-activity-predicitions.pdf'))
			keras_model = keras_models[model][min_mae_client_id]
			plot_activity(keras_model,min_mae_client_id, 'Min-class-activity-predictions', os.path.join(model_dir,'Min-Class-activity-predicitions.pdf'))
		#
		title = algorithm + win_len + '-Model-Metrics' 
		plot_bargraph_metrics(model_mae, model_mse, title, os.path.join(algo_dir, 'model-metrics.pdf'))
		title = algorithm + win_len + '-Model-QCI-MAPE-Distribution'
		plot_models_mape(model_mape, title, os.path.join(algo_dir,'Model-QCI-MAPE-Distribution.pdf'))
sys.stdout.close()
