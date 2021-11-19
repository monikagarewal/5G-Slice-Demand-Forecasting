#!/usr/bin/env python

import os 
import re
import sys
import collections

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp 

import seaborn
from matplotlib import pyplot

import tensorflow as tf
import tensorflow_federated as tff

# creating regex for matching files in dataset
clientFilePattern = re.compile('client-.*\.csv$')

baseResultsDir = '../Graphs/Standardization/'
clientDirectoryPath = '../Datasets/MI-2013-Averaged/ClientData/'

def generate_violinplot(dataset, x_label, y_label, title, plot_name, tick_labels):
	axes = seaborn.violinplot(x=x_label, y=y_label, data=dataset)
	axes.set_xticklabels(tick_labels, rotation=90)
	fig = axes.get_figure()
	fig.suptitle(title)
	fig.savefig(plot_name)
	return  

#Standard-Scaler
def scale_dataset(scaler, dataset):
	scaler = scaler.fit(dataset)
	scaledDataset = scaler.transform(dataset)
	return scaledDataset

##create test data in a similar fashion to the train dataset
##load dataset
def load_dataset(clientId, sliceId):
	filename = 'client-' + str(clientId) + '.csv'
	print('------------------------ Loading client-'+ str(clientId) +' and slice-'+ str(sliceId) +' test dataset ------------------------')
	clientData = pd.read_csv(os.path.join(clientDirectoryPath, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
	clientData = clientData.drop(['Date','StartTime'], axis=1)
	sliceData = clientData.loc[clientData['NSSAI'] == sliceId]
	sliceData = sliceData.drop(['NSSAI'], axis=1)
	numTestSamples = sliceData.shape[0] - (int(0.66*sliceData.shape[0]))
	slice_TestData = sliceData.tail(numTestSamples)
	slice_TestData.reset_index(drop=True, inplace=True)
	return slice_TestData

#tryout different data scalers
def instantiate_scalers():
	scalers = {}
	scalers['Normalizer'] =  skp.Normalizer()
	scalers['MinMaxScaler'] = skp.MinMaxScaler()
	scalers['RobustScaler'] = skp.RobustScaler()
	scalers['StandardScaler'] = skp.StandardScaler()
	scalers['PowerTransformer'] = skp.PowerTransformer() 
	scalers['QuantileTransformer'] = skp.QuantileTransformer(n_quantiles=250,random_state=5)
	return scalers

NUM_CLIENTS = 100
sliceSet = [0,39,44,86]
scalers = instantiate_scalers()
for clientId in range(NUM_CLIENTS):
	clientDir = baseResultsDir + 'client-' + str(clientId) + '/'
	os.makedirs(clientDir, exist_ok=True)
	sys.stdout = open(os.path.join(clientDir,'output.txt'), 'w')
	for sliceId in sliceSet:
		testDataset = load_dataset(clientId, sliceId)
		qci_list = list(testDataset.columns)
		for scaler_type in scalers.keys():
			scaledData = scale_dataset(scalers[scaler_type],testDataset.values)
			scaledDataset = pd.DataFrame(data=scaledData,columns=testDataset.columns)
			scaledDataset = scaledDataset.melt(var_name='5QI', value_name='Activity')
			plot_name = clientDir + 'Slice-' + str(sliceId) + '-' + scaler_type + '.pdf'
			generate_violinplot(scaledDataset,'5QI','Activity', scaler_type, plot_name, testDataset.columns)
			for index in range(scaledData.shape[1]):
				series = scaledData[:,index]
				print('------------------------- ('+ scaler_type +','+ str(sliceId)+', QCI-'+ str(index+1)+') -------------------------')
				print('max: '+ str(series.max()) +'| min: '+ str(series.max()) +'| mean : '+ str(series.mean()) +'| num_zeroes : ' + str(np.count_nonzero(series == 0)))
	sys.stdout.close()
