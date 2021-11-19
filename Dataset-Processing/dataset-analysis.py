#!/usr/bin/env python

import os
import sys
import re
import datetime
import collections

import numpy as np
import pandas as pd

from matplotlib import pyplot
import sklearn.preprocessing as skp 

# creating regex for matching files in dataset
dataFilePattern = re.compile('client-.*\.csv$')

# directory paths for the various type of datasets that are generated
dataDirectory = '../Datasets/MI-2013-Averaged/'

graphDir = os.path.join(dataDirectory,'Graphs/')
clientDir = os.path.join(dataDirectory,'ClientData/')
os.makedirs(graphDir, exist_ok=True)
os.makedirs(clientDir, exist_ok=True)

def plot_series(axes,time, series, format="-", start=0, end=None, label_plot='Observed Values', x_label='Time', y_label='Bytes'):
    axes.plot(time[start:end], series[start:end], format, label=label_plot)
    #axes.set(xlabel=x_label, ylabel=y_label)
    axes.grid(False)

#generate time-series graph per slice per client for 5 clients
nssai_set = [0,39,33,49]
qci_set = ['Q1','Q2','Q3','Q4','Q5']
def plot_activity_graphs(clientData):
	fig, axes = pyplot.subplots(nrows=4,ncols=2,sharex=True, figsize=(6,6))
	for i in range(len(nssai_set)):
		sliceData = clientData.loc[clientData['NSSAI']==nssai_set[i]]
		time = (sliceData['StartTime'] - sliceData['StartTime'].min())/600000
		sliceData = sliceData.drop(['Date','StartTime','NSSAI'],axis=1)
		sliceData.reset_index(drop=True,inplace=True)
		#scaler = skp.RobustScaler(unit_variance=True)
		#scaledData = scaler.fit_transform(sliceData.values)
		print('################# Slice - '+ str(nssai_set[i])+' Client-'+ str(clientId)+ ' #################')
		for index in range(len(qci_set)):
			#qci_scld_activity = scaledData[:,index]
			qci_scld_activity = (sliceData[qci_set[index]] - sliceData[qci_set[index]].min())/(sliceData[qci_set[index]].max() - sliceData[qci_set[index]].min())
			qci_real_activity = sliceData[qci_set[index]].values
			print(str(qci_set[index]) + '-Scaled - max: '+ str(qci_scld_activity.max()) +'| min: '+ str(qci_scld_activity.max()) +'| mean : '+ str(qci_scld_activity.mean()) +'| num_zeroes : ' + str(np.count_nonzero(qci_scld_activity == 0)))
			print(str(qci_set[index]) + '-Actual - max: '+ str(qci_real_activity.max()) +'| min: '+ str(qci_real_activity.max()) +'| mean : '+ str(qci_real_activity.mean()) +'| num_zeroes : ' + str(np.count_nonzero(qci_real_activity == 0)))
			print('################# Slice - '+ str(nssai_set[i])+' Client-'+ str(clientId)+ ' #################')
			plot_series(axes[i][0], time.values, qci_scld_activity.values, label_plot=qci_set[index])
			plot_series(axes[i][1], time.values, qci_real_activity, label_plot=qci_set[index])
			title = 'S-' + str(nssai_set[i]) + '-Scaled'
			axes[i][0].set_title(title)
			title = 'S-' + str(nssai_set[i]) + '-Actual'
			axes[i][1].set_title(title)
	#end of loop - plot the image and legend 
	handles, labels = axes[0][0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right')
	fig.text(0.5, 0.04, 'Time', ha='center')
	fig.text(0.04, 0.5, 'Activity', va='center', rotation='vertical')
	s_title = 'Time series plots for client -' + str(clientId) 
	fig.suptitle(s_title)
	plotName = graphDir + 'TimeSeries-client' + clientId + '.pdf'
	fig.savefig(plotName)
	fig.clf()

def plot_client_sample_distribution(clientSampleStats, plotTitle, plotPath):
	pyplot.bar(range(len(clientSampleStats.values())), clientSampleStats.values())
	pyplot.xticks(range(len(clientSampleStats.keys())), clientSampleStats.keys())
	pyplot.title(plotTitle)
	pyplot.xlabel('Client-ID')
	pyplot.ylabel('Number of Samples')
	pyplot.savefig(plotPath)


def generate_clientdata_plots():
	numFilesProcessed = 0
	clientSampleStats = {}
	clientSliceDistribution = {}
	print("Files processed :  " + str(numFilesProcessed))
	# load datasets into a dataset object
	for filename in os.listdir(clientDir):
		if dataFilePattern.match(filename) != None:
			clientData = pd.read_csv(os.path.join(clientDir, filename), sep=',', header=0, low_memory=False, infer_datetime_format=False)
			clientId = (((filename.split('-'))[1]).split('.'))[0]
			#update the client and slice sample collections
			clientSampleStats[clientId] = clientData.shape[0]
			sliceSet = set(clientData['NSSAI'])
			for slice_id in range(slice_set)
				clientSliceDistribution[]
			#now plot the per slice time series pattern for any 5 clients
			#if numFilesProcessed < 5:
			#	plot_activity_graphs(clientData)
			#else:
			#	continue
			numFilesProcessed += 1
		else:
			continue
	print("Files processed :  " + str(numFilesProcessed))
	plot_client_sample_distribution(clientSampleStats, 'Client-Sample-Distribution', os.path.join(graphDir, 'Client-Sample-Distribution.pdf'))
	plot_client_slice_sample_distribution(clientSliceDistribution, 'Client-Slice-Sample-Distribution', os.path.join(graphDir,'Client-Slice-Sample-Distribution.pdf'))
	return  

generate_clientdata_plots()
