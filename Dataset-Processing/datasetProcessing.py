#!/usr/bin/env python
import os
import sys
import re
import datetime
import collections

import numpy as np
import pandas as pd
from matplotlib import pyplot

# creating regex for matching files in dataset
gridFilePattern = re.compile('grid-.*\.csv$')
clntFilePattern = re.compile('client-.*\.csv$')
origFilePattern = re.compile('sms-call-internet-mi-2013-.*\.txt$')

# directory paths for the various type of datasets that are generated
baseDirectory = '../Datasets/MI-2013-Averaged/'
rawDatasetPath = '../Datasets/Raw-Data'

graphDir = os.path.join(baseDirectory,'Graphs/')
gridDir = os.path.join(baseDirectory,'GridData/')
clientDir = os.path.join(baseDirectory,'ClientData/')

#Check and create the base directories if they don't exist
os.makedirs(gridDir, exist_ok=True)
os.makedirs(graphDir, exist_ok=True)
os.makedirs(clientDir, exist_ok=True)

#printing monthly data for each feature at each site
def dataset_get_slice_stats():
	nssai_data_stats = {}
	nssai_grid_stats = {}
	nssai_grid_info = {}
	# load dataset into a dataset object
	for filename in os.listdir(rawDatasetPath):
		if origFilePattern.match(filename) != None:
			dataset = pd.read_csv(os.path.join(rawDatasetPath, filename), sep='\t', header=None, names=["GridId", "StartTime", "NSSAI", "Q1","Q2","Q3","Q4","Q5"], low_memory=False, infer_datetime_format=False)
			for sliceID in set(dataset['NSSAI']):
				sliceData = dataset.loc[dataset['NSSAI'] == sliceID]
				#update per slice data stats i.e. number of samples per slice over observed period
				if sliceID in nssai_data_stats:
					nssai_data_stats[sliceID] += sliceData.shape[0]
				else:
					nssai_data_stats[sliceID] = sliceData.shape[0]
				#update per slice grid stats i.e. number of samples per grid per slice over observed period
				if sliceID in nssai_grid_stats:
					nssai_grid_stats[sliceID].update(set(sliceData['GridId']))
				else:
					nssai_grid_stats[sliceID] = set(sliceData['GridId'])
		else:
			continue
	sortedItems = sorted(nssai_data_stats.items(), key=lambda x:x[1])
	mostUsedSlices = sortedItems[-15:]
	nssai_data_stats = dict(mostUsedSlices)
	for sliceID in nssai_data_stats.keys():
		nssai_grid_info[sliceID] = len(nssai_grid_stats[sliceID])
	plotPath = graphDir + 'Slice-Sample-Stats-Nov-2013.pdf'
	pyplot.bar(range(len(nssai_data_stats.values())), nssai_data_stats.values())
	pyplot.xticks(range(len(nssai_data_stats.keys())), nssai_data_stats.keys())
	pyplot.title('Slice Sample Stats for November')
	pyplot.xlabel('S-NSSAI')
	pyplot.ylabel('Number of Samples')
	pyplot.savefig(plotPath)
	return nssai_grid_info, nssai_data_stats 

#split dataset grid wise
def split_dataset_by_grid(selectedNssai):
	numFilesProcessed = 0
	# load datasets into a dataset object
	for filename in os.listdir(rawDatasetPath):
		if origFilePattern.match(filename) != None:
			numFilesProcessed += 1
			#print("Files processed : "+str(numFilesProcessed))
			dataset = pd.read_csv(os.path.join(rawDatasetPath, filename), sep='\t', header=None, names=["GridId", "StartTime", "NSSAI", "Q1","Q2","Q3","Q4","Q5"], low_memory=False, infer_datetime_format=False)
			#adding the date column to grid file as it now contains all dates
			substr = (filename.split("mi"))[1]
			date_str = ((substr.split('.'))[0])[1:]
			date = datetime.datetime.strptime(date_str, '%Y-%m-%d') 
			dateCol = [date]*(dataset.shape[0])
			dataset["Date"] = dateCol
			#extract the set of grids in the loaded dataset
			grid_set = set(dataset['GridId'])
			#for each grid filter the samples and write to the corresponding CSV file
			for grid in grid_set:
				gridfile = 'grid-'+str(grid)+'.csv'
				fullGridData = dataset.loc[dataset['GridId'] == grid]
				gridData = fullGridData.loc[fullGridData['NSSAI'].isin(selectedNssai)]
				if not os.path.isfile(os.path.join(gridDir,gridfile)):
					gridData.to_csv(os.path.join(gridDir,gridfile), index=False, header=["GridId", "StartTime", "NSSAI", "Q1","Q2","Q3","Q4","Q5","Date"])
				else:
					gridData.to_csv(os.path.join(gridDir,gridfile), mode='a', index = False, header=False)
		else:
			continue
	return  

#split dataset grid wise
def sort_csvdata_by_slice_and_date(filePattern, dataDir, destDir):
	numFilesProcessed = 0
	print("Files processed :  " + str(numFilesProcessed))
	# load datasets into a dataset object
	for filename in os.listdir(dataDir):
		if filePattern.match(filename) != None:
			dataset = pd.read_csv(os.path.join(dataDir, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
			#sort the grid dataset by the date and nssai values
			dataset.sort_values(by=['Date','NSSAI'], inplace=True)
			#replacing the 'na' values with 0s in the dataset (why not other imputation methods? - cause these values aren't missing, it's just no observed data)
			dataset.fillna(0, inplace=True)
			dataset.to_csv(os.path.join(destDir,filename), mode='w', index = False, header=list(dataset.columns))
			numFilesProcessed += 1
		else:
			continue
	print("Files processed :  " + str(numFilesProcessed))
	return  

#aggregate multiple grids data to form client data 
def generate_client_data():
	#merge every 100 grids data into one client data 
	for filename in os.listdir(gridDir):
		gridData = pd.read_csv(os.path.join(gridDir, filename), sep=',', header=0, low_memory=False, infer_datetime_format=False)
		gridId = gridData['GridId'].iloc[0]
		#calculate client ID and generate file name to write the grid data
		clientId = (int(gridId / 100) + 1) % 100
		clientFile = 'client-'+str(clientId)+'.csv'
		if not os.path.isfile(os.path.join(clientDir,clientFile)):
			gridData.to_csv(os.path.join(clientDir,clientFile), index=False, header=list(gridData.columns))
		else:
			gridData.to_csv(os.path.join(clientDir,clientFile), mode='a', index = False, header=False)
	print("All grid files processed!")
	return

def aggregate_client_data():
	#merge every 100 grids data into one client data 
	for filename in os.listdir(clientDir):
		clientData = pd.read_csv(os.path.join(clientDir, filename), sep=',', header=0, low_memory=False, infer_datetime_format=True)
		#create groups using date, time and slice ID. And drop the gridID as its no longer required (using sum or mean values?)
		aggData = clientData.groupby(['Date','StartTime','NSSAI']).mean().reset_index().drop('GridId', axis=1)
		aggData.sort_values(by=['Date','NSSAI'], inplace=True)
		aggData.to_csv(os.path.join(clientDir, filename), index=False, header=list(aggData.columns))
	print("All clients processed!")
	return

nssai_grid_stats, nssai_data_stats = dataset_get_slice_stats()
selectedNssai = nssai_grid_stats.keys()
print("Splitting Dataset")
split_dataset_by_grid(selectedNssai)
generate_client_data()
sort_csvdata_by_slice_and_date(clntFilePattern, clientDir, clientDir)
aggregate_client_data()
