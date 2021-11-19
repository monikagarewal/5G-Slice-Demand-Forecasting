# 5G-Slice-Demand-Forecasting
This repository contains the code for four appraoches to slice traffic forecasting: centralized, distributed, federated and hybrid. 
Details on the algorithm and individual approaches can be found in the paper (to be added)

How to run the code?
The code is organized into three major components: Dataset-Processing, Training and Evaluation Framework. 
The repository misses the dataset folder which should be located in the root dir of repository. We are using the telecom italia Milano city raw data. 
Refer to the os path variables listed in the dataset-processing code for accurate paths.
a) After setting up the dataset, generate federated dataset using datasetProcessing.py. Can run standardization analysis but not mandatory 
b) Execute the training scripts followed by evaluation framework 
NOTE: All paths are relative to the location the code is being run from. So run the code from its directory or ensure the paths are relative to your base directory 
Requirements
a) All packages are listed in requirements.txt and its also recommended to use a virtual environment to avoid any conflict with already installed python packages
