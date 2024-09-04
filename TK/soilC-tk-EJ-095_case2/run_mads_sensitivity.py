# Sensitivity adapted for the calibration type output
# uses calibration configuration file as an input
# Example: python3 run_mads_sensitivity.py /work/mads_calibration/config-step1-md1.yaml
# Author: Elchin Jafarov 
# Date: 03/27/2023

import os,sys
import json
import yaml
import numpy as np
import pandas as pd
import mads_sensitivity as Sensitivity

#read the config yaml file and 
if len(sys.argv) != 2:
    print("Usage: python run_mads_sensitivity.py <path/configfilename>")
    sys.exit(1)

config_file_name = sys.argv[1]
print(f"The filename you provided is: {config_file_name}")

with open(config_file_name, 'r') as config_data:
    config = yaml.safe_load(config_data)

#define the SA setup
driver = Sensitivity.SensitivityDriver(config_file=config_file_name)
driver.clean()
sample_size=500
driver.design_experiment(sample_size, driver.cmtnum,
  params=driver.paramnames,
  pftnums=driver.pftnums,
  percent_diffs=list(0.9999999995*np.ones(len(driver.pftnums))),
  sampling_method='uniform')

initial=config['mads_initial_guess']

#perturbation=0.75
#for i in range(len(driver.params)):
#    driver.params[i]['initial']=initial[i]
#    driver.params[i]['bounds']=[initial[i] - (initial[i]*perturbation), initial[i] + (initial[i]*perturbation)]

print('params:',driver.params)
#customize bounds
#new_bounds=[[1e-1, 2.0], [0.1, 0.99], [0.01, 0.1], [0.0001,0.01], [1e-7, 1e-4]]
#[ [0.1, 20.0],[0.1, 20.0],[0.1, 20.0],[0.1, 20.0],[0.1, 20.],[0.1,20.0],[0.1,20.0],[0.1,20.0]]
        #[1, 5], [1, 5], [1, 5], [1, 5], \
#        [-20, -0.1],[-20, -0.1],[-20, -0.1],[-20, -0.1],[-20, -0.1], \
#        [-20, -0.1],[-20, -0.1],[-20, -0.1],[-20, -0.1],[-20, -0.1], \
#        [0.00001, 0.2],[0.00001, 0.2],[0.00001, 0.2],[0.00001, 0.2],[0.00001, 0.2], \
#        [0.00001, 0.2],[0.00001, 0.2],[0.00001, 0.09],[0.00001, 0.09],[0.00001, 0.09] \
#        ]

#new_bounds=[ [0.1, 20],[.1, 50],[.1, 30],[.1, 10],[.1, 15],[.1, 20],[.1, 15] ]
#driver.params[2]['bounds']=new_bounds[2]
#for i in range(len(driver.params)):
#    driver.params[i]['bounds']=new_bounds[i]

#[2.325, 3.875],[1.875, 3.125],[1.875, 3.125],[1.575, 2.625], \
#            [-4.5, -7.5],[-2.5875, -4.3125],[-2.2125, -3.6875],[-3.4875, -5.8125],\
#           [-3.66, -6.1],[-3.863, -6.4375],[-4.99, -8.3125],\
#            [-6.15, -10.25],[-4.65, -7.75],[-2.4, -4.0]]

#{'name': 'micbnup', 'bounds': [0.004495000000000027, 0.894505], 'initial': 0.4495, 'cmtnum': 1, 'pftnum': 'None'}
#{'name': 'kdcrawc', 'bounds': [0.006340000000000012, 1.26166], 'initial': 0.634, 'cmtnum': 1, 'pftnum': 'None'}
#{'name': 'kdcsoma', 'bounds': [0.00539999999999996, 1.0746000000000002], 'initial': 0.54, 'cmtnum': 1, 'pftnum': 'None'}
#{'name': 'kdcsompr', 'bounds': [2.0000000000000052e-05, 0.00398], 'initial': 0.002, 'cmtnum': 1, 'pftnum': 'None'}
#{'name': 'kdcsomcr', 'bounds': [7.000000000000035e-07, 0.00013929999999999997], 'initial': 7e-05, 'cmtnum': 1, 'pftnum': 'None'}


for i in range(len(driver.pftnums)):
    print(driver.params[i])

#for i in range(len(driver.params)):
#    driver.params[i]['bounds']=new_bounds[i]

driver.generate_lhc(sample_size)
#print(driver.pftnums)
#print(driver.info())
#exit()
#setup folders based on a sample size  
try:
    driver.setup_multi(calib=True)
except ValueError:
    print("Oops!  setup_multi failed.  Check the setup...")

#run themads_sensitivity in parallel
try:
    driver.run_all_samples()
except ValueError:
    print("Oops!  run_all_samples failed.  Check the sample folders...")

#save results in the work_dir results.txt
#NOTE, that the last row in the results.txt is targets/observations
driver.save_results()

