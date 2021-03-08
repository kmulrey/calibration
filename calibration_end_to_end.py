import numpy as np
import pickle
from optparse import OptionParser
import scipy.fftpack as fftp
import scipy.interpolate as intp
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import datetime
from datetime import date
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cal_routines as cal
import os.path
from os import path

base_dir='/vol/astro7/lofar/kmulrey/calibration/'



parser = OptionParser()

parser.add_option('-f', '--model_folder', default = 'jones_standard', help = 'jones matrix folder')
(options, args) = parser.parse_args()

antenna_model_folder = options.model_folder

jones_dir=base_dir+antenna_model_folder

print(jones_dir)
#data_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final'

flag=0
# 0 if the averages already exist
# 1 if averages need to be done
# 2 if no path exists

if antenna_model_folder=='jones_standard':
    print('running calibration with standard CR calibration')
    
elif path.exists(jones_dir):
    print('running with {0}'.format(antenna_model_folder))
    for f in np.arange(51):
        freq=str(f+30)
        if path.exists(jones_dir+'/'+'jones_all_'+freq+'.p')==False:
            flag=1
    if flag==1:
        print('need to average model')

else:
    print('no valid antenna model')
    flag=2
    

if flag==1:

    cal.average_model(jones_dir)
    
find_simulated_power(jones_dir, 'power')
