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

parser.add_option('-n', '--model_name', default = 'standard', help = 'run type name')

parser.add_option('-r', '--reprocess_flag', default = 0, help = 'flag to reprocess model')
parser.add_option('-s', '--reprocess_power_flag', default = 0, help = 'flag to reprocess power')

(options, args) = parser.parse_args()

name = options.model_name

reprocess_flag = int(options.reprocess_flag)
reprocess_power_flag = int(options.reprocess_power_flag)

jones_dir=base_dir+'antenna_model/'+'jones_'+name
power_dir=base_dir+'power/power_'+name
consolidate_dir=base_dir+'consolidated_info/consolidated_'+name
data_dir='/vol/astro3/lofar/sim/kmulrey/calibration/TBBdata/'
fit_data_dir=base_dir+'fit_data'
fit_dir=base_dir+'fits/'


station='CS002'
antenna_model_folder='jones_'+name
print(jones_dir)
print(power_dir)
print(consolidate_dir)


#print(jones_dir)
#data_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final'

flag=0
# 0 if the averages already exist
# 1 if averages need to be done
# 2 if no path exists

if name=='standard':
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
    
# find average antenna model
flag=1
if flag==1 and name!='standard':

    cal.average_model(jones_dir)
    
    

    
# find correct LST power


if not os.path.exists(power_dir):
    os.makedirs(power_dir)
    
power_flag=0
#0 if directory exists with all freq files
#1 if not, or if reprocessing


for f in np.arange(51):
    freq=str(f+30)
    if path.exists(power_dir+'/'+'integrated_power_'+str(freq)+'.txt')==False:
        power_flag=1
        print('no power {0}, {1}'.format(freq,power_dir+'/'+'integrated_power_'+str(freq)+'.txt'))
   
if power_flag==1 or reprocess_power_flag==1:
    print('calculating power as a function of LST')
    cal.find_simulated_power(jones_dir, power_dir)
    #cal.find_simulated_power_single(jones_dir, power_dir, 2)



consol_flag=0
#0 if directory exists with all freq files
#1 if not, or if reprocessing


for f in np.arange(51):
    freq=str(f+30)
    if path.exists(consolidate_dir+'/power_all_80m.p')==False:
        consol_flag=1
        print('no consolidated info {0}'.format(consolidate_dir))
        
consol_flag=1
if consol_flag==1:
    print('now consolidating info')
    if not os.path.exists(consolidate_dir):
        os.makedirs(consolidate_dir)

    cal.consolidate(consolidate_dir,power_dir,data_dir,station,2)


print('doing fit')

    
cal.do_fit(consolidate_dir,fit_data_dir,fit_dir,name,station,2)

