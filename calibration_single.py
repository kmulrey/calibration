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

parser.add_option('-n', '--model_name', default = 'R700', help = 'run type name')
parser.add_option('-a', '--antenna', default = '664', help = 'antenna number')

parser.add_option('-r', '--reprocess_flag', default = 0, help = 'flag to reprocess model')
parser.add_option('-s', '--reprocess_power_flag', default = 0, help = 'flag to reprocess power')

(options, args) = parser.parse_args()

name = options.model_name

reprocess_flag = int(options.reprocess_flag)
antenna_no = int(options.antenna)

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
print(antenna_no)





if not os.path.exists(power_dir):
    os.makedirs(power_dir)


power_flag=0



for f in np.arange(51):
    freq=str(f+30)
    if path.exists(power_dir+'/integrated_power_'+str(freq)+'_antenna_'+str(antenna_no)+'.txt')==False:
        power_flag=1
        print('no power {0}, {1}'.format(freq,power_dir+'/'+'integrated_power_'+str(freq)+'.txt'))
   
if power_flag==1 or reprocess_power_flag==1:
    print('calculating power as a function of LST')
    #cal.find_simulated_power(jones_dir, power_dir)
    cal.find_simulated_power_single(jones_dir, power_dir, antenna_no)


if path.exists(consolidate_dir+'/power_'+station+'_antenna_'+str(antenna_no)+'.p')==False:
    consol_flag=1
    print('no consolidated info {0}'.format(consolidate_dir))
        
if consol_flag==1:
    print('now consolidating info')
    if not os.path.exists(consolidate_dir):
        os.makedirs(consolidate_dir)

    cal.consolidate_single(consolidate_dir,power_dir,data_dir,station,antenna_no)


print('doing calibration')

cal.do_fit_single(consolidate_dir,fit_data_dir,fit_dir,name,station,antenna_no,'X')
cal.do_fit_single(consolidate_dir,fit_data_dir,fit_dir,name,station,antenna_no,'y')
