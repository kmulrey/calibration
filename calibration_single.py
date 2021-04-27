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
