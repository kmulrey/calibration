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

base_dir='/vol/astro7/lofar/kmulrey/calibration'



parser = OptionParser()

parser.add_option('-f', '--model_folder', default = 'jones_standard', help = 'jones matrix folder')
(options, args) = parser.parse_args()

antenna_model_folder = int(options.antenna)

jones_dir=base_dir+antenna_model_folder

print(jones_dir)
#data_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final'
