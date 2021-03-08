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





parser = OptionParser()

parser.add_option("-a", "--antenna", default = "1", help = "antenna ID")
(options, args) = parser.parse_args()

AART_ant_id = int(options.antenna)



#data_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final'
