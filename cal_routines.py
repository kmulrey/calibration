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



kB=1.38064852e-23
c=3.0e8
Z0=120*np.pi
Z=120*np.pi

ZL=75
#data_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final'

