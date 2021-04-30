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
import glob
import re
from scipy import signal
from scipy.signal import resample




def correlate(data,sim,event,station,antenna,pol,caltype):
    a_raw=data
    b_raw=1*sim
    a=resample(a_raw,400)
    b=resample(b_raw,400)
    #scale=np.max(a)/np.max(b)
    #b=b*scale
    a=a/np.max(a)
    b=b/np.max(b)

    timestep = 1
    start = 0
    end = 400#80*51
    signal_size = abs(end-start)
    lag_values = np.arange(-signal_size+timestep, signal_size, timestep)
    t = np.linspace(start, end, int(end/timestep))
    
    crosscorr = signal.correlate(a,b)
    max_crosscorr_idx = np.argmax(crosscorr)
    
    lag = lag_values[max_crosscorr_idx]
    lag_timesteps = int(round(lag_values[max_crosscorr_idx]/timestep))

    if lag > 0:
        new_a = list(a) + [np.nan]*lag_timesteps
        new_b = [np.nan]*lag_timesteps + list(b)
        new_t = np.linspace(start, end+lag, int(round((end+lag)/timestep)))
    else:
        new_a = [np.nan]*abs(lag_timesteps) + list(a)
        new_b = list(b) + [np.nan]*abs(lag_timesteps)
        new_t = np.linspace(start+lag, end, int(round((end-(start+lag))/timestep)))
    
    return new_t, new_a, new_b, np.max(crosscorr)




def return_cal(caltype):

    calfile='/vol/astro7/lofar/kmulrey/calibration/fits/fits_'+caltype+'_CS002.p'
    antenna_response_dir='/vol/astro7/lofar/kmulrey/calibration/antenna_model/jones_'+caltype+'/'
    antenna_response_std='/vol/astro7/lofar/kmulrey/calibration/antenna_model/jones_standard/'
    fin=open(calfile,"rb")
    info=pickle.load(fin)
    fin.close()

    Calibration_curve_new=np.zeros(101)
    Calibration_curve_new[30:81]=info['cal']
    Calibration_curve=np.zeros(101)
    Calibration_curve[29:82]=np.array([0, 1.37321451961e-05,1.39846332239e-05,1.48748993821e-05,1.54402170354e-05,1.60684568225e-05,1.66241942741e-05,1.67039066047e-05,1.74480931848e-05,1.80525736486e-05,1.87066855054e-05,1.88519099831e-05,1.99625051386e-05,2.01878566584e-05,2.11573680797e-05,2.15829455528e-05,2.20133824866e-05,2.23736319125e-05,2.24484419697e-05,2.37802483891e-05,2.40581543111e-05,2.42020383477e-05,2.45305869187e-05,2.49399905965e-05,2.63774023804e-05,2.70334253414e-05,2.78034857678e-05,3.07147991391e-05,3.40755705892e-05,3.67311849851e-05,3.89987440028e-05,3.72257913465e-05,3.54293510934e-05,3.35552370942e-05,2.96529815929e-05,2.79271252352e-05,2.8818544973e-05,2.92478843809e-05,2.98454768706e-05,3.07045462103e-05,3.07210553534e-05,3.16442871206e-05,3.2304638838e-05,3.33203882046e-05,3.46651060935e-05,3.55193137077e-05,3.73919275937e-05,3.97397037914e-05,4.30625048727e-05,4.74612081994e-05,5.02345866124e-05, 5.53621848304e-05,0])

    return Calibration_curve,Calibration_curve_new
