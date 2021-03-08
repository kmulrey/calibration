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
        if path.exists(jones_dir+'/'+'jones_all_'+freq+'_average.p')==False:
            flag=1
    if flag==1:
        print('need to average model')

else:
    print('no valid antenna model')
    flag=2
    

if flag==1:
    array_ind_outer=np.arange(576,(576+96))[::2]   # indices for CSOO2 outer
    count=0
    jones_thetaX_total=np.zeros([361,91])
    jones_thetaY_total=np.zeros([361,91])
    jones_phiX_total=np.zeros([361,91])
    jones_phiY_total=np.zeros([361,91])
    for f in np.arange(51):
        freq=str(f+30)
        for i in np.arange(len(array_ind_outer)):
            ant_id=array_ind_outer[i]
            file=open(jones_dir+'/jones_all_'+freq+'_antenna_'+str(ant_id)+'.p','rb')
            info=pickle.load(file, encoding="latin1")
            file.close()
            jones_aartfaac=info['jones_aartfaac']
            jones_thetaX_total=jones_thetaX_total+np.abs(jones_aartfaac.T[0])
            jones_thetaY_total=jones_thetaY_total+np.abs(jones_aartfaac.T[1])
            jones_phiX_total=jones_phiX_total+np.abs(jones_aartfaac.T[2])
            jones_phiY_total=jones_phiY_total+np.abs(jones_aartfaac.T[3])
            count=count+1
   
        jones_thetaX_total=jones_thetaX_total/count
        jones_thetaY_total=jones_thetaY_total/count
        jones_phiX_total=jones_phiX_total/count
        jones_phiY_total=jones_phiY_total/count
    

    
        info={'jones_thetaX':jones_thetaX_total,'jones_thetaY':jones_thetaY_total,'jones_phiX':jones_phiX_total,'jones_phiY':jones_phiY_total}
        file2=open(jones_dir+'/jones_all_'+freq+'.p','wb')
        pickle.dump(info,file2)
        file2.close()
    
    

