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



def average_model(jone_dir):
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
    
    


