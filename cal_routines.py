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
LFmap_dir='/vol/astro3/lofar/sim/kmulrey/calibration/LFreduced/'


def average_model(jones_dir):
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
    
    


def do_integral_temp(theta_start,theta_stop, phi_start, phi_stop, temp, jones_theta, jones_phi):
    

    angle_th_stop=-0.5*np.cos(theta_stop*np.pi/180)*np.cos(theta_stop*np.pi/180)
    angle_th_start=-0.5*np.cos(theta_start*np.pi/180)*np.cos(theta_start*np.pi/180)
    angle_ph_stop=(phi_stop*np.pi/180)
    angle_ph_start=(phi_start*np.pi/180)
    antenna_response=jones_theta*jones_theta+jones_phi*jones_phi
    
    #print '{0}   {1}'.format(angle_start, angle_stop)
    return (angle_th_start-angle_th_stop)*(angle_ph_stop-angle_ph_start)*antenna_response*temp


def do_integral_freq(v_start, v_stop):
    return (1.0/3.0)*(v_stop*v_stop*v_stop-v_start*v_start*v_start)



def find_simulated_power(jones_dir, power_dir):

    for f in np.arange(51):
        freq=str(f+30)
        pickfile = open(LFmap_dir+'/LFreduced_'+str(freq)+'.p','rb')
        XX,YY,ZZ,XX2,YY2,times_utc,times_LST,ZZ2=pickle.load(pickfile, encoding="latin1")
    
        pickfile = open(jones_dir+'/jones_all_{0}.p'.format(freq),'rb')
        pickfile.seek(0)
        info=pickle.load(pickfile)
        pickfile.close()
        print(info['keys'])
        '''
            {'jones_thetaX':jones_thetaX_total,'jones_thetaY':jones_thetaY_total,'jones_phiX':jones_phiX_total,'jones_phiY':jones_phiY_total}
        jones=info['jones_cr']
        JJ=np.zeros([91,361,4],dtype='complex')
        '''
