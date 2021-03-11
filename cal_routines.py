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



def find_simulated_power(con_dir,jones_dir, power_dir):

    for f in np.arange(51):
        freq=str(f+30)
        print(freq)
        pickfile = open(LFmap_dir+'/LFreduced_'+str(freq)+'.p','rb')
        XX,YY,ZZ,XX2,YY2,times_utc,times_LST,ZZ2=pickle.load(pickfile, encoding="latin1")
    
        pickfile = open(jones_dir+'/jones_all_{0}.p'.format(freq),'rb')
        pickfile.seek(0)
        info=pickle.load(pickfile)
        pickfile.close()
        jones_thetaX=info['jones_thetaX']
        jones_thetaY=info['jones_thetaY']
        jones_phiX=info['jones_phiX']
        jones_phiY=info['jones_phiY']
        JJ=np.zeros([91,361,4],dtype='complex')

        for th in np.arange(90):
            for az in np.arange(360):
                phi=az
                i_az=az+180
                if az>180:
                    phi=az-360
                    i_az=az-180

                JJ[90-th][i_az][0]=jones_thetaX[i_az][th]
                JJ[90-th][i_az][1]=jones_thetaY[i_az][th]
                JJ[90-th][i_az][2]=jones_phiX[i_az][th]
                JJ[90-th][i_az][3]=jones_phiY[i_az][th]
                
                
                
                
        int_theta=np.arange(0.5,90,1.0)
        int_theta=np.append([0],int_theta)
        int_theta=np.append(int_theta,[90])

        int_phi=np.arange(-179.5,180,1.0)
        int_phi=np.append([-180],int_phi)
        int_phi=np.append(int_phi,[180])
        
        total_int_temp_X=np.zeros(len(times_LST))
        total_int_temp_Y=np.zeros(len(times_LST))
        total_int_temp=np.zeros(len(times_LST))

        total_int_v_A=np.zeros(len(times_LST))
        total_int_v=np.zeros(len(times_LST))

        total_int_X=np.zeros(len(times_LST))
        total_int_Y=np.zeros(len(times_LST))
        total_int=np.zeros(len(times_LST))



        v_start=(float(freq)-0.5)*1e6
        v_stop=(float(freq)+0.5)*1e6
    
        for t in np.arange(len(times_LST)):
            for theta in np.arange(len(int_theta)-1):
                for phi in np.arange(len(int_phi)-1):
                    temp=ZZ2[t][theta+90][phi]
                    if (float('-inf') < float(temp) < float('inf'))==False:
                        temp=0
            
                    jones_thetaX=np.abs(JJ[theta][phi][0])
                    jones_phiX=np.abs(JJ[theta][phi][1])
                    jones_thetaY=np.abs(JJ[theta][phi][2])
                    jones_phiY=np.abs(JJ[theta][phi][3])
                    theta_start=90-int_theta[theta]
                    theta_stop=90-int_theta[theta+1]
                    phi_start=int_phi[phi]
                    phi_stop=int_phi[phi+1]
                    jones_phi=1
                    jones_theta=1
            
            
                    intX=do_integral_temp(theta_start,theta_stop, phi_start, phi_stop, temp, jones_thetaX, jones_phiX)
                    intY=do_integral_temp(theta_start,theta_stop, phi_start, phi_stop, temp, jones_thetaY, jones_phiY)
                    int=do_integral_temp(theta_start,theta_stop, phi_start, phi_stop, temp, jones_theta, jones_phi)

                    total_int_temp_X[t]=total_int_temp_X[t]+intX
                    total_int_temp_Y[t]=total_int_temp_Y[t]+intY
                    total_int_temp[t]=total_int_temp[t]+int


            total_int_v[t]=do_integral_freq(v_start, v_stop)
            total_int[t]=(kB/(c*c))*total_int_v[t]*total_int_temp[t]
            total_int_X[t]=(kB/(c*c))*total_int_v[t]*total_int_temp_X[t]
            total_int_Y[t]=(kB/(c*c))*total_int_v[t]*total_int_temp_Y[t]

    
        inds=np.asarray(times_LST).argsort()


        times_sorted=np.asarray(times_LST)[inds]
        times_sorted_utc=np.asarray(times_utc)[inds]

        power_sorted=total_int[inds]
        power_sorted_X=total_int_X[inds]
        power_sorted_Y=total_int_Y[inds]


        outfile=open(power_dir+'/integrated_power_'+str(freq)+'.txt','w')
        print(power_dir+'/integrated_power_'+str(freq)+'.txt')
        for i in np.arange(len(times_LST)):
            outfile.write('{0}  {1}  {2}  {3}  {4} \n'.format(times_sorted[i],times_sorted_utc[i],power_sorted[i],power_sorted_X[i],power_sorted_Y[i]))
        outfile.close()



def consolidate(con_dir,power_dir,data_dir,station):

    nTimes=24
    nFreq=51
    nData=5
    frequencies=np.arange(30,80.5,1)
    power=np.zeros([nFreq,nTimes,nData])
    int_sim_X=np.zeros([nFreq,len(bins)])
    int_sim_Y=np.zeros([nFreq,len(bins)])


    for f in np.arange(nFreq):
        file=open(power_dir+'/integrated_power_'+str(f+30)+'.txt','rb')
        temp=np.genfromtxt(file)
    
        for t in np.arange(nTimes):
            power[f][t][0]=temp[t][0]
            power[f][t][1]=temp[t][1]
            power[f][t][2]=temp[t][2]
            power[f][t][3]=temp[t][4]
            power[f][t][4]=temp[t][3]
    

        fX = interp1d(power[f].T[0], power[f].T[3],kind='cubic',fill_value='extrapolate')
        fY = interp1d(power[f].T[0], power[f].T[4],kind='cubic',fill_value='extrapolate')
        
        int_sim_X[f]=fX(bins)
        int_sim_Y[f]=fY(bins)

        holdX=int_sim_X
        holdY=int_sim_Y

        # correct for hour offset
        for t in np.arange(nTimes-4):
            int_sim_X.T[t]=holdX.T[t+4]
            int_sim_Y.T[t]=holdY.T[t+4]

        int_sim_X.T[92]=holdX.T[0]
        int_sim_X.T[93]=holdX.T[1]
        int_sim_X.T[94]=holdX.T[2]
        int_sim_X.T[95]=holdX.T[3]

        int_sim_Y.T[92]=holdY.T[0]
        int_sim_Y.T[93]=holdY.T[1]
        int_sim_Y.T[94]=holdY.T[2]
        int_sim_Y.T[95]=holdY.T[3]
            
    cable_lengths=['50','80','115']
    
    for c in np.arange(len(cable_lengths)):
        dh=0.25
        bin_edges=np.arange(0,24.1,dh)
        bins=np.arange(0.0,23.9,dh)
        nTimes=len(bins)


        infile=open('/vol/astro3/lofar/sim/kmulrey/calibration/TBBdata/'+station+'_noise_power_'+cable_lengths[c]+'.p','rb')
        tbbInfo=pickle.load(infile, encoding="latin1")
        infile.close()

        avg_power_X=tbbInfo['avg_power_X_'+cable_lengths[c]].T
        std_power_X=tbbInfo['std_power_X_'+cable_lengths[c]].T
        avg_power_Y=tbbInfo['avg_power_Y_'+cable_lengths[c]].T
        std_power_Y=tbbInfo['std_power_Y_'+cable_lengths[c]].T

        pickfile = open(con_dir+'/power_all_'+cable_lengths[c]+'m.p','wb')
        pickle.dump((bins,int_sim_X,int_sim_Y,avg_power_X,std_power_X,avg_power_Y,std_power_Y),pickfile)
        pickfile.close()
