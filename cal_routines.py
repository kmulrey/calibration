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
from scipy.optimize import minimize


kB=1.38064852e-23
c=3.0e8
Z0=120*np.pi
Z=120*np.pi

ZL=75
#data_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final'
LFmap_dir='/vol/astro3/lofar/sim/kmulrey/calibration/LFreduced/'


jones_vals=np.genfromtxt('../fit_data/antenna_gain.txt')
RCU_gain=np.genfromtxt('../fit_data/RCU_gain_new_5.txt',usecols=0)

nFreq=51
nTimes=96


frequencies=np.arange(30,80.5,1)
times=np.arange(0,24.0,0.25)




def e_ACGMB_allCables(pars,data_X_50,std_X_50,sim_X_50,data_X_80,std_X_80,sim_X_80,data_X_115,std_X_115,sim_X_115,data_Y_50,std_Y_50,sim_Y_50,data_Y_80,std_Y_80,sim_Y_80,data_Y_115,std_Y_115,sim_Y_115,jones,cable_attenuation_50,cable_attenuation_80,cable_attenuation_115,RCU_gain,s):
    
    
    nF=len(data_X_50)
    nT=len(data_X_50[0])
    a=pars[0]
    c=pars[1]
    g=pars[2]
    
    #g=np.power(10,(g_par10)
    g115=np.power(10,(g)/10)
    g80=np.power(10,(g-1.5)/10)
    g50=np.power(10,(g-2.75)/10)
    
    
    b=pars[3]
    
    d=np.zeros([nFreq])
    
    for f in np.arange(nFreq):
        d[f]=b
    

    gain_curve=np.power(10,RCU_gain/10)
    
    rcu=g*gain_curve
    rcu50=g50*gain_curve
    rcu80=g80*gain_curve
    rcu115=g115*gain_curve
    
    
    A=np.zeros([nF])
    A_X_50=np.zeros([nF])
    A_Y_50=np.zeros([nF])
    A_X_80=np.zeros([nF])
    A_Y_80=np.zeros([nF])
    A_X_115=np.zeros([nF])
    A_Y_115=np.zeros([nF])
    
    
    data_corr_X_50=np.zeros([nF,nT])
    sim_corr_X_50=np.zeros([nF,nT])
    data_corr_X_80=np.zeros([nF,nT])
    sim_corr_X_80=np.zeros([nF,nT])
    data_corr_X_115=np.zeros([nF,nT])
    sim_corr_X_115=np.zeros([nF,nT])
    data_corr_Y_50=np.zeros([nF,nT])
    sim_corr_Y_50=np.zeros([nF,nT])
    data_corr_Y_80=np.zeros([nF,nT])
    sim_corr_Y_80=np.zeros([nF,nT])
    data_corr_Y_115=np.zeros([nF,nT])
    sim_corr_Y_115=np.zeros([nF,nT])
    
    sim_corrected_X_50=np.zeros([nF,nT])
    sim_corrected_X_80=np.zeros([nF,nT])
    sim_corrected_X_115=np.zeros([nF,nT])
    sim_corrected_Y_50=np.zeros([nF,nT])
    sim_corrected_Y_80=np.zeros([nF,nT])
    sim_corrected_Y_115=np.zeros([nF,nT])
    
    
    
    for f in np.arange(nF):
        for t in np.arange(nT):
            sim_corr_X_50[f][t]=(sim_X_50[f][t]+a*jones[f])
            sim_corr_X_80[f][t]=(sim_X_80[f][t]+a*jones[f])
            sim_corr_X_115[f][t]=(sim_X_115[f][t]+a*jones[f])
            sim_corr_Y_50[f][t]=(sim_Y_50[f][t]+a*jones[f])
            sim_corr_Y_80[f][t]=(sim_Y_80[f][t]+a*jones[f])
            sim_corr_Y_115[f][t]=(sim_Y_115[f][t]+a*jones[f])
            
            data_corr_X_50[f][t]=((data_X_50[f][t]-d[f])/(rcu50[f]*s)-c)*np.power(10.0,(cable_attenuation_50[f]/10.0))
            data_corr_X_80[f][t]=((data_X_80[f][t]-d[f])/(rcu80[f]*s)-c)*np.power(10.0,(cable_attenuation_80[f]/10.0))
            data_corr_X_115[f][t]=((data_X_115[f][t]-d[f])/(rcu115[f]*s)-c)*np.power(10.0,(cable_attenuation_115[f]/10.0))
            data_corr_Y_50[f][t]=((data_Y_50[f][t]-d[f])/(rcu50[f]*s)-c)*np.power(10.0,(cable_attenuation_50[f]/10.0))
            data_corr_Y_80[f][t]=((data_Y_80[f][t]-d[f])/(rcu80[f]*s)-c)*np.power(10.0,(cable_attenuation_80[f]/10.0))
            data_corr_Y_115[f][t]=((data_Y_115[f][t]-d[f])/(rcu115[f]*s)-c)*np.power(10.0,(cable_attenuation_115[f]/10.0))



    for f in np.arange(nF):
        A_X_50[f]=np.average(data_corr_X_50[f])/np.average(sim_corr_X_50[f])
        A_X_80[f]=np.average(data_corr_X_80[f])/np.average(sim_corr_X_80[f])
        A_X_115[f]=np.average(data_corr_X_115[f])/np.average(sim_corr_X_115[f])
        A_Y_50[f]=np.average(data_corr_Y_50[f])/np.average(sim_corr_Y_50[f])
        A_Y_80[f]=np.average(data_corr_Y_80[f])/np.average(sim_corr_Y_80[f])
        A_Y_115[f]=np.average(data_corr_Y_115[f])/np.average(sim_corr_Y_115[f])
    
    
    for f in np.arange(nF):
        for t in np.arange(nT):
            sim_corrected_X_50[f][t]=((((sim_X_50[f][t]+a*jones[f])*A_X_50[f])/np.power(10.0,(cable_attenuation_50[f]/10.0)))+c)*rcu50[f]*s+d[f]
            sim_corrected_X_80[f][t]=((((sim_X_80[f][t]+a*jones[f])*A_X_80[f])/np.power(10.0,(cable_attenuation_80[f]/10.0)))+c)*rcu80[f]*s+d[f]
            sim_corrected_X_115[f][t]=((((sim_X_115[f][t]+a*jones[f])*A_X_115[f])/np.power(10.0,(cable_attenuation_115[f]/10.0)))+c)*rcu115[f]*s+d[f]
            sim_corrected_Y_50[f][t]=((((sim_Y_50[f][t]+a*jones[f])*A_Y_50[f])/np.power(10.0,(cable_attenuation_50[f]/10.0)))+c)*rcu50[f]*s+d[f]
            sim_corrected_Y_80[f][t]=((((sim_Y_80[f][t]+a*jones[f])*A_Y_80[f])/np.power(10.0,(cable_attenuation_80[f]/10.0)))+c)*rcu80[f]*s+d[f]
            sim_corrected_Y_115[f][t]=((((sim_Y_115[f][t]+a*jones[f])*A_Y_115[f])/np.power(10.0,(cable_attenuation_115[f]/10.0)))+c)*rcu115[f]*s+d[f]

    X2=0
    
    X2_X_50=0
    X2_X_80=0
    X2_X_115=0
    
    X2_Y_50=0
    X2_Y_80=0
    X2_Y_115=0
    
    for f in np.arange(nF):
        for t in np.arange(nT):
            
            X2_X_50=np.power((data_X_50[f][t]-sim_corrected_X_50[f][t]),2)/(std_X_50[f][t]*std_X_50[f][t])
            X2_X_80=np.power((data_X_80[f][t]-sim_corrected_X_80[f][t]),2)/(std_X_80[f][t]*std_X_80[f][t])
            X2_X_115=np.power((data_X_115[f][t]-sim_corrected_X_115[f][t]),2)/(std_X_115[f][t]*std_X_115[f][t])
            X2_Y_50=np.power((data_Y_50[f][t]-sim_corrected_Y_50[f][t]),2)/(std_Y_50[f][t]*std_Y_50[f][t])
            X2_Y_80=np.power((data_Y_80[f][t]-sim_corrected_Y_80[f][t]),2)/(std_Y_80[f][t]*std_Y_80[f][t])
            X2_Y_115=np.power((data_Y_115[f][t]-sim_corrected_Y_115[f][t]),2)/(std_Y_115[f][t]*std_Y_115[f][t])

            X2=X2+X2_X_50+X2_X_80+X2_X_115+X2_Y_50+X2_Y_80+X2_Y_115

    return 100*X2/(6*nF*nT)




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
    
    #print '_____________________________'
    #print '{0}   {1}   {2}   {3}   {4}'.format(theta_start,theta_stop, phi_start, phi_stop, temp)
    
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
        print(freq)
        pickfile = open(LFmap_dir+'/LFreduced_'+str(freq)+'.p','rb')
        XX,YY,ZZ,XX2,YY2,times_utc,times_LST,ZZ2=pickle.load(pickfile, encoding="latin1")
        print(jones_dir+'/jones_all_{0}.p')
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
        print(outfile)



def consolidate(con_dir,power_dir,data_dir,station):
    
    nTimes1=24
    nFreq=51
    nData=5
    frequencies=np.arange(30,80.5,1)
    power=np.zeros([nFreq,nTimes1,nData])
    
    for f in np.arange(nFreq):
        file=open(power_dir+'/integrated_power_'+str(f+30)+'.txt','rb')
        temp=np.genfromtxt(file)
    
        for t in np.arange(nTimes1):
            power[f][t][0]=temp[t][0]
            power[f][t][1]=temp[t][1]
            power[f][t][2]=temp[t][2]
            power[f][t][3]=temp[t][4]
            power[f][t][4]=temp[t][3]
    
    dh=0.25
    bin_edges=np.arange(0,24.1,dh)
    bins=np.arange(0.0,23.9,dh)
    nTimes=len(bins)

    
    int_sim_X=np.zeros([nFreq,len(bins)])
    int_sim_Y=np.zeros([nFreq,len(bins)])
    int_sim_X_total=np.zeros([len(bins)])
    int_sim_Y_total=np.zeros([len(bins)])

    for f in np.arange(nFreq):

        fX = interp1d(power[f].T[0], power[f].T[3],kind='cubic',fill_value='extrapolate')
        fY = interp1d(power[f].T[0], power[f].T[4],kind='cubic',fill_value='extrapolate')

        int_sim_X[f]=fX(bins)
        int_sim_Y[f]=fY(bins)




    holdX=int_sim_X
    holdY=int_sim_Y
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

        infile=open('/vol/astro3/lofar/sim/kmulrey/calibration/TBBdata/'+station+'_noise_power_'+cable_lengths[c]+'.p','rb')
        tbbInfo=pickle.load(infile, encoding="latin1")
        infile.close()

        avg_power_X=tbbInfo['avg_power_X_'+cable_lengths[c]].T
        std_power_X=tbbInfo['std_power_X_'+cable_lengths[c]].T
        avg_power_Y=tbbInfo['avg_power_Y_'+cable_lengths[c]].T
        std_power_Y=tbbInfo['std_power_Y_'+cable_lengths[c]].T


        pickfile = open(con_dir+'/power_all_'+cable_lengths[c]+'m_'+station+'.p','wb')

        pickle.dump((bins,int_sim_X,int_sim_Y,avg_power_X,std_power_X,avg_power_Y,std_power_Y),pickfile)

        pickfile.close()





def do_fit(consolidate_dir,fit_data_dir,fit_dir,name,station):
    nFreq=51
    nTimes=96
    frequencies=np.arange(30,80.5,1)
    times=np.arange(0,24.0,0.5)
    
    
    #file50='../fit_data/power_all_50m.p'
    #file80='../fit_data/power_all_80m.p'
    #file115='../fit_data/power_all_115m.p'
    
    #file=open(file50,'rb')
    file=open(consolidate_dir+'/power_all_50m_'+station+'.p','rb')
    time_bins,sim_X,sim_Y,data_X_50,std_X_50,data_Y_50,std_Y_50=pickle.load(file, encoding="latin1")
    file.close()
    
    #file=open(file80,'rb')
    file=open(consolidate_dir+'/power_all_80m_'+station+'.p','rb')
    time_bins,sim_X,sim_Y,data_X_80,std_X_80,data_Y_80,std_Y_80=pickle.load(file, encoding="latin1")
    file.close()
    
    #file=open(file115,'rb')
    file=open(consolidate_dir+'/power_all_115m_'+station+'.p'','rb')
    time_bins,sim_X,sim_Y,data_X_115,std_X_115,data_Y_115,std_Y_115=pickle.load(file, encoding="latin1")
    file.close()
    
    
    
    sim_X_50=sim_X*337.0
    sim_X_80=sim_X*337.0
    sim_X_115=sim_X*337.0

    sim_Y_50=sim_Y*337.0
    sim_Y_80=sim_Y*337.0
    sim_Y_115=sim_Y*337.0
    
    
    cable_attenuation_50=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_50m.txt',usecols=1)
    cable_attenuation_80=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_80m.txt',usecols=1)
    cable_attenuation_115=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_115m.txt',usecols=1)
    RCU_gain=np.genfromtxt(fit_data_dir+'/RCU_gain_new_5.txt',usecols=0)
    jones_vals=np.genfromtxt(fit_data_dir+'/antenna_gain.txt')


    g=11.08423462
    c=3.0e-11
    a=5.0e-13
    s=9e7
    b=1.54646899e-02

    gain_curve=np.power(10,RCU_gain/10)
    
  
    
    
    
    res=minimize(e_ACGMB_allCables,[a,c,g,b],args=(data_X_50,std_X_50,sim_X_50,data_X_80,std_X_80,sim_X_80,data_X_115,std_X_115,sim_X_115,data_Y_50,std_Y_50,sim_Y_50,data_Y_80,std_Y_80,sim_Y_80,data_Y_115,std_Y_115,sim_Y_115,jones_vals,cable_attenuation_50,cable_attenuation_80,cable_attenuation_115,RCU_gain,s),method='Nelder-Mead', options={'disp': True})


    print('\n')
    print('___________________ done with fit _________________')
    print('\n')

    print(res['success'])
    print(res['fun'])
    pars=res['x']
    print(pars)

    a=pars[0]
    c=pars[1]
    g=pars[2]
    
    g115=pars[2]
    g80=pars[2]-1.5
    g50=pars[2]-2.75
    b=pars[3]*0.9

    print('a={0}'.format(a))
    print('c={0}'.format(c))
    print('g={0}'.format(g))
    print('d={0}'.format(b))
    print('s={0}'.format(s))


    A_X=np.zeros([nFreq])
    A_Y=np.zeros([nFreq])

    d=np.zeros([nFreq])

    for f in np.arange(nFreq):
        d[f]=b#m*(float(f))+b

    cable_atten=cable_attenuation_80
    rcu=np.power(10,(g80)/10)*gain_curve

    sim_corrX=np.zeros([nFreq,nTimes])
    sim_corrY=np.zeros([nFreq,nTimes])

    data_corrX=np.zeros([nFreq,nTimes])
    data_corrY=np.zeros([nFreq,nTimes])

    sim_to_dataX=np.zeros([nFreq,nTimes])
    sim_to_dataY=np.zeros([nFreq,nTimes])

    sim_to_data_rawX=np.zeros([nFreq,nTimes])
    sim_to_data_rawY=np.zeros([nFreq,nTimes])


    for f in np.arange(nFreq):
        for t in np.arange(nTimes):
            sim_corrX[f][t]=(sim_X_80[f][t]+a*jones_vals[f])
            data_corrX[f][t]=(((((data_X_80[f][t]-d[f])/s))/rcu[f])-c)*np.power(10.0,(cable_atten[f]/10.0))

            sim_corrY[f][t]=(sim_Y_80[f][t]+a*jones_vals[f])
            data_corrY[f][t]=(((((data_Y_80[f][t]-d[f])/s))/rcu[f])-c)*np.power(10.0,(cable_atten[f]/10.0))

    for f in np.arange(nFreq):
        A_X[f]=np.average(data_corrX[f])/np.average(sim_corrX[f])
        A_Y[f]=np.average(data_corrY[f])/np.average(sim_corrY[f])



    for f in np.arange(nFreq):
        for t in np.arange(nTimes):
            sim_to_dataX[f][t]=(((((sim_X_80[f][t]+a*jones_vals[f])*A_X[f])/np.power(10.0,(cable_atten[f]/10.0))+c)*rcu[f])*s+d[f])
            sim_to_dataY[f][t]=(((((sim_Y_80[f][t]+a*jones_vals[f])*A_Y[f])/np.power(10.0,(cable_atten[f]/10.0))+c)*rcu[f])*s+d[f])

    data=data_X_80
    std=std_X_80
    sim=sim_X_80
    cable_atten=cable_attenuation_80




    cal_X=np.sqrt(1/((A_X*(1/np.power(10.0,(cable_atten/10.0)))*rcu)*s))
    cal_Y=np.sqrt(1/((A_Y*(1/np.power(10.0,(cable_atten/10.0)))*rcu)*s))

    cal=(cal_Y+cal_X)/2.0

    cal_raw=np.sqrt(np.average(sim,axis=1)/np.average(data,axis=1))
    
    
    
    outputfile=fit_dir+'/fits_'+name+'.p'


    analysisinfo={'a':a,'c':c,'cal':cal,'g':g,'d':b,'sim_X':sim_X,'sim_Y':sim_Y,'sim_to_data_X':sim_to_dataX,'sim_to_data_Y':sim_to_dataY,'data_X':data_X_80,'data_Y':data_Y_80,'std_X':std_X_80,'std_Y':std_Y_80,'x2':res['fun']}
    fout=open(outputfile,"wb")
    pickle.dump(analysisinfo,fout)
    fout.close()
