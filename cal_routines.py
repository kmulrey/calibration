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

def e_ACGMB_single(pars,data,std,sim,jones,cable_attenuation,cable,RCU_gain,s):


    nF=len(data)
    nT=len(data[0])
    a=pars[0]
    c=pars[1]
    g=pars[2]

    if cable==50:
        gC=np.power(10,(g-2.75)/10)
    if cable==80:
        gC=np.power(10,(g-1.5)/10)
    if cable==115:
        gC=np.power(10,(g/10))


    b=pars[3]

    d=np.zeros([nFreq])

    for f in np.arange(nFreq):
        d[f]=b


    gain_curve=np.power(10,RCU_gain/10)

    rcu=g*gain_curve
    rcuC=gC*gain_curve


    A=np.zeros([nF])


    data_corr=np.zeros([nF,nT])
    sim_corr=np.zeros([nF,nT])
    sim_corrected=np.zeros([nF,nT])



    for f in np.arange(nF):
        for t in np.arange(nT):
            sim_corr[f][t]=(sim[f][t]+a*jones[f])

            data_corr[f][t]=((data[f][t]-d[f])/(rcuC[f]*s)-c)*np.power(10.0,(cable_attenuation[f]/10.0))



    for f in np.arange(nF):
        A[f]=np.average(data_corr[f])/np.average(sim_corr[f])


    for f in np.arange(nF):
        for t in np.arange(nT):

            sim_corrected[f][t]=((((sim[f][t]+a*jones[f])*A[f])/np.power(10.0,(cable_attenuation[f]/10.0)))+c)*rcuC[f]*s+d[f]

    X2=0

    for f in np.arange(nF):
        for t in np.arange(nT):


            X2=X2+np.power((data[f][t]-sim_corrected[f][t]),2)/(std[f][t]*std[f][t])


    return 100*X2/(nF*nT)


# find average antenna model for each direction, save a file for each frequency
# antenna model is for a specific antenna set, here from CS002

def average_model(jones_dir):
    array_ind_outer=np.arange(576,(576+96))[::2]   # indices for CSOO2 outer

    for f in np.arange(30,81):
        jones_thetaX_total=np.zeros([361,91])
        jones_thetaY_total=np.zeros([361,91])
        jones_phiX_total=np.zeros([361,91])
        jones_phiY_total=np.zeros([361,91])

        jones_thetaX_total_real=np.zeros([361,91])
        jones_thetaY_total_real=np.zeros([361,91])
        jones_phiX_total_real=np.zeros([361,91])
        jones_phiY_total_real=np.zeros([361,91])

        jones_thetaX_total_im=np.zeros([361,91])
        jones_thetaY_total_im=np.zeros([361,91])
        jones_phiX_total_im=np.zeros([361,91])
        jones_phiY_total_im=np.zeros([361,91])

        jones_thetaX_complex=np.zeros([361,91],dtype=complex)
        jones_thetaY_complex=np.zeros([361,91],dtype=complex)
        jones_phiX_complex=np.zeros([361,91],dtype=complex)
        jones_phiY_complex=np.zeros([361,91],dtype=complex)

        #freq=str(f+30)
        freq=str(f)

        count=0

        for i in np.arange(len(array_ind_outer)):

            try:
                ant_id=array_ind_outer[i]
                file=open(jones_dir+'/jones_all_'+freq+'_antenna_'+str(ant_id)+'.p','rb')
                info=pickle.load(file, encoding="latin1")
                file.close()
                jones_aartfaac=info['jones_aartfaac']

                for a in np.arange(361):
                    for z in np.arange(91):
                        jones_thetaX_total[a][z]=jones_thetaX_total[a][z]+np.abs(jones_aartfaac[z][a][0])
                        jones_thetaY_total[a][z]=jones_thetaY_total[a][z]+np.abs(jones_aartfaac[z][a][2])
                        jones_phiX_total[a][z]=jones_phiX_total[a][z]+np.abs(jones_aartfaac[z][a][1])
                        jones_phiY_total[a][z]=jones_phiY_total[a][z]+np.abs(jones_aartfaac[z][a][3])



                        jones_thetaX_total_real[a][z]=jones_aartfaac[z][a][0].real+jones_thetaX_total_real[a][z]
                        jones_phiX_total_real[a][z]=jones_aartfaac[z][a][2].real+jones_phiX_total_real[a][z]
                        jones_thetaY_total_real[a][z]=jones_aartfaac[z][a][1].real+jones_thetaY_total_real[a][z]
                        jones_phiY_total_real[a][z]=jones_aartfaac[z][a][3].real+jones_phiY_total_real[a][z]

                        jones_thetaX_total_im[a][z]=jones_aartfaac[z][a][0].imag+jones_thetaX_total_im[a][z]
                        jones_phiX_total_im[a][z]=jones_aartfaac[z][a][2].imag+jones_phiX_total_im[a][z]
                        jones_thetaY_total_im[a][z]=jones_aartfaac[z][a][1].imag+jones_thetaY_total_im[a][z]
                        jones_phiY_total_im[a][z]=jones_aartfaac[z][a][3].imag+jones_phiY_total_im[a][z]


                count=count+1
            except:
                print('can\'t find '+freq+'   '+str(ant_id))


        jones_thetaX_total=jones_thetaX_total/count
        jones_thetaY_total=jones_thetaY_total/count
        jones_phiX_total=jones_phiX_total/count
        jones_phiY_total=jones_phiY_total/count

        jones_thetaX_real=jones_thetaX_total_real/count
        jones_thetaY_real=jones_thetaY_total_real/count
        jones_phiX_real=jones_phiX_total_real/count
        jones_phiY_real=jones_phiY_total_real/count

        jones_thetaX_im=jones_thetaX_total_im/count
        jones_thetaY_im=jones_thetaY_total_im/count
        jones_phiX_im=jones_phiX_total_im/count
        jones_phiY_im=jones_phiY_total_im/count

        for a in np.arange(361):
            for z in np.arange(91):
                jones_thetaX_complex[a][z]=jones_thetaX_real[a][z]+jones_thetaX_im[a][z]*1j
                jones_thetaY_complex[a][z]=jones_thetaY_real[a][z]+jones_thetaY_im[a][z]*1j
                jones_phiX_complex[a][z]=jones_phiX_real[a][z]+jones_phiX_im[a][z]*1j
                jones_phiY_complex[a][z]=jones_phiY_real[a][z]+jones_phiY_im[a][z]*1j



        info={'jones_thetaX':jones_thetaX_total,'jones_thetaY':jones_thetaY_total,'jones_phiX':jones_phiX_total,'jones_phiY':jones_phiY_total,'jones_thetaX_complex':jones_thetaX_complex,'jones_thetaY_complex':jones_thetaY_complex,'jones_phiX_complex':jones_phiX_complex,'jones_phiY_complex':jones_phiY_complex}
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



# find modeled sky power.  Add averaged antenna model, and integrate over the whole sky.
# Save file with times, power
def find_simulated_power(jones_dir, power_dir):

    for f in np.arange(51):
        freq=str(f+30)
        #print(freq)
        pickfile = open(LFmap_dir+'/LFreduced_'+str(freq)+'.p','rb')
        XX,YY,ZZ,XX2,YY2,times_utc,times_LST,ZZ2=pickle.load(pickfile, encoding="latin1")
        #print(jones_dir+'/jones_all_{0}.p')
        JJ=np.zeros([91,361,4],dtype='complex')


        pickfile = open(jones_dir+'/jones_all_{0}.p'.format(freq),'rb')
        pickfile.seek(0)
        info=pickle.load(pickfile)
        pickfile.close()
        jones_thetaX=info['jones_thetaX']
        jones_thetaY=info['jones_thetaY']
        jones_phiX=info['jones_phiX']
        jones_phiY=info['jones_phiY']

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
        #print(power_dir+'/integrated_power_'+str(freq)+'.txt')
        for i in np.arange(len(times_LST)):
            outfile.write('{0}  {1}  {2}  {3}  {4} \n'.format(times_sorted[i],times_sorted_utc[i],power_sorted[i],power_sorted_X[i],power_sorted_Y[i]))
        outfile.close()
        #print(outfile)

def find_simulated_power_single(jones_dir, power_dir, antenna_no):

    for f in np.arange(51):
        freq=str(f+30)
        print(freq)
        pickfile = open(LFmap_dir+'/LFreduced_'+str(freq)+'.p','rb')
        XX,YY,ZZ,XX2,YY2,times_utc,times_LST,ZZ2=pickle.load(pickfile, encoding="latin1")
        #print(jones_dir+'/jones_all_{0}.p')
        JJ=np.zeros([91,361,4],dtype='complex')

        file=open(jones_dir+'/jones_all_'+freq+'_antenna_'+str(antenna_no)+'.p','rb')
        info=pickle.load(file, encoding="latin1")
        file.close()
        jones_aartfaac=info['jones_aartfaac']
        #jones_thetaX=np.abs(jones_aartfaac[z][a][0])
        #jones_thetaY=np.abs(jones_aartfaac[z][a][2])
        #jones_phiX=np.abs(jones_aartfaac[z][a][1])
        #jones_phiY=np.abs(jones_aartfaac[z][a][3]0

        for th in np.arange(90):
            for az in np.arange(360):
                phi=az
                i_az=az+180
                if az>180:
                    phi=az-360
                    i_az=az-180

                JJ[90-th][i_az][0]=jones_aartfaac[th][i_az][0]
                JJ[90-th][i_az][1]=jones_aartfaac[th][i_az][2]
                JJ[90-th][i_az][2]=jones_aartfaac[th][i_az][1]
                JJ[90-th][i_az][3]=jones_aartfaac[th][i_az][3]





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


        outfile=open(power_dir+'/integrated_power_'+str(freq)+'_antenna_'+str(antenna_no)+'.txt','w')
        #print(power_dir+'/integrated_power_'+str(freq)+'.txt')
        for i in np.arange(len(times_LST)):
            outfile.write('{0}  {1}  {2}  {3}  {4} \n'.format(times_sorted[i],times_sorted_utc[i],power_sorted[i],power_sorted_X[i],power_sorted_Y[i]))
        outfile.close()
        #print(outfile)



def consolidate(con_dir,power_dir,data_dir,station,ant_id):

    nTimes1=24
    nFreq=51
    nData=5
    frequencies=np.arange(30,80.5,1)
    power=np.zeros([nFreq,nTimes1,nData])

    for f in np.arange(nFreq):
        file=open(power_dir+'/integrated_power_'+str(f+30)+'.txt','rb')
        #file=open(power_dir+'/integrated_power_'+str(freq)+'_'+str(ant_id)+'.txt','w')
        temp=np.genfromtxt(file)

        for t in np.arange(nTimes1):
            power[f][t][0]=temp[t][0]
            power[f][t][1]=temp[t][1]
            power[f][t][2]=temp[t][2]
            power[f][t][3]=temp[t][4]
            power[f][t][4]=temp[t][3]   ## <---- this is where sim power is swapped!

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


        #pickfile = open(con_dir+'/power_all_'+cable_lengths[c]+'m_'+station+'_'+str(ant_id)+'.p','wb')
        pickfile = open(con_dir+'/power_all_'+cable_lengths[c]+'m_'+station+'.p','wb')

        pickle.dump((bins,int_sim_X,int_sim_Y,avg_power_X,std_power_X,avg_power_Y,std_power_Y),pickfile)

        pickfile.close()




def consolidate_single(con_dir,power_dir,data_dir,station,antenna_no):

    nTimes1=24
    nFreq=51
    nData=5
    frequencies=np.arange(30,80.5,1)
    power=np.zeros([nFreq,nTimes1,nData])

    for f in np.arange(nFreq):
        file=open(power_dir+'/integrated_power_'+str(f+30)+'_antenna_'+str(antenna_no)+'.txt','rb')
        #file=open(power_dir+'/integrated_power_'+str(freq)+'_'+str(ant_id)+'.txt','w')
        temp=np.genfromtxt(file)

        for t in np.arange(nTimes1):
            power[f][t][0]=temp[t][0]
            power[f][t][1]=temp[t][1]
            power[f][t][2]=temp[t][2]
            power[f][t][3]=temp[t][4]   ## <---- this is where sim power is swapped!
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

    #fig = plt.figure()
    #ax1 = fig.add_subplot(1,1,1)
    #ax1.plot(int_sim_X[40])
    #plt.show()


    ind_info=np.genfromtxt('/vol/astro7/lofar/kmulrey/calibration/corresponding_aartfaac_lofar_antennas_CS002.txt')
    aart_ind=ind_info.T[0]
    lofar_ind=ind_info.T[1]
    lofar_ant=lofar_ind[aart_ind==antenna_no]

    print('----> lofar antenna ',lofar_ant)




    file='/vol/astro3/lofar/sim/kmulrey/calibration/TBBdata/CS002_noise_power_antennas.p'
    file=open(file,'rb')
    info=pickle.load(file, encoding="latin1")

    #time_bins,sim_X_1,sim_Y_1,data_X_50_1,std_X_50_1,data_Y_50_1,std_Y_50_1=pickle.load(file, encoding="latin1")
    file.close()

    std_power_XY=info['std_power_XY']
    avg_power_XY=info['avg_power_XY']
    count_XY=info['count_XY']




    avg_power_X_raw=avg_power_XY[int(lofar_ant)].T
    std_power_X_raw=avg_power_XY[int(lofar_ant)].T
    avg_power_Y_raw=avg_power_XY[int(lofar_ant+1)].T
    std_power_Y_raw=avg_power_XY[int(lofar_ant+1)].T






    avg_power_X=np.zeros([nFreq,len(bins)])
    avg_power_Y=np.zeros([nFreq,len(bins)])
    std_power_X=np.zeros([nFreq,len(bins)])
    std_power_Y=np.zeros([nFreq,len(bins)])
    times_data=np.arange(0.0,23.9,.5)


    for f in np.arange(nFreq):

        fXdata = interp1d(times_data, avg_power_X_raw[f],kind='cubic',fill_value='extrapolate')
        fYdata = interp1d(times_data, avg_power_Y_raw[f],kind='cubic',fill_value='extrapolate')
        fXstd = interp1d(times_data, std_power_X_raw[f],kind='cubic',fill_value='extrapolate')
        fYstd = interp1d(times_data, std_power_Y_raw[f],kind='cubic',fill_value='extrapolate')

        avg_power_X[f]=fXdata(bins)
        avg_power_Y[f]=fYdata(bins)
        std_power_X[f]=fXstd(bins)
        std_power_Y[f]=fYstd(bins)


    print('----> int sim shape ',int_sim_X.shape)
    print('----> avg pow shape ',avg_power_X.shape)
    print('----> avg std shape ',std_power_X.shape)






    cable_info=np.genfromtxt('/vol/astro3/lofar/sim/kmulrey/calibration/TBBdata/cable_info/CS002_cables.txt')

    cables_X=cable_info[int(lofar_ant)]
    cables_Y=cable_info[int(lofar_ant+1)]

    pickfile = open(con_dir+'/power_'+station+'_antenna_'+str(antenna_no)+'.p','wb')

    pickle.dump((bins,int_sim_X,int_sim_Y,avg_power_X,std_power_X,avg_power_Y,std_power_Y,cables_X,cables_Y),pickfile)

    pickfile.close()














def do_fit(consolidate_dir,fit_data_dir,fit_dir,name,station,ant_id):
    nFreq=51
    nTimes=96
    frequencies=np.arange(30,80.5,1)
    times=np.arange(0,24.0,0.5)
    print(consolidate_dir+'/power_all_50m_'+station+'.p')
    print(consolidate_dir+'/power_all_80m_'+station+'.p')
    print(consolidate_dir+'/power_all_115m_'+station+'.p')

    file=open(consolidate_dir+'/power_all_50m_'+station+'.p','rb')
    time_bins,sim_X,sim_Y,data_X_50,std_X_50,data_Y_50,std_Y_50=pickle.load(file, encoding="latin1")
    file.close()

    file=open(consolidate_dir+'/power_all_80m_'+station+'.p','rb')
    time_bins,sim_X,sim_Y,data_X_80,std_X_80,data_Y_80,std_Y_80=pickle.load(file, encoding="latin1")
    file.close()

    file=open(consolidate_dir+'/power_all_115m_'+station+'.p','rb')
    time_bins,sim_X,sim_Y,data_X_115,std_X_115,data_Y_115,std_Y_115=pickle.load(file, encoding="latin1")
    file.close()


    # include impedence of free space
    sim_X_50=sim_X*337.0#*121#*2e4   # the 121 is here to account for LNA gain- check if this needs to be here for "standard"
    sim_X_80=sim_X*337.0#*121#*2e4
    sim_X_115=sim_X*337.0#*121#*2e4

    sim_Y_50=sim_Y*337.0#*121#*2e4
    sim_Y_80=sim_Y*337.0#*121#*2e4
    sim_Y_115=sim_Y*337.0#*121#*2e4


    cable_attenuation_50=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_50m.txt',usecols=1)
    cable_attenuation_80=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_80m.txt',usecols=1)
    cable_attenuation_115=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_115m.txt',usecols=1)
    RCU_gain=np.genfromtxt(fit_data_dir+'/RCU_gain_new_5.txt',usecols=0)
    jones_vals=np.genfromtxt(fit_data_dir+'/antenna_gain.txt')
    # ----> this needs to be updated with antenna again for AARTFAAC!!!
    #jones_vals=121*np.ones([51])


    g=11.08423462
    c=3.0e-11
    a=5.0e-15
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



    outputfile=fit_dir+'/fits_'+name+'_'+station+'.p'


    analysisinfo={'a':a,'c':c,'cal':cal,'g':g,'d':b,'sim_X':sim_X,'sim_Y':sim_Y,'sim_to_data_X':sim_to_dataX,'sim_to_data_Y':sim_to_dataY,'data_X':data_X_80,'data_Y':data_Y_80,'std_X':std_X_80,'std_Y':std_Y_80,'x2':res['fun'],'A_X':A_X,'A_Y':A_Y}
    fout=open(outputfile,"wb")
    pickle.dump(analysisinfo,fout)
    fout.close()


def do_fit_single(con_dir,fit_data_dir,fit_dir,name,station,ant_id,pol):
    nFreq=51
    nTimes=96
    frequencies=np.arange(30,80.5,1)
    times=np.arange(0,24.0,0.5)



    print(con_dir+'/power_'+station+'_antenna_'+str(ant_id)+'.p')



    file=open(con_dir+'/power_'+station+'_antenna_'+str(ant_id)+'.p','rb')
    time_bins,sim_X,sim_Y,data_X,std_X,data_Y,std_Y,cables_X,cables_Y=pickle.load(file, encoding="latin1")
    file.close()


    if pol=='X':
        sim=sim_X*337.0*121#*2e4
        data=data_X
        std=std_X
        cables=int(cables_X)
    if pol=='Y':
        sim=sim_Y*337.0*121#*2e4
        data=data_Y
        std=std_Y
        cables=int(cables_Y)


    cable_attenuation=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_'+str(int(cables))+'m.txt',usecols=1)
    #cable_attenuation_Y=np.genfromtxt(fit_data_dir+'/attenuation/attenuation_coax9_'+str(int(cables_Y))+'m.txt',usecols=1)
    RCU_gain=np.genfromtxt(fit_data_dir+'/RCU_gain_new_5.txt',usecols=0)
    jones_vals=np.genfromtxt(fit_data_dir+'/antenna_gain.txt')
    # jone_vals here is the actual antenna gain.  This is actually included in the antenna model, but it is normalized out, so that gain can be applied to the antenna noise.


    g=21.08423462
    c=3.0e-11
    a=5.0e-15
    s=9e7
    b=1.54646899e-02

    gain_curve=np.power(10,RCU_gain/10)

    #fig = plt.figure()
    ##ax1 = fig.add_subplot(1,2,1)
    #ax2 = fig.add_subplot(1,2,2)

    #ax1.plot(int_sim_X[20])
    #print(data_X[20].shape)
    #ax1.plot(data_Y[20],'.')
    ##ax2.plot(sim_Y[20])
    #plt.show()

    res=minimize(e_ACGMB_single,[a,c,g,b],args=(data,std,sim,jones_vals,cable_attenuation,int(cables),RCU_gain,s),method='Nelder-Mead', options={'disp': True})


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


    if int(cables)==50:
        gC=np.power(10,(g-2.75)/10)
    if int(cables)==80:
        gC=np.power(10,(g-1.5)/10)
    if int(cables)==115:
        gC=np.power(10,(g/10))



    b=pars[3]*0.9

    print('a={0}'.format(a))
    print('c={0}'.format(c))
    print('g={0}'.format(g))
    print('d={0}'.format(b))
    print('s={0}'.format(s))


    A=np.zeros([nFreq])

    d=np.zeros([nFreq])

    for f in np.arange(nFreq):
        d[f]=b#m*(float(f))+b



    rcuC=gC*gain_curve


    sim_corr=np.zeros([nFreq,nTimes])

    data_corr=np.zeros([nFreq,nTimes])

    sim_to_data=np.zeros([nFreq,nTimes])

    sim_to_data_raw=np.zeros([nFreq,nTimes])


    for f in np.arange(nFreq):
        for t in np.arange(nTimes):
            sim_corr[f][t]=(sim[f][t]+a*jones_vals[f])
            data_corr[f][t]=(((((data[f][t]-d[f])/s))/rcuC[f])-c)*np.power(10.0,(cable_attenuation[f]/10.0))




    for f in np.arange(nFreq):
        A[f]=np.average(data_corr[f])/np.average(sim_corr[f])


    for f in np.arange(nFreq):
        for t in np.arange(nTimes):
            sim_to_data[f][t]=(((((sim[f][t]+a*jones_vals[f])*A[f])/np.power(10.0,(cable_attenuation[f]/10.0))+c)*rcuC[f])*s+d[f])



    cal_total=np.sqrt(1/((A*(1/np.power(10.0,(cable_attenuation/10.0)))*rcuC)*s))


    cal_raw=np.sqrt(np.average(sim,axis=1)/np.average(data,axis=1))



    outputfile=fit_dir+'/fits_'+name+'_'+station+'_antenna_'+str(ant_id)+'_'+pol+'.p'
    print(outputfile)

    analysisinfo={'a':a,'c':c,'cal':cal_total,'g':g,'d':b,'sim':sim,'sim_to_data':sim_to_data,'data':data,'std':std,'x2':res['fun'],'A':A}
    fout=open(outputfile,"wb")
    pickle.dump(analysisinfo,fout)
    fout.close()
