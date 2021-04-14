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
    
    
    
    
    
    
    time0=new_t*1e-9
    frequencies = np.fft.rfftfreq(len(new_t), time0[1]-time0[0]) # Ends at 5000 MHz as it should for tstep=0.1 ns
    fft_a=np.fft.rfft(np.nan_to_num(new_a))
    fft_b=np.fft.rfft(np.nan_to_num(new_b))
    
    
    

    ##fig = plt.figure(figsize=(10,5))
    #ax1 = fig.add_subplot(2,1,1)
    #ax2 = fig.add_subplot(2,2,3)
    #ax3 = fig.add_subplot(2,2,4)
    
    fig = plt.figure(figsize=(12,3))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,4,3)
    ax3 = fig.add_subplot(1,4,4)
    
    ax1.set_title('{0}'.format(caltype))
    
    ax1.plot(new_t, new_a, linewidth=3, alpha=0.7, color='blue', label='data '+pol)
    ax1.plot(new_t, new_b, linewidth=3, alpha=0.7, color='red', label='sim '+pol)
    ax1.legend(loc='lower left')


    
    ax2.plot(frequencies*1e-6,np.abs(fft_a),color='blue',linewidth=3, alpha=0.7,label='data')
    ax2.plot(frequencies*1e-6,np.abs(fft_b),color='red',linewidth=3, alpha=0.7,label='sim')


    ax3.plot(frequencies*1e-6,(np.unwrap(2 * np.angle(fft_a)) / 2),color='blue',linewidth=3, alpha=0.7,label='data')
    ax3.plot(frequencies*1e-6,(np.unwrap(2 * np.angle(fft_b)) / 2),color='red',linewidth=3, alpha=0.7,label='sim')


    ax2.set_yscale('log')

    ax1.set_xlim([100,350])

    ax2.set_xlim([15,90])
    ax2.set_ylim([1e-1,1e2])

    ax3.set_xlim([15,90])
    ax3.set_ylim([-12,10])

    ax1.legend(frameon=False,loc="upper left")
    ax2.legend(frameon=False,loc="upper left")
    ax3.legend(frameon=False,loc="lower left")

    ax1.set_ylabel('amplitude (V)')
    ax1.set_xlabel('time (ns)')

    ax2.set_ylabel('magnitude (V)')
    ax2.set_xlabel('freq (MHz)')

    ax3.set_ylabel('unwrapped phase (rad)')
    ax3.set_xlabel('freq (MHz)')


    # show the pretty plots
    plt.tight_layout()
    plt.savefig('plots/correlated/'+event+'_'+station+'_'+antenna+'_pol'+pol+'_'+caltype+'.png',dpi=300)

    #plt.show()
    plt.close()
    return np.max(crosscorr)



parser = OptionParser()

parser.add_option("-e", "--event", default = '81147431', help = "event ID")
parser.add_option("-s", "--station", default = 'CS002', help = "event ID")
parser.add_option("-c", "--cal", default = 'R700', help = "calibration type")

(options, args) = parser.parse_args()

event = options.event
station= options.station
caltype = options.cal




calfile='/vol/astro7/lofar/kmulrey/calibration/fits/fits_'+caltype+'_CS002.p'
antenna_response_dir='/vol/astro7/lofar/kmulrey/calibration/antenna_model/jones_'+caltype+'/'
antenna_response_std='/vol/astro7/lofar/kmulrey/calibration/antenna_model/jones_standard/'

lba_response=np.genfromtxt('../fit_data/complex_lba_response.txt')
data_dir='/vol/astro3/lofar/sim/kmulrey/frequency_analysis/collect_info/info/'

fin=open(calfile,"rb")
info=pickle.load(fin)
fin.close()
print(info.keys())

Calibration_curve_new=np.zeros(101)
Calibration_curve_new[30:81]=info['cal']

Calibration_curve=np.zeros(101)
Calibration_curve[29:82]=np.array([0, 1.37321451961e-05,1.39846332239e-05,1.48748993821e-05,1.54402170354e-05,1.60684568225e-05,1.66241942741e-05,1.67039066047e-05,1.74480931848e-05,1.80525736486e-05,1.87066855054e-05,1.88519099831e-05,1.99625051386e-05,2.01878566584e-05,2.11573680797e-05,2.15829455528e-05,2.20133824866e-05,2.23736319125e-05,2.24484419697e-05,2.37802483891e-05,2.40581543111e-05,2.42020383477e-05,2.45305869187e-05,2.49399905965e-05,2.63774023804e-05,2.70334253414e-05,2.78034857678e-05,3.07147991391e-05,3.40755705892e-05,3.67311849851e-05,3.89987440028e-05,3.72257913465e-05,3.54293510934e-05,3.35552370942e-05,2.96529815929e-05,2.79271252352e-05,2.8818544973e-05,2.92478843809e-05,2.98454768706e-05,3.07045462103e-05,3.07210553534e-05,3.16442871206e-05,3.2304638838e-05,3.33203882046e-05,3.46651060935e-05,3.55193137077e-05,3.73919275937e-05,3.97397037914e-05,4.30625048727e-05,4.74612081994e-05,5.02345866124e-05, 5.53621848304e-05,0])



file=open(data_dir+'info_'+event+'_'+station+'.dat','rb')
core_x, core_y , stname, positions, signals, power11, rms, noisepower, time, lora_x, lora_y, lora_dens, azimuth, zenith, lora_zenith, xyz_files=pickle.load(file, encoding="latin1")
file.close()

file2=open('/vol/astro3/lofar/vhecr/lora_triggered/results_with_abs_calibration/'+event+'/calibrated_pulse_block-'+event+'-'+station+'.npy','rb')
info=np.load(file2,encoding='bytes')
file2.close()
file3=open('/vol/astro3/lofar/vhecr/lora_triggered/results_with_abs_calibration/'+event+'/xyz_calibrated_pulse_block-'+event+'-'+station+'.npy','rb')
info_xyz=np.load(file3,encoding='bytes')
file3.close()

nAntData=len(info[::2])
dt=5.0e-9
time0=np.arange(0,len(info[0]))*dt
freq=np.fft.rfftfreq(len(info[0]),dt)

data_0pol=info[::2]
data_1pol=info[1::2]

data_0pol_fft=np.fft.rfft(data_0pol)
data_1pol_fft=np.fft.rfft(data_1pol)

Calibration_curve_interp = interp1d(np.linspace(0.e6,100e6,101), Calibration_curve, kind='linear')
Calibration_curve_interp = Calibration_curve_interp(freq)

Calibration_curve_new_interp = interp1d(np.linspace(0.e6,100e6,101), Calibration_curve_new, kind='linear')
Calibration_curve_new_interp = Calibration_curve_new_interp(freq)

data_0pol_fft_new=np.nan_to_num((Calibration_curve_new_interp/Calibration_curve_interp))*data_0pol_fft
data_1pol_fft_new=np.nan_to_num((Calibration_curve_new_interp/Calibration_curve_interp))*data_1pol_fft


data_0pol_new=np.fft.irfft(data_0pol_fft_new)
data_1pol_new=np.fft.irfft(data_1pol_fft_new)

window=100



sim_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final/compare/sims/corsika/'+event+'/'


list_file=glob.glob(sim_dir+'*.list')[0]
print(list_file)
RUNNR=list_file.split('SIM')[1].split('.list')[0]
print(RUNNR)
coreas_dir=sim_dir+'SIM'+RUNNR+'_coreas/'
print(coreas_dir)

file3 = open(list_file, 'r')
Lines = file3.readlines()

x_sim_pos=[]
y_sim_pos=[]
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if station in line:
        #print("Line{}: {}".format(count, line.strip()))
        x_sim_pos.append(float(line.strip().split('=')[1].split('760.0')[0].split(' ')[1])/100)
        y_sim_pos.append(float(line.strip().split('=')[1].split('760.0')[0].split(' ')[2])/100)
x_sim_pos=np.asarray(x_sim_pos)
y_sim_pos=np.asarray(y_sim_pos)

fitfile='/vol/astro3/lofar/sim/kmulrey/calibration/final/compare/sims/make_sims/CR_event_info.txt'
info=np.genfromtxt(fitfile)

event_number=info.T[0].astype(int)

index=event_number.tolist().index(int(event))

core_x_fit=info[index][2]
core_y_fit=info[index][3]

azimuth_fit=info[index][6]
zenith_fit=info[index][5]

print((np.rad2deg(azimuth_fit))%360,np.rad2deg(zenith_fit))

x_temp=x_sim_pos
y_temp=y_sim_pos


x_sim_pos2=-1.0*y_temp+core_x_fit#-12
y_sim_pos2=x_temp+core_y_fit#12



steerfile='/vol/astro3/lofar/sim/kmulrey/calibration/final/compare/sims/corsika/'+event+'/RUN'+RUNNR+'.inp'

file4 = open(steerfile, 'r')
Lines = file4.readlines()
for line in Lines:
    count += 1
    if 'THETAP' in line:
        sim_zenith=float(line.strip().split(' ')[1])
    if 'PHIP' in line:
        sim_azimuth=float(line.strip().split(' ')[1])


antenna_model_standard=np.zeros([100,4],dtype=complex)
antenna_model_new=np.zeros([100,4],dtype=complex)



for f in np.arange(0,100):
    frequency=str(f)
    file=open(antenna_response_std+'/jones_all_'+frequency+'.p','rb')
    info=pickle.load(file, encoding="latin1")
    file.close()

    antenna_model_standard[f][0]=info['jones_thetaX_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]
    antenna_model_standard[f][1]=info['jones_phiX_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]
    antenna_model_standard[f][2]=info['jones_thetaY_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]
    antenna_model_standard[f][3]=info['jones_phiY_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]
    
for f in np.arange(30,81):
    frequency=str(f)#str(f+30)
    file=open(antenna_response_dir+'/jones_all_'+frequency+'.p','rb')

    info=pickle.load(file, encoding="latin1")
    file.close()

    antenna_model_new[f][0]=info['jones_thetaX_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]
    antenna_model_new[f][1]=info['jones_phiX_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]
    antenna_model_new[f][2]=info['jones_thetaY_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]
    antenna_model_new[f][3]=info['jones_phiY_complex'][int((sim_azimuth+720)%360)][int(sim_zenith)]


sim_list=glob.glob(coreas_dir+'*{0}*'.format(station))
nantennas=len(sim_list)
print('{0} antennas'.format(nantennas))

rawcoreas=[]
XYZcoreas=[]
pol=[]
filteredsignal=[]
hilbertsignal=[]
rawfft=[]
antennafft=[]
onskypower=np.zeros([nantennas,2])
    
    
frequencies_50=1e6*np.arange(0,5002,1)
jm_new=np.zeros([len(frequencies_50),4],dtype=complex)
jm_new_lna=np.zeros([len(frequencies_50),4],dtype=complex)

jm_std=np.zeros([len(frequencies_50),4],dtype=complex)
lba_full=np.zeros([len(frequencies_50)],dtype=complex)

#jm_new[30:81]=antenna_model_new
jm_std[30:81]=antenna_model_standard[30:81]
jm_new[30:81]=antenna_model_new[30:81]
#jm_new[0:100]=antenna_model_new[0:100]


frequencies_lba=lba_response.T[0]
lba_full[0:len(lba_response)]=lba_response.T[1]+1j*lba_response.T[2]

jm_new_lna.T[0]=jm_new.T[0]*lba_full
jm_new_lna.T[1]=jm_new.T[1]*lba_full
jm_new_lna.T[2]=jm_new.T[2]*lba_full
jm_new_lna.T[3]=jm_new.T[3]*lba_full#


procesed_length=80
data_standard=np.zeros([nantennas,2,procesed_length])
data_new=np.zeros([nantennas,2,procesed_length])
sim_standard=np.zeros([nantennas,2,procesed_length])
sim_new=np.zeros([nantennas,2,procesed_length])
sim_new_lna=np.zeros([nantennas,2,procesed_length])

f_real_theta0_new = interp1d(frequencies_50, jm_new.T[0].real)
f_imag_theta0_new = interp1d(frequencies_50, jm_new.T[0].imag)
f_real_phi0_new = interp1d(frequencies_50, jm_new.T[1].real)
f_imag_phi0_new = interp1d(frequencies_50, jm_new.T[1].imag)
f_real_theta1_new = interp1d(frequencies_50, jm_new.T[2].real)
f_imag_theta1_new = interp1d(frequencies_50, jm_new.T[2].imag)
f_real_phi1_new = interp1d(frequencies_50, jm_new.T[3].real)
f_imag_phi1_new = interp1d(frequencies_50, jm_new.T[3].imag)
    
f_real_theta0_new_lna = interp1d(frequencies_50, jm_new_lna.T[0].real)
f_imag_theta0_new_lna = interp1d(frequencies_50, jm_new_lna.T[0].imag)
f_real_phi0_new_lna = interp1d(frequencies_50, jm_new_lna.T[1].real)
f_imag_phi0_new_lna = interp1d(frequencies_50, jm_new_lna.T[1].imag)
f_real_theta1_new_lna = interp1d(frequencies_50, jm_new_lna.T[2].real)
f_imag_theta1_new_lna = interp1d(frequencies_50, jm_new_lna.T[2].imag)
f_real_phi1_new_lna = interp1d(frequencies_50, jm_new_lna.T[3].real)
f_imag_phi1_new_lna = interp1d(frequencies_50, jm_new_lna.T[3].imag)
    
f_real_theta0_std = interp1d(frequencies_50, jm_std.T[0].real)
f_imag_theta0_std = interp1d(frequencies_50, jm_std.T[0].imag)
f_real_phi0_std = interp1d(frequencies_50, jm_std.T[1].real)
f_imag_phi0_std = interp1d(frequencies_50, jm_std.T[1].imag)
f_real_theta1_std = interp1d(frequencies_50, jm_std.T[2].real)
f_imag_theta1_std = interp1d(frequencies_50, jm_std.T[2].imag)
f_real_phi1_std = interp1d(frequencies_50, jm_std.T[3].real)
f_imag_phi1_std = interp1d(frequencies_50, jm_std.T[3].imag)











