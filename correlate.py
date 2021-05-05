import numpy as np
import pickle
from optparse import OptionParser
import glob
import os
import re
import sys
sys.path.insert(1, '/vol/astro7/lofar/kmulrey/calibration/calibration/')
import trace_functions as trace



parser = OptionParser()

parser.add_option('-e', '--event', default = '105465463', help = 'event number')
parser.add_option('-s', '--station', default = 'CS002', help = 'station')
parser.add_option('-c', '--caltype', default = 'standard', help = 'type of calibration')

(options, args) = parser.parse_args()

event = str(options.event)
station = str(options.station)
caltype = str(options.caltype)


station_info_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final/compare/sims/make_sims/station_info/'
antenna_response_std='/vol/astro7/lofar/kmulrey/calibration/antenna_model/jones_standard/'
lba_response=np.genfromtxt('/vol/astro7/lofar/kmulrey/calibration/fit_data/complex_lba_response.txt')
data_dir='/vol/astro3/lofar/sim/kmulrey/frequency_analysis/collect_info/info/'
calfile='/vol/astro7/lofar/kmulrey/calibration/fits/fits_'+caltype+'_CS002.p'
antenna_response_dir='/vol/astro7/lofar/kmulrey/calibration/antenna_model/jones_'+caltype+'/'
output_dir='/vol/astro7/lofar/kmulrey/calibration/compare/correlations/'


    
file=open(data_dir+'info_'+event+'_'+station+'.dat','rb')
core_x, core_y , stname, positions, signals, power11, rms, noisepower, time, lora_x, lora_y, lora_dens, azimuth, zenith, lora_zenith, xyz_files=pickle.load(file, encoding="latin1")
file.close()
data_positions=positions[::2].T[0:2]

cal_old, cal_new=trace.return_cal(caltype)

file=open(data_dir+'info_'+event+'_'+station+'.dat','rb')
core_x, core_y , stname, positions, signals, power11, rms, noisepower, time, lora_x, lora_y, lora_dens, azimuth, zenith, lora_zenith, xyz_files=pickle.load(file, encoding="latin1")

time_data,data_all=trace.get_data(event,station,cal_old, cal_new)
data=data_all[::2]

time_sim, sim, sim_positions,sim_azimuth,sim_zenith=trace.get_simulation(event,station, caltype)
data_reduced=trace.reduce_data(data,len(time_sim))

sim=np.swapaxes(sim,1,0)
data_reduced=np.swapaxes(data_reduced,1,0)

if len(sim)==len(data_reduced):
    time_corr_0, data_corr_0, sim_corr_0, correlation_value_0=trace.run_correlation(data_reduced[0],sim[0])
    time_corr_1, data_corr_1, sim_corr_1, correlation_value_1= trace.run_correlation(data_reduced[1],sim[1])


    data_corr=np.swapaxes(np.stack((data_corr_0,data_corr_1)),1,0)
    sim_corr=np.swapaxes(np.stack((sim_corr_0,sim_corr_1)),1,0)
    time_corr=np.swapaxes(np.stack((time_corr_0,time_corr_1)),1,0)
    correlation_value=np.swapaxes(np.stack((correlation_value_0,correlation_value_1)),1,0)
    #signs=np.swapaxes(np.stack((sign_0,sign_1)),1,0)
    #flipped=np.asarray([flip_0,flip_1])

    pearson_values, chi2=trace.find_pearsonnr(data_corr,sim_corr)


    info={'time_corr':time_corr,'data_corr':data_corr,'sim_corr':sim_corr,'correlation_value':correlation_value,'sim_positions':sim_positions,'pearson_values':pearson_values,'chi2':chi2}

    outfile=open(output_dir+event+'_'+station+'_'+caltype+'.p','wb')
    pickle.dump(info,outfile)
    outfile.close()

