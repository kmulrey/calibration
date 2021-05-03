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

parser.add_option('-e', '--event', default = '105465463', help = 'event number)

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


os.chdir(station_info_dir)
for file in glob.glob('*'+str(event)+'*'):
    station_file=file
    
with open(station_file) as f:
    lines = [line.rstrip() for line in f]
    
file=open(data_dir+'info_'+event+'_'+station+'.dat','rb')
core_x, core_y , stname, positions, signals, power11, rms, noisepower, time, lora_x, lora_y, lora_dens, azimuth, zenith, lora_zenith, xyz_files=pickle.load(file, encoding="latin1")
file.close()
data_positions=positions[::2].T[0:2]

cal_old, cal_new=trace.return_cal(caltype)

file=open(data_dir+'info_'+event+'_'+station+'.dat','rb')
core_x, core_y , stname, positions, signals, power11, rms, noisepower, time, lora_x, lora_y, lora_dens, azimuth, zenith, lora_zenith, xyz_files=pickle.load(file, encoding="latin1")

time_data,data_all=trace.get_data(event,station,cal_old, cal_new)
data=data_all[::2]

time_sim, sim, sim_positions=trace.get_simulation(event,station, caltype)
data_reduced=trace.reduce_data(data,len(time_sim))

time_corr, data_corr, sim_corr, correlation_value=trace.run_correlation(data_reduced,sim)

info={'time_corr':time_corr,'data_corr':data_corr,'sim_corr':sim_corr,'correlation_value':correlation_value,'sim_positions':sim_positions}

outfile=open(output_dir+event+'_'+station+'_'+caltype+'.p','wb')
pickle.dump(info,outfile)
outfile.close()
