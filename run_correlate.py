import numpy as np
import pickle
from optparse import OptionParser
import glob
import os
import re
import sys

station_info_dir='/vol/astro3/lofar/sim/kmulrey/calibration/final/compare/sims/make_sims/station_info/'

events=[105465463,  123314680,  127163030,  168608146,  183321178,  207028458,  212089170,72094854,  82321543,  92380604,
118956923,  126484310,  148673810,  160673071,  173797298,  193057293,  207593143,  45616753,81147431,  86122409,  94175691,
121029645,  127108374,  157129793,  167252541,  181699538,  198490827,  211084564,  48361669,81409140,  86129434,  95228015]


parser = OptionParser()
parser.add_option('-c', '--caltype', default = 'standard', help = 'type of calibration')

(options, args) = parser.parse_args()
caltype = str(options.caltype)


for e in np.arange(len(events)):
#for e in np.arange(1):

    os.chdir(station_info_dir)

    for file in glob.glob('*'+str(events[e])+'*'):
        station_file=file
    
    with open(station_file) as f:
        stations = [line.rstrip() for line in f]
        
    for s in np.arange(len(stations)):
        print(str(events[e]),stations[s])
        os.chdir('/vol/astro7/lofar/kmulrey/calibration/calibration/')
        os.system('python3 correlate.py -e {0} -s {1} -c {2}'.format(str(events[e]),stations[s],caltype))
