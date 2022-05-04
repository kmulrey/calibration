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
import AARTFAAC_model as aa




parser = OptionParser()

parser.add_option("-a", "--antenna", default = "1", help = "antenna ID")
(options, args) = parser.parse_args()

AART_ant_id = int(options.antenna)



AARTFAAC_model = aa.aartfaac_LBA_model()
#AART_ant_id = 2
antenna_XYZ = np.array([ AARTFAAC_model.AART_ant_positions[0, AART_ant_id], AARTFAAC_model.AART_ant_positions[1, AART_ant_id], 0.0 ])
antenna_model = AARTFAAC_model.get_antenna_model( antenna_XYZ=antenna_XYZ )

for f in np.arange(30,81):
    print(f)
    JJ=np.zeros([91,361,4],dtype='complex')
    JJ_aartfaac=np.zeros([91,361,4],dtype='complex')

    #CR_jones_file='/vol/astro3/lofar/sim/kmulrey/calibration/final/antenna_model/jones/jones_matrix_'+str(f)+'.p'
    #jonesfile=open(CR_jones_file,'rb')
    #jonesfile.seek(0)
    #jones_matrix = pickle.load(jonesfile, encoding='latin1')

    for el in np.arange(90):
        for az in np.arange(360):
            theta=90-el
            phi=az
            i_az=az+180
            if az>180:
                phi=az-360
                i_az=az-180

            Jones_matrices = antenna_model.Jones_Matrices( [float(f)*1e6], zenith=theta, azimuth=i_az )

            #JJ[el][i_az][0]=jones_matrix[az][el][0]
            #JJ[el][i_az][1]=jones_matrix[az][el][1]
            #JJ[el][i_az][2]=jones_matrix[az][el][2]
            #JJ[el][i_az][3]=jones_matrix[az][el][3]
            JJ_aartfaac[90-el][i_az][0]=Jones_matrices[0][0][0]
            JJ_aartfaac[90-el][i_az][1]=Jones_matrices[0][0][1]
            JJ_aartfaac[90-el][i_az][2]=Jones_matrices[0][1][0]
            JJ_aartfaac[90-el][i_az][3]=Jones_matrices[0][1][1]

    #info={'jones_aartfaac':JJ_aartfaac,'jones_cr':JJ,'antenna_XYZ':antenna_XYZ,'AART_ant_id':AART_ant_id}
    info={'jones_aartfaac':JJ_aartfaac,'antenna_XYZ':antenna_XYZ,'AART_ant_id':AART_ant_id}

    pickfile = open('jones_all_antennas/jones_all_'+str(int(f))+'_antenna_'+str(AART_ant_id)+'.p','wb')
    pickle.dump(info,pickfile)
    pickfile.close()
