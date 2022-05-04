#!/usr/bin/env python3


##external 
import numpy as np
from scipy.interpolate import RegularGridInterpolator, pchip_interpolate
from scipy.io import loadmat


C = 299792458.0
RTD = 180.0/3.1415926 ##radians to degrees
n_air = 1.000293
v_air = C/n_air


class aartfaac_LBA_model:
    
    def __init__(self, model_loc="./AARTFAAC_LBA_MODEL/", R=700, C=15e-12):
        self.model_loc = model_loc
        
        self.fine_model_loc = self.model_loc + "LBA_core_fine"
        self.course_model_loc = self.model_loc + "LBA_core"
        
        ## get base data, assume is same between models
        data_fine = loadmat( self.fine_model_loc , variable_names=['Positions', 'Theta', 'Phi', 'Freq', 'Zant', 'g_z'] ) ## Zant is antenna impedences, g_z is angular dependence
        self.AART_ant_positions = data_fine['Positions']
        self.AART_thetas = data_fine['Theta'][0]  ## zenithal angle in degrees
        self.AART_Phis = data_fine['Phi'][0]      ## azimuthal angle in degrees
        
        self.AART_fineFreqs = data_fine['Freq'][0]
        fine_gz = data_fine['g_z']
        fine_Zant =  data_fine['Zant']
        
        data_course = loadmat( self.course_model_loc , variable_names=['Freq', 'Zant', 'g_z'] )
        self.AART_courseFreqs = data_course['Freq'][0]
        course_gz = data_course['g_z']
        course_Zant =  data_course['Zant']
        
        
        
        
        
        ### figure out how to fold fine and course models together
        fineFreq_low = self.AART_fineFreqs[0]
        # fineFreq_high = self.AART_fineFreqs[-1]
        
        freqs = []
        course_range_1 = [0]
        fine_range = []
        course_range_2 = []
        
        ## first we search AART_courseFreqs to find highest frequency lower than fine-range, adding them to the frequency list
        for i in range(len(self.AART_courseFreqs)): 
            Fc = self.AART_courseFreqs[i]
            if Fc < fineFreq_low:
                freqs.append(Fc)
            else:
                break
        course_range_1.append(i)
        fine_range.append(i)
        
        ## add all fine frequencies
        freqs += list(self.AART_fineFreqs)
        fine_range.append(len(freqs))
        course_range_2.append(len(freqs))
        
        ## find first course frequency greater than fine frequencies
        for i in range(i, len(self.AART_courseFreqs)):
            if self.AART_courseFreqs[i] > freqs[-1]:
                break
            
        ## add rest of course frequencies to list
        freqs += list(self.AART_courseFreqs[i:]) 
        course_range_2.append(len(freqs))
        
        self.all_frequencies = np.array(freqs, dtype=np.double)
        self.course_frequencyRange_low = np.array(course_range_1)
        self.fine_frequencyRange = np.array(fine_range)
        self.course_frequencyRange_high = np.array(course_range_2)
        
        
        
        #### combine antenna impedences
        num_ants = len(fine_Zant)
        num_thetas = len(self.AART_thetas)
        num_phis = len(self.AART_Phis)
        num_freqs = len(self.all_frequencies)
        
        self.antenna_Z = np.empty([num_ants, num_ants, num_freqs ], dtype=np.complex)
        
        self.antenna_Z[:,:, :self.course_frequencyRange_low[1]] = course_Zant[:,:,  :self.course_frequencyRange_low[1]]
        self.antenna_Z[:,:, self.fine_frequencyRange[0]:self.fine_frequencyRange[1]] = fine_Zant[:,:,  :]
        self.antenna_Z[:,:, self.fine_frequencyRange[1]:] = course_Zant[:,:, self.fine_frequencyRange[1]-len(self.all_frequencies):]
        
        
        self.total_impedence = np.empty((num_ants, num_ants, num_freqs), dtype=np.complex)
        self.set_RC( R, C ) ## this sets self.total_impedence
        
        
        
        
        
        ### now we combine total G_z, which is voltage on antenna
        self.total_Gz = np.empty( (num_ants,2,num_thetas,num_phis,num_freqs), dtype=np.complex )
        
        # for ant_i in range(num_ants):
            # for pol_i in range(2):
        self.total_Gz[:,:, :,:,                             : self.course_frequencyRange_low[1]] = course_gz[:,:, :,:,    : self.course_frequencyRange_low[1]]
        self.total_Gz[:,:, :,:, self.fine_frequencyRange[0] : self.fine_frequencyRange[1]]       = fine_gz  [:,:, :,:,    : ]
        self.total_Gz[:,:, :,:, self.fine_frequencyRange[1] :]                                   = course_gz[:,:, :,:, self.fine_frequencyRange[1]-len(self.all_frequencies): ]
        
    
    
    def set_RC(self, R, C ):
        self.R = R
        self.C = C
        
        
        num_ants = len(self.antenna_Z)
        num_freqs = len(self.all_frequencies)
        
        
        tmp_mat = np.empty((num_ants, num_ants), dtype=np.complex)
        ZLMA_matrix =  np.zeros((num_ants, num_ants), dtype=np.complex)
        
        for Fi in range(num_freqs):
            # set LNA matrix
            LNA = R/(1+self.all_frequencies[Fi]*(2*np.pi*1j*R*C)) 
            np.fill_diagonal(ZLMA_matrix,  LNA)
            
            ## caclulate denominator bit
            tmp_mat[:,:] = self.antenna_Z[:,:,Fi]
            tmp_mat += ZLMA_matrix
            
            ## try sign difference
            # tmp_mat *= -1
            # np.fill_diagonal(tmp_mat,  np.diagonal(tmp_mat)*-1)
            
            ## invert and multiple
            inv = np.linalg.inv( tmp_mat )
            np.matmul(ZLMA_matrix, inv, out=tmp_mat)
            
            # finally
            self.total_impedence[:,:,Fi] = tmp_mat
            

    def loc_to_anti(self, antenna_XYZ):
        
        KX, KY, KZ = antenna_XYZ
        
        AARFAACC_X_locations = self.AART_ant_positions[0, 0::2]
        AARFAACC_Y_locations = self.AART_ant_positions[1, 0::2]
        
        found_index = None
        err2 = 0.1*0.1
        for AAR_i in range(len(AARFAACC_X_locations)):
            AX = AARFAACC_X_locations[AAR_i]
            AY = AARFAACC_Y_locations[AAR_i]
            R2 = (AX-KX)**2 + (AY-KY)**2
            if R2<err2:
                found_index = AAR_i*2
                break
        if found_index is None:
            print("ERROR! AARTFAAC model cannot find your antenna loc:", antenna_XYZ)
            quit()
            
        X_ant_i = found_index
        return X_ant_i
        
        
    def get_antenna_model(self, antenna_i= None, antenna_XYZ=None):
        """Give antenna_i or XYZ, and get full antenna model. WARNING: antenna_i has some internal definition, and has nothing to do with pyCRtools antenna indececs or RCU ids"""
            
        if antenna_i is None:
            X_ant_i = self.loc_to_anti( antenna_XYZ )
            KX, KY, KZ = antenna_XYZ
        else:
            X_ant_i = int(antenna_i/2)*2
            KX = self.AART_ant_positions[0, X_ant_i]
            KY = self.AART_ant_positions[1, X_ant_i]
            KZ = 0
        
        
        num_thetas = len(self.AART_thetas)
        num_phis = len(self.AART_Phis)
        num_frequencies = len( self.all_frequencies)
        num_antennas = len( self.total_Gz )
        
    #### some setup
        ## phase shifter to remove geometric delay
        shifter = np.empty([num_thetas, num_phis, num_frequencies ], dtype=np.complex)
        for zi in range( num_thetas ):
            for ai in range( num_phis ):
                Zenith = self.AART_thetas[zi]/RTD
                Azimuth = self.AART_Phis[ai]/RTD
                dt = ( KX*np.sin(Zenith)*np.cos(Azimuth) + KY*np.sin(Zenith)*np.sin(Azimuth))/v_air
                np.exp( self.all_frequencies*(1j*2*np.pi*(-dt)), out=shifter[zi,ai] )
                

    
        ## frequency interpolation
        frequency_bin = 0.5e6
        minF = self.all_frequencies[0]
        maxF = self.all_frequencies[-1]
        num_interpolant_freqs = int((maxF+minF)/frequency_bin )
        interpolant_frequencies = np.linspace(minF, maxF, num_interpolant_freqs)
        
        ## memory
        tmp1 = np.empty( (num_antennas,num_frequencies), dtype=np.complex )
        tmp2 = np.empty(num_frequencies, dtype=np.complex )
        tmp3 = np.empty(num_frequencies, dtype=np.double )


        def make_interpolant(antenna, polarization):
            """ antenna is 0 or 1 for X or Y, pol is 0 or 1 for zenith or azimithal component"""
            nonlocal tmp1, tmp2, tmp3
            
            antenna = antenna + X_ant_i
            
            grid = np.empty([num_thetas, num_phis, len(interpolant_frequencies) ], dtype=np.complex)
        
            for theta_i in range(num_thetas):
                for phi_i in range(num_phis):
                    
                    ## dot product between voltages and impedences
                    tmp1[:,:]  = self.total_Gz[:, polarization,theta_i,phi_i, :] 
                    tmp1 *= self.total_impedence[antenna, :, :] # this shouldn't matter??
                    # tmp1 *= self.total_impedence[:, antenna, :]
                    
                    
                    
                    np.sum( tmp1, axis=0, out = tmp2 )
                    
                    ## shift phase due to arival direction
                    tmp2 *= shifter[theta_i,phi_i]
                    
                    ## interpolate amplitude and phase
                    np.abs(tmp2, out=tmp3)
                    interp_ampltude = pchip_interpolate(self.all_frequencies,tmp3, interpolant_frequencies)
                    
                    # if theta_i==0 and phi_i==0:
                        # print('amp')
                        # plt.plot( interpolant_frequencies, interp_ampltude, 'o' )
                        # plt.plot( self.all_frequencies, tmp3, 'o' )
                        # plt.show()
                    
                    angles = np.angle(tmp2)
                    angles = np.unwrap(angles)
                    
                    interp_angle = pchip_interpolate(self.all_frequencies,angles, interpolant_frequencies)
                    
                    # if theta_i==0 and phi_i==0:
                    #     print('angle')
                    #     plt.plot( interpolant_frequencies, interp_angle, 'o' )
                    #     plt.plot( self.all_frequencies, angles, 'o' )
                    #     plt.show()
                    

                        
                    ## now convert back to real and imag
                    interp_angle = interp_angle*1j
                    np.exp( interp_angle, out=grid[theta_i,phi_i] )
                    grid[theta_i,phi_i] *= interp_ampltude
                    
            ## correct for different in definition
            if polarization==0: ## zenith points in oppisite directions in two models??
                grid *= -1
            ## and final angle-frequency interpolant
            interpolant = RegularGridInterpolator((self.AART_thetas, self.AART_Phis, interpolant_frequencies),  grid,  bounds_error=False,fill_value=0.0)
            return interpolant
        
        J00_interpolant = make_interpolant(0, 0)
        J01_interpolant = make_interpolant(0, 1)
        J10_interpolant = make_interpolant(1, 0)
        J11_interpolant = make_interpolant(1, 1)
    
        class single_LBA_model:
            def __init__(self, jones_functions, freq_bounds):
                self.jones_functions = jones_functions
                self.freq_bounds = freq_bounds
                
            def Jones_Matrices(self, frequencies, zenith, azimuth, freq_fill=1.0):
                """ if frequencies is numpy array in Hz, zenith and azimuth in degrees, than return numpy array of jones matrices,
                that when doted with [zenithal,azimuthal] component of incidident E-field, then will give [X,Y] voltages on dipoles"""
                
                return_matrices = np.empty( (len(frequencies), 2,2), dtype=np.complex )
                
                if zenith<0:
                    zenith = 0
                elif zenith > 90:
                    zenith = 90
                    
                while azimuth<0:
                    azimuth += 360
                while azimuth>360:
                    azimuth -= 360
                
                
                J00,J01 = self.jones_functions[0]
                J10,J11 = self.jones_functions[1]
                
                for fi,f in enumerate(frequencies):
                    if f<=self.freq_bounds[0] or f>=self.freq_bounds[-1]:
                        return_matrices[fi, 0,0] = freq_fill
                        return_matrices[fi, 1,1] = freq_fill
                        return_matrices[fi, 0,1] = 0.0
                        return_matrices[fi, 1,0] = 0.0
                    else:
                        return_matrices[fi, 0,0] = J00( (zenith,azimuth,f) )
                        return_matrices[fi, 0,1] = J01( (zenith,azimuth,f) )
                        return_matrices[fi, 1,0] = J10( (zenith,azimuth,f) )
                        return_matrices[fi, 1,1] = J11( (zenith,azimuth,f) )
                        
                        # print(fi)
                        # M = return_matrices[fi]
                        # print("J:", M)
                        # print( M[ 0,0]*M[ 1,1] - M[  0,1]*M[  1,0] )
                        # print()
                        
                return return_matrices
        
        return single_LBA_model([[J00_interpolant,J01_interpolant],[J10_interpolant,J11_interpolant]],  [interpolant_frequencies[0],interpolant_frequencies[-1]])
    
    
if __name__ == "main":
    
    ## assumes both mat files are in directory: "./AARTFAAC_LBA_MODEL/", and that LNA has: R=700 ohms, C=15e-12 farads, as per Maria Krause thesis
    AARTFAAC_model = aartfaac_LBA_model()  ### note, this is memory intensive. Most time is taking opening the mat files
    
    
    ## need to specify which antenna you want. This model's ids have nothing to do with LOFAR RCU ids, thus should mostly be ignored.
    ## only way to find which antenna to use is to specify the position, which is the same between this model and pyCRtools
    ## here I use this model to get position, just to show code. In your code you could use pyCRtools to get the location of the antenna of interest... or something..
    AART_ant_id = 2  ## number_antennas = AARTFAAC_model.AART_ant_positions.shape[1]
    antenna_XYZ = np.array([ AARTFAAC_model.AART_ant_positions[0, AART_ant_id], AARTFAAC_model.AART_ant_positions[1, AART_ant_id], 0.0 ])
    
    antenna_model = AARTFAAC_model.get_antenna_model( antenna_XYZ=antenna_XYZ )
    
    Jones_matrices = antenna_model.Jones_Matrices( [30e6, 50e6, 80e6], zenith=10, azimuth=136 ) ## or whatever, you get the point. This is a list of jones matrices for each frequency. 
    ##For each jones matrix, dot with zenithal and azimuthal fields to get voltage out of X and Y antennas.
    
    
    
    
    ### what if you want a different R and C?
    # 1) reset R and C
    AARTFAAC_model.set_RC(R=42, C=20e-12) ## who knows. Could be right...
    # 2) recalculate individual antenna model
    antenna_model = AARTFAAC_model.get_antenna_model( antenna_XYZ=antenna_XYZ )
    # 3) recaculate your jones matrix 
    Jones_matrices = antenna_model.Jones_Matrices( [30e6, 50e6, 80e6], zenith=10, azimuth=136 )
    
    
    
    
    
