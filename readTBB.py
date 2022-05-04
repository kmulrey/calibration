# this script reads in TBB data and processes it up to the point in the pipeline where the calibration would happen
# Note- this uses python2 and old pycrtools that are now broken


import pickle
from pycrtools import *
from pycrtools import lora
from pycrtools import xmldict
from pycrtools import crdatabase as crdb
from pycrtools.tasks import averagespectrum, beamformer2, fitbaseline, pulsecal, crosscorrelateantennas, directionfitplanewave, crimager
import pycrtools as cr
import numpy as np
from pycrtools import crdatabase as crdb
from pycrtools import metadata as md
from pycrtools import tools
from pycrtools import lora
from pycrtools import simhelp

from pycrtools.tasks import antennaresponse
from pycrtools.tasks import findrfi

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from scipy import signal
from scipy.interpolate import interp1d

from optparse import OptionParser


# for event 127417834, station 002
nAvg=100  # number of blocks to averate
nResample=51
blocksize=2**10
lora_logfile="LORAtime4"
lora_directory="/vol/astro3/lofar/vhecr/lora_triggered/LORA/"
bandpass_filter = cr.hArray(float, blocksize / 2 + 1)

loc = EarthLocation(lat = 52.905329712*u.deg, lon = 6.867996528*u.deg, height = 7.6*u.m)


dbManager = crdb.CRDatabase("crdb", host="coma00.science.ru.nl",user="crdb", password="crdb", dbname="crdb")
db = dbManager.db



## take as input an event index to read from the list in LBA_events.txt

parser = OptionParser()
parser.add_option("-i", "--id", type="int", help="event ID", default=1)

(options, args) = parser.parse_args()
event_ids=np.genfromtxt(open('LBA_events.txt'))




def get_noise(event_id):


    nstations=0

    try:
        #for u in np.arange(1):
        event = crdb.Event(db=db, id=event_id)

        event_time=np.asarray(event["lora_time"])
        event_time=event_time[event_time>1.0]
        event_time=np.min(event_time)
        
        #if event_time<100.0:
        #   break
        print(event_time)
        time = Time(event_time, format='unix', scale='utc',location=loc)
        time.delta_ut1_utc = 0.
        
        LST=time.sidereal_time('apparent').hour

        print('event time UTC: {0}'.format(event_time))
        print('event time LST: {0}'.format(LST))


        stations = []
        #collect stations with "GOOD" status for event
        for f in event.datafiles:
            stn=[]
            stn.extend(f.stations)
            #print stn[0].stationname
            #print f.stations.stationname
            if stn[0].stationname == "CS001" or stn[0].stationname == "CS002" or stn[0].stationname == "CS003" or stn[0].stationname == "CS004" or stn[0].stationname == "CS005" or stn[0].stationname == "CS006" or stn[0].stationname == "CS007" or stn[0].stationname == "CS011" or stn[0].stationname == "CS013" or stn[0].stationname == "CS017" or stn[0].stationname == "CS021" or stn[0].stationname == "CS026" or stn[0].stationname == "CS028" or stn[0].stationname == "CS030" or stn[0].stationname == "CS031" or stn[0].stationname == "CS032" or stn[0].stationname == "CS101" or stn[0].stationname == "CS103" or stn[0].stationname == "CS301" or stn[0].stationname == "CS302" or stn[0].stationname == "CS401" or stn[0].stationname == "CS501":
                if stn[0].status=="GOOD":
                    stations.extend(f.stations)
    
        nstations=len(stations)

    except:
        print('no event at this point')


    for s in np.arange(nstations):

        station_flag=0
        station=stations[s]

        # The following steps are copied from cr_physics pipeline
        # there are a million try/excepts because I ran into lots of specific errors I didn't want to handle
        
        try:
            # Open file
            f = cr.open(station.datafile.settings.datapath + '/' + station.datafile.filename)
            antenna_set= f["ANTENNA_SET"]
            
            # Check if we are dealing with LBA or HBA observations
            if "LBA" in f["ANTENNA_SET"]:
                print ('LBA event')
            else:
                print ('HBA event')
                continue
        except:
            print ('no event at antennas')
            continue
            
            
            
        # Read LORA information
        try:
            tbb_time = f["TIME"][0]
            max_sample_number = max(f["SAMPLE_NUMBER"])
            min_sample_number = min(f["SAMPLE_NUMBER"])

            (tbb_time_sec, tbb_time_nsec) = lora.nsecFromSec(tbb_time, logfile=os.path.join(lora_directory,lora_logfile))
            (block_number_lora, sample_number_lora) = lora.loraTimestampToBlocknumber(tbb_time_sec, tbb_time_nsec, tbb_time, max_sample_number, blocksize=blocksize)
        except:
            continue

       # Check if starting time in sample units (SAMPLE_NUMBER) does not deviate among antennas
        try:
            sample_number_per_antenna = np.array(f["SAMPLE_NUMBER"])
            median_sample_number = np.median(sample_number_per_antenna)
            data_length = np.median(np.array(f["DATA_LENGTH"]))
            deviating_antennas = np.where( np.abs(sample_number_per_antenna - median_sample_number) > data_length/4)[0]
            nof_deviating_antennas = len(deviating_antennas)
            print ('Number of deviating antennas: %d' % nof_deviating_antennas)
        except:
            continue


        try:
            frequencies = f["FREQUENCY_DATA"]
            print ('blocksize:  {0}'.format(f["BLOCKSIZE"]))
            
            # Get bandpass filter
            nf = f["BLOCKSIZE"] / 2 + 1
            ne = int(10. * nf / f["CLOCK_FREQUENCY"])
            bandpass_filter.fill(0.)

            bandpass_filter[int(nf * 30.0 / 100.)-(ne/2):int(nf * 80.0 / 100.)+(ne/2)] = 1.0
            gaussian_weights = cr.hArray(cr.hGaussianWeights(ne, 4.0))
            cr.hRunningAverage(bandpass_filter, gaussian_weights)
        except:
            continue
        
        try:
            raw_data = f["TIMESERIES_DATA"].toNumpy()
            # Find outliers
            tmp = np.max(np.abs(raw_data), axis=1)
            outlier_antennas = np.argwhere(np.abs(tmp-np.median(tmp[tmp>0.1])) > 2*np.std(tmp[tmp>0.1])).ravel()
            print("Outlier antennas", outlier_antennas)


        except:
            print 'no raw data'
            continue


        try:
            # Get calibration delays to flag antennas with wrong calibration values
            try:
                cabledelays = cr.hArray(f["DIPOLE_CALIBRATION_DELAY"])
                cabledelays = np.abs(cabledelays.toNumpy())
            except:
                print 'problem with cable delays'
                continue
                
            # Find RFI and bad antennas
            findrfi = cr.trun("FindRFI", f=f, nofblocks=10, plotlist=[], apply_hanning_window=True, hanning_fraction=0.2, bandpass_filter=bandpass_filter)
            print "Bad antennas", findrfi.bad_antennas
            antenna_ids_findrfi = f["SELECTED_DIPOLES"]
            nAnt=len(f["SELECTED_DIPOLES"])
            bad_antennas_spikes = []
            bad_antennas = findrfi.bad_antennas[:]

            dipole_names = f["SELECTED_DIPOLES"]
            good_antennas = [n for n in dipole_names if n not in bad_antennas]
            station["crp_bad_antennas_power"] = findrfi.bad_antennas
            station["crp_bad_antennas_spikes"] = bad_antennas_spikes
            selected_dipoles = []

            for i in range(len(dipole_names) / 2):
                if dipole_names[2 * i] in good_antennas and dipole_names[2 * i + 1] in good_antennas and f.nof_consecutive_zeros[2 * i] < 512 and f.nof_consecutive_zeros[2 * i + 1] < 512 and cabledelays[2 * i] < 150.e-9 and cabledelays[2 * i + 1] < 150.e-9:
                    selected_dipoles.extend([dipole_names[2 * i], dipole_names[2 * i + 1]])
            
            f["SELECTED_DIPOLES"] = selected_dipoles
            station["crp_selected_dipoles"] = selected_dipoles



            nDipoles=len(selected_dipoles)


        except:
            print 'issue with RFI'
            continue

        try:
            print block_number_lora
            nF= len(frequencies.toNumpy())
            all_ffts=np.zeros([nAvg,nDipoles,nF])
            all_ffts_cleaned=np.zeros([nAvg,nDipoles,nF])
        except:
            continue
        
        #______________________________________________________________________
        
        block_number=0

        for i in np.arange(nAvg):
            try:
        
                # make sure not to include the signal window in the average
                if abs(block_number-block_number_lora)<5:
                    block_number=block_number+10
    
                fft_data = f.empty("FFT_DATA")
                #f.getFFTData(fft_data, block_number_lora, True, hanning_fraction=0.2, datacheck=True)   # this is what is in the pipeline
                f.getFFTData(fft_data, block_number, True, hanning_fraction=0.2, datacheck=True)

                # Apply bandpass
                fft_data[...].mul(bandpass_filter)
    
                # Normalize spectrum
                fft_data /= f["BLOCKSIZE"]
                fft_hold=fft_data
                # Reject DC component
                fft_data[..., 0] = 0.0
            
                # Also reject 1st harmonic (gives a lot of spurious power with Hanning window)
                fft_data[..., 1] = 0.0
                
                # Flag dirty channels (from RFI excission)
                fft_data[..., cr.hArray(findrfi.dirty_channels)] = 0


                # factor of two because reall FFT
                all_ffts[i]=2*np.abs(fft_hold.toNumpy())**2
                all_ffts_cleaned[i]=2*np.abs(fft_data.toNumpy())**2
                
                
                badFreq=frequencies.toNumpy()[findrfi.dirty_channels]/1e6
                
                
                nBadChannelsFilt=len(badFreq[(badFreq>=30.0)*(badFreq<=80.0)])
                
                if nBadChannelsFilt>1:
                    print 'n bad channels: {0}'.format(nBadChannelsFilt)
                    continue


                block_number=block_number+1
                
            except:
                print 'error'
                continue

        try:
            #for y in np.arange(1):

            fft_avg=np.average(all_ffts_cleaned,axis=0)
            
            
            freq=frequencies.toNumpy()
            df=(freq[1]-freq[0])/1e6
            freq_new=np.arange(30,81,1)
            fft_resample=np.zeros([nAnt,nResample])

        except:
            print 'error in average'
            continue


        for n in np.arange(nDipoles):
            try:
                
                fft_use=fft_avg[n][fft_avg[n]>1e-100]
                freq_use=freq[fft_avg[n]>1e-100]
                
                if len(fft_avg[n])>len(fft_use):
                    station_flag=1

                f=interp1d(freq_use/1e6,fft_use)
                
                f_new=f(freq_new)
            
                start_f=np.argmin(np.abs((freq/1e6)-30))
                stop_f=np.argmin(np.abs((freq/1e6)-80))


                fft_resample[n]=f_new*(1/df)
            except:
                station_flag=1
                print 'issue with interp'

        analysisinfo={'event_number': event_id,'station': station.stationname,'UTC_time':event_time,'LST':LST,'frequencies':freq,'FFT_data':fft_avg,'frequencies_50':freq_new,'FFT_data_resampled':fft_resample,'flag': station_flag,'antenna_set':antenna_set,'selected_dipoles': dipole_names,'bad_dipoles':bad_antennas,'nBadChannelsFilt':nBadChannelsFilt}

        outputfile=open(station.stationname+'/'+str(int(event_id))+'_noise_OUTER.p','w')
                
        pickle.dump(analysisinfo,outputfile)
        outputfile.close()

        print '{0} done'.format(station.stationname)
      
                
    print 'done with event'





# find average noise for event with index from OptionParser

id_use=int(options.id)
print 'running event {0}'.format(event_ids[id_use])
get_noise(event_ids[id_use])




