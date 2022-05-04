"""CR pipeline.
"""

import logging

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG)

import matplotlib
matplotlib.use("Agg")

import os
import sys
import time
import datetime
import pickle
import pytmf
import numpy as np
import matplotlib.pyplot as plt
import pycrtools as cr
from pycrtools import crdatabase as crdb
from pycrtools import metadata as md
from pycrtools import tools
from pycrtools import lora
from pycrtools import simhelp

from pycrtools.tasks import antennaresponse
from pycrtools.tasks import findrfi
from pycrtools.tasks import shower
from pycrtools.tasks import galaxy
from pycrtools.tasks import minibeamformer
from pycrtools.tasks import directionfitplanewave
from pycrtools.tasks import pulseenvelope
from pycrtools.tasks import hypwavefront
from pycrtools.tasks import directionfitbf
from pycrtools.tasks import polarization
from pycrtools.tasks import ldf

from optparse import OptionParser
from contextlib import contextmanager

def readCalibratedClockOffsets(eventID, indir='/vol/astro3/lofar/vhecr/lora_triggered/calibratedClockOffsets'):
# Outputs dict with keys = stations, values = full clock offsets.

    filename = str(eventID) + '-calibrateFM.dat'
    filename = os.path.join(indir, filename)
    if not os.path.isfile(filename):
        print 'WARNING: no clock (FM) calibration file for this event, returning empty dict...'
        return {}
    infile = open(filename, 'r')
    infile.readline() # skip header
    offsetsPerStation_88 = {}
    finalOffsetsPerStation = {}
    # Station pol Status nofLines line0 line1 line2 delay0 delay1 delay2 final_delay
    for line in infile:
        cells = line.split()
        #nofFreqs = cells[3]
        for i, cell in enumerate(cells):
            if i > 3:
                cell = float(cell)
            if i == 1 or i == 3:
                cell = int(cell)
        (station, pol, status, noflines, line0, line1, line2, delay0, delay1, delay2, final_delay, goodness) = cells
        offsetsPerStation_88[station] = float(delay0) * 1.0e-9
        finalOffsetsPerStation[station] = float(final_delay) * 1.0e-9

    return finalOffsetsPerStation

# Error handling
class PipelineError(Exception):
    """Base class for pipeline exceptions."""

    def __init__(self, message, category="OTHER"):
        self.message = message
        self.category = category

class EventError(PipelineError):
    """Raised when an unhandlable error occurs at event level."""
    pass

class StationError(PipelineError):
    """Raised when an unhandlable error occurs at station level."""
    pass

class PolarizationError(PipelineError):
    """Raised when an unhandlable error occurs at polarization level."""
    pass

class Skipped(PipelineError):
    """Base class for everything that needs to lead to a skipped state."""
    pass

class EventSkipped(Skipped):
    """Raised when event is skipped."""
    pass

class StationSkipped(Skipped):
    """Raised when station is skipped."""
    pass

class PolarizationSkipped(Skipped):
    """Raised when polarization is skipped."""
    pass

@contextmanager
def process_event(event):
    start = time.clock()

    print "-- event {0}".format(event._id)

    event.status = "PROCESSING"
    event.statusmessage = ""
    event["crp_plotfiles"] = []
    event["wavefront_fit_output"] = ""

    event.write()

    for station in event.stations:
        station.status = "NEW"
        station.statusmessage = ""
        station["crp_plotfiles"] = []

        for p in station.polarization.keys():
            station.polarization[p].status = "NEW"
            station.polarization[p].statusmessage = ""
            station.polarization[p]["crp_plotfiles"] = []

    try:
        yield event
    except EventSkipped as e:
        logging.info("event skipped because: {0}".format(e.message))
        event.status = "SKIPPED"
        event.statusmessage = e.message
        event.simulation_status = "NOT_DESIRED"
        event.simulation_statusmessage = "event skipped"
    except EventError as e:
        logging.exception(e.message)
        event.status = "ERROR"
        event.statusmessage = e.message
    except Exception as e:
        event.status = "ERROR"
        event.statusmessage = e.message
        raise
    except BaseException as e:
        event.status = "ERROR"
        event.statusmessage = "sigterm recieved"
        raise
    finally:
        event["last_processed"] = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        event.write()
        print "-- event {0} completed in {1:.3f} s".format(event._id, time.clock() - start)

@contextmanager
def process_station(station):
    start = time.clock()

    print "-- station {0}".format(station.stationname)

    station.status = "PROCESSING"
    station.statusmessage = ""
    station.statuscategory = ""
    station["crp_plotfiles"] = []

    try:
        yield station
    except StationSkipped as e:
        logging.info("station skipped because: {0}".format(e.message))
        station.status = "SKIPPED"
        station.statusmessage = e.message
    except Skipped as e:
        logging.info("station processing aborted due to container skipping")
        station.status = "ABORTED"
        station.statusmessage = e.message
        station.statuscategory = e.category
        raise
    except StationError as e:
        logging.exception(e.message)
        station.status = "ERROR"
        station.statusmessage = e.message
    except Exception as e:
        station.status = "ERROR"
        station.statusmessage = e.message
        raise
    except BaseException as e:
        station.status = "ERROR"
        station.statusmessage = "sigterm recieved"
        raise
    finally:
        print "-- station {0} completed in {1:.3f} s".format(station.stationname, time.clock() - start)

@contextmanager
def process_polarization(polarization, *args, **kwargs):
    start = time.clock()

    print "-- polarization",
    for p in args:
        print p,
    print "\n"

    if not 'reset' in kwargs or kwargs['reset']:
        for p in args:
            polarization[p].status = "PROCESSING"
            polarization[p].statusmessage = ""
            polarization[p]["crp_plotfiles"] = []

    try:
        yield polarization
    except PolarizationSkipped as e:
        logging.info("polarization skipped because: {0}".format(e.message))
        for p in args:
            polarization[p].status = "SKIPPED"
            polarization[p].statusmessage = e.message
    except Skipped as e:
        logging.info("polarization processing aborted due to container skipping")
        for p in args:
            polarization[p].status = "ABORTED"
            polarization[p].statusmessage = e.message
        raise
    except PolarizationError as e:
        logging.exception(e.message)
        for p in args:
            polarization[p].status = "ERROR"
            polarization[p].statusmessage = e.message
    except Exception as e:
        for p in args:
            polarization[p].status = "ERROR"
            polarization[p].statusmessage = e.message
        raise
    except BaseException as e:
        for p in args:
            polarization[p].status = "ERROR"
            polarization[p].statusmessage = "sigterm recieved"
        raise
    finally:
        print "-- polarization",
        for p in args:
            print p,
        print "completed in {0:.3f} s".format(time.clock() - start)

# Don't write output to all tasks
cr.tasks.task_write_parfiles = False

# Parse commandline options
parser = OptionParser()
parser.add_option("-i", "--id", type="int", help="event ID", default=1)
parser.add_option("-b", "--blocksize", type="int", default=2 ** 16)
parser.add_option("-d", "--database", default="crdb.sqlite", help="filename of database")
parser.add_option("-o", "--output-dir", default="./", help="output directory")
parser.add_option("-s", "--station", action="append", help="only process given station")
parser.add_option("-a", "--accept_snr", type="int", default=3, help="accept pulses with snr higher than this")
parser.add_option("--maximum_nof_iterations", type="int", default=2, help="maximum number of iterations in antenna pattern unfolding loop")
parser.add_option("--maximum_angular_diff", type="float", default=0.5, help="maximum angular difference in direction fit iteration (in degrees), corresponds to angular resolution of a LOFAR station")
parser.add_option("--maximum_allowed_residual_delay", type="float", default=9e-8, help="average delay that is still allowed for a station to be called good")
parser.add_option("--maximum_allowed_outliers", type="float", default=4, help="number of outliers that can be ignored when calculating the average residuals")
parser.add_option("--broad_search_window_width", type="int", default=2 ** 13, help="width of window around expected location for first pulse search")
parser.add_option("--narrow_search_window_width", type="int", default=2 ** 8, help="width of window around expected location for subsequent pulse search")
parser.add_option("-l", "--lora_directory", default="./", help="directory containing LORA information")
parser.add_option("--lora_logfile", default="LORAtime4", help="name of LORA logfile with timestamps")
parser.add_option("--host", default=None, help="PostgreSQL host.")
parser.add_option("--user", default=None, help="PostgreSQL user.")
parser.add_option("--password", default=None, help="PostgreSQL password.")
parser.add_option("--dbname", default=None, help="PostgreSQL dbname.")
parser.add_option("--plot-type", default="png", help="Plot type (e.g. png, jpeg, pdf.")
parser.add_option("--beamform-outer-stations", default=False, action="store_true", help="Beamform outer stations (slow).")
parser.add_option("--debug", default=False, action="store_true", help="Generate additional plots for debugging.")
parser.add_option("--debug-test-pulse", default=False, action="store_true", help="Replace data from file by a known test pulse (delta pulse, high SNR, from zenith), to check e.g. calibration scale factors.")
parser.add_option("--store-calibrated-pulse-block", default=False, action="store_true", help="Store calibrated pulse block for offline analysis.")
parser.add_option("--gain-calibration-type", default="none", help="Type of gain calibration to use: 'crane', 'galaxy' or 'none'.")
parser.add_option("--publication-quality-plots", default=False, action="store_true", help="Produce publication quality plots.")
parser.add_option("--initial-direction-from-db", default=False, action="store_true", help="Use initial pulse direction from crp_average_direction parameter in database.")
parser.add_option("--initial-direction-az", default=None, type="float", help="Initial pulse direction azimuth (LOFAR convention)")
parser.add_option("--initial-direction-el", default=None, type="float", help="Initial pulse direction elevation (LOFAR convention)")
parser.add_option("--run-wavefront", default=False, action="store_true", help="Run wavefront task.")
parser.add_option("--wavefront-brute-force", default=False, action="store_true", help="Use brute force grid search to find core confidence level in wavefront task.")
parser.add_option("--run-polarization", default=False, action="store_true", help="Run polarization task.")
parser.add_option("--run-ldf", default=False, action="store_true", help="Run ldf task.")
parser.add_option("--only-inner-core-stations", default=False, action="store_true", help="Only process stations CS0??, i.e. superterp and inner core stations")
(options, args) = parser.parse_args()

db_filename = options.database
dbManager = crdb.CRDatabase(db_filename, host=options.host, user=options.user, password=options.password, dbname=options.dbname)
db = dbManager.db

superterpStations = ['CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007']

start = time.clock()

# Get event from database
with process_event(crdb.Event(db=db, id=options.id)) as event:

    # Keep track of large files that might turn out to be not needed
    event['extra_files'] = []

    # Read in FM-calibration values. Returns {} if file is not there.
    offsetsPerStation = readCalibratedClockOffsets(options.id)

    # Create output directory
    directory = os.path.join(options.output_dir, str(options.id))
    if not os.path.exists(directory):
        os.makedirs(directory)

    event_plot_prefix = os.path.join(directory, "cr_physics-{0}-".format(options.id))

    cr_found = False

    # Get first estimate of pulse direction
    try:
        lora_direction = list(event["lora_direction"])
    except KeyError:
        raise EventSkipped("have no lora_direction")

    # Create FFTW plans
    fftwplan = cr.FFTWPlanManyDftR2c(options.blocksize, 1, 1, 1, 1, 1, cr.fftw_flags.ESTIMATE)
    ifftwplan = cr.FFTWPlanManyDftC2r(options.blocksize, 1, 1, 1, 1, 1, cr.fftw_flags.ESTIMATE)

    # Create bandpass filter
    bandpass_filter = cr.hArray(float, options.blocksize / 2 + 1)

    # Loop over all stations in event
    stations = []
    for f in event.datafiles:
        stations.extend(f.stations)

    # Process only given stations if explicitly requested
    all_station_timeseries = []
    for station in stations:
        station['extra_files'] = []

        if options.station and station.stationname not in options.station:
            continue
        if options.only_inner_core_stations and "CS0" not in station.stationname:
            continue
        # Keep track of large files that might turn out to be not needed

        with process_station(station) as station:
            if options.initial_direction_az and options.initial_direction_el:
                pulse_direction = [options.pulse_az, options.pulse_el]
                print "Initial pulse direction from commandline", pulse_direction
            elif options.initial_direction_from_db:
                try:
                    pulse_direction = event["crp_average_direction"]
                    print "Initial pulse direction from crp_average_direction", pulse_direction
                except:
                    # Reset to LORA direction for initial values
                    pulse_direction = lora_direction
                    print "Initial pulse direction from LORA", pulse_direction
            else:
                # Reset to LORA direction for initial values
                pulse_direction = lora_direction
                print "Initial pulse direction from LORA", pulse_direction

            station_plot_prefix = event_plot_prefix + "{0}-".format(station.stationname)

            # Open file
            f = cr.open(station.datafile.settings.datapath + '/' + station.datafile.filename)

            # Check if we are dealing with LBA or HBA observations
            if "HBA" in f["ANTENNA_SET"]:
                hba = True
            elif "LBA" in f["ANTENNA_SET"]:
                hba = False
            else:
                raise EventSkipped("unsupported antennaset {0}".format(f["ANTENNA_SET"]))

            # Fix (Dec 16, 2019): remove RCUs that have no odd/even counterpart (i.e. antennas with only one dipole in data file)
            # Message to self (cc all): this requires a separate data integrity / readability module...!

            all_dipoles = [int(x) % 100 for x in f["SELECTED_DIPOLES"]]
            all_rcu_ids = np.array(f["SELECTED_DIPOLES"])
            antennas_missing_dipole = [x for x in all_dipoles if (x + (1 - 2*(x%2))) not in all_dipoles]
            # one-liner to catch where x+1 is not there if x is even, or when x-1 is not there when x is odd
            print 'Number of antennas with missing dipoles: %d: %s' % (len(antennas_missing_dipole), antennas_missing_dipole)

            # Check if starting time in sample units (SAMPLE_NUMBER) does not deviate among antennas
            sample_number_per_antenna = np.array(f["SAMPLE_NUMBER"])
            median_sample_number = np.median(sample_number_per_antenna)
            # Allow for one RCU to deviate (cf. CS004 RCU 75 issue, found Sept / Oct 2016)
            # and allow shifts up to 1/4 of typical data length (2 ms i.e. 400000 samples), which is 100000 samples.
            #nof_deviating_antennas = np.count_nonzero(sample_number_per_antenna - median_sample_number)
            median_data_length = np.median(np.array(f["DATA_LENGTH"]))

            deviating_antennas = np.where( np.abs(sample_number_per_antenna - median_sample_number) > median_data_length/4)[0]

            # Oct 28, 2019: also remove antennas with sample_number > median, otherwise unable
            # to read block 0 as this antenna has no data there.

            deviating_antennas_starting_later = np.where( sample_number_per_antenna > median_sample_number )[0]
            deviating_antennas = np.unique(np.concatenate( (deviating_antennas, deviating_antennas_starting_later)))

            data_length_per_antenna = np.array(f["DATA_LENGTH"])
            max_data_length = np.max(np.array(f["DATA_LENGTH"]))

            # FIX Oct 3, 2018
            print 'Number of deviating antennas (SAMPLE_NUMBER): %d' % len(deviating_antennas)

            # FIX Oct 23, 2019
            #more_deviating_antennas = np.where( data_length_per_antenna < max_data_length)[0]
            more_deviating_antennas = np.where( np.abs(data_length_per_antenna - median_data_length) > median_data_length / 10)[0]

            print 'Number of deviating antennas (DATA_LENGTH): %d' % len(more_deviating_antennas)
            deviating_antennas = np.unique(np.concatenate( (deviating_antennas, more_deviating_antennas)) )
            # END FIX

            nof_deviating_antennas = len(deviating_antennas) + len(antennas_missing_dipole)

            print 'Number of deviating antennas (total, unique): %d' % nof_deviating_antennas
            if nof_deviating_antennas > 8:
                raise StationError("Starting time (SAMPLE_NUMBER) deviates in %d > 8 antennas" % nof_deviating_antennas)
            elif nof_deviating_antennas > 0:
                #print 'Warning: one RCU found with deviating SAMPLE_NUMBER'
                other_polarization_rcus = []
                for thisRCU in deviating_antennas:
                    # Remove this RCU and its odd/even counterpart from the selected_dipoles list
                    # This prevents a read error later on
                    if thisRCU % 2 == 0:
                        other_polarization = thisRCU + 1
                    else:
                        other_polarization = thisRCU - 1
                    if other_polarization not in other_polarization_rcus:
                        other_polarization_rcus.append(other_polarization)
                            
                nof_antennas = len(sample_number_per_antenna)
                # Does not work for antennas_missing_dipole (and others?) if RCU numbers do not start at 0. Need to reference RCU ids, not just the array index...
                #selected_dipoles = [i for i in range(nof_antennas) if (i not in deviating_antennas) and (i not in other_polarization_rcus) and (i not in antennas_missing_dipole)]
                selected_dipoles = [rcu for rcu in all_rcu_ids if ((int(rcu) % 100) not in deviating_antennas) and ((int(rcu) % 100) not in other_polarization_rcus) and ((int(rcu) % 100) not in antennas_missing_dipole)]

                #print 'Now selecting out RCUs %d and %d' % (thisRCU, other_polarization)
                print 'Removing: %s' % deviating_antennas
                print 'Removing: %s' % other_polarization_rcus
                print 'Selected dipoles after removing %d rcus:' % (nof_antennas - len(selected_dipoles))
                print selected_dipoles
                
                if len(selected_dipoles) >= 8:
                    #print 'DISABLED selecting dipoles!!! (hack)'
                    f["SELECTED_DIPOLES"] = selected_dipoles
                else:
                    raise StationError("Less than 8 rcus remaining after outlier removal (data length, starting time)")

            # Read LORA information (only here, as max(SAMPLE_NUMBER) may be different after selection
            tbb_time = f["TIME"][0]
            max_sample_number = max(f["SAMPLE_NUMBER"])
            min_sample_number = min(f["SAMPLE_NUMBER"])

            # Continue reading LORA data
            print "reading LORA data"
            try:
                (tbb_time_sec, tbb_time_nsec) = lora.nsecFromSec(tbb_time, logfile=os.path.join(options.lora_directory, options.lora_logfile))

                (block_number_lora, sample_number_lora) = lora.loraTimestampToBlocknumber(tbb_time_sec, tbb_time_nsec, tbb_time, max_sample_number, blocksize=options.blocksize)
            except Exception:
                raise StationError("could not get expected block number from LORA data")
            print "have LORA data"

            # Center readout block around LORA pulse location
            shift = sample_number_lora - (options.blocksize / 2)
            pulse_search_window_start = (options.blocksize / 2) - (options.broad_search_window_width / 2)
            pulse_search_window_end = (options.blocksize / 2) + (options.broad_search_window_width / 2)

            print "shifting block by {0} samples to center lora pulse at sample {1}".format(shift, options.blocksize / 2)

            # Set file parameters to match LORA block
            f["BLOCKSIZE"] = options.blocksize
            f["BLOCK"] = block_number_lora
            try:
                f.shiftTimeseriesData(shift)
            except ValueError as e:
                raise StationError("{0}, signal is at edge of file".format(e.message))

            station["blocksize"] = options.blocksize
            station["block"] = block_number_lora
            station["shift"] = shift

            #Check for strange things in the datafile, especially clockfrequency
            if f["CLOCK_FREQUENCY"] == 0.:
                raise StationError("Clock frequency is set to zero, skipping station.")

            # Get frequencies
            frequencies = f["FREQUENCY_DATA"]

            # Get bandpass filter
            nf = f["BLOCKSIZE"] / 2 + 1
            ne = int(10. * nf / f["CLOCK_FREQUENCY"])
            bandpass_filter.fill(0.)
            if hba:
                fr = frequencies.toNumpy()

                if fr[0] > 170.e6:
                    bandpass_filter.fill(1.0)
                else:
                    bandpass_filter[:int(np.max(np.argwhere(fr < 168.21e6)))] = 1.0
                    bandpass_filter[int(np.min(np.argwhere(fr > 171.21e6))):] = 1.0

                gaussian_weights = cr.hArray(cr.hGaussianWeights(ne, 8.0))
            else: 
                bandpass_filter[int(nf * 30.0 / 100.)-(ne/2):int(nf * 80.0 / 100.)+(ne/2)] = 1.0
                gaussian_weights = cr.hArray(cr.hGaussianWeights(ne, 4.0))

            cr.hRunningAverage(bandpass_filter, gaussian_weights)

            # Make plot of timeseries data in both polarizations of first selected antenna
            try:
                raw_data = f["TIMESERIES_DATA"].toNumpy()
            except RuntimeError as e:
                raise StationError("Cannot get the raw data, hdf5 problem detected: {0}".format(e.message))

            if raw_data.shape[0] < 8:
                raise StationError("File of station contains less than 8 antennas.")

            # Find outliers
            tmp = np.max(np.abs(raw_data), axis=1)
            outlier_antennas = np.argwhere(np.abs(tmp-np.median(tmp[tmp>0.1])) > 2*np.std(tmp[tmp>0.1])).ravel()

            print "Outlier antennas", outlier_antennas

            # Optionally plot raw data
            if options.debug:
                t = np.arange(raw_data.shape[1]) / f["CLOCK_FREQUENCY"]
                for i in range(raw_data.shape[0]):
                    plt.clf()
                    plt.plot(t, raw_data[i])
                    plt.xlabel(r"Time ($\mu s$)")
                    plt.ylabel("Amplitude (ADU)")
                    plt.title("Timeseries raw dipole {0}".format(i))
                    plotfile = station_plot_prefix + "raw_data-{0}.{1}".format(i, options.plot_type)
                    plt.savefig(plotfile)
                    station["crp_plotfiles"].append(plotfile)

            with process_polarization(station.polarization, '0', '1') as polarization:

                # Get calibration delays to flag antennas with wrong calibration values
                try:
                    cabledelays = cr.hArray(f["DIPOLE_CALIBRATION_DELAY"])
                    cabledelays = np.abs(cabledelays.toNumpy())
                except Exception:
                    raise StationSkipped("do not have DIPOLE_CALIBRATION_DELAY value")

                # Find RFI and bad antennas
                try:
                    findrfi = cr.trun("FindRFI", f=f, nofblocks=10, save_plots=True, plot_prefix=station_plot_prefix, plot_type=options.plot_type, plotlist=[], apply_hanning_window=True, hanning_fraction=0.2, bandpass_filter=bandpass_filter)
                    print "Bad antennas", findrfi.bad_antennas
                    antenna_ids_findrfi = f["SELECTED_DIPOLES"]
                except ZeroDivisionError as e:
                    raise StationError("findrfi reports NaN in file {0}".format(e.message))

                print "blocks_with_dataloss", findrfi.blocks_with_dataloss

                for plot in findrfi.plotlist:
                    station["crp_plotfiles"].append(plot)

                # Select antennas which are marked good for both polarization
                dipole_names = f["SELECTED_DIPOLES"]

                bad_antennas = findrfi.bad_antennas[:]
                bad_antennas_spikes = []

                if hba:
                    if len(outlier_antennas) >= 1:
                        bad_antennas_spikes = [dipole_names[i] for i in outlier_antennas]
                        bad_antennas += bad_antennas_spikes

                        print "Adding outlier", bad_antennas_spikes, "to bad_antennas"

                good_antennas = [n for n in dipole_names if n not in bad_antennas]

                station["crp_bad_antennas_power"] = findrfi.bad_antennas
                station["crp_bad_antennas_spikes"] = bad_antennas_spikes

                print "NOF consecutive zeros", f.nof_consecutive_zeros
                selected_dipoles = []
                for i in range(len(dipole_names) / 2):
                    if dipole_names[2 * i] in good_antennas and dipole_names[2 * i + 1] in good_antennas and f.nof_consecutive_zeros[2 * i] < 512 and f.nof_consecutive_zeros[2 * i + 1] < 512 and cabledelays[2 * i] < 150.e-9 and cabledelays[2 * i + 1] < 150.e-9:
                        selected_dipoles.extend([dipole_names[2 * i], dipole_names[2 * i + 1]])
                print 'Selecting dipoles after RFI cleaning: '
                print selected_dipoles 
                try:
                    f["SELECTED_DIPOLES"] = selected_dipoles
                except ValueError as e:
                    raise StationError("Data problem, cannot read selected dipoles: {0}".format(e.message))

                station["crp_selected_dipoles"] = selected_dipoles

                # Read FFT data
                fft_data = f.empty("FFT_DATA")
                f.getFFTData(fft_data, block_number_lora, True, hanning_fraction=0.2, datacheck=True)

                print "NOF consecutive zeros", f.nof_consecutive_zeros

                # If debug-test-pulse option, replace fft_data by the fft of a test pulse
                if options.debug_test_pulse:
                    print '### Replacing LOFAR data by a test pulse! ###'
                    #Y = fft_data.toNumpy()
                    testpulse = np.zeros( (len(selected_dipoles), options.blocksize) )
                    testpulse[:, options.blocksize/2] += 100.0
                    testpulse += 1.0e-6 * np.random.randn(len(selected_dipoles), options.blocksize)
                    
                    # (AC) Delta pulse, height 100 (only positive for 1 sample, rest zero), middle of the block
                    # noise floor at 1e-6, make variable as option...
                    Y = np.fft.rfft(testpulse, axis=1) # This is the same FFT (convention) as is used in f.getFFTData
                    fft_data = cr.hArray(Y) # Replace real data with test pulse

                    # Make zenith angle = elevation = 45 degrees
                    # get time delays for a pulse from az=0, el=45
                    from pycrtools import srcfind as sf
                    antenna_positions = f["SCR_ANTENNA_POSITION"].toNumpy().ravel()
                    arrival_delays = sf.timeDelaysFromDirection(antenna_positions, (0.0, 45.0*np.pi/180))
                    arrival_delays = cr.hArray(arrival_delays)
                    
                    # Apply time delay. With or without minus sign? With minus sign, apparently...
                    weights = cr.hArray(complex, dimensions=fft_data, name="Complex Weights")
                    phases = cr.hArray(float, dimensions=fft_data, name="Phases", xvalues=frequencies)
    
                    cr.hDelayToPhase(phases, frequencies, -1.0 * arrival_delays)
                    cr.hPhaseToComplex(weights, phases)
            
                    fft_data.mul(weights)

                    # Set LORA direction to test pulse direction
                    lora_direction = (0.0, 45.0) # az, el


                # Apply bandpass
                fft_data[...].mul(bandpass_filter)

                # Normalize spectrum
                fft_data /= f["BLOCKSIZE"]

                # Reject DC component
                fft_data[..., 0] = 0.0

                # Also reject 1st harmonic (gives a lot of spurious power with Hanning window)
                fft_data[..., 1] = 0.0

                # Flag dirty channels (from RFI excision)
                fft_data[..., cr.hArray(findrfi.dirty_channels)] = 0
                station["crp_dirty_channels"] = findrfi.dirty_channels

                # Get integrated power / amplitude spectrum for each antenna
                # now selected_dipoles contains less antennas (or the same) as earlier, as bad antennas have been removed
                antennas_cleaned_sum_amplitudes = []
                antennas_cleaned_power = []
                for id in selected_dipoles:
                    index_findrfi_dipoles = antenna_ids_findrfi.index(id)
                    antennas_cleaned_sum_amplitudes.append(findrfi.antennas_cleaned_sum_amplitudes[index_findrfi_dipoles])
                    antennas_cleaned_power.append(findrfi.antennas_cleaned_power[index_findrfi_dipoles])
                #antennas_cleaned_sum_amplitudes = cr.hArray([findrfi.antennas_cleaned_sum_amplitudes[] for i in f["SELECTED_DIPOLES_INDEX"]])
                #                antennas_cleaned_power = cr.hArray([findrfi.antennas_cleaned_power[i] for i in f["SELECTED_DIPOLES_INDEX"]])

                antennas_cleaned_sum_amplitudes = cr.hArray(antennas_cleaned_sum_amplitudes)
                antennas_cleaned_power = cr.hArray(antennas_cleaned_power)

                station["crp_antennas_cleaned_sum_amplitudes"] = antennas_cleaned_sum_amplitudes
                station["crp_antennas_cleaned_power"] = antennas_cleaned_power

                if not hba:
                    print "Gain calibrating using Galactic Noise or crane calibration"
                    print options.gain_calibration_type
                    print (options.gain_calibration_type == 'none')
                    print (options.gain_calibration_type != 'none')
                    use_gain_curve = True if options.gain_calibration_type != 'none' else False
                    use_gain_galaxy = True if options.gain_calibration_type == 'galaxy' else False
                    
                    # Normalize received power to that expected for a Galaxy dominated reference antenna
                    galactic_noise = cr.trun("GalacticNoise", fft_data=fft_data, frequencies=frequencies, channel_width=f["SAMPLE_FREQUENCY"][0] / f["BLOCKSIZE"], timestamp=tbb_time, antenna_set=f["ANTENNA_SET"], original_power=antennas_cleaned_power, use_gain_curve=use_gain_curve, use_gain_galaxy=use_gain_galaxy)
                    calibrated = True
                    
                    station["crp_galactic_noise"] = galactic_noise.galactic_noise_power
                    station["crp_relative_gain_correction_factor"] = np.square(galactic_noise.correction_factor.toNumpy())
                else:
                    print "Gain calibrating using LOFAR CalTables for HBA"

                    # Get relative gains from LOFAR CalTables
                    try:
                        relative_gains = f["STATION_GAIN_CALIBRATION"]
                    except Exception as e:
                        raise StationSkipped("do not have STATION_GAIN_CALIBRATION value")

                    fft_data[...].mul(relative_gains[...])

                    station["crp_relative_gain_correction_factor"] = relative_gains.toNumpy()
                    calibrated = False
                    
                # Apply calibration delays
                try:
                    cabledelays = cr.hArray(f["DIPOLE_CALIBRATION_DELAY"])
                except Exception:
                    raise StationSkipped("do not have DIPOLE_CALIBRATION_DELAY value")

                if options.debug_test_pulse:
                    nof_ants = len(cabledelays.toNumpy())
                    cabledelays[0] = 0.0
                    cabledelays[1] = 0.0
                    cabledelays[2] = 1.0
                    cabledelays[3] = 1.0
                    cabledelays[4] = 2.0
                    cabledelays[5] = 2.0
                    #cabledelays = np.arange(0.0, 5.0, 0.5)
                    #cabledelays = np.repeat(cabledelays, 2)
                    #cabledelays = np.append(cabledelays, 2.0 * np.random.randn(nof_ants - len(cabledelays)))
                    
                    print 'Dipole calibration delays for test pulse: '
                    print cabledelays
                    cabledelays = cr.hArray(cabledelays)

                weights = cr.hArray(complex, dimensions=fft_data, name="Complex Weights")
                phases = cr.hArray(float, dimensions=fft_data, name="Phases", xvalues=frequencies)

                cr.hDelayToPhase(phases, frequencies, cabledelays)
                cr.hPhaseToComplex(weights, phases)

                fft_data.mul(weights)

                # Get timeseries data
                timeseries_data = f.empty("TIMESERIES_DATA")
                nantennas = timeseries_data.shape()[0] / 2
                
                print "(DEBUG) setting nantennas to %d" % nantennas
                # Get antennas positions
                antenna_positions = f["SCR_ANTENNA_POSITION"]

                # Swap dipoles if needed
                if f["ANTENNA_SET"] == "LBA_OUTER":
                    print "LBA_OUTER, swapping 0,1 dipoles"
                    cr.hSwap(fft_data[0:2*nantennas:2, ...], fft_data[1:2*nantennas:2, ...])

                # Beamform in LORA direction for both polarizations
                fft_data_0 = cr.hArray(complex, dimensions=(nantennas, options.blocksize / 2 + 1))
                fft_data_1 = cr.hArray(complex, dimensions=(nantennas, options.blocksize / 2 + 1))

                fft_data_0[...].copy(fft_data[0:2*nantennas:2, ...])
                fft_data_1[...].copy(fft_data[1:2*nantennas:2, ...])

                antenna_positions_one = cr.hArray(float, dimensions=(nantennas, 3))
                antenna_positions_one[...].copy(antenna_positions[0:2*nantennas:2, ...])

                mb0 = cr.trun("MiniBeamformer", fft_data=fft_data_0, frequencies=frequencies, antpos=antenna_positions_one, direction=pulse_direction)
                mb1 = cr.trun("MiniBeamformer", fft_data=fft_data_1, frequencies=frequencies, antpos=antenna_positions_one, direction=pulse_direction)

                beamformed_timeseries = cr.hArray(float, dimensions=(2, options.blocksize))

                print "calculating inverse FFT"

                cr.hFFTWExecutePlan(beamformed_timeseries[0], mb0.beamformed_fft, ifftwplan)
                cr.hFFTWExecutePlan(beamformed_timeseries[1], mb1.beamformed_fft, ifftwplan)

                print "starting pulse envelope"

                # Look for significant pulse in beamformed signal
                pulse_envelope_bf = cr.trun("PulseEnvelope", timeseries_data=beamformed_timeseries, pulse_start=pulse_search_window_start, pulse_end=pulse_search_window_end, nsigma=options.accept_snr, save_plots=True, plot_prefix=station_plot_prefix+"bf-", plot_type=options.plot_type, plotlist=[])
                polarization['0']['crp_plotfiles'].append(pulse_envelope_bf.plotlist[0])
                polarization['1']['crp_plotfiles'].append(pulse_envelope_bf.plotlist[1])

                polarization['0']['crp_bf_peak_amplitude'] = pulse_envelope_bf.peak_amplitude[0]
                polarization['1']['crp_bf_peak_amplitude'] = pulse_envelope_bf.peak_amplitude[1]

                polarization['0']['crp_bf_rms'] = pulse_envelope_bf.rms[0]
                polarization['1']['crp_bf_rms'] = pulse_envelope_bf.rms[1]

                polarization['0']['crp_bf_mean'] = pulse_envelope_bf.mean[0]
                polarization['1']['crp_bf_mean'] = pulse_envelope_bf.mean[1]

                # Get pulse window (no longer used but kept for information)
                station['crp_bf_pulse_position'] = pulse_envelope_bf.bestpos

                pulse_start = pulse_search_window_start + pulse_envelope_bf.bestpos - options.narrow_search_window_width / 2
                pulse_end = pulse_search_window_start + pulse_envelope_bf.bestpos + options.narrow_search_window_width / 2

                print "now looking for pulse in narrow range between samples {0:d} and {1:d}".format(pulse_start, pulse_end)

                # skip this station for further processing when no cosmic ray signal is found
                cr_found_in_station = False
                for i in [0, 1]:
                    if i in pulse_envelope_bf.antennas_with_significant_pulses:
                        cr_found_in_station = True
                        polarization[str(i)].status = "GOOD"
                        polarization[str(i)].statusmessage = ""
                    else:
                        polarization[str(i)].status = "BAD"
                        polarization[str(i)].statusmessage = "no significant pulse found in beamformed signal"

                if cr_found_in_station:
                    station.status = "GOOD"
                    station.statusmessage = ""
                else:
                    station.status = "BAD"
                    station.statusmessage = "no significant pulse found in beamformed signal for either polarization"
                    continue

            # Store calibrated pulse block
            if options.store_calibrated_pulse_block:

                # Get timeseries data
                cr.hFFTWExecutePlan(timeseries_data[...], fft_data[...], ifftwplan)

                outfile = "calibrated_pulse_block-{0}-{1}.npy".format(options.id, station.stationname)
                np.save(os.path.join(directory, outfile), timeseries_data.toNumpy())
                station['extra_files'].append(outfile)

            # Start direction fitting loopim
            n = 0
            direction_fit_converged = False
            direction_fit_successful = True
            while True:

                # Unfold antenna pattern
                if hba:
                    # Get timeseries data
                    cr.hFFTWExecutePlan(timeseries_data[...], fft_data[...], ifftwplan)

                else:
                    antenna_response = cr.trun("AntennaResponse", instrumental_polarization=fft_data, frequencies=frequencies, direction=pulse_direction)

                    # Get timeseries data
                    cr.hFFTWExecutePlan(timeseries_data[...], antenna_response.on_sky_polarization[...], ifftwplan)

                # Run beamforming direction finder once
                # not to find the direction but to at least have one point for outer stations
                if options.beamform_outer_stations and n==0 and not hba:
                    fft_data_0[...].copy(antenna_response.on_sky_polarization[0:2*nantennas:2, ...])
                    fft_data_1[...].copy(antenna_response.on_sky_polarization[1:2*nantennas:2, ...])

                    dbf_theta = cr.trun("DirectionFitBF", fft_data=fft_data_0, frequencies=frequencies, antpos=antenna_positions, start_direction=lora_direction)
                    dbf_phi = cr.trun("DirectionFitBF", fft_data=fft_data_1, frequencies=frequencies, antpos=antenna_positions, start_direction=lora_direction)

                    timeseries_data_dbf = cr.hArray(float, dimensions=(2,dbf_theta.beamformed_timeseries.shape()[0]))
                    timeseries_data_dbf[0].copy(dbf_theta.beamformed_timeseries)
                    timeseries_data_dbf[1].copy(dbf_phi.beamformed_timeseries)

                    pulse_envelope_dbf = cr.trun("PulseEnvelope", timeseries_data=timeseries_data_dbf, pulse_start=pulse_start, pulse_end=pulse_end, resample_factor=16, save_plots=True, plot_prefix=station_plot_prefix+"dbf-", plot_type=options.plot_type, plotlist=station["crp_plotfiles"], extra=True, plot_antennas=[])

                    station["crp_integrated_pulse_power_dbf"] = cr.hArray(pulse_envelope_dbf.integrated_pulse_power).toNumpy()
                    station["crp_integrated_noise_power_dbf"] = cr.hArray(pulse_envelope_dbf.integrated_noise_power).toNumpy()

                # Calculate delays using Hilbert transform
                pulse_envelope = cr.trun("PulseEnvelope", timeseries_data=timeseries_data, pulse_start=pulse_start, pulse_end=pulse_end, resample_factor=16, npolarizations=2)

                delays = pulse_envelope.delays

                # Use current direction if not enough significant pulses are found for direction fitting
                if len(pulse_envelope.antennas_with_significant_pulses) < 3:
                    if n == 0:
                        logging.info("less than 3 antennas with significant pulses in first iteration")
                        raise StationError("less than 3 antennas with significant pulses")
                    else:
                        logging.info("less than 3 antennas with significant pulses, using previous direction")
                        station["crp_pulse_direction"] = pulse_direction
                        station.statusmessage = "less than 3 antennas in last iteration"
                        break

                # Fit pulse direction

                direction_fit_plane_wave = cr.trun("DirectionFitPlaneWave", positions=antenna_positions, timelags=delays, good_antennas=pulse_envelope.antennas_with_significant_pulses, reference_antenna=pulse_envelope.refant, verbose=True)

                pulse_direction = direction_fit_plane_wave.meandirection_azel_deg

                print "Hilbert envelope direction:", direction_fit_plane_wave.meandirection_azel_deg

                # Check if fitting was succesful
                if direction_fit_plane_wave.fit_failed:
                    break

                # Check for convergence of iterative direction fitting loop
                if n > 0:
                    angular_diff = np.rad2deg(tools.spaceAngle(np.deg2rad((90 - last_direction[1])), np.deg2rad((90 - last_direction[0])), np.deg2rad((90 - pulse_direction[1])), np.deg2rad((90 - pulse_direction[0]))))

                    if angular_diff < options.maximum_angular_diff:
                        direction_fit_converged = True

                last_direction = pulse_direction

                n += 1
                if direction_fit_converged:
                    print "fit converged"
                    station["crp_pulse_direction"] = pulse_direction
                    break

                # Check if maximum number of iterations is reached (will avoid infinite loop)
                if n > options.maximum_nof_iterations:
                    print "maximum number of iterations reached"
                    station["crp_pulse_direction"] = pulse_direction
                    station.statusmessage = "maximum number of iterations reached"
                    break

                if hba:
                    station["crp_pulse_direction"] = pulse_direction
                    break

            # Get final set of delays again, but now with plots and timing error estimates
            pulse_envelope = cr.trun("PulseEnvelope", timeseries_data=timeseries_data, pulse_start=pulse_start, pulse_end=pulse_end, resample_factor=32, npolarizations=2, save_plots=True, plot_prefix=station_plot_prefix+"pe-", plot_type=options.plot_type, plotlist=station["crp_plotfiles"], estimate_timing_error=True, plot_antennas=[0, 1])

            delays = pulse_envelope.delays

            polarization['0'].status = "GOOD"
            polarization['1'].status = "GOOD"
            polarization['0'].statusmessage = ""
            polarization['1'].statusmessage = ""

            # Check if result of planewave fit is reasonable
            residual_delays = direction_fit_plane_wave.residual_delays.toNumpy()
            residual_delays = np.abs(residual_delays)

            average_residual = residual_delays.mean()

            # Plot residual delays
            plt.clf()
            plt.plot(direction_fit_plane_wave.residual_delays.toNumpy(), "ro")
            plt.xlabel("Antenna number")
            plt.ylabel("Residual delay (s)")
            plotfile = station_plot_prefix + "residual_delay.{0}".format(options.plot_type)
            plt.savefig(plotfile)
            station["crp_plotfiles"].append(plotfile)

            # Plot spectrum after antenna model unfolding
            if not hba:
                plt.clf()
                plt.plot(frequencies.toNumpy() / 1.e6, np.log(np.median(np.square(np.abs(antenna_response.on_sky_polarization.toNumpy())), axis=0)))
                plotfile = station_plot_prefix + "spectrum_after_antenna_model.{0}".format(options.plot_type)
                plt.title("Median spectrum after antenna model.")
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Log-Spectral Power [ADU]")
                plt.savefig(plotfile)
                station["crp_plotfiles"].append(plotfile)

            # Add parameters
            station["crp_estimate_timing_timediff"] = pulse_envelope.estimate_timing_timediff.toNumpy().reshape((nantennas, 2, 10))[:,pulse_envelope.strongest_polarization,:]
            station["crp_estimate_timing_std"] = pulse_envelope.estimate_timing_std.reshape((nantennas, 2))[:,pulse_envelope.strongest_polarization]

            station["crp_pulse_delay"] = delays.toNumpy().reshape((nantennas, 2))[:,pulse_envelope.strongest_polarization]
            station["crp_pulse_snr"] = pulse_envelope.snr.toNumpy().reshape((nantennas, 2))[:,pulse_envelope.strongest_polarization]
            station["crp_median_pulse_snr"] = np.median(station["crp_pulse_snr"])
            print 'Strongest polarization = %d' % pulse_envelope.strongest_polarization

            station["clock_offset"] = f["CLOCK_OFFSET"][0]
            # Adapt outer-core clock offsets until the day that the common clock was installed
            # Use data from FM-radio calibration. Otherwise, use zero (0.0), which should be within +/- 50 ns from reality.
            if datetime.datetime.utcfromtimestamp(tbb_time) <= datetime.datetime(2012, 10, 10):
                if station.stationname not in superterpStations: # superterp is fine already
                    station["clock_offset"] = offsetsPerStation[station.stationname] if station.stationname in offsetsPerStation.keys() else 0.0
                    print 'Setting clock offset %s to %3.3f' % (station.stationname, station["clock_offset"]*1.0e9)

            station["crp_pulse_delay"] += float(block_number_lora * options.blocksize + max_sample_number + shift + pulse_start) / f["SAMPLE_FREQUENCY"][0] + station["clock_offset"]
                        
            print 'Station: %s' % station.stationname
            print 'BLock nr Lora = %d, max-min sample nr = %d, shift = %d, pulse_start = %d, clock offset = %f' % (block_number_lora, max_sample_number - min_sample_number, shift, pulse_start, station["clock_offset"])
            station["crp_pulse_delay_fit_residual"] = direction_fit_plane_wave.residual_delays.toNumpy()

            station["scr_antenna_position"] = f["SCR_ANTENNA_POSITION"].toNumpy()
            print '(DEBUG) Writing antenna positions of %d antennas to database (lcr_antenna_position)' % len(f["LCR_ANTENNA_POSITION"].toNumpy())
            print f["LCR_ANTENNA_POSITION"].toNumpy().shape
            station["lcr_antenna_position"] = f["LCR_ANTENNA_POSITION"].toNumpy()

            # Keep copy with old name for now, cleanup later
            station["local_antenna_positions"] = f["LCR_ANTENNA_POSITION"].toNumpy()

            if direction_fit_plane_wave.fit_failed:
                station.status = "BAD"
                station.statusmessage = "direction fit failed"
                station.statuscategory = "fit_failed"
                pulse_direction = list(event["lora_direction"])
                direction_fit_successful = False


            if direction_fit_plane_wave.goodcount < nantennas / 2:
                station.status = "BAD"
                station.statusmessage = "goodcount {0} < nantennas / 2 [= {1}]".format(direction_fit_plane_wave.goodcount, nantennas / 2)
                station.statuscategory = "good_count_ant"
                pulse_direction = list(event["lora_direction"])
                direction_fit_successful = False


            if average_residual > options.maximum_allowed_residual_delay:
                limited_residuals = residual_delays
                limited_residuals.sort()
                limited_residuals = limited_residuals[:-1*options.maximum_allowed_outliers]
                limited_residuals_mean = limited_residuals.mean()
                if limited_residuals_mean > options.maximum_allowed_residual_delay:
                    print "direction fit residuals too large, average_residual = {0}, limited = {1}".format(average_residual,limited_residuals_mean)
                    station.status = "BAD"
                    station.statusmessage = "average_residual = {0}, limited = {1}".format(average_residual,limited_residuals_mean )
                    station.statuscategory = "average_residual"
                    pulse_direction = list(event["lora_direction"])
                    direction_fit_successful = False

                else:
                    print "direction fit limited residuals ok, average_residual = {0}, limited = {1}".format(average_residual,limited_residuals_mean)
            else:
                print "direction fit residuals ok, average_residual = {0}".format(average_residual)

            # Adding cut for too many flagged antennas
            if len(selected_dipoles) < 48:
                station.status = "BAD"
                station.statusmessage = "more than 48 dipoles removed in rfi and spikes finding"
                station.statuscategory = "good_count_ant"
                direction_fit_successful = False
                print "Too many flagged dipoles, fit cannot be reliable."


            with process_polarization(station.polarization, 'xyz') as polarization:
                print "(DEBUG) Creating xyz_timeseries_data: nantennas = %d" % nantennas
                xyz_timeseries_data = cr.hArray(float, dimensions=(3 * nantennas, options.blocksize))

                cr.hProjectPolarizations(xyz_timeseries_data[0:3 * nantennas:3, ...], xyz_timeseries_data[1:3 * nantennas:3, ...], xyz_timeseries_data[2:3 * nantennas:3, ...], timeseries_data[0:2 * nantennas:2, ...], timeseries_data[1:2 * nantennas:2, ...], pytmf.deg2rad(pulse_direction[0]), pytmf.deg2rad(pulse_direction[1]))

                if options.store_calibrated_pulse_block:
                    outfile = "xyz_calibrated_pulse_block-{0}-{1}.npy".format(options.id, station.stationname)
                    np.save(os.path.join(directory, outfile), xyz_timeseries_data.toNumpy())
                    station['extra_files'].append(outfile)

                # Get Stokes parameters
#                stokes_parameters = cr.trun("StokesParameters", timeseries_data=xyz_timeseries_data, pulse_start=pulse_start, pulse_end=pulse_end, resample_factor=16)

                # Get pulse strength
                pulse_envelope_xyz = cr.trun("PulseEnvelope", timeseries_data=xyz_timeseries_data, Efield_unit=True, pulse_start=pulse_start, pulse_end=pulse_end, resample_factor=16, npolarizations=3, save_plots=True, plot_prefix=station_plot_prefix, plot_type=options.plot_type, plotlist=polarization['xyz']["crp_plotfiles"], extra=True, plot_antennas=[])

                # Do noise characterization
#                noise = cr.trun("Noise", timeseries_data=xyz_timeseries_data, histrange=(-3 * pulse_envelope_xyz.rms[0], 3 * pulse_envelope_xyz.rms[0]), save_plots=True, plot_prefix=station_plot_prefix, plot_type=options.plot_type, plotlist=polarization['xyz']["crp_plotfiles"])

                # Calculate time delay of pulse with respect to the start time of the file (e.g. f["TIME"])
                time_delays = pulse_envelope_xyz.pulse_maximum_time.toNumpy().reshape((nantennas, 3))
                time_delays += float(block_number_lora * options.blocksize + max_sample_number) / f["SAMPLE_FREQUENCY"][0] + station["clock_offset"]

                station["crp_pulse_time"] = time_delays

                polarization['xyz']["crp_pulse_peak_amplitude"] = cr.hArray(pulse_envelope_xyz.peak_amplitude).toNumpy().reshape((nantennas, 3))
                polarization['xyz']["crp_integrated_pulse_power"] = cr.hArray(pulse_envelope_xyz.integrated_pulse_power).toNumpy().reshape((nantennas, 3))
                polarization['xyz']["crp_integrated_noise_power"] = cr.hArray(pulse_envelope_xyz.integrated_noise_power).toNumpy().reshape((nantennas, 3))
                polarization['xyz']["crp_rms"] = cr.hArray(pulse_envelope_xyz.rms).toNumpy().reshape((nantennas, 3))
#                polarization['xyz']["crp_stokes"] = pulse_envelope_xyz.stokes.toNumpy()
#                polarization['xyz']["crp_polarization_angle"] = pulse_envelope_xyz.polarization_angle.toNumpy()

                # Exclude failed fits and average residuals and set polarization status
                if  direction_fit_successful == False:
                    polarization['xyz'].status = "BAD"
                    polarization['xyz'].statusmessage = "direction fit not successful"

                else:
                    polarization['xyz'].status = "GOOD"
                    polarization['xyz'].statusmessage = ""

                    cr_found = True
                    
                    all_station_timeseries.append(timeseries_data.toNumpy()) # For later analysis e.g. in Wavefront task using cross-correlation timing. Only when cr_found is True.

            with process_polarization(station.polarization, '0', '1', reset=False) as polarization:

                # Get original gain calibrated (but not antenna response corrected) timeseries data
                timeseries_data.fill(0.)
                cr.hFFTWExecutePlan(timeseries_data[...], fft_data[...], ifftwplan)

                # Get pulse strength
                pulse_envelope_01 = cr.trun("PulseEnvelope", timeseries_data=timeseries_data, pulse_start=pulse_start, pulse_end=pulse_end, resample_factor=16, npolarizations=2, save_plots=True, plot_prefix=station_plot_prefix+'pol01', plot_type=options.plot_type, plotlist=[], extra=True, plot_antennas=[0, 1])

                station['crp_pulse_start'] = pulse_start
                station['crp_pulse_end'] = pulse_end
                station['crp_pulse_maxpos_01'] = pulse_envelope_01.maxpos

                polarization['0']['crp_plotfiles'].extend(pulse_envelope_01.plotlist[0:len(pulse_envelope_01.plotlist):2])
                polarization['1']['crp_plotfiles'].extend(pulse_envelope_01.plotlist[1:len(pulse_envelope_01.plotlist):2])

                polarization['0']["crp_pulse_peak_amplitude"] = cr.hArray(pulse_envelope_01.peak_amplitude).toNumpy().reshape((nantennas, 2))
                polarization['0']["crp_integrated_pulse_power"] = cr.hArray(pulse_envelope_01.integrated_pulse_power).toNumpy().reshape((nantennas, 2))
                polarization['0']["crp_integrated_noise_power"] = cr.hArray(pulse_envelope_01.integrated_noise_power).toNumpy().reshape((nantennas, 2))
                polarization['0']["crp_integrated_pulse_power_wide"] = cr.hArray(pulse_envelope_01.integrated_pulse_power_wide).toNumpy().reshape((nantennas, 2))
                polarization['0']["crp_integrated_noise_power_wide"] = cr.hArray(pulse_envelope_01.integrated_noise_power_wide).toNumpy().reshape((nantennas, 2))
                polarization['0']["crp_integrated_pulse_power_double_wide"] = cr.hArray(pulse_envelope_01.integrated_pulse_power_double_wide).toNumpy().reshape((nantennas, 2))
                polarization['0']["crp_integrated_noise_power_double_wide"] = cr.hArray(pulse_envelope_01.integrated_noise_power_double_wide).toNumpy().reshape((nantennas, 2))
                polarization['0']["crp_rms"] = cr.hArray(pulse_envelope_01.rms).toNumpy().reshape((nantennas, 2))

    if cr_found:

        # Get combined parameters from (cached) database
        all_station_selected_dipole_ids = []
        all_station_lcr_antenna_position = []
        all_station_pulse_delays = []
        all_station_pulse_snr = []
        all_station_fit_residuals = []
        all_station_pulse_peak_amplitude = []
        all_station_integrated_pulse_power = []
        all_station_integrated_noise_power = []
        all_station_rms = []
        all_station_direction = []
        all_station_names = []
        all_station_antennas_stationnames = []
        
        all_stations_flagged = False
#        all_station_polarization_angle = []

        nof_good_stations = 0
        for station in stations:
            if station.status == "GOOD":
                if not station.flagged and not station.stationname == 'CS001':
                    nof_good_stations += 1
                    try:
                        all_station_direction.append(station["crp_pulse_direction"])
                        all_station_pulse_delays.append(station["crp_pulse_delay"])# - station["clock_offset"])
                        all_station_pulse_snr.append(station["crp_pulse_snr"])# - station["clock_offset"])
                        all_station_fit_residuals.append(station["crp_pulse_delay_fit_residual"])
                        all_station_lcr_antenna_position.append(station["lcr_antenna_position"])
                        all_station_selected_dipole_ids.append(station["crp_selected_dipoles"])
                        all_station_pulse_peak_amplitude.append(station.polarization['xyz']["crp_pulse_peak_amplitude"])
                        all_station_integrated_pulse_power.append(station.polarization['xyz']["crp_integrated_pulse_power"]) #  in proper units
                        all_station_integrated_noise_power.append(station.polarization['xyz']["crp_integrated_noise_power"]) # in proper units
                        all_station_rms.append(station.polarization['xyz']["crp_rms"])
                        all_station_names.append(station.stationname)
                        all_station_antennas_stationnames.extend([station.stationname] * len(station["crp_pulse_delay"]) ) # gives station name for every antenna in the combined array
    #                    all_station_polarization_angle.append(station.polarization['xyz']["crp_polarization_angle"])
                    except Exception as e:
                        raise EventError("{0} when attempting to obtain parameters for station {1}".format(e.message, station.stationname))
                else: # station was flagged e.g. manually
                    logging.info('Not using station %s in wavefront, it was flagged.' % station.stationname)

        try:
            all_station_lcr_antenna_position = np.vstack(all_station_lcr_antenna_position)
            all_station_pulse_delays = np.hstack(all_station_pulse_delays)
            all_station_pulse_snr = np.hstack(all_station_pulse_snr)
            all_station_pulse_delays -= all_station_pulse_delays.min() # Subtract global offset
            all_station_fit_residuals = np.hstack(all_station_fit_residuals)
            all_station_fit_residuals = all_station_fit_residuals[0::2] # only one delay per antenna, i.e. same for odd/even rcus
            all_station_pulse_peak_amplitude = np.vstack(all_station_pulse_peak_amplitude)
            all_station_integrated_pulse_power = np.vstack(all_station_integrated_pulse_power)
            all_station_integrated_noise_power = np.vstack(all_station_integrated_noise_power)
            all_station_rms = np.vstack(all_station_rms)
            all_station_direction = np.asarray(all_station_direction)
    #        all_station_polarization_angle = np.hstack(all_station_polarization_angle)
            all_station_antennas_stationnames = np.array(all_station_antennas_stationnames)
            all_station_selected_dipole_ids = np.hstack(all_station_selected_dipole_ids)
            # Convert to contiguous array of correct shape
            shape = all_station_lcr_antenna_position.shape
            all_station_lcr_antenna_position = all_station_lcr_antenna_position.reshape((shape[0] / 2, 2, shape[1]))[:, 0]
            all_station_lcr_antenna_position = all_station_lcr_antenna_position.copy()
        
        except:
            logging.info('Only signals in flagged stations found')
            all_stations_flagged = True
            
        # Calculate average direction and store it
        if not all_stations_flagged:
            average_direction = tools.averageDirectionLOFAR(all_station_direction[:, 0], all_station_direction[:, 1])
            event["crp_average_direction"] = average_direction

        # Beamform with all stations can be done at some point here

        # Collecting LORA parameters for LDF fitting initial values
        partial_lora = False 
        lora_positions = np.array(zip(event["lora_posx"],event["lora_posy"],event["lora_posz"]))
        lora_signals = event["lora_particle_density__m2"]      
        
        try:
            core = list(event["lora_core"])
            core_uncertainties = list(event["lora_coreuncertainties"])          
        except:
            partial_lora = True
            core = [0.,0.,0.]
            core_uncertainties = [0.,0.,0.]

        try:
            corr = event["lora_corcoef_xy"]
            cov = corr*core_uncertainties[0]*core_uncertainties[1]
        except:
            print "Invalid value for correlation coefficient, file probably needs to be updated or incomplete reconstruction"
            cov = 0.
            partial_lora = True
            
        if partial_lora == True:
            print "LORA data not complete."
            
        core_uncertainties[2] = cov
        #Assume a resolution of 3 degrees (conservative assumption)
        direction_uncertainties = [3., 3., 0]
           
        if all_stations_flagged == True:
            if options.debug:
                event.status = "DEBUG_CR_FOUND"
            elif options.debug_test_pulse:
                event.status = "DEBUG_PULSE_FOUND"
            else:
                event.status = "CR_FOUND"
            event.statusmessage = "All stations flagged, no LDF and no wavefront computed"
            
        else: 
            ldf = cr.trun("Shower", positions=all_station_lcr_antenna_position, signals_uncertainties=all_station_rms, core=core, direction=average_direction, timelags=all_station_pulse_delays, core_uncertainties=core_uncertainties, signals=all_station_pulse_peak_amplitude, direction_uncertainties=direction_uncertainties, all_directions=all_station_direction, all_stations=all_station_names, ldf_enable=True, footprint_enable=True, footprint_use_background=True, skyplot_of_directions_enable=True, calibrated=calibrated, lora_direction=lora_direction, lora_positions=lora_positions, lora_signals=lora_signals, save_plots=True, plot_prefix=event_plot_prefix, plot_type=options.plot_type, plotlist=event["crp_plotfiles"])
    
            ldf_total = cr.trun("Shower", positions=all_station_lcr_antenna_position, signals_uncertainties=all_station_rms, core=core, direction=average_direction, core_uncertainties=core_uncertainties, signals=all_station_pulse_peak_amplitude, direction_uncertainties=direction_uncertainties, all_directions=all_station_direction, calibrated=calibrated, all_stations=all_station_names, ldf_enable=True, ldf_total_signal=True, save_plots=True, plot_prefix=event_plot_prefix, plot_type=options.plot_type, plotlist=event["crp_plotfiles"])
    
            ldf_power = cr.trun("Shower", positions=all_station_lcr_antenna_position, signals_uncertainties=all_station_integrated_noise_power, core=core, direction=average_direction, timelags=all_station_pulse_delays, core_uncertainties=core_uncertainties, calibrated=calibrated, signals=all_station_integrated_pulse_power, direction_uncertainties=direction_uncertainties, ldf_enable=True, ldf_integrated_signal=True, ldf_logplot=False, footprint_enable=False, save_plots=True, plot_prefix=event_plot_prefix, plot_type=options.plot_type, plotlist=event["crp_plotfiles"])

            if options.run_polarization and not hba:
                polarization = cr.trun("Polarization", eventid = options.id, event = event, data_path=directory, core = core, core_uncertainties=core_uncertainties, direction=average_direction, save_plots=True, plot_prefix=event_plot_prefix, plot_type=options.plot_type, plotlist=event["crp_plotfiles"], pulse_integration_half_width=2)
                event['polarization_mean_vxvxb'] = polarization.mean_vxvxb
                event['polarization_mean_vxb'] = polarization.mean_vxb
                event['polarization_std_vxvxb'] = polarization.std_vxvxb
                event['polarization_std_vxb'] = polarization.std_vxb
                
#                print 'TEST! Check polarization output parameters!'
#                print '-------------------------------------------'
                event['polarization_stokes_i'] = polarization.stokes_I
#                print 'POLARIZATION STOKES I'
#                print polarization.stokes_I
#                print '------------------------'
                event['polarization_stokes_q'] = polarization.stokes_Q
#                print 'POLARIZATION STOKES Q'
#                print polarization.stokes_Q
#                print '------------------------'
                event['polarization_stokes_u'] = polarization.stokes_U
#                print 'POLARIZATION STOKES U'
#                print polarization.stokes_U
#                print '------------------------'
#                print 'POLARIZATION STOKES V'
                event['polarization_stokes_v'] = polarization.stokes_V
#                print polarization.stokes_V
#                print '------------------------'
                
                event['polarization_stokes_i_uncertainty_mc'] = polarization.stokes_I_uncertainty_MC
                event['polarization_stokes_q_uncertainty_mc'] = polarization.stokes_Q_uncertainty_MC
                event['polarization_stokes_u_uncertainty_mc'] = polarization.stokes_U_uncertainty_MC
                event['polarization_stokes_v_uncertainty_mc'] = polarization.stokes_V_uncertainty_MC
                
                event['polarization_stokes_i_uncertainty'] = polarization.stokes_I_uncertainty_analytic
                event['polarization_stokes_q_uncertainty'] = polarization.stokes_Q_uncertainty_analytic
                event['polarization_stokes_u_uncertainty'] = polarization.stokes_U_uncertainty_analytic
                event['polarization_stokes_v_uncertainty'] = polarization.stokes_V_uncertainty_analytic
                
                event['polarization_angle'] = polarization.polarization_angle

                # Cuts on polarization to flag events as disregardable for simulation
                # These cuts follow from the figure Polarization_inspection.pdf in
                # https://svn.science.ru.nl/repos/lofar_crp/trunk/Sim_Pipeline/images
                if ((polarization.mean_vxvxb > 0.45) | (polarization.std_vxvxb > 0.23)):
                    if event.flagged != True:
                        event.flagged = True
                        event.flagged_reason = 'polarization_outlier'
                else:
                    # Clear status if single stations have been flagged in an earlier iteration
                    if event.flagged_reason == 'polarization_outlier':
                        event.flagged = False # (AC 24 Sep 2020: bugfix. Was set to True here)
                        event.flagged_reason = ''
                        

            if options.run_ldf and not hba:
                ldf_2d = cr.trun("ldf", eventid = options.id, antenna_positions=all_station_lcr_antenna_position, signal_power=all_station_integrated_pulse_power, noise_power=all_station_integrated_noise_power, pulse_direction=average_direction, particle_core=core, particle_direction=lora_direction, particle_densities=lora_signals, save_plots=True, plot_prefix=event_plot_prefix, plot_type=options.plot_type, plotlist=event["crp_plotfiles"],debug=options.debug)
                event['ldf_fit_output'] = ldf_2d.ldf_fit_output
                event['ldf_fit_core'] = ldf_2d.ldf_fit_core
                event['ldf_fit_energy'] = ldf_2d.ldf_fit_energy
                event['ldf_fit_xmax'] = ldf_2d.ldf_fit_xmax
                event['ldf_fit_energy_particle'] = ldf_2d.ldf_fit_energy_particle
                
                
            if options.run_wavefront:
                try:
                    # Plot wavefront shape using arrival times (from all_station_pulse_delays)
                    # filter out all channels with > +/- 3 ns fit residual. This removes all 5-ns glitches as well as bad pulse position estimates.
                    # Later we'll try to correct for glitches by collecting database values of fit residuals.
                    noGlitchIndices = np.where( (abs(all_station_fit_residuals) < 3e-9) )[0] # make plain array, hence index [0]
        #            print 'no glitch indices arE:'
        #            print noGlitchIndices
        #            restrictToSuperterpIndices = []
        #            for index in noGlitchIndices[0]: # HACK to exclude outer core stations.
        #                if all_station_antennas_stationnames[index] in ['CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007', 'CS011', 'CS017']:
        #                    restrictToSuperterpIndices.append(index)
        #            noGlitchIndices = np.array(restrictToSuperterpIndices)
        
                    if len(noGlitchIndices) > 48:
                        
                        full_wavefront_input = dict(eventID=options.id, pulse_delays=all_station_pulse_delays[noGlitchIndices], antenna_positions=all_station_lcr_antenna_position[noGlitchIndices], station_names=all_station_antennas_stationnames[noGlitchIndices], particle_core=core, particle_shower_direction=lora_direction, use_title=False, save_plots=True, plot_prefix=event_plot_prefix, plot_type=options.plot_type, plotlist=event["crp_plotfiles"], plot_publishable=options.publication_quality_plots, pulse_delay_uncertainties=12.65e-9/all_station_pulse_snr[noGlitchIndices], optimize_core_brute_force=options.wavefront_brute_force, optimize_core_brute_force_Ns=11, optimize_core_fixed_to=None, refit_particle=True, particles_lora=lora_signals, add_lora_chi2=True, shower_core=event['ldf_fit_core'],verbose=2)

        
                        pickle.dump(full_wavefront_input, open(event_plot_prefix + "wavefront_input.pickle", "wb"))
                        
                        shower_core = event['ldf_fit_core']
                        if shower_core is not None:
                            hyp_wavefront_input = dict(eventID=options.id, pulse_delays=all_station_pulse_delays[noGlitchIndices], pulse_delay_uncertainties=(12.65e-9/all_station_pulse_snr[noGlitchIndices]), antenna_positions=all_station_lcr_antenna_position[noGlitchIndices], antenna_ids=all_station_selected_dipole_ids[noGlitchIndices], all_antenna_ids_pipeline=all_station_selected_dipole_ids, station_names=all_station_antennas_stationnames[noGlitchIndices], no_glitch_indices=noGlitchIndices, shower_core=shower_core, save_plots=True, plot_prefix=event_plot_prefix, plot_publishable=options.publication_quality_plots, plotlist=event["crp_plotfiles"], verbose=2)

                            pickle.dump(hyp_wavefront_input, open(event_plot_prefix + "hypwavefront_input.pickle", "wb"))
                                
                            print 'Going to save all station timeseries...'
                            all_station_timeseries = np.vstack(all_station_timeseries)
                            outfile = "onsky_pulse_block_timinganalysis-{0}.npy".format(options.id)
                            np.save(os.path.join(directory, outfile), all_station_timeseries)
                            event['extra_files'].append(outfile)
                            print 'Saved on-sky time series block for additional timing analysis.'
                                                                                         
                            wavefront = cr.trun("HypWavefront", **hyp_wavefront_input)
            
                            event["wavefront_fit_output"] = wavefront.wavefront_output
                            event["wavefront_fit_parameters"] = wavefront.fitParams
                            event["wavefront_fit_errors"] = wavefront.fitParamErrors
                            event["wavefront_fit_direction"] = wavefront.fitAzEl
                        else:
                            wavefront = None
                            print "! Cannot run Wavefront task, no core fit position from LDF task..."
                            
                    else:
                        wavefront = None
                        print "No channels with < +/- 3 ns fit residual within core stations, cannot run Wavefront"
                        print "--- all_station_fit_residuals for this event: "
                        print all_station_fit_residuals
                        print "---"
               
                except Exception as e:
                    wavefront = None
                    print "[Wavefront] Error:"
                    print e
                    print "Skipping"
        
            else:
                print "Wavefront deselected, not running wavefront"
                wavefront = None
                
            if options.debug:
                event.status = "DEBUG_CR_FOUND"
            elif options.debug_test_pulse:
                event.status = "DEBUG_PULSE_FOUND"
            else:
                event.status = "CR_FOUND"
                
            event.statusmessage = ""

    else:

        if options.debug:
            event.status = "DEBUG_CR_NOT_FOUND"
        elif options.debug_test_pulse:
            event.status = "DEBUG_PULSE_NOT_FOUND"
        else:
            event.status = "CR_NOT_FOUND"

    # Cleanup large files
    for station in stations:
        if station.status != "GOOD":
            for f in station['extra_files']:
                try:
                    os.remove(os.path.join(directory, f))
                except:
                    print "warning: could not remove file", f

    if event.status != "CR_FOUND":
        for f in event['extra_files']:
            try:
                os.remove(os.path.join(directory, f))
            except:
                print "warning: could not remove file", f

    # ---------- Checking whether an event should be marked for simulation
    # Contains hardcoded choices for cuts 
    
    if event.status == "CR_FOUND":
        if not all_stations_flagged:
            event.simulation_status, event.simulation_statusmessage, refill = simhelp.setSimulationStatus(event.simulation_status,event.simulation_statusmessage, nof_good_stations,event.flagged, hba, average_direction,lora_direction, angle = 10., overwrite=False)
            print event.simulation_status, event.simulation_statusmessage, refill
                  
            # Checking whether LDF fit converged and delivers reasonable results as start values
            if event.simulation_status == "DESIRED":
                if refill:
                    print "Refilling the simulation parameters"
                    ldf_quality = simhelp.checkLDFFitQuality(ldf_2d.ldf_fit_output)
                    event["simulation_xmax"], event["simulation_xmax_reason"] = simhelp.setSimulationXmax(ldf_2d.ldf_fit_output,ldf_quality,average_direction[1])
                    event["simulation_energy"], event["simulation_energy_reason"] = simhelp.setSimulationEnergy(ldf_2d.ldf_fit_energy,ldf_quality,event['lora_energy'])
                    if wavefront:
                        event["simulation_direction"], event["simulation_direction_reason"] = simhelp.setSimulationDirection(options.run_wavefront,wavefront.wavefront_output,wavefront.wavefront_output["fitChi2"],average_direction,wavefront.fitAzEl)
                    else:
                        event["simulation_direction"], event["simulation_direction_reason"] = simhelp.setSimulationDirection(False,None,None,average_direction,[0.,0.])
        else:
            event.simulation_status = "NOT_DESIRED"
            event.simulation_statusmessage = "All stations flagged"                        
    else:
        if not options.debug_test_pulse: # leave simulation settings alone when in debug mode
            event.simulation_status = "NOT_DESIRED"
            event.simulation_statusmessage = "No CR found"

# HACK: disable simulations until GDAS is implemented completely.
#if event.simulation_status == 'DESIRED':
#    event.simulation_status = 'LATER_DESIRED'

print "[cr_physics] completed in {0:.3f} s".format(time.clock() - start)

