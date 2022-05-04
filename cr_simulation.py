"""CR simulation pipeline.
"""

# ./gdastool --observatory=lofar --utctimestamp=1343951431 --gdaspath=/vol/astro7/lofar/gdas --output=ATMOSPHERE_TESTGDAS.DAT


import logging

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG)

import os
import sys
import time
import datetime
import pytmf
import subprocess 
import pycrtools as cr
import numpy as np

from pycrtools import crdatabase as crdb
from pycrtools import simhelp

from pycrtools.tasks import xmaxmethod

from optparse import OptionParser
from contextlib import contextmanager

class PipelineError(Exception):
    """Base class for pipeline exceptions."""

    def __init__(self, message, category="OTHER"):
        self.message = message
        self.category = category

class EventError(PipelineError):
    """Raised when an unhandlable error occurs at event level."""
    pass

class Skipped(PipelineError):
    """Base class for everything that needs to lead to a skipped state."""
    pass

class EventSkipped(Skipped):
    """Raised when event is skipped."""
    pass

@contextmanager
def process_event(event):
    start = time.clock()

    print "-- event {0}".format(event._id)

    try:
        yield event
    except EventSkipped as e:
        logging.info("event skipped because: {0}".format(e.message))

        event.simulation_status = "SKIPPED"
        event.simulation_statusmessage = e.message
    except EventError as e:
        logging.exception(e.message)
        event.simulation_status = "ERROR"
        event.simulation_statusmessage = e.message
    except Exception as e:
        event.simulation_status = "ERROR"
        event.simulation_statusmessage = e.message
        raise
    except BaseException as e:
        event.simulation_status = "ERROR"
        event.simulation_statusmessage = "sigterm recieved"
        raise
    finally:
        event["simulation_last_processed"] = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        event.write()
        print "-- event {0} completed in {1:.3f} s".format(event._id, time.clock() - start)

# Parse commandline options
parser = OptionParser()
parser.add_option("-i", "--id", type="int", help="event ID", default=1)
parser.add_option("-d", "--database", default="crdb.sqlite", help="filename of database")
parser.add_option("--host", default=None, help="PostgreSQL host.")
parser.add_option("--user", default=None, help="PostgreSQL user.")
parser.add_option("--password", default=None, help="PostgreSQL password.")
parser.add_option("--dbname", default=None, help="PostgreSQL dbname.")
parser.add_option("--simdir", default="/vol/astro7/lofar/sim/pipeline", help="Simulation output directory")
parser.add_option("--plot-directory", default="/vol/astro3/lofar/vhecr/lora_triggered/results", help="Plot output directory")
parser.add_option("--iteration", type=int, default=-1, help="Iteration")
parser.add_option("--azimuth", type=float, default=None, help="If given use this azimuth, otherwise fall back to database value (in degrees)")
parser.add_option("--elevation", type=float, default=None, help="If given use this elevation, otherwise fall back to database value (in degrees)")
parser.add_option("--energy", type=float, default=None, help="If given use this energy, otherwise fall back to database value (in eV)")
parser.add_option("--xmax", type=float, default=None, help="If given use this xmax, otherwise fall back to database value")
parser.add_option("--nof-conex-proton", type=int, default=150)
parser.add_option("--nof-conex-iron", type=int, default=50)
parser.add_option("--showers-around-xmax-estimate", type=int, default=11, help="Number of showers to simulate in a narrow region around the estimated Xmax. Default 11.")
parser.add_option("--width-around-xmax-estimate", type=float, default=20.0, help="Width of densely simulated region around Xmax estimate. Default 20 g/cm2")
parser.add_option("--no-atmosphere", default=False, action="store_true")
parser.add_option("--skip-conex", default=False, action="store_true")
parser.add_option("--skip-coreas", default=False, action="store_true")
parser.add_option("--skip-analysis", default=False, action="store_true")
parser.add_option("--ignore-suspended-jobs", default=False, action="store_true", help="Ignore suspended jobs in queue, i.e. submit duplicate jobs for these (if no other jobs RUNNING or PENDING for the same event)")

parser.add_option("--hadronic-interaction-model", type=str, default="QGSII")
(options, args) = parser.parse_args()

db_filename = options.database
dbManager = crdb.CRDatabase(db_filename, host=options.host, user=options.user, password=options.password, dbname=options.dbname)
db = dbManager.db

valid_status = ["DESIRED", "CONEX_STARTED", "CONEX_DONE", "COREAS_STARTED", "COREAS_DONE"]

print "skipping conex:", options.skip_conex
print "skipping coreas:", options.skip_coreas
print "skipping analysis:", options.skip_analysis

# Ignore if simulations are already scheduled
if options.id in simhelp.running_jobs(ignore_suspended=options.ignore_suspended_jobs):
    print "Event {0} already scheduled, skipping...".format(options.id)
    sys.exit(0)

# Get event from database and run pipeline on it
with process_event(crdb.Event(db=db, id=options.id)) as event:

    if event.simulation_status not in valid_status:
        sys.exit(1)

    # Use options
    az = options.azimuth
    if az is None:
        az = event["simulation_direction"][0]
    az = pytmf.deg2rad(90. - az) # LOFAR convention has east = 0, north = 90 deg. Convert to 0=North, 90=East for further use
    
    zen = options.elevation
    if zen is None:
        zen = event["simulation_direction"][1]
    zen = pytmf.deg2rad(90. - zen)
    
    energy = options.energy
    if energy is None:
        energy = event["simulation_energy"]
    
    xmax = options.xmax
    if xmax is None:
        xmax = event["simulation_xmax"]
    energy /= 1.e9

    if options.iteration >= 0 and event.simulation_status=="DESIRED":
        # AC: added condition to only reset iteration before start of simulations ('DESIRED' state)
        iteration = options.iteration
        event["simulation_current_iteration"] = iteration
    else:
        try:
            iteration = event["simulation_current_iteration"]
        except:
            iteration = 0
            event["simulation_current_iteration"] = iteration

    atmosphere_file = None # if not changed below
    if not options.no_atmosphere: # Get GDAS atmosphere profile if needed
        atmosphere_file = "/vol/astro7/lofar/sim/pipeline/atmosphere_files/ATMOSPHERE_{0}.DAT".format(int(options.id))
        #atmosphere_file = "ATMOSPHERE_{0}.DAT".format(int(options.id)) # Without directory to work around Fortran upper case bug
        
        if os.path.exists(atmosphere_file):
            print "Found atmosphere file: %s" % atmosphere_file
        else: # # Run gdastool to extract info from GDAS and produce file
            utctimestamp = int(options.id) + 1262304000 # Add timestamp of Jan 1, 2010 to get Unix UTC timestamp
            runCommand = 'python /vol/optcoma/cr-simulations/corsika_production/src/utils/gdastool --observatory=lofar --utctimestamp={0} --output=/vol/astro7/lofar/sim/pipeline/atmosphere_files/ATMOSPHERE_{1}.DAT --gdaspath=/vol/astro7/lofar/gdas'.format(utctimestamp, int(options.id))
            #runCommand = 'python /vol/optcoma/cr-simulations/corsika_final_75700/gdastool --observatory=lofar --utctimestamp={0} --output={1}.DAT --gdaspath=/vol/astro7/lofar/gdas'.format(utctimestamp, int(options.id))

            print 'Running command: %s' % runCommand
            process = subprocess.Popen([runCommand], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            output, error = process.communicate()
            if process.returncode != 0:
                print "Error running subprocess!"
                print output
                print error
                event.simulation_status = "ERROR"
                event.simulation_statusmessage = "Cannot get GDAS profile"
                sys.exit(1)
            else:
                print output

    event_output_directory = os.path.join(options.simdir, "events/{0}/{1}".format(event.id, iteration))

    # Setup first run parameters
    if event.simulation_status == "DESIRED":
        event["simulation_nof_scheduled"] = {'conex_proton' : 0, 'conex_iron' : 0, 'coreas_proton' : 0, 'coreas_iron' : 0}

    # Check holds
    if "simulation_cleared_holds" not in event.parameter.keys() or not isinstance(event["simulation_cleared_holds"], list):
        event["simulation_cleared_holds"] = []

    if not "energy" in event["simulation_cleared_holds"] and (energy < 1.e16 / 1.e9 or energy > 3.e18 / 1.e9):
        event.simulation_status = "HOLD"
        event.simulation_statusmessage = "improbable energy, manual check needed"

#    if not "xmax" in event["simulation_cleared_holds"] and (xmax < 550. or xmax > 1000.):
#        event.simulation_status = "HOLD"
#        event.simulation_statusmessage = "improbable xmax, manual check needed"
# let it run, if too many proton / iron conex showers without getting in range, then set to 'scatter'
    ska_pattern = (event.status == "SKA") # use N=208 antenna pattern with more coverage near the core, for SKA-simulations
    ska_site_params = (event.status == "SKA")
    if ska_pattern:
        print "  !! Using SKA antenna pattern for this event."

    # Run CONEX
    if (event.simulation_status == "DESIRED" or event.simulation_status == "CONEX_STARTED") and not options.skip_conex:
        if event["simulation_xmax_reason"] == "scatter":
            nevents_proton, nevents_iron, proton_ok, iron_ok, runno_proton, runno_iron = simhelp.check_conex(event_output_directory + "/conex/", 650.0, n_around_x_estimate=0, n_proton=25, n_iron=15)
        else:
            nevents_proton, nevents_iron, proton_ok, iron_ok, runno_proton, runno_iron = simhelp.check_conex(event_output_directory + "/conex/", xmax, n_around_x_estimate=options.showers_around_xmax_estimate, width_around_x_estimate=options.width_around_xmax_estimate)

        print nevents_proton, nevents_iron, proton_ok, iron_ok, runno_proton, runno_iron

        if not proton_ok:
            # Safety check
            if not event["simulation_xmax_reason"] == "scatter" and event["simulation_nof_scheduled"]["conex_proton"] > 3 * options.nof_conex_proton:
                event["simulation_xmax_reason"] = "scatter"
                # Set to scatter, don't raise ERROR state
                #raise EventError("Too many conex proton simulations requested, Xmax still out of range")

            print "Scheduling CONEX proton simulations for event", event.id

            simhelp.generate_cr_simulation(event.id, event_output_directory + "/conex/", 'proton', energy, az, zen, runnum=range(event["simulation_nof_scheduled"]["conex_proton"], event["simulation_nof_scheduled"]["conex_proton"] + options.nof_conex_proton), atmosphere_file=atmosphere_file, conex=True, ska_pattern=ska_pattern, ska_site_params=ska_site_params, h_model=options.hadronic_interaction_model)

            event["simulation_nof_scheduled"]["conex_proton"] += options.nof_conex_proton
            event.simulation_status = "CONEX_STARTED"
            event.simulation_statusmessage = ""

        if not iron_ok:
            # Safety check
            if not event["simulation_xmax_reason"] == "scatter" and event["simulation_nof_scheduled"]["conex_iron"] > 3 * options.nof_conex_iron:
                event["simulation_xmax_reason"] = "scatter"
                #raise EventError("Too many conex iron simulations requested, Xmax still out of range")

            print "Scheduling CONEX iron simulations for event", event.id

            simhelp.generate_cr_simulation(event.id, event_output_directory + "/conex/", 'iron', energy, az, zen, runnum=range(event["simulation_nof_scheduled"]["conex_iron"], event["simulation_nof_scheduled"]["conex_iron"] + options.nof_conex_iron), atmosphere_file=atmosphere_file, conex=True, ska_pattern=ska_pattern, ska_site_params=ska_site_params, h_model=options.hadronic_interaction_model)

            event["simulation_nof_scheduled"]["conex_iron"] += options.nof_conex_iron
            event.simulation_status = "CONEX_STARTED"
            event.simulation_statusmessage = ""

        if proton_ok and iron_ok:
            event.simulation_status = "CONEX_DONE"
            #event.simulation_statusmessage = ""
                

    # Run CoREAS
    if (event.simulation_status == "CONEX_DONE" or event.simulation_status == "COREAS_STARTED") and not options.skip_coreas:
        if event["simulation_xmax_reason"] == "scatter":
            nevents_proton, nevents_iron, proton_ok, iron_ok, t_runno_proton, t_runno_iron = simhelp.check_conex(event_output_directory + "/conex/", xmax, n_around_x_estimate=0, n_proton=25, n_iron=15)
        else:
            nevents_proton, nevents_iron, proton_ok, iron_ok, t_runno_proton, t_runno_iron = simhelp.check_conex(event_output_directory + "/conex/", xmax, n_around_x_estimate=options.showers_around_xmax_estimate, width_around_x_estimate=options.width_around_xmax_estimate)

        print nevents_proton, nevents_iron, proton_ok, iron_ok, t_runno_proton, t_runno_iron

        # Check if simulations are complete
        runno_proton = []
        for i in t_runno_proton:
            if not simhelp.simulation_complete(event.id, i, "proton", iteration):
                print "event", event.id, "simulation", i, "proton", "NOT COMPLETE"
                runno_proton.append(i)
            else:
                print "event", event.id, "simulation", i, "proton", "COMPLETE"

        runno_iron = []
        for i in t_runno_iron:
            if not simhelp.simulation_complete(event.id, i, "iron", iteration):
                print "event", event.id, "simulation", i, "iron", "NOT COMPLETE"
                runno_iron.append(i)
            else:
                print "event", event.id, "simulation", i, "iron", "COMPLETE"

        nof_split_runs = 2 # Strategy to split antenna list into N parts, to have one shower taking < 48h on one core. Then, can run in the 'normal' queue at normal priority.
        if np.log10(energy) > 17.7 - 9:
            nof_split_runs = 3
        if np.log10(energy) > 18.2 - 9:
            nof_split_runs = 4 # These numbers should be on the conservative side (Coma)...
        
        elevation_deg = event["simulation_direction"][1]
        print 'Elevation is %3.2f deg' % elevation_deg
        if elevation_deg < 30.0:
            nof_split_runs *= 2 # Takes longer for inclined showers... Twice as many parts just to be sure.
        if ska_pattern:
            nof_split_runs += 1 # just in case


        if len(runno_proton) > 0:
            print "Scheduling CoREAS proton simulations for event", event.id
            print "Splitting CoREAS antenna list into %d parts" % nof_split_runs
            simhelp.generate_cr_simulation(event.id, event_output_directory + "/coreas/", 'proton', energy, az, zen, runno_proton, atmosphere_file=atmosphere_file, nof_split_runs=nof_split_runs, ska_pattern=ska_pattern, ska_site_params=ska_site_params, h_model=options.hadronic_interaction_model)
        if len(runno_iron) > 0:
            print "Scheduling CoREAS iron simulations for event", event.id
            print "Splitting CoREAS antenna list into %d parts" % nof_split_runs
            simhelp.generate_cr_simulation(event.id, event_output_directory + "/coreas/", 'iron', energy, az, zen, runno_iron, atmosphere_file=atmosphere_file, nof_split_runs=nof_split_runs, ska_pattern=ska_pattern, ska_site_params=ska_site_params, h_model=options.hadronic_interaction_model)

        if len(runno_proton) == 0 and len(runno_iron) == 0:
            event.simulation_status = "COREAS_DONE"
            #event.simulation_statusmessage = ""
        else:
            event.simulation_status = "COREAS_STARTED"
            #event.simulation_statusmessage = ""
            event["simulation_nof_scheduled"]["coreas_proton"] += len(runno_proton)
            event["simulation_nof_scheduled"]["coreas_iron"] += len(runno_iron)

    # Run Analysis
    if (event.simulation_status == "COREAS_DONE") and not options.skip_analysis:

        event_plot_prefix = os.path.join(options.plot_directory, "{0}/cr_simulation-{0}-".format(options.id))

        xmaxmethod = cr.trun("XmaxMethod", eventid=options.id, event=event, simulation_directory=event_output_directory, plotlist=event["crp_plotfiles"], plot_prefix=event_plot_prefix)

    print event.simulation_status

