"""
Module for interfacing with CoREAS. Used by ''cr_simulation.py''.

.. moduleauthor:: Pim Schellart <P.Schellart@astro.ru.nl>
.. moduleauthor:: Arthur Corstanje <A.Corstanje@astro.ru.nl>

"""

import os
import re
import glob
import subprocess
import numpy as np
import pytmf
import time 

def running_jobs(ignore_suspended=False):
    """Returns a list of jobs that are currently running in the scheduler.
    
    """

    if ignore_suspended:
        print '### Ignoring SUSPENDED jobs in queue'
        o1 = subprocess.check_output(['/usr/local/slurm/bin/squeue', '-tRUNNING', '-o "%.40j'])
        o2 = subprocess.check_output(['/usr/local/slurm/bin/squeue', '-tPENDING', '-o "%.40j'])
        o = o1 + o2
    else:
        o = subprocess.check_output(['/usr/local/slurm/bin/squeue', '-o "%.40j'])

    j = re.compile(r"([0-9]+)_((coreas)|(conex))_((iron)|(proton))\.q")
    
    pipeline_jobs = []
    for line in o.split():
        m = j.search(line)
    
        if m:
            pipeline_jobs.append(int(m.group(1)))
    
    return list(set(pipeline_jobs))


def simulation_complete(eventID, runno, particle_type, iteration):
    """Returns the completeness of the requested simulation as *True* or *False*.
    
    **Properties**
    
    ============== ===================================================================
    Parameter      Description
    ============== ===================================================================
    *eventID*      event id from the database
    *runno*        run number from the scheduler
    *particle_type numerical type of the primary particle in CORSIKA notation
    *iteration*    iteration of the simulation
    ============== ===================================================================
    
    """
    #os.path.isfile(datadir+"/DAT{0}.lora".format(str(runno).zfill(6))) and
    # Temporarily removed; LORA simulation postponed (Dec 2017)
    datadir="/vol/astro7/lofar/sim/pipeline/events/{0}/{1}/coreas/{2}".format(eventID, iteration, particle_type)
    if (os.path.isdir(datadir+"/SIM{0}_coreas".format(str(runno).zfill(6))) and
            os.path.isfile(datadir+"/DAT{0}.long".format(str(runno).zfill(6))) and
            os.path.isfile(datadir+"/steering/RUN{0}.inp".format(str(runno).zfill(6))) and
            os.path.isfile(datadir+"/steering/SIM{0}.list".format(str(runno).zfill(6)))):
        return True
    return False


def check_conex(datadir, x_estimate, n_around_x_estimate=11, width_around_x_estimate=20.0, n_proton=11, n_iron=5):
    """Check if CONEX simulations are complete.

    Selects *n_around_x_estimate* simulations around the *x_estimate* value.
    Additionally *n_proton* and *n_iron* simulations are selected from their full
    respective ranges.

    **Properties**
    
    ===================== =====================================================
    Parameter             Description
    ===================== =====================================================
    *datadir*             directory where simulations are located
    *x_estimate*          X_max estimate arround which simulations are selected 
    *n_around_x_estimate* number of simulations to select around X_max estimate
    *n_proton*            number of proton simulations to select
    *n_iron*              number of iron simulations to select
    ===================== =====================================================
    
    **Returns**

    ``(nevents_proton, nevents_iron, proton_ok, iron_ok, runno_proton, runno_iron)``
    
    =============== ==============================================================
    Field           Description
    =============== ==============================================================
    nevents_proton  number of completed proton simulations
    nevents_iron    number of completed iron simulations
    proton_ok       do we have sufficient (i.e. *n_proton*) proton simulations
    iron_ok         do we have sufficient (i.e. *n_iron*) iron simulations
    runno_proton    what are the runnumbers of the selected proton simulations
    runno_iron      what are the runnumbers of the selected iron simulations
    =============== ==============================================================

    """

    print "checking datadir", datadir
    print "with x_estimate", x_estimate

    if not os.path.exists(datadir):
        return 0, 0, False, False, None, None 

    proton_sims = glob.glob(("{0}/proton/DAT??????.long").format(datadir))
    iron_sims = glob.glob(("{0}/iron/DAT??????.long").format(datadir))

    nevents_proton = len(proton_sims)
    nevents_iron = len(iron_sims)

    if nevents_proton == 0 or nevents_iron == 0:
        return nevents_proton, nevents_iron, False, False, None, None 

    all_proton_runno = np.zeros(nevents_proton, dtype='int64')
    all_proton_xmax = np.zeros(nevents_proton, dtype='float64')

    for i, name in enumerate(proton_sims):
        m = re.search('DAT([0-9]+).long', name)
        all_proton_runno[i] = int(m.group(1))
        all_proton_xmax[i] = np.genfromtxt(re.findall("PARAMETERS.*",open(name,'r').read()))[4]
    
    # Conex / Corsika Bug workaround: Xmax can be >> 1000 sometimes, and run is invalid.
    # Possibly because extreme-Xmax showers are not sampled beyond 1056 g/cm2 of depth.

    selection_indices = np.where(all_proton_xmax < 1030)
    if len(selection_indices[0]) < len(all_proton_xmax):
        print 'WARNING: %d proton Conex showers discarded due to invalid Xmax (> 1030)' % (len(all_proton_xmax) - len(selection_indices[0]))
        all_proton_runno = all_proton_runno[selection_indices]
        all_proton_xmax = all_proton_xmax[selection_indices]
        proton_sims = list(np.array(proton_sims)[selection_indices])
        nevents_proton = len(proton_sims)
    # end workaround

    all_iron_runno = np.zeros(nevents_iron, dtype='int64')
    all_iron_xmax = np.zeros(nevents_iron, dtype='float64')

    for i, name in enumerate(iron_sims):
        m = re.search('DAT([0-9]+).long', name)
        all_iron_runno[i] = int(m.group(1))
        all_iron_xmax[i] = np.genfromtxt(re.findall("PARAMETERS.*",open(name,'r').read()))[4]

    #selection array, initialized at zero
    select_proton = np.zeros(nevents_proton, dtype=bool)
    select_iron = np.zeros(nevents_iron, dtype=bool)
    
    x_aim = x_estimate + np.linspace(-width_around_x_estimate, +width_around_x_estimate, n_around_x_estimate) # default 20 g/cm2 window around Xmax estimate
    for i in np.arange(n_around_x_estimate):
        # select n_around_x_estimate showers in a range [-20:20] around estimated value
        nearest_proton = np.argmin(np.abs(all_proton_xmax - x_aim[i]))
        select_proton[nearest_proton] = True
        nearest_iron = np.argmin(np.abs(all_iron_xmax - x_aim[i]))
        select_iron[nearest_iron] = True

    x_aim = np.linspace(np.min(all_proton_xmax), np.max(all_proton_xmax), n_proton)
    for i in np.arange(n_proton):
        # select n_proton showers in a range [Xmin:Xmax]
        rng_proton = np.max(all_proton_xmax) - np.min(all_proton_xmax)
        nearest = np.argmin(np.abs(all_proton_xmax - x_aim[i]))
        select_proton[nearest] = True
    
    x_aim = np.linspace(np.min(all_iron_xmax), np.max(all_iron_xmax), n_iron)
    for i in np.arange(n_iron):
        # select n_iron showers in a range [Xmin:Xmax]
        rng_iron = np.max(all_iron_xmax) - np.min(all_iron_xmax)
        nearest = np.argmin(np.abs(all_iron_xmax - x_aim[i]))
        select_iron[nearest] = True
    
    x_selection_proton = all_proton_xmax[select_proton]
    x_selection_iron = all_iron_xmax[select_iron]
    
    no_of_sims_proton = len(x_selection_proton)
    no_of_sims_iron = len(x_selection_iron)
    no_of_sims = no_of_sims_proton + no_of_sims_iron
    no_of_sims_below_x_est = len(x_selection_proton[x_selection_proton < x_estimate]) + len(x_selection_iron[x_selection_iron < x_estimate])
    no_of_sims_above_x_est = len(x_selection_proton[x_selection_proton > x_estimate]) + len(x_selection_iron[x_selection_iron > x_estimate])
    
    # make list of sim ids to run full CORSIKA, for example:
    iron_ok = False
    proton_ok = False
    
    # The number of proton / iron simulations in the selection needs to be 0.5 times the number of target X-values.
    # In general, multiple target X values can have the same selected shower.
    if (no_of_sims_below_x_est > 2) and (no_of_sims_iron > 0.5 * n_iron):
        iron_ok = True
    if (no_of_sims_above_x_est > 2) and (no_of_sims_proton > 0.5 * n_proton):
        proton_ok = True
    
    runno_proton = all_proton_runno[select_proton]
    runno_iron = all_iron_runno[select_iron]
    
    print x_selection_proton
    print x_selection_iron

    return nevents_proton, nevents_iron, proton_ok, iron_ok, runno_proton, runno_iron


def create_antenna_position_file(eventID, az, zen, length_meters=500.0, number_of_arms=8, arm_length=20, ska_pattern=False, ska_site_params=False):
    # ska_pattern=True: an N=208 antenna pattern with extra coverage near the core, and less coverage further out (r > 200 m in shower plane)
    """Creates a file with antenna positions for input into CoREAS.
        
        **Properties**
        
        ============== ===================================================================
        Parameter      Description
        ============== ===================================================================
        *eventID*      event id from the database
        *az*           arrival direction azimuth (Measured from East towards North in degrees)
        *zen*          arrival direction zenith (Zenith angle in degrees)
        ============== ===================================================================
        
        """
    
    simdir = '/vol/astro7/lofar/sim/pipeline'
    listname = simdir+'/run/SIM{0}.list'.format(eventID)
    f = open(listname, 'w')
    
    # Set LOFAR-site parameters, except when ska_site_params==True
    if not ska_site_params:
        inc = 67.8 / 180.0 * np.pi
        altitude_in_cm = 760.0
    else:
        inc = np.arctan(-48.27 / 27.6) # ~ -1.05138
        altitude_in_cm = 46000.0

    B = np.array([0, np.cos(inc), -np.sin(inc)]) #LOFAR coordinates
    v = np.array([-np.cos(az)*np.sin(zen), -np.sin(az)*np.sin(zen), -np.cos(zen)])
    vxB = np.array([v[1]*B[2]-v[2]*B[1], v[2]*B[0]-v[0]*B[2], v[0]*B[1]-v[1]*B[0]])
    vxB = vxB / np.linalg.norm(vxB)
    vxvxB = np.array([v[1]*vxB[2]-v[2]*vxB[1], v[2]*vxB[0]-v[0]*vxB[2], v[0]*vxB[1]-v[1]*vxB[0]])
    
    if not ska_pattern:
        meters_step_radial = length_meters / arm_length
        degrees_step = 360.0 / number_of_arms
        radians_step = degrees_step * np.pi/180.0
        
        for i in np.arange(1, 1+arm_length):
            for j in np.arange(number_of_arms):
                xyz = i * meters_step_radial * (np.cos(j*radians_step) * vxB + np.sin(j*radians_step) * vxvxB)
                c = xyz[2] / v[2]
                name="pos_{0}_{1}".format(int(np.round(i*meters_step_radial)), int(np.round(j*degrees_step)))
                f.write('AntennaPosition = {0} {1} {2} {3}\n'.format(100*(xyz[1]-c*v[1]), -100*(xyz[0]-c*v[0]), altitude_in_cm, name))
    else: # SKA footprint pattern, N=208
        radius = np.linspace(12.5, 200.0, 16)
        radius = np.concatenate( (np.array([12.5/8, 12.5/4, 12.5/2]), radius))
        radius_ext = np.array([225.0, 250.0, 275.0, 300.0, 350.0, 400.0, 500.0])
        radius = np.concatenate( (radius, radius_ext))
        print radius
        print '---'
        print len(radius)
        print '---'
        degrees_step = 360.0 / number_of_arms
        radians_step = degrees_step * np.pi/180.0
        for R in radius:
            for j in np.arange(number_of_arms):
                xyz = R * (np.cos(j*radians_step) * vxB + np.sin(j*radians_step) * vxvxB)
                c = xyz[2] / v[2]
                name="pos_{0}_{1}".format(int(np.round(R)), int(np.round(j*degrees_step)))
                f.write('AntennaPosition = {0} {1} {2} {3}\n'.format(100*(xyz[1]-c*v[1]), -100*(xyz[0]-c*v[0]), altitude_in_cm, name))

    f.close()

def split_antenna_position_file(eventID, Nparts=None):
    # Split antenna positions file by lines, in Nparts.
    # Also works when Nparts does not divide the number of antennas.
    # Outputs files SIMxxxxxx_Part1.list, for part 1..N

    simdir = '/vol/astro7/lofar/sim/pipeline'
    listname = simdir+'/run/SIM{0}.list'.format(eventID)
    f = open(listname, 'r')
    lines = f.readlines()
    f.close()
    Nlines = len(lines)
    
    test = []
    start = 0
    end = 0
    i = 0 # this is not the shortest way to do it, but it works and this isn't a job interview...
    while end < Nlines:
        start = end
        i += 1
        end = (i * Nlines) / Nparts
        test.extend(lines[start:end])
        filename = simdir + '/run/SIM{0}_Part{1}.list'.format(eventID, i)
        split_file = open(filename, 'w')
        for line in lines[start:end]:
            split_file.write(line)
        split_file.close()
    
    assert test == lines


#def writefile(name, eventID, energy, az, zen, particle_type, simdir, atmosphere_file=None, conex=False, h_model='QGSII'):

def writefile_CONEX(name, eventID, energy, az, zen, particle_type, simdir, atmosphere_file=None, h_model='QGSII', SKA=False):

    """Creates a CONEX input file. Do not allow for splitting in CoREAS antenna list
    
    **Properties**
    
    =============== =======================================================================
    Parameter       Description
    =============== =======================================================================
    *name*          filename
    *eventID*       event id from the database
    *energy*        energy in GeV
    *az*            arrival direction azimuth (Measured from East towards North in radians)
    *zen*           arrival direction zenith (Zenith angle in radians)
    *particle_type* particle type in CORSIKA numerical notation
    *simdir*        directory containing simulations
    *h_model*       hadronic interaction model: QGSII (default), EPOS or SIBYLL)
    =============== =======================================================================
    
    """

    seed=(int(eventID) * int(particle_type)) % 100000
    scratchdir = '/scratch/crsim_pipeline/{0}/{1}/'.format(eventID, int(particle_type))
    f = open(name, 'w')
    f.write('#! /bin/bash\n')
    f.write('#SBATCH --time=03:00:00\n')
    f.write('#SBATCH --output /vol/astro7/lofar/sim/pipeline/run/output/{0}_conex_{1}-%j\n'.format(eventID, particle_type))
    f.write('#SBATCH --error /vol/astro7/lofar/sim/pipeline/run/output/{0}_conex_{1}-ERROR-%j\n'.format(eventID, particle_type))

    f.write('\n')
    f.write('hostname\n')
    f.write('umask 002\n')
    #    f.write('use geant\n')
    f.write('. /vol/optcoma/geant4_9.6_install/share/Geant4-9.6.4/geant4make/geant4make.sh\n')
    f.write('export RUNNR=`printf "%06d" $SLURM_ARRAY_TASK_ID`\n')
    f.write('export FLUPRO=/vol/optcoma/cr-simulations/fluka64\n')
    f.write('cd /vol/astro7/lofar/sim/pipeline/run/\n')
    f.write('mkdir -p {0}steering/\n'.format(simdir))

    f.write('rm -rf {0}$RUNNR\n'.format(scratchdir))
    f.write('mkdir -p {0}$RUNNR\n'.format(scratchdir))
    #f.write('chmod -R 770 /scratch/crsim_pipeline\n')
    atmosphere_options = ""
    if atmosphere_file is not None:
        atmosphere_options = "--atmosphere --atmfile={0}".format(atmosphere_file)

    ska_flag = "--ska" if SKA else ""
    f.write('python /vol/optcoma/pycrtools/src/PyCRTools/extras/geninp.py {6} {7} --conex -r $RUNNR -s {0} -u {1} -a {2} -z {3} -t {4} -d {5}$RUNNR/ > {5}$RUNNR/RUN$RUNNR.inp\n'.format(seed, energy, 180.*az/np.pi, 180*zen/np.pi, particle_type, scratchdir, atmosphere_options, ska_flag))

    source_REAS = "SIM.reas" if not SKA else "SIM_SKA.reas"
    f.write('cp /vol/astro7/lofar/sim/pipeline/run/{0} {1}$RUNNR/SIM$RUNNR.reas\n'.format(source_REAS, scratchdir))
    f.write('cp /vol/astro7/lofar/sim/pipeline/run/SIM{0}.list {1}$RUNNR/SIM$RUNNR.list\n'.format(eventID, scratchdir))

    f.write('cd /vol/optcoma/cr-simulations/corsika_production/run\n')
    f.write('./corsika77100Linux_{1}_fluka_thin_conex_coreas < /{0}$RUNNR/RUN$RUNNR.inp\n'.format(scratchdir, h_model))

    f.write('cd {0}$RUNNR\n'.format(scratchdir))
    f.write('mv RUN$RUNNR.inp {0}steering/RUN$RUNNR.inp\n'.format(simdir))
    f.write('mv *.long {0}\n'.format(simdir))
    # (AC, Jan 2020) NOT NEEDED for CONEX??? f.write('cp -r * {0}\n'.format(simdir)) # mv * fails if the target directory (simdir) already exists and is not empty
    f.write('rm -rf {0}$RUNNR/*\n'.format(scratchdir))
    f.close()


def writefile_CoREAS(name, eventID, energy, az, zen, particle_type, simdir, atmosphere_file=None, nof_split_runs=1, h_model='QGSII', SKA=False):

    """Creates a CoREAS input file.
    
    **Properties**
    
    =============== =======================================================================
    Parameter       Description
    =============== =======================================================================
    *name*          filename
    *eventID*       event id from the database
    *energy*        energy in GeV
    *az*            arrival direction azimuth (Measured from East towards North in radians)
    *zen*           arrival direction zenith (Zenith angle in radians)
    *particle_type* particle type in CORSIKA numerical notation
    *simdir*        directory containing simulations
    *h_model*       hadronic interaction model: QGSII (default), EPOS or SIBYLL)
    =============== =======================================================================
    
    """

    seed=(int(eventID) * int(particle_type)) % 100000
    scratchdir='/scratch/crsim_pipeline/{0}/{1}/'.format(eventID, int(particle_type))
    f = open(name, 'w')
    f.write('#! /bin/bash\n')
    f.write('#SBATCH --time=1-23:59:00\n')
    f.write('#SBATCH --output /vol/astro7/lofar/sim/pipeline/run/output/{0}_coreas_{1}-%j\n'.format(eventID, particle_type))
    f.write('#SBATCH --error /vol/astro7/lofar/sim/pipeline/run/output/{0}_coreas_{1}-ERROR-%j\n'.format(eventID, particle_type))
    f.write('#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL\n') # Want to get a mail if 90 and 100% of timelimit is reached, or other error occurs.
    f.write('#SBATCH --mail-user=a.corstanje@astro.ru.nl\n') # Change mail address accordingly.
    f.write('#SBATCH -x coma02\n')

    if nof_split_runs > 1:
        f.write('#SBATCH -N 1 -n %d\n' % nof_split_runs) # reserve # cores on same node for split runs

    f.write('\n')
    f.write('hostname\n')
    f.write('umask 002\n')
#    f.write('use geant\n')
    f.write('. /vol/optcoma/geant4_9.6_install/share/Geant4-9.6.4/geant4make/geant4make.sh\n')
    f.write('export RUNNR=`printf "%06d" $SLURM_ARRAY_TASK_ID`\n')
    f.write('export FLUPRO=/vol/optcoma/cr-simulations/fluka64\n')
    f.write('cd /vol/astro7/lofar/sim/pipeline/run/\n')
    f.write('mkdir -p {0}steering/\n'.format(simdir))

    # Loop over split simulations
    f.write('for i in {1..%d}; do\n' % nof_split_runs)

    scratchdir_split = '/scratch/crsim_pipeline_Part$i/{0}/{1}/'.format(eventID, int(particle_type))
    
    f.write('    rm -rf {0}$RUNNR\n'.format(scratchdir_split))
    f.write('    mkdir -p {0}$RUNNR\n'.format(scratchdir_split))
    f.write('    chmod -R 770 /scratch/crsim_pipeline_Part$i\n')

    atmosphere_options = ""
    if atmosphere_file is not None:
        atmosphere_options = "--atmosphere --atmfile={0}".format(atmosphere_file)
    ska_flag = "--ska" if SKA else ""

    f.write('    python /vol/optcoma/pycrtools/src/PyCRTools/extras/geninp.py {6} {7} -r $RUNNR -s {0} -u {1} -a {2} -z {3} -t {4} -d {5}$RUNNR/ > {5}$RUNNR/RUN$RUNNR.inp\n'.format(seed, energy, 180.*az/np.pi, 180*zen/np.pi, particle_type, scratchdir_split, atmosphere_options, ska_flag))

    source_REAS = "SIM.reas" if not SKA else "SIM_SKA.reas"
    f.write('    cp /vol/astro7/lofar/sim/pipeline/run/{0} {1}$RUNNR/SIM$RUNNR.reas\n'.format(source_REAS, scratchdir_split))
    f.write('    cp /vol/astro7/lofar/sim/pipeline/run/SIM{0}_Part$i.list {1}$RUNNR/SIM$RUNNR.list\n'.format(eventID, scratchdir_split)) # Part i of simulated antennas list
    f.write('done\n\n') # end for

    f.write('cd /vol/optcoma/cr-simulations/corsika_production/run\n')

    f.write('for i in {1..%d}; do\n' % nof_split_runs)
    f.write('    srun -N 1 -n 1 ./corsika77100Linux_{1}_fluka_thin_conex_coreas < /{0}$RUNNR/RUN$RUNNR.inp &\n'.format(scratchdir_split, h_model)) # & for spawning subprocess
    f.write('done\n\n')

    f.write('wait # Wait for completion of subtasks\n\n')

    f.write('for i in {1..%d}; do\n' % nof_split_runs)
    f.write('    cd {0}$RUNNR\n'.format(scratchdir_split))
    f.write('    mv RUN$RUNNR.inp {0}steering/RUN$RUNNR.inp\n'.format(simdir))
    f.write('    mv SIM$RUNNR.reas {0}steering/SIM$RUNNR.reas\n'.format(simdir))
    f.write('    if [ $i -eq 1 ]; then\n')
    f.write('        mv SIM$RUNNR.list {0}steering/SIM$RUNNR.list\n'.format(simdir))
    f.write('        mv SIM$RUNNR\_coreas.bins {0}\n'.format(simdir))
    f.write('    else\n')
    # Concatenate SIMxxxxxx_coreas.bins and SIMxxxxxx.list files from their parts
    f.write('        cat SIM$RUNNR\_coreas.bins >> {0}SIM$RUNNR\_coreas.bins\n'.format(simdir))
    f.write('        cat SIM$RUNNR.list >> {0}steering/SIM$RUNNR.list\n'.format(simdir))
    f.write('        rm SIM$RUNNR\_coreas.bins\n')
    f.write('        rm SIM$RUNNR.list\n')
    f.write('    fi\n')
    f.write('    rsync -av . {0} # rsync the rest to the event directory \n'.format(simdir))
    f.write('    # /vol/optcoma/cr-simulations/LORAtools_GeVfix/DAT2txt DAT$RUNNR DAT$RUNNR.tmp\n')
    f.write('    # /vol/optcoma/cr-simulations/LORAtools_GeVfix/LORA_simulation DAT$RUNNR.tmp DAT$RUNNR.lora\n')
    f.write('    # rm DAT$RUNNR.tmp\n')
    f.write('    # (UGLY, REMOVED) mv * {0}\n'.format(simdir))
    f.write('    rm -rf {0}$RUNNR/*\n'.format(scratchdir_split))
    f.write('done\n\n')
    f.close()


def generate_cr_simulation(eventID, event_output_directory, particle_type, energy, az, zen, runnum, atmosphere_file=None, nof_split_runs=1, conex=False, ska_pattern=False, ska_site_params=False, h_model='QGSII'):
    """Schedule a CONEX or CoREAS simulation.
    
    **Properties**
    
    =============== ============================================
    Parameter       Description
    =============== ============================================
    *name*          filename
    *eventID*       event id from the database
    *energy*        energy in GeV
    *az*            arrival direction azimuth (Measured from East towards North in radians)
    *zen*           arrival direction zenith (Zenith angle in radians)
    *particle_type* particle type in CORSIKA numerical notation
    *runnum*        scheduler run number
    *conex*         CONEX *True* or CoREAS *False* simulations
    *nof_split_runs* Number of cores to use / split CoREAS antenna list into N parts
    =============== ============================================
    
    """

    if conex:
        simfile='{0}_conex_{1}.q'.format(eventID, particle_type)
    else:
        simfile='{0}_coreas_{1}.q'.format(eventID, particle_type)

    simdir=event_output_directory + '/{0}/'.format(particle_type)

    create_antenna_position_file(eventID, az, zen, ska_pattern=ska_pattern, ska_site_params=ska_site_params)
    split_antenna_position_file(eventID, Nparts=nof_split_runs)

    if particle_type == 'proton':
        if conex:
            writefile_CONEX(simfile, eventID, energy, az, zen, 14, simdir, atmosphere_file=atmosphere_file, h_model=h_model, SKA=ska_pattern)
        else:
            writefile_CoREAS(simfile, eventID, energy, az, zen, 14, simdir, atmosphere_file=atmosphere_file, nof_split_runs=nof_split_runs, h_model=h_model, SKA=ska_pattern)
    elif particle_type == 'iron':
        if conex:
            writefile_CONEX(simfile, eventID, energy, az, zen, 5626, simdir, atmosphere_file=atmosphere_file, h_model=h_model, SKA=ska_pattern)
        else:
            writefile_CoREAS(simfile, eventID, energy, az, zen, 5626, simdir, atmosphere_file=atmosphere_file, nof_split_runs=nof_split_runs, h_model=h_model, SKA=ska_pattern)
    else:
        raise ValueError("invalid particle_type {0}".format(particle_type))

    queuetype = 'normal' # Yay, no more backfill! (Jan 2020) #  if conex else 'backfill'

    #reservation_identifier = '--reservation=stijnbui_7' if (np.random.rand() > 0.0) else '' # and not conex else ''
    reservation_identifier = '' # used when running jobs in a reservation on the cluster
    priority_identifier = ''
    # Check if eventID is in Nature set. If not, run with --nice i.e. lowered priority.
    # If in Nature set, at normal priority
    #eventlist_file = '/vol/astro3/lofar/sim/pipeline/eventlist_Nature.txt'
    #with open(eventlist_file) as f:
    #    Nature_events = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    #Nature_eventlist = [int(x) for x in Nature_events]
    #if int(eventID) in Nature_eventlist:
    #    reservation_identifier = '--reservation=acorstanje_4' # always within reservation if in Nature set
    #    print 'Event in Nature event list, running with NORMAL priority'
    #elif len(reservation_identifier) > 1: # not in Nature list, and inside reservation
    #    priority_identifier = '--nice'
    #    print 'Submitting job with --nice flag, LOWER priority'

    if len(reservation_identifier) > 1:
        print 'Submitting job inside reservation'
    else:
        print 'Submitting job OUTSIDE reservation'

    nofRetries = 3
    delay_to_retry = 1800
    while nofRetries > 0:
        retcode = subprocess.call("/usr/local/slurm/bin/sbatch -p {2} {3} {4} --array {0} {1}".format(','.join([str(i) for i in runnum]), simfile, queuetype, reservation_identifier, priority_identifier), shell=True)
        if retcode == 0:
            break # done
        else:
            print '!! Submitting job (sbatch) failed, retrying in %d seconds...' % delay_to_retry
            time.sleep(delay_to_retry)
            nofRetries -= 1

    # removed -x coma[18-20]
    return retcode

# Functions to select useful events for simulations

def GetAngleMagneticField(theta, phi):

    """Returns the angle between the shower and the magnetic field.
    
    **Properties**
    
    ============== =========================================================
    Parameter      Description
    ============== =========================================================
    *theta*        elevation angle of shower in LOFAR convention and degrees
    *phi*          azimuth angle of shower in LOFAR convention and degrees
    ============== =========================================================
    
    """

    alpha = 0
    # Rough magnetic field at LOFAR
    t_b = np.radians(157.88)
    p_b = np.radians(89.35)
    # Shower in radians and cartesian
    t_s = np.radians(90.- theta)
    p_s = np.radians(90. - phi)

    #Vectors
    B = np.array([np.sin(t_b)*np.cos(p_b),np.sin(t_b)*np.sin(p_b),np.cos(t_b)])
    shower = np.array([-1.*np.sin(t_s)*np.cos(p_s),-1.*np.sin(t_s)*np.sin(p_s),-1.*np.cos(t_s)])
    
    alpha = np.dot(B,shower)
    alpha = np.arccos(alpha)
    return alpha


def checkLDFFitQuality(ldf_fit_output):

    """Returns *True* or *False* for the quality of the LDF reconstruction for simulation purposes. Thus, somewhat conservative cuts are required. 
    
    **Properties**
    
    ============== ===========================================
    Parameter      Description
    ============== ===========================================
    *ldf_output*   Dictionary as generated by the task ldf.py
    ============== ===========================================
    
    """
    quality = False
    if ldf_fit_output['red_chi_2'] < 2.5:
        if ldf_fit_output['fit_parameter_uncertainties'][2] < 100:
            if ldf_fit_output['nfev'] < 600:
                if ldf_fit_output['fit_parameters'][0] < 1000:
                    if ldf_fit_output['fit_parameters'][1] < 1000:
                        quality = True

    return quality

def setSimulationXmax(ldf_fit_output,quality,elevation,xmax_conversion=[230.,0.91,0.008],full_atmosphere=1036.0):

    """Calculates the Xmax from the LDF fit parameters and sets the corresponding value in the database. 
    
    **Properties**
    
    =================== ==========================================================
    Parameter           Description
    =================== ==========================================================
    *ldf_output*        Dictionary as generated by the task ldf.py
    *quality*           Return value of checkLDFFitQuality
    *elevation*         Elevation angle of shower in LOFAR convention and degrees
    *xmax_conversion*   Values to convert fitted sigma to xmax (from LDF paper)
    *full_atmosphere*   US Standard atmosphere above LOFAR
    =================== ==========================================================
    
    """
    if quality == True:
        p = ldf_fit_output['fit_parameters']
        zenith = np.deg2rad(90.-elevation)
        simulation_xmax = xmax_conversion[0] + xmax_conversion[1]*p[2] + xmax_conversion[2]*p[2]**2
        simulation_xmax *= -1.
        simulation_xmax += full_atmosphere/np.cos(zenith)
        
        simulation_xmax_reason = "Xmax from 2D LDF"
    
    else:
        simulation_xmax = 650.
        simulation_xmax_reason= "Default Xmax of 650"    
    

    return simulation_xmax, simulation_xmax_reason
    
 
def setSimulationEnergy(ldf_fit_energy,quality,lora_energy):

    """Sets the energy from the LDF fit parameters or the original LORA energy in the database. 
    
    **Properties**
    
    =================== ==============================================================
    Parameter           Description
    =================== ==============================================================
    *ldf_fit_energy*    Output of task ldf.py
    *quality*           Return value of checkLDFFitQuality
    *lora_energy*       Database values stored from the original LORA reconstruction
    =================== ==============================================================
    """

    if quality:
        simulation_energy = ldf_fit_energy
        simulation_energy_reason = "Energy from radio LDF"  
  
    
    else:
        simulation_energy = lora_energy
        simulation_energy_reason = "LDF fit of poor quality, default LORA energy"        

    return simulation_energy, simulation_energy_reason
    


def setSimulationDirection(run_wavefront,wavefront_output,chi2,average_direction,fitAzEl,angle=3.):

    """Sets the direction to use for the automated simulations. Checks whether wavefront has run reliably.  
    
    **Properties**
    
    =================== ============================================================================
    Parameter           Description
    =================== ============================================================================
    *run_wavefront*     True or False value if wavefront task was run 
    *wavefront_output*  Output dictionary of wavefront.py
    *chi2*              Chi^2 of wavefront fit
    *average_direction* Average direction of plane wave fit. 
    *fitAzEl*           Wavefront return value for direction. 
    *angle*             Value allowed for deviation (in degree) between wavefront and plane wave fit
    =================== ============================================================================
    """

    if run_wavefront:
        if wavefront_output is not None:
            if chi2 < 8.0:
                if pytmf.angular_separation(pytmf.deg2rad(average_direction[0]),pytmf.deg2rad(average_direction[1]),pytmf.deg2rad(fitAzEl[0]),pytmf.deg2rad(fitAzEl[1])) < pytmf.deg2rad(angle):
                    simulation_direction = fitAzEl
                    simulation_direction_reason = "Wavefront direction"
                else:
                    simulation_direction = average_direction
                    simulation_direction_reason = "Planewave direction, direction deviated"
            else:
                simulation_direction = average_direction
                simulation_direction_reason = "Planewave direction, fit did not converge"
        else:
            simulation_direction = average_direction
            simulation_direction_reason = "Planewave direction, wavefront not run"
    else:
        simulation_direction = average_direction
        simulation_direction_reason = "Planewave direction, wavefront not run"


    return simulation_direction, simulation_direction_reason
    

def setSimulationStatus(old_status,old_statusmessage, nof_good_stations,flagged, hba, average_direction,lora_direction, angle = 10., overwrite=False):

    """Logic to decide whether a shower should be simulated automatically by the pipeline.  
    
    **Properties**
    
    =================== ===================================================================
    Parameter           Description
    =================== ===================================================================
    *old_status*        Status of event before the pipeline was run again.
    *old_statusmessage* Statusmessage of event before the pipeline was run again.
    *nof_good_stations* Number of good stations identified by pipeline   
    *flagged*           Parameter to indicate whether event was flagged (thunderstorm or other)
    *hba*               Parameter to indicate whether event is HBA event
    *average_direction* Average direction from plane wave fit
    *lora_direction*    Direction reconstructed with LORA (database parameter)
    *angle*             Difference in degree allowed between average_direction and lora_direction
    *overwrite*         Should the old status be overwritten?
    =================== ===================================================================
    """
    refill = False
    if ((old_status == 'NEW')| (old_status == 'NOT_DESIRED') | (overwrite)):
        if not hba:
            if nof_good_stations >= 3:
                if not flagged:
                    if pytmf.angular_separation(pytmf.deg2rad(average_direction[0]),pytmf.deg2rad(average_direction[1]),pytmf.deg2rad(lora_direction[0]),pytmf.deg2rad(lora_direction[1])) < pytmf.deg2rad(angle):
                        refill = True
                        simulation_status = "DESIRED"
                        simulation_statusmessage = 'Good simulation set'

                    else:
                        simulation_status = "NOT_DESIRED"
                        simulation_statusmessage ="direction (LORA,LOFAR) deviates"
                else:
                    simulation_status = "NOT_DESIRED"
                    simulation_statusmessage = 'event flagged'
            else:
                simulation_status = "NOT_DESIRED"
                simulation_statusmessage = 'too few good stations'
        else:
            simulation_status = "NOT_DESIRED"
            simulation_statusmessage = 'HBA'     
        
    else:
        print "Event has already been processed before, skip this step to not overwrite choices"
        simulation_status = old_status
        simulation_statusmessage = old_statusmessage
        
    return simulation_status, simulation_statusmessage, refill

