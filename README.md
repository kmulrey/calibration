# calibration
use galactic emission to calibrate antennas


This project is current run on coma /vol/astro7/lofar/kmulrey/calibration.


Generate an antenna model:


AARTFAAC antennna model can be found here: https://www.astro.rug.nl/~hare/public/AARTFAAC_antmodel/

The write_and_save_model.py file will generate an AARTFAAC antenna model using Brian's AARTFAAC_model.py script.  You can specify R and C values in AARTFAAC_model.py

The standard LOFAR CR antenna model is at /vol/astro7/lofar/kmulrey/calibration/antenna_model/jones_standard/




Run a calibration:

To run a calibration, where the noise measured in the TBBs is compared to modeled sky noise as measured with a specific antenna model, you should run calibration_end_to_end.py.  You can specify which antenna model to use there.  

This script will first generate the average antenna response per arrival direction and frequency for the chosen atenna model.  A file is saved for each frequency (1 MHz steps).

Then, the corresponding TBB data is found, and a file is created with all the relevant info for the fit.

Finally, the fit is done, and then the output saved.





