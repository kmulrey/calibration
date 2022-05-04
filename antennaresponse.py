"""
AntennaResponse
===============

.. moduleauthor:: Pim Schellart <p.schellart@astro.ru.nl>

"""

from pycrtools.tasks import Task
import pycrtools as cr
import numpy as np
import os


class AntennaResponse(Task):
    """Calculates and unfolds the LOFAR (LBA or HBA) antenna response.

    Given an array with *instrumental_polarization* and a *direction* as (Azimuth, Elevation) the Jones matrix
    containing the LOFAR (LBA or HBA) antenna response is calculated.

    Mixing the two instrumental polarizations by multiplying with the inverse
    Jones matrix gives the resulting on-sky polarizations (e.g. perpendicular to the direction and
    parallel to and perpendicular to the horizon).

    Note that, due to the intrinsic symmetry of the antenna configuration, the direction of Azimuth (CW or CCW)
    is not relevant.

    .. seealso:: Schellart et al., Detecting cosmic rays with the LOFAR radio telescope, Astronomy and Astrophysics, 560, A98, (2013) and Nelles, Karskens, Krause et al., Calibration Paper in prep (2015).

    """

    parameters = dict(
        instrumental_polarization=dict(default=None,
            doc="FFT data."),
        on_sky_polarization=dict(default=lambda self: self.instrumental_polarization.new(), output=True,
            doc="FFT data corrected for element response (contains on sky polarizations)."),
        frequencies=dict(default=None,
            doc="Frequencies in Hz."),
        direction=dict(default=(0, 0),
            doc="Direction in degrees as a (Azimuth, Elevation) tuple LOFAR convention for azimuth (eastwards positive from north)."),
        nantennas = dict(default=lambda self: self.instrumental_polarization.shape()[0],
            doc="Number of antennas."),
        antennaset = dict(default="LBA_OUTER",
            doc="Antennaset."),
        jones_matrix = dict(default=lambda self: cr.hArray(complex, dimensions=(self.frequencies.shape()[0], 2, 2)),
            doc = "Jones matrix for each frequency."),
        inverse_jones_matrix=dict(default = lambda self: cr.hArray(complex, dimensions=(self.frequencies.shape()[0], 2, 2)),
            doc="Inverse Jones matrix for each frequency."),
        swap_dipoles=dict(default=False,
            doc="Swap dipoles before mixing."),
        backwards=dict(default=False,
            doc="Apply antenna response backwards (e.g. without inverting the Jones matrix)."),
        apply_to_data=dict(default=True,
            doc="Apply to data, set to False if you only need the (inverse) Jones matrix."),
        test_with_identity_matrix=dict(default=False,
            doc="Don't apply the model, use identity matrix for mixing. For testing purposes only."),
        vt=dict(default=None,
                doc="Table with complex antenna response for X dipole to wave purely polarized in theta direction."),
        vp=dict(default=None,
                doc="Table with complex antenna response for X dipole to wave purely polarized in phi direction."),
    )

    def run(self):
        """Run.
        """

        if "LBA" in self.antennaset:
            if self.vt is None:
                self.vt = np.loadtxt(os.environ["LOFARSOFT"] + "/data/lofar/antenna_response_model/LBA_Vout_theta.txt", skiprows=1)
            if self.vp is None:
                self.vp = np.loadtxt(os.environ["LOFARSOFT"] + "/data/lofar/antenna_response_model/LBA_Vout_phi.txt", skiprows=1)

            cvt = cr.hArray(self.vt[:, 3] + 1j * self.vt[:, 4])
            cvp = cr.hArray(self.vp[:, 3] + 1j * self.vp[:, 4])

            fstart = 10.0 * 1.e6
            fstep = 1.0 * 1.e6
            fn = 101
            tstart = 0.0
            tstep = 5.0
            tn = 19
            pstart = 0.0
            pstep = 10.0
            pn = 37
        elif "HBA" in self.antennaset:
            if self.vt is None:
                self.vt = np.loadtxt(os.environ["LOFARSOFT"] + "/data/lofar/antenna_response_model/HBA_Vout_theta.txt", skiprows=1)
            if self.vp is None:
                self.vp = np.loadtxt(os.environ["LOFARSOFT"] + "/data/lofar/antenna_response_model/HBA_Vout_phi.txt", skiprows=1)

            cvt = cr.hArray(self.vt[:, 3] + 1j * self.vt[:, 4])
            cvp = cr.hArray(self.vp[:, 3] + 1j * self.vp[:, 4])

            fstart = 100.0 * 1.e6
            fstep = 10.0 * 1.e6
            fn = 21
            tstart = 0.0
            tstep = 1.0
            tn = 91
            pstart = 0.0
            pstep = 10.0
            pn = 37
        else:
            raise ValueError("Unsupported antennaset {0}".format(self.antennaset))

        # Get inverse Jones matrix for each frequency
        print "obtaining Jones matrix for direction", self.direction[0], self.direction[1]
        for i, f in enumerate(self.frequencies):
            cr.hGetJonesMatrix(self.jones_matrix[i], f, self.direction[0], self.direction[1], cvt, cvp, fstart, fstep, fn, tstart, tstep, tn, pstart, pstep, pn)

        print "inverting Jones matrix"
        if not self.backwards:
            cr.hInvertComplexMatrix2D(self.inverse_jones_matrix, self.jones_matrix)

        if self.test_with_identity_matrix:
            print "overriding antenna model Jones matrix with identity."
            identity = cr.hArray(complex, 2, fill=1)
            for i in range(self.inverse_jones_matrix.shape()[0]):
                cr.hDiagonalMatrix(self.inverse_jones_matrix[i], identity)

        # Unfold the antenna response and mix polarizations according to the Jones matrix to get the on-sky polarizations
        if self.apply_to_data:
            # Copy FFT data over for correction
            self.on_sky_polarization.copy(self.instrumental_polarization)

            if self.swap_dipoles:
                print "swapping dipoles"
                cr.hSwap(self.on_sky_polarization[0:self.nantennas:2, ...], self.on_sky_polarization[1:self.nantennas:2, ...])

            if not self.backwards:
                print "unfolding antenna pattern"
                cr.hMatrixMix(self.on_sky_polarization[0:self.nantennas:2, ...], self.on_sky_polarization[1:self.nantennas:2, ...], self.inverse_jones_matrix)
            else:
                print "unfolding antenna pattern (backwards)"
                cr.hMatrixMix(self.on_sky_polarization[0:self.nantennas:2, ...], self.on_sky_polarization[1:self.nantennas:2, ...], self.jones_matrix)
