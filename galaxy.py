"""
Galaxy
======

.. moduleauthor:: Pim Schellart <p.schellart@astro.ru.nl> 

"""

from pycrtools.tasks import Task
import pycrtools as cr
import numpy as np
import pytmf
import datetime
from scipy.interpolate import interp1d

def calibratedGaincurve(freq, NrAntennas,galaxy = True):
    """
    Function delivers calibration curve as::

        Data * Calibration curve = Simulated voltage
                                 = Expected electric field * Antenna model

    Hence, it's the gain-factor by which the data should be multiplied in order to match the expected voltages.
    """

    
    if galaxy == True:
        Calibration_curve = np.zeros(101)
        Calibration_curve[29:82] = np.array([0, 1.37321451961e-05,
                                             1.39846332239e-05,
                                             1.48748993821e-05,
                                             1.54402170354e-05,
                                             1.60684568225e-05,
                                             1.66241942741e-05,
                                             1.67039066047e-05,
                                             1.74480931848e-05,
                                             1.80525736486e-05,
                                             1.87066855054e-05,
                                             1.88519099831e-05,
                                             1.99625051386e-05,
                                             2.01878566584e-05,
                                             2.11573680797e-05,
                                             2.15829455528e-05,
                                             2.20133824866e-05,
                                             2.23736319125e-05,
                                             2.24484419697e-05,
                                             2.37802483891e-05,
                                             2.40581543111e-05,
                                             2.42020383477e-05,
                                             2.45305869187e-05,
                                             2.49399905965e-05,
                                             2.63774023804e-05,
                                             2.70334253414e-05,
                                             2.78034857678e-05,
                                             3.07147991391e-05,
                                             3.40755705892e-05,
                                             3.67311849851e-05,
                                             3.89987440028e-05,
                                             3.72257913465e-05,
                                             3.54293510934e-05,
                                             3.35552370942e-05,
                                             2.96529815929e-05,
                                             2.79271252352e-05,
                                             2.8818544973e-05,
                                             2.92478843809e-05,
                                             2.98454768706e-05,
                                             3.07045462103e-05,
                                             3.07210553534e-05,
                                             3.16442871206e-05,
                                             3.2304638838e-05,
                                             3.33203882046e-05,
                                             3.46651060935e-05,
                                             3.55193137077e-05,
                                             3.73919275937e-05,
                                             3.97397037914e-05,
                                             4.30625048727e-05,
                                             4.74612081994e-05,
                                             5.02345866124e-05,
                                             5.53621848304e-05,
 0])
                ## 30 - 80 MHz, derived from average galaxy model + electronics  
    else:
        Calibration_curve = np.array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 9.99000000e-07,   9.13000000e-07,   1.05000000e-06,
                 1.16000000e-06,   1.14000000e-06,   1.14000000e-06,
                 1.13000000e-06,   1.21000000e-06,   1.25000000e-06,
                 1.25000000e-06,   1.21000000e-06,   1.29000000e-06,
                 1.31000000e-06,   1.33000000e-06,   1.24000000e-06,
                 1.22000000e-06,   1.32000000e-06,   1.30000000e-06,
                 1.24000000e-06,   1.20000000e-06,   1.23000000e-06,
                 1.22000000e-06,   1.50000000e-06,   1.42000000e-06,
                 1.41000000e-06,   1.50000000e-06,   1.63000000e-06,
                 1.71000000e-06,   1.83000000e-06,   2.04000000e-06,
                 1.96000000e-06,   1.69000000e-06,   1.51000000e-06,
                 1.53000000e-06,   1.41000000e-06,   1.30000000e-06,
                 1.40000000e-06,   1.43000000e-06,   1.46000000e-06,
                 1.49000000e-06,   1.57000000e-06,   1.59000000e-06,
                 1.68000000e-06,   1.75000000e-06,   1.81000000e-06,
                 1.96000000e-06,   2.20000000e-06,   2.46000000e-06,
                 3.01000000e-06,   3.66000000e-06,   4.68000000e-06,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00])
                ## 30 - 80 MHz, derived from crane calibration   

    Calibration_curve_interp = interp1d(np.linspace(0.e6,100e6,101), Calibration_curve, kind='linear')
    Calibration_curve_interp = Calibration_curve_interp(freq)

    Calibration_curve_interp = cr.hArray(Calibration_curve_interp)
    Calibration_curve_interp_array = cr.hArray(np.zeros(NrAntennas * len(freq)).reshape(NrAntennas, len(freq)))
    Calibration_curve_interp_array[...] = (Calibration_curve_interp)
    return Calibration_curve_interp_array


class GalacticNoise(Task):
    """Task to normalize noise levels in both dipoles to the expected Galactic noise level and to apply calibration of dipoles as developed in master thesis Tijs Karskens. 

    Evaluates a partial Fourier series fit to the Galactic response as a function of Local Apparant Siderial Time.
    
    Before 2015 the following coefficients were used: np.array([ 0.01620088, -0.00143372,  0.00099162, -0.00027658, -0.00056887]), np.array([  1.44219822e-02,  -9.51155631e-04,   6.51046296e-04, 8.33650041e-05,  -4.91284500e-04])
    
    After a slight offset for unknown reasons was found the following coefficients were put in place:
    
    np.array([ 0.01489468, -0.00129305,  0.00089477, -0.00020722, -0.00046507]) , np.array([ 0.01347391, -0.00088765,  0.00059822,  0.00011678, -0.00039787]
    

    It also multiplies the data with the calibration curve to physical units. 

    For example::

        # Normalize recieved power to that expected for a Galaxy dominated reference antenna
        galactic_noise = cr.trun("GalacticNoise", fft_data=fft_data, channel_width=f["SAMPLE_FREQUENCY"][0] / f["BLOCKSIZE"], timestamp=tbb_time, antenna_set=f["ANTENNA_SET"], original_power=antennas_cleaned_power)

    .. seealso:: Schellart et al., Detecting cosmic rays with the LOFAR radio telescope, Astronomy and Astrophysics, 560, A98, (2013) and Nelles, Karskens, Krause et al., Calibration Paper in prep (2015).

    """

    parameters = dict(
        fft_data=dict(default=None, doc="FFT data to correct."),
        frequencies=dict(default=None, doc="Frequencies in Hz."),
        original_power=dict(default=None, doc="Original power to normalize by (i.e. output of :class:`findrfi.antennas_cleaned_power`)."),
        antenna_set=dict(default="", doc="Antenna set"),
        channel_width=dict(default=1., doc="Width of a single frequency channel in Hz"),
        galactic_noise_power=dict(default=(0, 0), doc="Galactic noise power per Hz", output=True),
        timestamp=dict(default=None, doc="Observation time"),
        longitude=dict(default=pytmf.deg2rad(6.869837540), doc="Observer longitude in radians"),
        coefficients_lba=dict(default=( np.array([ 0.01489468, -0.00129305,  0.00089477, -0.00020722, -0.00046507]),
                                        np.array([ 0.01347391, -0.00088765,  0.00059822,  0.00011678, -0.00039787])
                                        ), doc="Tuple with coefficients for partial Fourier series describing galaxy response in Hz for polarization 0 and 1 respectively"),
        coefficients_hba=dict(default=lambda self : (np.array([2.0 / self.channel_width]), np.array([2.0 / self.channel_width])),
            doc="Tuple with coefficients for partial Fourier series describing galaxy response in Hz for polarization 0 and 1 respectively"),
        use_gain_curve=dict(default=False, doc="Use gain curve to correct to physical units (Volts) (note that applying the antenna model will add meters to the units)."),
        use_gain_galaxy=dict(default=False, doc="Use gain curve from Galaxy to correct to physical units (Volts) (note that applying the antenna model will add meters to the units). If set False, the crane calibration will be used.")
    )

    def fourier_series(self, x, p):
        """Evaluates a partial Fourier series

        .. math::

            F(x) \\approx \\frac{a_{0}}{2} + \\sum_{n=1}^{\\mathrm{order}} a_{n} \\sin(nx) + b_{n} \\cos(nx)
            
        

        """

        r = p[0] / 2

        order = (len(p) - 1) / 2

        for i in range(order):

            n = i + 1

            r += p[2*i + 1] * np.sin(n * x) + p[2*i + 2] * np.cos(n * x)

        return r

    def run(self):
        """Run.
        
        
        
        """

        # Convert timestamp to datetime object
        t = datetime.datetime.utcfromtimestamp(self.timestamp)

        # Calculate JD(UT1)
        ut = pytmf.gregorian2jd(t.year, t.month, float(t.day) + ((float(t.hour) + float(t.minute) / 60. + float(t.second) / 3600.) / 24.))

        # Calculate JD(TT)
        dtt = pytmf.delta_tt_utc(pytmf.date2jd(t.year, t.month, float(t.day) + ((float(t.hour) + float(t.minute) / 60. + float(t.second) / 3600.) / 24.)))
        tt = pytmf.gregorian2jd(t.year, t.month, float(t.day) + ((float(t.hour) + float(t.minute) / 60. + (float(t.second) + dtt / 3600.)) / 24.))

        # Calculate Local Apparant Sidereal Time
        self.last = pytmf.rad2circle(pytmf.last(ut, tt, self.longitude))

        # Evaluate Fourier series for calculated LST
        if "LBA" in self.antenna_set:
            self.galactic_noise_power = (self.fourier_series(self.last, self.coefficients_lba[0]), self.fourier_series(self.last, self.coefficients_lba[1]))
        elif "HBA" in self.antenna_set:
            self.galactic_noise_power = (self.fourier_series(self.last, self.coefficients_hba[0]), self.fourier_series(self.last, self.coefficients_hba[1]))
        else:
            raise ValueError("Unsupported antenna_set {0}".format(self.antenna_set))

        if self.fft_data is not None:

            print "correcting power per channel to", self.galactic_noise_power[0] * self.channel_width, self.galactic_noise_power[1] * self.channel_width

            # Calculate correction factor
            self.correction_factor = self.original_power.new()
            self.correction_factor.copy(self.original_power)

            ndipoles = self.correction_factor.shape()[0]

            cr.hInverse(self.correction_factor)
            cr.hMul(self.correction_factor[0:ndipoles:2, ...], self.galactic_noise_power[0] * self.channel_width)
            cr.hMul(self.correction_factor[1:ndipoles:2, ...], self.galactic_noise_power[1] * self.channel_width)
            cr.hSqrt(self.correction_factor)

            # Correct FFT data, for relative deviations
            cr.hMul(self.fft_data[...], self.correction_factor[...])

            
            # Correct FFT data, for absolute values
            if self.use_gain_curve:
                print "Applying gain curve from:"
                if self.use_gain_galaxy:
                    print "Using GALAXY calibration"
                else:
                    print "Using CRANE calibration"
                
                gc = calibratedGaincurve(self.frequencies.toNumpy(),self.fft_data.shape()[0] ,self.use_gain_galaxy)

                cr.hMul(self.fft_data[...], gc[...])
            else:
                print "Applying NO gain calibration curve."


