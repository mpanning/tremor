This contains the model output from the Bayesian runs in the Kedar et al. (2021) tremor paper.  There are 4 runs compatible with the two events (S0189a and S0105a) and the two velocity models used to derive the amplitude scaling relationship.

All directories are compressed with tar and xzip.

In order to make the runs, some parameters need to be set in the code.  The following values were used for the runs included in this directory:

Here is the section of the code that selects the data vector component for amplitude and frequency for the two events:

```
# Make data vector.  Right now is hard-coded, but may adjust to read from file
# New ampltiude estimates based on peak amplitude in body wave window after
# filtering to region with SNR > 1
# evt 0 is S0105a, while evt 1 is S0189a
# Amplitude picked between 0.2 and 0.6 Hz for s0105a, and
# between 0.4 and 0.9 Hz for S0189a
# Amplitude is picked on acceleration traces as the peak to 2 significant
# figures
fobs_vector = [0.35, 0.6]
ampobs_vector = [1.5e-9, 1.4e-9]
evt_select = 0
```

Toggling between the events is accomplished by chagning evt_select to 0 for S0105a or 1 or S0189a.  The actual inverted vector uses period rather than frequency.  The following uncertainty estimates are used:

```
# Uncertainty estimates - 1 sigma
wsig_freq = 0.175 * freq_obs
wsig_period = 0.175 * period_obs
wsig_amp = 0.5e-9
```

A third data value is added to constrain the R parameter to be 0.95 with a 1 sigma uncertainty of 0.05.

The amplitude scaling relationship used varies by model selected as defined in this code block:

```
# TAYAK
# c0 = [1.132e-19, 9.116e-21]
# alpha = [0.608, 0.740]
# EH45Tcold
c0 = [5.294e-22, 8.095e-23]
alpha = [0.687, 0.789]
```

where the two values are for the two different source depths considered (6 km and 60 km).  Switching between models is accomodating by uncommenting the appropriate block for c0 and alpha.

Finally, the standard deviation for new model proposals needs to be defined for each parameter.  While all model chains are initiated by models generated from uniform porbabilities, we actually use a log-normal distibution for generation of model perturbations for all parameters except h0 fraction, which indicates a prior distribution that approaches log uniform for L, eta, pressure ratio, and aspect ratio, and approaches uniform for h0 fraction.  Here are the definitions for the standard deviations for model perturbations:

```
# Standard deviations for the Gaussian proposal distributions

# Given that many of these vary over orders of magnitude, maybe should use
# log value as input, and thus it becomes log-normal perturbation.  Consider
# this and whether it should effect acceptance criteria
thetaL = np.log(1.10) # Length perturbations
thetaETA = np.log(1.10) # Using a log normal perturbation instead
thetaPR = np.log(1.01) # Pressure ratio perturbatio
thetaWL = np.log(1.10) # Using a log normal perturbation instead
thetaH0 = 0.002
```




