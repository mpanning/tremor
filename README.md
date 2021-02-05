# tremor
Python implementation of tremor modeling based on Julian (1994).

The bulk of calculations are performed in the module `tremor.py`, while `tremor_test.py` calls the functions for a demonstration of the calculations.

`fig9-10-12.py` attempts to reproduce calculations for some figures in the Julian (1994) paper.  Note that the calculations for figure 12 do not apparently match the paper results, but figures 9 and 10 are well reproduced.

`forward_model.py` times the calculation of relevant parameters for a forward calculation to be used in a single iteration of future MCMC inversion

`tremor_source.py` calculates the tremor model and then computes seismograms from the model using InstaSeis.

Currently includes all codes used to calculate figures in the Kedar et al. (2020) revised paper on tremor modeling, although the documentation will be clarified and the codes slightly cleaned up to correctly reference the subdirectory structure before the final archived version is included with an accepted paper.

The models derived in the Bayesian MCMC codes are all stored compressed in xz format in the `models` subdirectory.  xz is open source, freely available compression software more efficient than the standard gzip used to create tar.gz files, which was required to keep file sizes under the 100 MB github limit.

## References
[B.R. Julian (1994) Volcanic tremor: Nonlinear excitation by fluid flow, J. Geophys. Res., 99, B6, 11859-11877, doi: /10.1029/93JB03129](https://doi.org/10.1029/93JB03129)
