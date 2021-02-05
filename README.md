# tremor
Python implementation of tremor modeling based on Julian (1994).

The bulk of calculations are performed in the module `tremor.py`, while `tremor_test.py` calls the functions for a demonstration of the calculations.

`fig9-10-12.py` attempts to reproduce calculations for some figures in the Julian (1994) paper.  Note that the calculations for figure 12 do not apparently match the paper results, but figures 9 and 10 are well reproduced.

`forward_model.py` times the calculation of relevant parameters for a forward calculation to be used in a single iteration of future MCMC inversion

`tremor_source.py` calculates the tremor model and then computes seismograms from the model using InstaSeis.

Currently includes all codes used to calculate figures in the Kedar et al. (2021) paper on tremor modeling.

The models derived in the Bayesian MCMC codes are all stored compressed in xz format in the `models` subdirectory.  xz is open source, freely available compression software more efficient than the standard gzip used to create tar.gz files, which was required to keep file sizes under the 100 MB github limit.

For the Kedar et al. (2021) study of potential Martian tremor sources, Bayesian modeling was used to estimate source properties required to produce the observed signals for 2 potential events assumed to be located at Cerberus Fossae recorded at InSight.  Inversion was performed using the code `MCMC_main.py`, which calls functions in `MCMC_functions.py`.  Inversions were run on JPL high performance computing clusters, and a sample run script is included as `PBS_node_script.bash`.

All figures displaying the modeling results (and many more possible parameter combinations) are calculated using the codes `make_plots.py`, `make_plots_combined.py` and make_plots_combined_4.py`.  Note that all of these require editing of the code to include the path to the model directories (by setting `model_dir` in `make_plots.py`, or the similarly named variables ending in integers 1-4 in the other two codes.

The amplitude scaling figures are produced with the code `amp_scaling.py`, which requires a working installation of the python Instaseis package, and accesses and online waveform database located at http://instaseis.ethz.ch/blindtest_1s which was developed for the InSight Marsquake Service blindtest (Clinton et al., 2017, Ceylan et al., 2017).

## References
[B.R. Julian (1994) Volcanic tremor: Nonlinear excitation by fluid flow, J. Geophys. Res., 99, B6, 11859-11877, doi: /10.1029/93JB03129](https://doi.org/10.1029/93JB03129)
