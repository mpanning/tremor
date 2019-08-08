# tremor
Python implementation of tremor modeling based on Julian (1994).

The bulk of calculations are performed in the module `tremor.py`, while `tremor_test.py` calls the functions for a demonstration of the calculations.

`fig9-10-12.py` attempts to reproduce calculations for some figures in the Julian (1994) paper.  Note that the calculations for figure 12 do not apparently match the paper results, but figures 9 and 10 are well reproduced.

`forward_model.py` times the calculation of relevant parameters for a forward calculation to be used in a single iteration of future MCMC inversion

## References
[B.R. Julian (1994) Volcanic tremor: Nonlinear excitation by fluid flow, J. Geophys. Res., 99, B6, 11859-11877, doi: /10.1029/93JB03129](https://doi.org/10.1029/93JB03129)
