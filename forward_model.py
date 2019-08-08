"""
Test code to time forward model calculation for single tremor model

Does calculations necessary to get frequency, duration, and average moment
  for a particular set of tremor model parameters (i.e. one iteration of 
  MCMC inversion)
"""

import numpy as np
import timeit
import tremor

# Try first with default values
start_time = timeit.default_timer()
model = tremor.TremorModel()
model.calc_derived()
eta = 1.1390625
model.set_eta(eta)
model.calc_R()
model.calc_f()

# Now calculate the time series
tmax = 2000.0
dt = 0.1

# Initial conditions
vi = 0.0 # initial fluid velocity (m/s)
hi = 1.0 # initial wall position (m)
ui = 0.0 # initial wall velocity (m/s)
w0 = [vi, hi, ui]

t, wsol = model.generate_tremor(tmax, dt, w0)
durations = model.get_durations(taper=0.02, threshold=0.1)
m0_total, m0_average = model.get_moments(window=durations)
elapsed = timeit.default_timer() - start_time
print("Full forward calculation: {:.3f} ms".format(elapsed*1.e3))

print("Model output:", model.f, durations, m0_average)

    
