"""
Calculate equilibrium values and time series for a tremor model
"""

import math
import numpy as np
# from tqdm import tqdm
import tremor
import timeit
import csv

# Create models and change from default parameters if desired

h0 = -3.8
start_time = timeit.default_timer()
model = tremor.TremorModel(h0=h0)
model.calc_derived()
elapsed = timeit.default_timer() - start_time
print("Initiate model time: {:.3f} ms".format(1.e3*elapsed))

# eta = np.array([0.1, 0.15, 0.225, 0.338, 0.506, 0.759, 1.14])
n_eta = 7
eta = np.zeros(n_eta)
eta[0] = 0.1
for i in range(1,len(eta)):
    eta[i] = eta[i-1] * 1.5
model.set_eta(eta)

start_time = timeit.default_timer()
# This does all intermediate calculations (hs, vs, a, m, r1, r2, r3)
model.calc_R() 
elapsed = timeit.default_timer() - start_time
print("R calculation time: {:.3f} ms".format(1.e3*elapsed))

model.calc_f()

flux = model.rho*model.vs
flow = flux*model.width*model.hs/model.rho

print("eta ",model.eta)
print("R ",model.R)
print("f ",model.f)
print("flux ",flux)
print("flow ",flow)

# Now calculate the time series
duration = 2000.0
dt = 0.1

# Initial conditions
vi = 0.0 # initial fluid velocity (m/s)
hi = 1.0 # initial wall position (m)
ui = 0.0 # initial wall velocity (m/s)
w0 = [vi, hi, ui]

start_time = timeit.default_timer()
t, wsol = model.generate_tremor(duration, dt, w0)
elapsed = timeit.default_timer() - start_time
print("Time series integration: {:.3f} ms".format(1.e3*elapsed))

for i, val in enumerate(eta):
    outfile = "tremor_{:.3f}.csv".format(val)
    with open(outfile, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['Time', 'Fluid velocity', 'Wall position',
                            'Wall velocity'])
        for t1, w1 in zip(t, wsol[i, :, :]):
            csvwriter.writerow(["{:.3f}".format(t1), "{:.5f}".format(w1[0]),
                                "{:.5f}".format(w1[1]), "{:.5f}".format(w1[2])])
        
start_time = timeit.default_timer()
durations = model.get_duration(taper=0.02, threshold=0.1)
elapsed = timeit.default_timer() - start_time
print("Picking duration: {:.3f} ms".format(1.e3*elapsed))

print("duration", durations)
