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
# Fig. 9 parameters Julian paper
k = 600.e6
M = 3.e5
rho = 2.5e3
p1 = 10.e6
p2 = 0.1e6
h0 = 1.0
L = 10.
g = 9.8
start_time = timeit.default_timer()
model = tremor.TremorModel(k=k, M=M, rho=rho, p1=p1, p2=p2, h0=h0, L=10., g=g)
model.calc_derived()
elapsed = timeit.default_timer() - start_time
print("Initiate model time: {:.3f} ms".format(1.e3*elapsed))

eta = [500.]
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
duration = 15.0
dt = 0.01

# Initial conditions
vi = 0.0 # initial fluid velocity (m/s)
hi = 1.05 # initial wall position (m)
ui = 0.0 # initial wall velocity (m/s)
w0 = [vi, hi, ui]

start_time = timeit.default_timer()
t, wsol = model.generate_tremor(duration, dt, w0)
elapsed = timeit.default_timer() - start_time
print("Time series integration: {:.3f} ms".format(1.e3*elapsed))

for i, val in enumerate(eta):
    outfile = "fig9_{:.3f}.csv".format(val)
    with open(outfile, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['Time', 'Fluid velocity', 'Wall position',
                            'Wall velocity'])
        for t1, w1 in zip(t, wsol[i, :, :]):
            csvwriter.writerow(["{:.3f}".format(t1), "{:.5f}".format(w1[0]),
                                "{:.5f}".format(w1[1]), "{:.5f}".format(w1[2])])
        
# Fig 10
p1 = 3.e6
start_time = timeit.default_timer()
model = tremor.TremorModel(k=k, M=M, rho=rho, p1=p1, p2=p2, h0=h0, L=10., g=g)
model.calc_derived()
elapsed = timeit.default_timer() - start_time
print("Initiate model time: {:.3f} ms".format(1.e3*elapsed))

eta = [500.]
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
duration = 15.0
dt = 0.01

start_time = timeit.default_timer()
t, wsol = model.generate_tremor(duration, dt, w0)
elapsed = timeit.default_timer() - start_time
print("Time series integration: {:.3f} ms".format(1.e3*elapsed))

for i, val in enumerate(eta):
    outfile = "fig10_{:.3f}.csv".format(val)
    with open(outfile, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['Time', 'Fluid velocity', 'Wall position',
                            'Wall velocity'])
        for t1, w1 in zip(t, wsol[i, :, :]):
            csvwriter.writerow(["{:.3f}".format(t1), "{:.5f}".format(w1[0]),
                                "{:.5f}".format(w1[1]), "{:.5f}".format(w1[2])])

# Fig 12
p1 = 18.e6
M = 3.e5
A = 1.e7
start_time = timeit.default_timer()
model = tremor.TremorModel(k=k, M=M, rho=rho, p1=p1, p2=p2, h0=h0, L=10., g=g)
model.calc_derived()
elapsed = timeit.default_timer() - start_time
print("Initiate model time: {:.3f} ms".format(1.e3*elapsed))

eta = [50.]
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
duration = 15.0
dt = 0.01

# Initial conditions
vi = 0.0 # initial fluid velocity (m/s)
hi = 1.05 # initial wall position (m)
ui = -10.0 # initial wall velocity (m/s)
w0 = [vi, hi, ui]

start_time = timeit.default_timer()
t, wsol = model.generate_tremor(duration, dt, w0)
elapsed = timeit.default_timer() - start_time
print("Time series integration: {:.3f} ms".format(1.e3*elapsed))

for i, val in enumerate(eta):
    outfile = "fig12_{:.3f}.csv".format(val)
    with open(outfile, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['Time', 'Fluid velocity', 'Wall position',
                            'Wall velocity'])
        for t1, w1 in zip(t, wsol[i, :, :]):
            csvwriter.writerow(["{:.3f}".format(t1), "{:.5f}".format(w1[0]),
                                "{:.5f}".format(w1[1]), "{:.5f}".format(w1[2])])
