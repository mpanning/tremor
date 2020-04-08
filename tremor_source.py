"""
Create seismograms from Julian 1994 tremor model
"""
import math
import obspy
from obspy.core import UTCDateTime
import instaseis
import numpy as np
import matplotlib.pyplot as plt
import tremor
import timeit

start_time = timeit.default_timer()
db_short = "EH45TcoldCrust1b"
# Instaseis parameters
# db_short = "TAYAK"
instaseisDB = "http://instaseis.ethz.ch/blindtest_1s/{}_1s/".format(db_short)
maxRetry = 25
db = instaseis.open_db(instaseisDB)
f1 = 0.005
f2 = 2.0
depth_in_km = 6.0
t0 = UTCDateTime(2019,3,1)
dbdt = db.info['dt']
dbnpts = db.info['npts']

# Tremor parameters Should define a way to set this from a file or command line
depth = 1.e3*depth_in_km
pratio = 1.01
mu = 7.e9
rho = 2.7e3
L = 200.
aspect_ratio = 12.5
width = L*aspect_ratio
eta = 10.0

# Define unit M0 CLVD moment tensor, lined up such that it represents a
# vertical crack aligned E-W
scale = 1.0/math.sqrt(3.0) # Factor to correct for non-unit M0
m_rr = -1.0*scale
m_tt = 2.0*scale
m_pp = -1.0*scale
m_rt = 0.0*scale
m_rp = 0.0*scale
m_tp = 0.0*scale

slat = 11.28 # Assumed at Cerberus Fossae
slon = 166.37

rlat = 4.5 #Landing site ellipse
rlon = 135.9

# Do tremor calculations
model = tremor.TremorModel(depth=depth, pratio=pratio, mu=mu, rho=rho, L=L,
                           width=width)
model.calc_derived()
model.set_eta(eta)
model.calc_R()
model.calc_f()

print("R: ", model.R)

# Calculate the crack motions
tmax = 1000.0
tremordt = 0.1
vi = 0.0
hi = 1.0
ui = 0.0
w0 = [vi, hi, ui]
tarray, wsol = model.generate_tremor(tmax, tremordt, w0)

fig = plt.figure()
plt.plot(tarray, wsol[0,:,0])
plt.xlim((0, tmax))
plt.savefig("tremor_v.png")
plt.close(fig)

fig = plt.figure()
plt.plot(tarray, wsol[0,:,1])
plt.xlim((0, tmax))
plt.xlabel("Time (s)")
plt.ylabel("h (m)")
plt.savefig("tremor_h.png")
plt.close(fig)

fig = plt.figure()
plt.plot(tarray, wsol[0,:,2])
plt.xlim((0, tmax))
plt.savefig("tremor_u.png")
plt.close(fig)

# Taper tremor time series and calculate moments
taper_frac = 0.02
m0_total, m0_average = model.get_moments(taper=taper_frac)
M0 = m0_total[0]

print("Estimated seismic moment: {:.3E}".format(m0_total[0]))

# Set the Instaseis source parameters
sliprate = model.u[0]
slipdt = tremordt

source = instaseis.source.Source(latitude=slat, longitude=slon,
                                 depth_in_m=depth,
                                 m_rr=m_rr*M0, m_tt=m_tt*M0, m_pp=m_pp*M0,
                                 m_rt=m_rt*M0, m_rp=m_rp*M0, m_tp=m_tp*M0,
                                 origin_time=t0)
source.set_sliprate(sliprate, slipdt)
source.resample_sliprate(dt=dbdt, nsamp=len(model.u[0]))

# Renormalize sliprate with absolute value appropriate for oscillatory
# sliprates with negative values
source.sliprate /= np.trapz(np.absolute(source.sliprate), dx=source.dt)

fig = plt.figure()
plt.plot(np.arange(0, len(source.sliprate)*source.dt, source.dt),
         source.sliprate)
plt.xlim((0, tmax))
plt.xlabel("Time (s)")
plt.ylabel("Normalized wall velocity")
plt.savefig("sliprate.png")
plt.close(fig)

fig = plt.figure()
plt.plot(tarray[:-1], model.h[0])
plt.xlim((0, tmax))
plt.savefig("model_h.png")
plt.close(fig)

receiver = instaseis.Receiver(latitude=rlat, longitude=rlon, network='XB',
                              station='ELYSE')

st = db.get_seismograms(source=source, receiver=receiver, kind='acceleration',
                        remove_source_shift=False, reconvolve_stf=True)

st.plot(outfile="seismograms.png")

st.write("{}_{:d}km_.sac".format(db_short, int(depth_in_km)),
         format='sac')
st.write("{}_{:d}km.mseed".format(db_short, int(depth_in_km)),
         format='mseed')
elapsed = timeit.default_timer() - start_time
print("Elapsed time: {:.3f} s".format(elapsed))
