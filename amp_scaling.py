"""
Make plots to determine scaling from tremor model to seismic amplitude
"""
import math
import obspy
from obspy.core import UTCDateTime
import instaseis
import numpy as np
import matplotlib.pyplot as plt
import tremor
from scipy.optimize import curve_fit

symbols = ['ro', 'bo', 'go', 'ko']

# db_short = "EH45TcoldCrust1b"
# Instaseis parameters
db_short = "EH45Tcold"
instaseisDB = "http://instaseis.ethz.ch/blindtest_1s/{}_1s/".format(db_short)
maxRetry = 25
db = instaseis.open_db(instaseisDB)
f1 = 0.005
f2 = 2.0
t0 = UTCDateTime(2019,3,1)
dbdt = db.info['dt']
dbnpts = db.info['npts']

# Tremor parameters Should define a way to set this from a file or command line
# depth_in_km = 6.0
# depth = 1.e3*depth_in_km
depths = [2000., 6000., 60000.]
# pratio = 1.01
# mu = 7.e9
# rho = 2.7e3
Ls = [50., 200., 600.]
# aspect_ratio = 7.
# width = L*aspect_ratio
# Test a range of eta to get different behaviors
# n_eta = 10
n_eta = 10
eta = np.zeros(n_eta)
eta[0] = 0.1
for i in range(1,len(eta)):
    eta[i] = eta[i-1] * 1.5



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

sts = [[]]
durs = [[]]
m0ts = [[]]
m0as = [[]]

for i, depth in enumerate(depths):
    sts.append([])
    durs.append(np.array([]))
    m0ts.append(np.array([]))
    m0as.append(np.array([]))
    for L in Ls:
        # Do tremor calculations
        model = tremor.TremorModel(depth=depth, L=L)
        model.calc_derived()
        model.set_eta(eta)
        model.calc_R()
        model.calc_f()

        # Calculate the crack motions
        tmax = 2000.0
        tremordt = 0.1
        vi = 0.0
        hi = 1.0
        ui = 0.0
        w0 = [vi, hi, ui]
        tarray, wsol = model.generate_tremor(tmax, tremordt, w0)

        # Taper tremor time series and calculate moments
        taper_frac = 0.05
        durations = model.get_durations(taper=taper_frac, threshold=0.1)
        durs[i] = np.concatenate((durs[i], durations))
        m0_total, m0_average = model.get_moments(taper=taper_frac,
                                                 window=durations)
        m0ts[i] = np.concatenate((m0ts[i], m0_total))
        m0as[i] = np.concatenate((m0as[i], m0_average))

        print("Total moment: ", ' '.join('{:.3E}'.format(k) for k in m0_total))
        print("Average moment: ", ' '.join('{:3E}'.format(k)
                                           for k in m0_average))
        print("Durations: ", ' '.join('{}'.format(k) for k in durations))
        print("Frequencies: ", ' '.join('{:.5f}'.format(k) for k in model.f))

        # Set the Instaseis source parameters
        for j in range(n_eta):
            sliprate = model.u[j]
            slipdt = tremordt
            M0 = m0_total[j]
            source = instaseis.source.Source(latitude=slat, longitude=slon,
                                             depth_in_m=depth,
                                             m_rr=m_rr*M0, m_tt=m_tt*M0,
                                             m_pp=m_pp*M0,
                                             m_rt=m_rt*M0, m_rp=m_rp*M0,
                                             m_tp=m_tp*M0,
                                             origin_time=t0)
            source.set_sliprate(sliprate, slipdt)
            source.resample_sliprate(dt=dbdt, nsamp=len(model.u[j]))

            # Renormalize sliprate with absolute value appropriate for oscillatory
            # sliprates with negative values
            source.sliprate /= np.trapz(np.absolute(source.sliprate),
                                        dx=source.dt)

            receiver = instaseis.Receiver(latitude=rlat, longitude=rlon,
                                          network='XB',
                                          station='ELYSE')

            sts[i].append(db.get_seismograms(source=source, receiver=receiver,
                                             kind='acceleration',
                                             remove_source_shift=False,
                                             reconvolve_stf=True))


# Make some plots
vamps = []
for i in range(len(depths)):
    vamps.append(np.array([st[0].max() for st in sts[i]]))
    print("Max vertical amplitude at depth {}: ".format(depths[i]),
          ' '.join('{:.3E}'.format(k) for k in vamps[i]))
print(sts)
print(vamps)
print(durs)
print(m0ts)
print(m0as)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(m0ts[i], np.abs(vamps[i]), symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("m0total_vamp.png")
plt.close(fig)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(m0as[i], np.abs(vamps[i]), symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("m0average_vamp.png")
plt.close(fig)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(durs[i], np.abs(vamps[i]), symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))             
plt.yscale('log')
plt.xscale('linear')
plt.legend(loc='lower right')
plt.savefig("durations_vamp.png")
plt.close(fig)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(durs[i], m0ts[i], symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
plt.yscale('log')
plt.xscale('linear')
plt.legend(loc='lower right')
plt.savefig("durations_m0total.png")
plt.close(fig)





