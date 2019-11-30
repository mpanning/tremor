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
# from scipy.optimize import curve_fit
from scipy.stats import linregress
from tqdm import tqdm
from pandas import DataFrame
import statsmodels.api as sm

symbols = ['ro', 'bo', 'go', 'ko']
lines = ['r-', 'b-', 'g-', 'k-']
verbose = False

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
# depths = [2000.]
# pratio = 1.01
pratios = [1.001, 1.01, 1.10]
# pratios = [1.001, 1.01]
# mu = 7.e9
# rho = 2.7e3
Ls = [50., 200., 600.]
wls = [5., 15., 50.]
# Ls = [200.]
# aspect_ratio = 7.
# width = L*aspect_ratio
# Test a range of eta to get different behaviors
# n_eta = 10
n_eta = 5
eta = np.zeros(n_eta)
eta[0] = 0.1
for i in range(1,len(eta)):
    eta[i] = eta[i-1] * 2.5



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

sts = []
durs = []
m0ts = []
m0as = []
fs = []
Rs = []

for i, depth in enumerate(depths):
    print("\n\n\nWorking on depth {} of {}\n".format(i+1, len(depths)))
    sts.append([])
    durs.append(np.array([]))
    m0ts.append(np.array([]))
    m0as.append(np.array([]))
    fs.append(np.array([]))
    Rs.append(np.array([]))
    for pratio in tqdm(pratios):
        for L in tqdm(Ls):
            for wl in tqdm(wls):
                # Do tremor calculations
                width = wl*L
                model = tremor.TremorModel(depth=depth, L=L, pratio=pratio,
                                           width=width)
                model.calc_derived()
                model.set_eta(eta)
                model.calc_R()
                model.calc_f()
                fs[i] = np.concatenate((fs[i], model.f))
                Rs[i] = np.concatenate((Rs[i], model.R))

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
                durations = model.get_durations(taper=taper_frac, threshold=0.2)
                durs[i] = np.concatenate((durs[i], durations))
                # print("Test", durations)
                m0_total, m0_average = model.get_moments(taper=taper_frac,
                                                         window=durations)
                m0ts[i] = np.concatenate((m0ts[i], m0_total))
                m0as[i] = np.concatenate((m0as[i], m0_average))

                if verbose:
                    print("Total moment: ", ' '.join('{:.3E}'.format(k)
                                                     for k in m0_total))
                    print("Average moment: ", ' '.join('{:3E}'.format(k)
                                                       for k in m0_average))
                    print("Durations: ", ' '.join('{}'.format(k) for k in
                                                  durations))
                    print("Frequencies: ", ' '.join('{:.5f}'.format(k)
                                                    for k in model.f))

                # Set the Instaseis source parameters
                for j in tqdm(range(n_eta)):
                    sliprate = model.u[j]
                    slipdt = tremordt
                    M0 = m0_total[j]
                    source = instaseis.source.Source(latitude=slat,
                                                     longitude=slon,
                                                     depth_in_m=depth,
                                                     m_rr=m_rr*M0, m_tt=m_tt*M0,
                                                     m_pp=m_pp*M0,
                                                     m_rt=m_rt*M0, m_rp=m_rp*M0,
                                                     m_tp=m_tp*M0,
                                                     origin_time=t0)
                    source.set_sliprate(sliprate, slipdt)
                    source.resample_sliprate(dt=dbdt, nsamp=len(model.u[j]))

                    # Renormalize sliprate with absolute value appropriate for
                    # oscillatory
                    # sliprates with negative values
                    source.sliprate /= np.trapz(np.absolute(source.sliprate),
                                                dx=source.dt)

                    receiver = instaseis.Receiver(latitude=rlat, longitude=rlon,
                                                  network='XB',
                                                  station='ELYSE')

                    sts[i].append(db.get_seismograms(source=source,
                                                     receiver=receiver,
                                                     kind='acceleration',
                                                     remove_source_shift=False,
                                                     reconvolve_stf=True))


# Make some plots
vamps = []
for i in range(len(depths)):
    # vamps.append(np.array([np.sqrt(np.mean(st[0].data**2)) for st in sts[i]]))
    vamps.append([])
    for j in range(len(sts[i])):
        imax = np.where(sts[i][j][0].data == sts[i][j][0].data.max())[0][0]
        i1 = max(imax - int(50.0/dbdt), 0)
        i2 = min(imax + int(durs[i][j]/dbdt), len(sts[i][j][0].data))
        vamps[i].append(np.sqrt(np.mean(sts[i][j][0].data[i1:i2]**2)))
        # Pick 50 s before and go to duration for RMS calculation
    vamps[i] = np.array(vamps[i])
vamps_flat = np.array(vamps).flatten()
m0ts_flat = np.array(m0ts).flatten()
m0as_flat = np.array(m0as).flatten()
fs_flat = np.array(fs).flatten()
Rs_flat = np.array(Rs).flatten()
    
fig = plt.figure()
print("M0 total vs. vertical amplitude")
for i in range(len(depths)):
    a, b, r, p, std_err = linregress(np.log(m0ts[i]), np.log(vamps[i]))
    coeff = math.exp(b)
    plt.plot(m0ts[i], np.abs(vamps[i]), symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
    newX = [m0ts[i].min(), m0ts[i].max()]
    plt.plot(newX, np.power(newX, a)*coeff, lines[i],
             label=r'y={:.3e} $x^{{{:.3f}}}$'.format(coeff, a))
    print("\tDepth {} km coefficents: {:.4e}*x**{:.4f}".format(int(depths[i]*1.e-3),
                                                               coeff, a))
    print("\tR squared {}".format(r))
# Fit all data as well
a, b, r, p, std_err = linregress(np.log(m0ts_flat), np.log(vamps_flat))
coeff = math.exp(b)
newX = [m0ts_flat.min(), m0ts_flat.max()]
plt.plot(newX, np.power(newX, a)*coeff, 'k-',
         label=r'y={:.3e} $x^{{{:.3f}}}$'.format(coeff, a))
print("Overall coefficents: {:.4e}*x**{:.4f}".format(coeff, a))
print("R squared {}".format(r))          
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("{}_m0total_vamp.png".format(db_short))
plt.close(fig)

fig = plt.figure()
print("M0 average vs. vertical amplitude")
for i in range(len(depths)):
    a, b, r, p, std_err = linregress(np.log(m0as[i]), np.log(vamps[i]))
    plt.plot(m0as[i], np.abs(vamps[i]), symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
    newX = [m0as[i].min(), m0as[i].max()]
    coeff = math.exp(b)
    plt.plot(newX, np.power(newX, a)*coeff, lines[i],
             label=r'y={:.3e} $x^{{{:.3f}}}$'.format(coeff, a))
    print("\tDepth {} km coefficents: {:.4e}*x**{:.4f}".format(int(depths[i]*1.e-3), coeff, a))
    print("\tR squared {}".format(r))
# Fit all data as well
a, b, r, p, std_err = linregress(np.log(m0ts_flat), np.log(vamps_flat))
coeff = math.exp(b)
newX = [m0ts_flat.min(), m0ts_flat.max()]
plt.plot(newX, np.power(newX, a)*coeff, 'k-',
         label=r'y={:.3e} $x^{{{:.3f}}}$'.format(coeff, a))
print("Overall coefficents: {:.4e}*x**{:.4f}".format(coeff, a))
print("R squared {}".format(r))          
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("{}_m0average_vamp.png".format(db_short))
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
plt.savefig("{}_durations_m0total.png".format(db_short))
plt.close(fig)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(durs[i], Rs[i], symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
plt.yscale('log')
plt.xscale('linear')
plt.legend(loc='lower right')
plt.savefig("durations_R.png")
plt.close(fig)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(m0ts[i], Rs[i], symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
plt.yscale('linear')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("{}_m0total_R.png".format(db_short))
plt.close(fig)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(m0ts[i], fs[i], symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("{}_m0total_f.png".format(db_short))
plt.close(fig)

print("Frequency vs. vertical amplitude")
fig = plt.figure()
for i in range(len(depths)):
    a, b, r, p, std_err = linregress(np.log(fs[i]), np.log(vamps[i]))
    coeff = math.exp(b)
    plt.plot(fs[i], vamps[i], symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
    newX = [fs[i].min(), fs[i].max()]
    plt.plot(newX, np.power(newX, a)*coeff, lines[i],
             label=r'y={:.3e} $x^{{{:.3f}}}$'.format(coeff, a))
    print("\tDepth {} km coefficents: {:.4e}*x**{:.4f}".format(int(depths[i]*1.e-3), coeff, a))
    print("\tR squared {}".format(r))
a, b, r, p, std_err = linregress(np.log(fs_flat), np.log(vamps_flat))
coeff = math.exp(b)
newX = [fs_flat.min(), fs_flat.max()]
plt.plot(newX, np.power(newX, a)*coeff, 'k-',
         label=r'y={:.3e} $x^{{{:.3f}}}$'.format(coeff, a))
print("Overall coefficents: {:.4e}*x**{:.4f}".format(coeff, a))
print("R squared {}".format(r))          

plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("{}_f_vamp.png".format(db_short))
plt.close(fig)

fig = plt.figure()
for i in range(len(depths)):
    plt.plot(Rs[i], vamps[i], symbols[i],
             label='{:d}km'.format(int(1.e-3*depths[i])))
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')
plt.savefig("{}_R_vamp.png".format(db_short))
plt.close(fig)

# Try multiple linear regression on log values
TremorValues = {'Vertical_Amplitude': np.log(vamps_flat),
                'Seismic_Moment_Total': np.log(m0ts_flat),
                'Seismic_Moment_Average': np.log(m0as_flat),
                'Frequency': np.log(fs_flat),
                'R_Value': np.log(Rs_flat)
                }
df = DataFrame(TremorValues, columns=['Vertical_Amplitude',
                                      'Seismic_Moment_Total', 'Frequency',
                                      'R_Value', 'Seismic_Moment_Average'])
X = df[['Seismic_Moment_Total']]
Y = df['Vertical_Amplitude']

X = sm.add_constant(X)
smodel = sm.OLS(Y, X).fit()
predictions = smodel.predict(X)
print_model = smodel.summary()
print(print_model)


               


