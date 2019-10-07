"""
Program to perform Markov chain Monte Carlo inversion of seismic observables
for acceptable range of tremor parameters based on the work of Julian, 1994
"""

import numpy as np
import math
from tqdm import tqdm
import datetime
import os
from MCMC_functions import (startmodel, startchain, finderror, accept_reject,
			    errorfig, accratefig, set_global)
import tremor
from obspy.core import UTCDateTime
import instaseis
import string
import copy
from random import randint
import pickle
import matplotlib.pyplot as plt
import pylab as P
import sys
import argparse

VERSION = 0.1

# from MCMC_functions import (startmodel,MODEL,startchain,runmodel,finderror,
# 			    randINTF,accept_reject,mintwo,runmodel_bw,errorfig,
			    # accratefig,nlhist,sighhist,modfig,vdispfig)

# ----------------------------------------------------------------------------
# ****************************************************************************
# --------------------------Set up INPUT information--------------------------
# # directory where working
# MAIN = '/nobackup/planetseismology/panning'
# os.chdir(MAIN)
# # MAIN = os.getcwd()

# Check to see if we're running in parallel mode
ifParallel = False

# Parse command line with argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", help="display version number and exit",
		    action="store_true")
parser.add_argument("-V", "--verbose", help="display full diagnostic output",
		    action="store_true")
parser.add_argument("-p", "--parallel",
		    help="set parallel processing index number", type=int)
parser.add_argument("-d", "--directory",
		    help="set root directory for model output")
args = parser.parse_args()
if args.version:
	print("MCMC_main version {}".format(VERSION))
	exit()
if args.verbose:
	ifVerbose = True
else:
	ifVerbose = False
set_global(verbflag=ifVerbose)
if args.parallel is not None:
	ifParallel = True
	parallelIndex = args.parallel
	print("Setting parallel mode with index {}".format(parallelIndex))
if args.directory is not None:
	MAIN = args.directory
	os.chdir(MAIN)
else:
	MAIN = os.getcwd()

now = datetime.datetime.now()
foldername = now.strftime("%m_%d_%Y_%H_%M")
if ifParallel:
        foldername = foldername + "_" + "{:04d}".format(parallelIndex)
os.mkdir(foldername)

SAVEMs = MAIN + '/' + foldername

# Overall parameters
max_duration = 2000. # length of tremor record to calculate
tremor_dt = 0.2
tremor_vi = 0.0
tremor_hi = 1.0
tremor_ui = 0.0
tremor_w0 = np.array([tremor_vi, tremor_hi, tremor_ui]) # initial conditions
dur_taper = 0.05
dur_threshold = 0.2
minR = 0.1 # Set these ridiculously wide to test implementing R as data
maxR = 2.0
fmax = 0.8

# Instaseis stuff.  Need to make alternate method for amplitude that skips this
ifInstaseis = False
if ifInstaseis:
	if ifVerbose:
		print("Initializing Instaseis database")
	db_short = "EH45Tcold"
	instaseisDB = ("http://instaseis.ethz.ch/blindtest_1s/" +
                       "{}_1s/".format(db_short))
	maxRetry = 25
	db = instaseis.open_db(instaseisDB)
	f1 = 0.005
	f2 = 2.0
	t0 = UTCDateTime(2019,3,1)
	dbdt = db.info['dt']
	dbnpts = db.info['npts']

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

# Scaling parameters
# Assume a power law scaling to estimated seismic moment: A = c0*M0^alpha
# Parameters are estimated from regression of numbers from amp_scaling.py
c0 = 3.523e-19
alpha = 0.607


# Make data vector.  Right now is hard-coded, but will adjust to read from file
freq_obs = 0.35 # Dominant frequency of signal (Hz)
period_obs = 1./freq_obs
amp_obs = 1.5e-9 # Acceleration amplitude (m/s^2)
# dur_obs = 1000.0 # Duration of observed signal (s)
# Instead of unstable dur measurement, just use R value instead
# dobs = np.array([freq_obs, amp_obs])
dobs = np.array([period_obs, amp_obs])
ndata = len(dobs)

# Uncertainty estimates - 1 sigma
wsig_freq = 0.05
wsig_period = 0.5
wsig_amp = 1.e-9
# wsig_dur = 300.
# wsig = np.array([wsig_freq, wsig_amp])
wsig = np.array([wsig_period, wsig_amp])

# Additional prior constraints
# Add in a misfit constraint based on R values and append to data vector
R_expected = 0.95
R_sigma = 0.05
dobs = np.append(dobs, R_expected)
ndata += 1
wsig = np.append(wsig, R_sigma)

# create boss matrix to control all combinations of starting depths
depopt = ([6.0, 60.0])
muopt = ([7.e9, 70.e9])
# repeat = 1
all_letters = list(string.ascii_lowercase)
# letters = all_letters[0:repeat]
abc=[]

DRAW = ['CHANGE CHANNEL LENGTH: Perturb the length of oscillating channel',
        'CHANGE VISCOSITY: Change the viscosity of the fluid in the channel',
        'CHANGE PRATIO: Change the overpressure ratio of the lower reservoir',
        'CHANGE ASPECT: Change the width to length aspect ratio of channel',]
# ---------------------------------------------------------------------------------------
# ----------------
# totch = 10                    # Total number of chains
# numm = 1000
totch = 1
numm = 10000                      # Number of iterations per chain
# ----------------
# ---------------------------------------------------------------------------------------
# ----------------
BURN = 1000
# BURN = 2000           # Number of models designated BURN-IN, gets discarded
# M = 10                        # Interval to keep models (e.g. keep every 100th model, M=100)
M = 10
MMM = np.arange(BURN-1,numm+M,M)
# ----------------
# ---------------------------------------------------------------------------------------
#########Option to weight data points######### 
######### by variance or stand. dev. ######### 
weight_opt = 'ON'
#weight_opt = 'OFF'
# --------------------------------------------
doptnum = len(depopt)
# numrun = doptnum*repeat
numrun = doptnum

boss = np.zeros((numrun, 2))
k=0
for i, dep in enumerate(depopt):
        if ifVerbose:
                print('starting depth option: {}'.format(dep))
        # r = 0
        # while r < repeat:
        #         abc.append(letters[r])
        #         r = r + 1
        #         boss[k:k+repeat,0]=dep
        #         boss[k:k+repeat,1]=muopt[i]
        #         k=k+repeat
        abc.append(all_letters[i])
        boss[i, 0] = dep
        boss[i, 1] = muopt[i]

reprunsPHI = []
reprunsdiagCE = []
reprunsHIST = []

# Standard deviations for the Gaussian proposal distributions

# Given that many of these vary over orders of magnitude, maybe should use
# log value as input, and thus it becomes log-normal perturbation.  Consider
# this and whether it should effect acceptance criteria
thetaL = np.log(1.05) # Length perturbations
thetaETA = np.log(1.05) # Using a log normal perturbation instead
thetaPR = np.log(1.005) # Pressure ratio perturbatio
thetaWL = np.log(1.05) # Using a log normal perturbation instead

# Use higher values during burn-in time period if highburn True
highburn = False
if highburn:
        thetaL_burn = np.log(1.2)
        thetaETA_burn = np.log(1.2)
        thetaPR_burn = np.log(1.02)
        thetaWL_burn = np.log(1.2)
else:
        thetaL_burn = thetaL
        thetaETA_burn = thetaETA
        thetaPR_burn = thetaPR
        thetaWL_burn = thetaWL
# ---------------------------------------------------------------------------------------

savefname = "saved_models"
SAVEF = SAVEMs + '/' + savefname
os.mkdir(SAVEF)

bossfile = open(SAVEMs+'/boss.txt', 'w')
bossfile.write(now.strftime("%m_%d_%Y, %H:%M")+'\n')
bossfile.write('   ' + '\n')
bossfile.write('Depth of source (km)   Mu at source depth (Pa)\n')
k=0
while (k < numrun):
        # maxdepth = boss[k,1]
        depth = boss[k, 0]
        mu = boss[k, 1]
        writestring = "          {:6.2f}              {:e}\n".format(depth, mu)
        bossfile.write(writestring)
        k = k + 1
bossfile.write('   ' + '\n')
bossfile.write('TOTAL # OF CHAINS: '+str(totch)+'    ITERATIONS: '+str(numm)+
               '\n')
bossfile.write('   ' + '\n')
bossfile.close()

Elog = open(SAVEMs+'/'+'errorlog.txt','w')

#### Select the colormaps desired for the output pdf figures
#### options are GREY, GREY_rev, HOT, HOT_rev
pdfcmap=('GREYS_rev','HOT')

RUNMODS = []
runPHI = []
rnummods = len(RUNMODS)         
# runINTF=[]    
# runNM=[]      
# runSIGH=[]
# rep_cnt = 0
runLENGTH = []
runETA = []
runWL = []
runPR = []

print("Starting loop on {} runs".format(numrun))
for run in range(numrun):       
        print("Working on run {}".format(run))
        # -------------------------------------------------------------------
        # --------------------------Set up MODEL information-----------------
        boss0 = boss[run,0]
        boss1 = boss[run,1]

        # Set parameter bounds
        Lmin = 1.0
        Lmax = 1000.0

        etamin = 1.0
        etamax = 1000.0

        prmin = 1.00001
        prmax = 1.2

        wlmin = 5.0
        wlmax = 100.0

        # if rep_cnt == repeat:
        #         rep_cnt = 0
        #         RUNMODS = []
        #         reprunsPHI = []
        #         reprunsHIST = []
        #         BEST_RUNMODS = []
        #         BESTreprunPHI = []
        #         BESTreprunNL = []
        #         BESTrerunsSIGH = []
        #         savefname = 'saved_initial'
        #         SAVEF = SAVEMs + '/' + savefname
        #         os.mkdir(SAVEF)


        CHMODS = []
        BEST_CHMODS = []

        # Set up totch chains:------------------------
        stL, steta, stpratio, stwl = startmodel(totch, Lmin, Lmax, etamin,
                                                etamax, prmin, prmax, wlmin,
                                                wlmax)


        acc_rate = np.zeros(totch)
        draw_acc_rate = np.zeros((len(DRAW),totch))

        """      ------  Loop through multiple chains:  -------------     """
        if ifVerbose:
                print("Starting loop on chains")
        for chain in range(totch):

                # Create new tremor model with starting parameters
                if ifVerbose:
                        print("Initializing starting model")
                x = tremor.TremorModel(depth=boss0*1.e3,
                                       pratio=stpratio[chain], mu=boss1,
                                       L=stL[chain],
                                       width=stwl[chain]*stL[chain])

                # Get basic parameters
                eta = steta[chain]
                x.set_eta(eta)
                x.calc_derived()
                x.calc_R()
                x.calc_f()
                # Only do a model if f is less than observed + 2 sigma
                # freq_limit = freq_obs + 2.*wsig_freq
                if (x.f[0] > fmax or x.R[0] < minR or x.R[0] > maxR):
                        if ifVerbose:
                                print("Frequency too high or R out of range, "
                                      + "generating new startmodel")
                        istart=0
                        while (x.f[0] > fmax or x.R[0] < minR
                               or x.R[0] > maxR):
                                # Generate a new starting model and calc f
                                istart += 1
                                if (istart % 10 == 0 and ifVerbose):
                                        print("Attempt {}".format(istart))
                                L, eta, pratio, wl = startchain(Lmin, Lmax,
                                                                etamin, etamax,
                                                                prmin, prmax,
                                                                wlmin, wlmax)
                                x = tremor.TremorModel(depth=boss0*1.e3,
                                                       pratio=pratio, mu=boss1,
                                                       L=L, width=wl*L)
                                x.set_eta(eta)
                                x.calc_derived()
                                x.calc_R()
                                x.calc_f()

                                
                # Calculate other data parameters
                if ifVerbose:
                        print("Running tremor model")
                x.generate_tremor(max_duration, tremor_dt, tremor_w0)
                duration = x.get_durations(taper=dur_taper,
                                           threshold=dur_threshold)
                dur_pre = duration[0]

                # For now, use instaseis for amplitudes... may be slow
                # Takes the RMS amplitude of vertical component over a window
                # starting 50 seconds before the max amplitude and extending
                # over the calculated source duration
                m0_total, m0_average = x.get_moments(window=duration)
                if ifInstaseis:
                        if ifVerbose:
                                print("Running Instaseis modeling")
                        sliprate = x.u[0]
                        slipdt = tremor_dt
                        M0 = m0_total[0]
                        source = instaseis.source.Source(latitude=slat,
                                                         longitude=slon,
                                                         depth_in_m=depth,
                                                         m_rr=m_rr*M0,
                                                         m_tt=m_tt*M0,
                                                         m_pp=m_pp*M0,
                                                         m_rt=m_rt*M0,
                                                         m_rp=m_rp*M0,
                                                         m_tp=m_tp*M0,
                                                         origin_time=t0)
                        source.set_sliprate(sliprate, slipdt)
                        source.resample_sliprate(dt=dbdt, nsamp=len(x.u[0]))
                        # Renormalize sliprate with absolute value appropriate
                        # for oscillatory sliprates with negative values
                        source.sliprate /= np.trapz(np.absolute(source.sliprate), dx=source.dt)
                        receiver = instaseis.Receiver(latitude=rlat,
                                                      longitude=rlon,
                                                      network='XB',
                                                      station='ELYSE')

                        st = db.get_seismograms(source=source,
                                                receiver=receiver,
                                                kind='acceleration',
                                                remove_source_shift=False,
                                                reconvolve_stf=True)

                        imax = np.where(st[0].data == st[0].data.max())[0][0]
                        i1 = max(imax - int(50.0/dbdt), 0)
                        i2 = min(imax + int(dur_pre/dbdt), len(st[0].data))
                        vamp = np.sqrt(np.mean(st[0].data[i1:i2]**2))
                else:
                        M0 = m0_total[0]
                        vamp = c0*math.pow(M0, alpha)
                        # raise NotImplementedError("Amplitude by scaling is " +
                        #                           "not yet implemented")
                
                dpre = np.zeros((ndata,numm))
                # dpre[:,0] = np.array([x.f[0], vamp, x.R[0]])
                dpre[:,0] = np.array([1./x.f[0], vamp, x.R[0]])
                x.number = 0
                x.dpre = copy.deepcopy(dpre[:,0])

                # dpre = np.zeros((ndata,numm))
                # dpre[:,0] = np.concatenate([np.concatenate(dpre_sw), 
                #                        np.concatenate(dpre_bw)])
                # x.dpre = dpre[:,0]

                # CALCULATE MISFIT BETWEEN D_PRE AND D_OBS:
                # CONTINUE BODY WAVE MOD FROM HERE
                misfit = np.zeros(ndata)
                newmis = np.zeros(ndata)
                diagCE = np.zeros((ndata,numm))
                PHI = np.zeros(numm)
                
                misfit,newmis,PHI,x,diagCE = finderror((-1),x,ndata,dpre,dobs,
                                                       misfit,newmis,wsig,PHI,
                                                       diagCE,weight_opt)
                                                                              
                ITMODS = []
                x.reduce_size() # Remove some large arrays from x to save space
                ITMODS.append(x)

                numreject = 0.0
                numaccept = 0.0
                drawnumreject = np.zeros(len(DRAW))
                drawnumaccept = np.zeros(len(DRAW))

                keep_cnt = 0
                
                # =============================================================
                k = 0
                while (k < (numm-1)):
                        if ifVerbose:        
                                print("=======================================")
                                print("                   CHAIN # [" +
                                      str(chain)+ "]    ITERATION # [" +
                                      str(k) + "]" )
                                print(" ")

                        # Set previous model object as "oldx" so can call on 
                        # it's attributes when needed
                        oldx = copy.deepcopy(ITMODS[k])

                        # Copy old model to new model and update number
                        # Perturbation steps then only need to change
                        # elements that are perturbed
                        x = copy.deepcopy(oldx)
                        x.number = k+1

                        WARN_BOUNDS = False

                        ########### Draw a new model ########################
                        # Choose random integer between 0 and 3 such that each 
                        # of the 4 options
                        # (Change L, Change Viscosity, Change Pressure Ratio,
                        # Change Aspect Ratio) have
                        #  a 1/4 chance of being selected

                        pDRAW = randint(0,3)

                        # Change channel length
                        if pDRAW == 0:
                                if ifVerbose:
                                        print(DRAW[pDRAW])

                                if k < BURN:
                                        theta = thetaL_burn
                                else:
                                        theta = theta

                                wL = np.exp(np.random.normal(0,theta))
                                newL = wL * x.L
                                Ldiff = newL - x.L
                                
                                if ifVerbose:
                                        print('Perturb channel length by ' + 
                                              str(Ldiff) + ' km')

                                if ((newL < Lmin) or 
                                    (newL > Lmax)):
                                        if ifVerbose:
                                                print("!!! Outside channel " +
                                                      "length range")
                                                print("Automatically REJECT " +
                                                      "model")
                                        WARN_BOUNDS = True
                                else:
                                        x = tremor.TremorModel(depth=boss0*1.e3,
                                                               pratio=oldx.pratio,
                                                               mu=boss1,
                                                               L=newL,
                                                               width=oldx.wl*newL)
                                        x.set_eta(oldx.eta)
                                # vsOUT = copy.deepcopy(vsIN)
                                # BDi = 0
                                # delv2 = 0

                        # Change viscosity
                        if pDRAW == 1:
                                if ifVerbose:
                                        print(DRAW[pDRAW])

                                if k < BURN:
                                        theta = thetaETA_burn
                                else:
                                        theta = thetaETA

                                wETA = np.exp(np.random.normal(0,theta))

                                newETA = x.eta[0] * wETA
                                ETAdiff = newETA - x.eta[0]
                                
                                if ifVerbose:
                                        print('Perturb viscosity by ' +
                                              str(ETAdiff) + ' Pa s')

                                # newETA = x.eta[0] + wETA

                                if ((newETA < etamin) or 
                                    (newETA > etamax)):
                                        if ifVerbose:
                                                print ("!!! Outside viscosity "
                                                       + "range\n")
                                                print("Automatically REJECT " +
                                                      "model")
                                        WARN_BOUNDS = True
                                else:
                                        x.set_eta(newETA)
                                # vsOUT = copy.deepcopy(vsIN)
                                # BDi = 0
                                # delv2 = 0

                        # Change pressure ratio
                        if pDRAW == 2:
                                if ifVerbose:
                                        print(DRAW[pDRAW])

                                if k < BURN:
                                        theta = thetaPR_burn
                                else:
                                        theta = thetaPR

                                wPR = np.exp(np.random.normal(0, theta))
                                newPR = wPR * x.pratio
                                PRdiff = newPR - x.pratio

                                if ifVerbose:
                                        print('Perturb pressure ratio by ' +
                                              str(PRdiff))

                                # newPR = x.pratio + wPR

                                if (newPR < prmin) or (newPR > prmax):
                                        if ifVerbose:
                                                print("!!! Outside pressure " +
                                                      "ratio range\n")
                                                print("Automatically REJECT " +
                                                      "model")
                                        WARN_BOUNDS = True
                                else:
                                        x = tremor.TremorModel(depth=boss0*1.e3,
                                                               pratio=newPR,
                                                               mu=boss1,
                                                               L=oldx.L,
                                                               width=oldx.wl*oldx.L)
                                        x.set_eta(oldx.eta)


                        # Change aspect ratio        
                        elif pDRAW == 3:
                                if ifVerbose:
                                        print(DRAW[pDRAW])

                                if k < BURN:
                                        theta = thetaWL_burn
                                else:
                                        theta = thetaWL

                                wWL = np.exp(np.random.normal(0, theta))

                                newWL = x.wl * wWL
                                WLdiff = newWL - x.wl

                                if ifVerbose:
                                        print('Perturb aspect ratio by ' +
                                              str(WLdiff))

                                # newWL = x.wl + wWL

                                if (newWL < wlmin) or (newWL > wlmax):
                                        if ifVerbose:
                                                print("!!! Outside aspect " +
                                                      "ratio range\n")
                                                print("Automatically REJECT " +
                                                      "model")
                                        WARN_BOUNDS = True
                                else:
                                        x = tremor.TremorModel(depth=boss0*1.e3,
                                                               pratio=oldx.pratio,
                                                               mu=boss1,
                                                               L=oldx.L,
                                                               width=newWL*oldx.L)
                                        x.set_eta(oldx.eta)
                        # Calculate frequency
                        x.calc_derived()
                        x.calc_R()
                        x.calc_f()

                        if x.f[0] > fmax:
                                if ifVerbose:
                                        print("Frequency too high")
                                        print("Automatically REJECT model")
                                WARN_BOUNDS = True
                        elif (x.R[0] < minR or x.R[0] > maxR):
                                if ifVerbose:
                                        print("R out of range")
                                        print("Automatically REJECT model")
                                WARN_BOUNDS = True


                        # Continue as long as proposed model is not out of 
                        # bounds:
                        if WARN_BOUNDS:
                                if ifVerbose:
                                        print(' == ! == ! == ! ' +
                                              'REJECT NEW MODEL  ' +
                                              '! == ! == ! ==  ')
                                del x
                                numreject = numreject + 1
                                drawnumreject[pDRAW] = drawnumreject[pDRAW] + 1
                                continue
                                # re-do iteration -- DO NOT increment (k)

                        # Calculate predicted data
                        if ifVerbose:
                                print("Running tremor model")
                        x.generate_tremor(max_duration, tremor_dt, tremor_w0)
                        duration = x.get_durations(taper=dur_taper,
                                                   threshold=dur_threshold)
                        dur_pre = duration[0]

                        # For now, use instaseis for amplitudes... may be slow
                        # Takes the RMS amplitude of vertical component over a window
                        # starting 50 seconds before the max amplitude and extending
                        # over the calculated source duration
                        m0_total, m0_average = x.get_moments(window=duration)
                        if ifInstaseis:
                                if ifVerbose:
                                        print("Running Instaseis modeling")
                                sliprate = x.u[0]
                                slipdt = tremor_dt
                                M0 = m0_total[0]
                                source = instaseis.source.Source(latitude=slat,
                                                                 longitude=slon,
                                                                 depth_in_m=depth,
                                                                 m_rr=m_rr*M0,
                                                                 m_tt=m_tt*M0,
                                                                 m_pp=m_pp*M0,
                                                                 m_rt=m_rt*M0,
                                                                 m_rp=m_rp*M0,
                                                                 m_tp=m_tp*M0,
                                                                 origin_time=t0)
                                source.set_sliprate(sliprate, slipdt)
                                source.resample_sliprate(dt=dbdt,
                                                         nsamp=len(x.u[0]))
                                # Renormalize sliprate with absolute value
                                # appropriate
                                # for oscillatory sliprates with negative values
                                source.sliprate /= np.trapz(np.absolute(source.sliprate),
                                                            dx=source.dt)
                                receiver = instaseis.Receiver(latitude=rlat,
                                                              longitude=rlon,
                                                              network='XB',
                                                              station='ELYSE')

                                st = db.get_seismograms(source=source,
                                                        receiver=receiver,
                                                        kind='acceleration',
                                                        remove_source_shift=False,
                                                        reconvolve_stf=True)

                                imax = np.where(st[0].data == st[0].data.max())[0][0]
                                i1 = max(imax - int(50.0/dbdt), 0)
                                i2 = min(imax + int(dur_pre/dbdt),
                                         len(st[0].data))
                                vamp = np.sqrt(np.mean(st[0].data[i1:i2]**2))
                                # print("Amp stuff {} {} {} {:.1f} {:.3E}".format(imax, i1, i2, dur_pre, vamp))
                        else:
                                M0 = m0_total[0]
                                vamp = c0 * math.pow(M0, alpha)
                                # raise NotImplementedError("Amplitude by scaling is " +
                                #                           "not yet implemented")
                        # dpre[:, k+1] = np.array([x.f[0], vamp, x.R[0]])
                        dpre[:, k+1] = np.array([1./x.f[0], vamp, x.R[0]])
                        x.dpre = copy.deepcopy(dpre[:,k+1])
                        x.number = k + 1

                        # Calculate error of the new model:
                        if ifVerbose:
                                print("Freq {:.4f}, Amplitude {:.3E}, Duration {:.1f}, R {:.3f}".format(x.f[0], vamp, dur_pre, x.R[0]))
                                print("L {:.1f} eta {:.1f} pratio {:.6f} aspect {:.1f}".format(x.L, x.eta[0], x.pratio, x.wl))
                                print("M0 {:.3E}".format(M0))
                        misfit,newmis,PHI,x,diagCE = finderror(k,x,ndata,dpre,
                                                               dobs,misfit,
                                                               newmis,wsig,PHI,
                                                               diagCE,
                                                               weight_opt)

                        pac,q = accept_reject(PHI,k,pDRAW,WARN_BOUNDS)
        
                        if pac < q:
                                if ifVerbose:
                                        print (' == ! == ! == !  ' +
                                               'REJECT NEW MODEL  ' +
                                               '! == ! == ! ==  ')
                                del x
                                PHI[k+1] = PHI[k]

                                numreject = numreject + 1
                                drawnumreject[pDRAW] = drawnumreject[pDRAW] + 1
                                # re-do iteration -- DO NOT increment (k)
                        else: 
                                if ifVerbose:
                                        print('******   ******  ' +
                                              'ACCEPT NEW MODEL  ' +
                                              '******   ******')

                                # Retain MODEL
                                x.reduce_size()
                                ITMODS.append(x)

                                numaccept = numaccept + 1
                                drawnumaccept[pDRAW] = drawnumaccept[pDRAW] + 1
                                
                                if k == MMM[keep_cnt]:
                                        print ('Adding model #' + str(k + 1) +
                                               ' to the CHAIN ensemble')
                                        CHMODS.append(x)
                                        # modl = x.filename
                                        # curlocation = MAIN + '/' + modl
                                        # newfilename = ('M' + str(ii) + '_' + 
                                        #                abc[run] + '_' + 
                                        #                str(chain)+ '_' + modl)
                                        # newlocation = SAVEF + '/' + newfilename
                                        # shutil.copy2(curlocation, newlocation)
                                        # Try to also save a binary version of the
                                        # MODEL class object
                                        modl = "{:06d}".format(x.number)
                                        if ifParallel:
                                                parl = "{:04d}".format(parallelIndex)
                                                classOutName = ('M_' + abc[run]
                                                                + '_' +
                                                                str(chain) +
                                                                '_' + modl +
                                                                '_' + parl +
                                                                '.pkl')
                                        else:
                                                classOutName = ('M' + '_' + 
                                                                abc[run] + '_'
                                                                + str(chain) +
                                                                '_' + modl +
                                                                '.pkl')
                                        newlocation = SAVEF + '/' + classOutName
                                        with open(newlocation, 'wb') as output:
                                                pickle.dump(x, output)
                                        keep_cnt = keep_cnt + 1

                                # #### Remove all models from current chain ####
                                # for filename in glob(MAIN+"/*.swm"):
                                #         os.remove(filename) 
                                # for filename in glob(MAIN+"/*.tvel"):
                                #         os.remove(filename) 
                                # for filename in glob(MAIN+"/*.npz"):
                                #         os.remove(filename) 
                                
                                # move to next iteration
                                
                                k = k + 1
                                
        
                # Calculate the acceptance rate for the chain:
                numgen = numreject + numaccept
                drawnumgen = drawnumreject + drawnumaccept
                acc_rate[chain] = (numaccept/numgen)*100
                draw_acc_rate[:,chain] = (drawnumaccept[:]/drawnumgen[:])*100.0
                # print(draw_acc_rate)

                """
                if errorflag1 == 'on':
                        print " "
                        print " error occurred on first model generation, "
                        print " no disp.out file found, try to re-do start of chain "
                        print " "
                        errorflag1 = 'off'
                else:
                """
                if ifVerbose:
                        print(PHI)
                inumm = numm + 0.0
                cc = ([plt.cm.brg(columnno/inumm) 
                       for columnno in range(numm)])
                plt.close('all')

                # # Ignore first BURN # of models as burn-in period
                # keepM = ITMODS[BURN:numm]
                # numremain = len(keepM)

                # # Keep every Mth model from the chain
                # ii = BURN - 1
                # while (ii < numm):
                #         sample = copy.deepcopy(ITMODS[ii])
                #         print ('Adding model #  [ '+str(sample.number)+
                #                ' ]  to the CHAIN ensemble')
                #         CHMODS.append(sample)
                #         modl = sample.filename
                #         curlocation = MAIN + '/' + modl
                #         newfilename = ('M'+str(ii)+'_'+abc[run]+ '_' 
                #                        + str(chain)+ '_' + modl)
                #         newlocation = SAVEF + '/' + newfilename
                #         shutil.copy2(curlocation, newlocation)
                #         # Try to also save a binary version of the
                #         # MODEL class object
                #         classOutName = ('M' + str(ii) + '_' + abc[run] +
                #                         '_' + str(chain) + '_' + modl +
                #                         '.pkl')
                #         newlocation = SAVEF + '/' + classOutName
                #         with open(newlocation, 'wb') as output:
                #                 pickle.dump(sample, output, -1)
                #         ii = ii + M

                # #### Remove all models from current chain ####
                # for filename in glob(MAIN+"/*.swm"):
                #         os.remove(filename) 
                # for filename in glob(MAIN+"/*.tvel"):
                #         os.remove(filename) 
                # for filename in glob(MAIN+"/*.npz"):
                #         os.remove(filename) 

                #### Plot the error at each iterations ####
                errorfig(PHI, BURN, chain, abc, run, SAVEF)

                # # Keep lowest error models from posterior distribution
                # realmin = np.argsort(PHI)
                # jj=0
                # while (jj < keep):                                
                #         ind=realmin[jj]
                #         sample = copy.deepcopy(ITMODS[ind])
                #         BEST_CHMODS.append(sample)
                #         jj = jj + 1        

                #### Advance to next chain ####
                chain = chain + 1
        

        #### Plot acceptance rate ####
        accratefig(totch, acc_rate, draw_acc_rate, abc, run, SAVEF)
        
        keptPHI = []
        nummods = len(CHMODS)                
        # INTF=[]        
        # NM=[]        
        # SIGH=[]
        LENGTH = []
        ETA = []
        WL = []
        PR = []
        jj = 0
        while (jj < nummods):
                sample = copy.deepcopy(CHMODS[jj])
                RUNMODS.append(sample)
                
                curPHI = sample.PHI
                keptPHI = np.append(keptPHI, curPHI)
                
                # newcol = 1000*(sample.intf)
                # newcol = np.array(sample.mantleR)
                # INTF = np.append(INTF, newcol)        
                # newnm = sample.nmantle
                # NM = np.append(NM, newnm)        
                # newsighyp = sample.sighyp
                LENGTH = np.append(LENGTH, sample.L)
                ETA = np.append(ETA, sample.eta[0])
                WL = np.append(WL, sample.wl)
                PR = np.append(PR, sample.pratio)
                # if (jj == 0):
                #         SIGH = copy.deepcopy(newsighyp)
                # else:
                #         SIGH=np.vstack((SIGH, newsighyp))
                jj = jj + 1
                
        # runINTF = np.append(runINTF, INTF)
        # runNM = np.append(runNM, NM)
        # if (run == 0):
        #         runSIGH = copy.deepcopy(SIGH)
        # else:
        #         runSIGH = np.vstack((runSIGH, SIGH))
        runPHI = np.append(runPHI, keptPHI)
        runLENGTH = np.append(runLENGTH, LENGTH)
        runETA = np.append(runETA, ETA)
        runWL = np.append(runWL, WL)
        runPR = np.append(runPR, PR)
        
        PHIind = np.argsort(keptPHI)
        Ult_ind = PHIind[0]
        revPHIind = PHIind[::-1]
        
        """                        
        #### Plot histogram of the number of layers ####                
        nlhist(rep_cnt,repeat, NM, nmin, nmax, maxz_m, abc, run, SAVEF)
        
        #### Plot histogram of the hyperparameter SIGH ####
        sighhist(rep_cnt,repeat, SIGH, hypmin, hypmax, maxz_m, abc, run, SAVEF)
        
        # Specify colormap for plots
        chosenmap='brg_r'        
        
        # ==================== PLOT [1] ==================== 
        # ================ Velocity Models =================
        CS3,scalarMap=modfig(rep_cnt,repeat,keptPHI,vmin,vmax,chosenmap,
                             nummods,revPHIind,CHMODS,Ult_ind,maxz_m,abc,run,
                             SAVEF)
        
        # ==================== PLOT [2] ==================== 
        # =========== Dispersion Curve Vertical ============
        vdispfig(rep_cnt,repeat,nummods,revPHIind,keptPHI,CHMODS,scalarMap,
                 dobs_sw,cp,Ult_ind,weight_opt,wsig,cpmin,cpmax,vmin,vmax,CS3,
                 maxz_m,abc,run,SAVEF)
        
        # ==================== PLOT [3] ==================== 
        # ========== Dispersion Curve Horizontal ===========
        # hdispfig(rep_cnt,repeat,nummods,revPHIind,keptPHI,CHMODS,scalarMap,
        #          dobs,instpd,Ult_ind,weight_opt,wsig,pmin,pmax,vmin,vmax,CS3,
        #          maxz_m,abc,run,SAVEF)
        
        #### Plot histogram of ALL interface depths ####
        # intffig(rep_cnt,repeat,INTF,maxz_m,abc,run,SAVEF)

        # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
        # = # = # = # = # = # PROBABILITY DENSITY FUNCTIONS # = # = # = # = # = # = # 
        # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
        #vh,maxline,maxlineDISP,newpin,newzin,newvin,cutpin,newvinD,normvh,normph=pdfdiscrtze(maxz_m,vmax,instpd,nummods,CHMODS,pdf_connect,pmin,pmax)
        
        #### Create the pdf figures for disperions curves and models ####
        #setpdfcmaps(pdfcmap,rep_cnt,repeat,weight_opt,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax)
        
        if rep_cnt == (repeat - 1):
                rnummods = len(RUNMODS)                
                                
                #### Plot histogram of the number of layers ####                
                nlhist(rep_cnt,repeat,runNM, nmin, nmax, maxz_m, abc, run, 
                       SAVEF)
        
                #### Plot histogram of the hyperparameter SIGH ####
                sighhist(rep_cnt,repeat,runSIGH, hypmin, hypmax, maxz_m, abc, 
                         run, SAVEF)
        
                runPHIind = np.argsort(runPHI)
                rUlt_ind = runPHIind[0]
                revrunind = runPHIind[::-1]
        
                # ==================== PLOT [1] ==================== 
                # ================ Velocity Models =================
                CS3,scalarMap=modfig(rep_cnt,repeat,runPHI,vmin,vmax,chosenmap,
                                     rnummods,revrunind,RUNMODS,rUlt_ind,
                                     maxz_m,abc,run,SAVEF)
                     
                # ==================== PLOT [2] ==================== 
                # =========== Dispersion Curve Vertical ============
                vdispfig(rep_cnt,repeat,rnummods,revrunind,runPHI,RUNMODS,
                         scalarMap,dobs_sw,cp,rUlt_ind,weight_opt,wsig,cpmin,
                         cpmax,vmin,vmax,CS3,maxz_m,abc,run,SAVEF)

                # ==================== PLOT [3] ==================== 
                # ========== Dispersion Curve Horizontal ===========
                # hdispfig(rep_cnt,repeat,rnummods,revrunind,runPHI,RUNMODS,
                #          scalarMap,dobs,instpd,rUlt_ind,weight_opt,wsig,pmin,
                #          pmax,vmin,vmax,CS3,maxz_m,abc,run,SAVEF)

                #### Plot histogram of ALL interface depths ####
                # intffig(rep_cnt,repeat,runINTF,maxz_m,abc,run,SAVEF)

                # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
                # = # = # = # = # = # PROBABILITY DENSITY FUNCTIONS # = # = # = # = # = # = # 
                # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
                # vh,maxline,maxlineDISP,newpin,newzin,newvin,cutpin,newvinD,normvh,normph=pdfdiscrtze(maxz_m,vmax,instpd,rnummods,RUNMODS,pdf_connect,pmin,pmax)

                #### Create the pdf figures for disperions curves and models ####
                # setpdfcmaps(pdfcmap,rep_cnt,repeat,weight_opt,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax)
"""                                        
        # rep_cnt = rep_cnt + 1
"""
Elog.close()

"""

