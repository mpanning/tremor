"""
Program to perform Markov chain Monte Carlo inversion of seismic observables
for acceptable range of tremor parameters based on the work of Julian, 1994
"""

import numpy as np
import math
from tqdm import tqdm
import datetime
import os
from MCMC_functions import startmodel, startchain
import tremor
from obspy.core import UTCDateTime
import instaseis
import string

# from numpy import inf, log, cos, array
# from glob import glob
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as PathEffects
# import matplotlib.colors as colors
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.cm as cmx
# from matplotlib import rc
# import math
# import os
# import operator
# import datetime
# import sys
# import copy
# import subprocess
# import random
# from random import randint
# import shutil
# import pylab as P
# import string
# from pprint import pprint
# from scipy.interpolate import interp1d
# import cPickle as pickle

# from MCMC_functions import (startmodel,MODEL,startchain,runmodel,finderror,
# 			    randINTF,accept_reject,mintwo,runmodel_bw,errorfig,
			    # accratefig,nlhist,sighhist,modfig,vdispfig)

# ----------------------------------------------------------------------------
# ****************************************************************************
# --------------------------Set up INPUT information--------------------------
# directory where working
MAIN = '/Users/panning/work_local/Insight/tremor/MCMC'
os.chdir(MAIN)
# MAIN = os.getcwd()

now = datetime.datetime.now()
foldername = now.strftime("%m_%d_%Y_%H:%M")
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
dur_threshold = 0.1

# Instaseis stuff.  Need to make alternate method for amplitude that skips this
ifInstaseis = True
if ifInstaseis:
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


# Make data vector.  Right now is hard-coded, but will adjust to read from file
freq_obs = 0.4 # Dominant frequency of signal (Hz)
amp_obs = 1.e-9 # Acceleration amplitude (m/s^2)
dur_obs = 1000.0 # Duration of observed signal (s)
dobs = np.array([freq_obs, amp_obs, dur_obs])
ndata = len(dobs)

# Uncertainty estimates - 1 sigma
wsig_freq = 0.3
wsig_amp = 3.e-10
wsig_dur = 200.
wsig = np.array([wsig_freq, wsig_amp, wsig_dur])

# create boss matrix to control all combinations of starting number of layers 
depopt = ([6.0, 60.0])
muopt = ([7.e9, 70.e9])
repeat = 1
all_letters = list(string.ascii_lowercase)
letters = all_letters[0:repeat]
abc=[]

DRAW = ['CHANGE CHANNEL LENGTH: Perturb the length of oscillating channel',
        'CHANGE VISCOSITY: Change the viscosity of the fluid in the channel',
        'CHANGE PRATIO: Change the overpressure ratio of the lower reservoir',
        'CHANGE ASPECT: Change the width to length aspect ratio of channel',]
# ---------------------------------------------------------------------------------------
# ----------------
# totch = 10			# Total number of chains
# numm = 1000
totch = 1
numm = 1000			# Number of iterations per chain
# ----------------
# ---------------------------------------------------------------------------------------
# ----------------
BURN = 200
# BURN = 2000		# Number of models designated BURN-IN, gets discarded
# M = 10			# Interval to keep models (e.g. keep every 100th model, M=100)
M = 3
MMM = np.arange(BURN-1,numm,M)
# ----------------
# ---------------------------------------------------------------------------------------
#########Option to weight data points######### 
######### by variance or stand. dev. ######### 
weight_opt = 'ON'
#weight_opt = 'OFF'
# --------------------------------------------
########## Options for weighting ############
##########   sigd   or   sigd_n  ############
#if weight_opt == 'ON':
#	wsig = sigd
#	weight_sig = "sigd"
	#wsig = sigd_n
	#weight_sig = "sigd_n"
#else:
#	wsig = np.zeros(fnum)

# --------------------------------------------
doptnum = len(depopt)
numrun = doptnum*repeat

boss = np.zeros((numrun, 2))
k=0
for i, dep in enumerate(depopt):
        print('starting depth option: {}'.format(dep))
        r = 0
        while r < repeat:
                abc.append(letters[r])
                r = r + 1
                boss[k:k+repeat,0]=dep
                boss[k:k+repeat,1]=muopt[i]
                k=k+repeat

reprunsPHI = []
reprunsdiagCE = []
reprunsHIST = []

# Standard deviations for the Gaussian proposal distributions

# Given that many of these vary over orders of magnitude, maybe should use
# log value as input, and thus it becomes log-normal perturbation.  Consider
thetaL = 25.0 # Length perturbations
thetaETA = 10.0 # Viscosity perturbations
thetaP = 0.005 # Pressure ratio perturbatio
thetaWL = 0.5 # Aspect ratio perturbation
# ---------------------------------------------------------------------------------------

savefname = "saved_initial_m"
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
runINTF=[]	
runNM=[]	
runSIGH=[]
rep_cnt = 0

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
        prmax = 1.1

        wlmin = 5.0
        wlmax = 100.0

        if rep_cnt == repeat:
                rep_cnt = 0
                RUNMODS = []
                reprunsPHI = []
                reprunsHIST = []
                BEST_RUNMODS = []
                BESTreprunPHI = []
                BESTreprunNL = []
                BESTrerunsSIGH = []
                savefname = 'saved_initial'
                SAVEF = SAVEMs + '/' + savefname
                os.mkdir(SAVEF)


        CHMODS = []
        BEST_CHMODS = []

        # Set up totch chains:------------------------
        stL, steta, stpratio, stwl = startmodel(totch, Lmin, Lmax, etamin,
                                                etamax, prmin, prmax, wlmin,
                                                wlmax)


        acc_rate = np.zeros(totch)
        draw_acc_rate = np.zeros((len(DRAW),totch))

        """      ------  Loop through multiple chains:  -------------     """
        print("Starting loop on chains")
        for chain in range(totch):

                # Create new tremor model with starting parameters
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
                freq_limit = freq_obs + 2.*wsig_freq
                if x.f[0] > freq_limit:
                        print("Frequency too high, generating new startmodel")
                        while x.f[0] > freq_limit:
                                # Generate a new starting model and calc f
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
                        raise NotImplementedError("Amplitude by scaling is " +
                                                  "not yet implemented")
                dpre = np.array([x.f[0], vamp, dur_pre])

                # dpre = np.zeros((ndata,numm))
                # dpre[:,0] = np.concatenate([np.concatenate(dpre_sw), 
                # 		       np.concatenate(dpre_bw)])
                # x.dpre = dpre[:,0]

                # CALCULATE MISFIT BETWEEN D_PRE AND D_OBS:
                # CONTINUE BODY WAVE MOD FROM HERE
                misfit = np.zeros(ndata)
                newmis = np.zeros(ndata)
                diagCE = np.zeros((ndata,numm))
                PHI = np.zeros(numm)
                """
		misfit,newmis,PHI,x,diagCE = finderror((-1),x,ndsub,dpre,dobs,
						       misfit,newmis,wsig,PHI,
						       diagCE,weight_opt)
								   	   
		ITMODS = []
		ITMODS.append(x)

		numreject = 0.0
		numaccept = 0.0
		drawnumreject = np.zeros(len(DRAW))
		drawnumaccept = np.zeros(len(DRAW))

		keep_cnt = 0

		# =============================================================
		k = 0
		while (k < (numm-1)):
				
			print "================================================"
			print ("                   CHAIN # [" + str(chain)+
			       "]    ITERATION # ["+str(k)+"]" )
			print " "
					
			# Set previous model object as "oldx" so can call on 
			# it's attributes when needed
			oldx = copy.deepcopy(ITMODS[k])
					
			# Copy old model to new model and update number
			# Perturbation steps then only need to change
			# elements that are perturbed
			x = copy.deepcopy(oldx)
			x.number = k+1
					
			curNM = copy.deepcopy(oldx.nmantle)
			WARN_BOUNDS = 'OFF'
				
			# save original velocities
			vsIN = np.zeros(curNM+3)
			vsIN[0] = copy.deepcopy(oldx.crustVs)
			vsIN[1:] = copy.deepcopy(oldx.mantleVs)

		        ########### Draw a new model ########################
			# Choose random integer between 0 and 6 such that each 
			# of the 7 options
			# (Change epidist, Change otime, Change Velocity, Move,
			# Birth, Death, Change Hyper-parameter) have
			#  a 1/7 chance of being selected
		
			# pDRAW = randint(0,6)
			# Change odds so that 50% change of changing epicentral
			# parameters
			epichange = randint(0,1)
			if epichange == 0:
				pDRAW = randint(0,1)
			else:
				pDRAW = randint(2,6)
					
			# Change epicentral distance
			if pDRAW == 0:
				
				print DRAW[pDRAW]

				wEPI = np.random.normal(0,thetaEPI)
				ievt = randint(0,nevts-1)

				newdist = x.epiDistkm[ievt] + wEPI
				print ('Perturb epicentral distance of evt[' +
				       str(ievt) + '] by ' + 
				       str(wEPI) + ' km')

				if ((newdist < epimin[ievt]) or 
				    (newdist > epimax[ievt])):
					print ("!!! Outside epicentral " +
					       "distance range")
					print "Automatically REJECT model"
					WARN_BOUNDS = 'ON'
				else:
					x.epiDistkm[ievt] = newdist
				vsOUT = copy.deepcopy(vsIN)
				BDi = 0
				delv2 = 0

			# Change origin time
			if pDRAW == 1:
				
				print DRAW[pDRAW]

				wOT = np.random.normal(0,thetaOT)
				ievt = randint(0,nevts-1)
				
				print ('Perturb origin time of evt[' + 
				       str(ievt) + '] by ' + str(wOT)
				       + 's')

				newtime = x.epiTime[ievt] + wOT

				if ((newtime < otmin[ievt]) or 
				    (newtime > otmax[ievt])):
					print ("!!! Outside origin time " +
					       "range\n")
					print "Automatically REJECT model"
					WARN_BOUNDS = 'ON'
				else:
					x.epiTime[ievt] = newtime
				vsOUT = copy.deepcopy(vsIN)
				BDi = 0
				delv2 = 0

			# Change velocity of a layer (vi)
			if pDRAW == 2:
						
				print DRAW[pDRAW]
						
				# Randomly select a perturbable velocity
				pV = randint(0,curNM+2)
						
				# Generate random perturbation value
				wV1 = np.random.normal(0,thetaV1)
						
						
		
				# initialize all velocities as the same as 
				# previously
				vsOUT = copy.deepcopy(vsIN)

				# target layer being perturbed and add in 
				# random wV
				vsOUT[pV] = vsOUT[pV] + wV1
	
				if pV == 0:
					print ('Perturb crust VS[' + str(pV) +
					       ']\n')
					print 'wV1 = ' + str(wV1)
							
					# if the new value is outside of 
					# velocity range or creates a negative
					# velocity gradient - REJECT
					if ((vsOUT[pV] < cvmin) or 
					    (vsOUT[pV] > cvmax) or
					    (vsOUT[pV] > vsOUT[pV+1])):
						print ('!!! Outside velocity ' +
						       'range allowed!!')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
				elif pV == curNM+2:
					print ('Perturb base VS[' + str(pV) +
					       ']\n')
					print 'wV1 = ' + str(wV1) + '\n'

					# if the new value is outside of 
					# velocity range or creates a negative
					# velocity gradient - REJECT
					if ((vsOUT[pV] < vsOUT[pV-1]) or 
					    (vsOUT[pV] > vmax)):
						print ('!!! Outside velocity ' +
						       'range allowed!!')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
				else:
					print 'Perturb VS[' + str(pV) +']'
					print 'wV1 = ' + str(wV1) 
		
					# if the new value is outside of 
					# velocity range or creates a negative
					# velocity gradient - REJECT
					if ((vsOUT[pV] < vsOUT[pV-1]) or 
					    (vsOUT[pV] > vsOUT[pV+1]) or
					    (vsOUT[pV] < vmin) or 
					    (vsOUT[pV] > vmax)):
						print ('!!! Outside velocity ' +
						       'range allowed!!')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
		
				BDi = 0
				delv2 = 0
				x.crustVs = vsOUT[0]
				x.mantleVs = []
				x.mantleVs = vsOUT[1:]
									
			# Change an interface depth	
			elif pDRAW == 3:
						
				# MOVE an interface!!!
				nintf = curNM+1
				print DRAW[pDRAW]
						
				# initialize all velocities and interfaces the 
				# same as previously
				vsOUT = copy.deepcopy(vsIN)
				intfIN = np.zeros(curNM+1)
				intfIN = oldx.radius - oldx.mantleR
				intfOUT = copy.deepcopy(intfIN)
						
				# Choose an interface at random to perturb
				pI = randint(0,(nintf-1))
				print 'Perturbing INTF['+str(pI)+']'
						
				# Generate random perturbation value
				wI = np.random.normal(0,thetaI)
				print 'wI = ' + str(wI)
									
				# select the interface being perturbed and add 
				# in random wI and then resort
				intfOUT[pI] = intfIN[pI] + wI
				# sorting avoids interfaces overtaking each 
				# other
				tmpint = np.array(sorted(intfOUT))
				intfOUT = tmpint

				# Check if crustal thickness has changed
				if not (intfOUT[0] == oldx.crustThick):
					x.crustThick = intfOUT[0]
					if ((x.crustThick < chmin) or 
					    (x.crustThick > chmax)):
						print ('!!! Crustal thickness '
						       + 'outside of depth ' + 
						       'bounds')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'

				# if the new value is outside of depth bounds 
				# - REJECT
				if ((intfOUT[pI] < x.crustThick) or 
				    (intfOUT[pI] > maxz)):
					print ('!!! Outside of depths ' +
					       'allowed!!!')
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'
	

				# if any layers are now thinner than hmin - 
				# REJECT
				for i in range(1,curNM+1):
					if ((intfOUT[i] - intfOUT[i-1]) < hmin):
						print '!!! Layer too thin!!!\n'
						print ('Automatialy REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
				if (x.radius - x.cmbR - intfOUT[curNM]) < hmin:
					print '!!! Layer too thin!!!'
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'

				x.crustThick = intfOUT[0]
				x.mantleR = x.radius - intfOUT
				BDi = 0
				delv2 = 0

			# Create a new layer
			elif pDRAW == 4:
						
				# ADD a layer in!!!
				print DRAW[pDRAW]
						
				# initialize all velocities as the same as 
				# previously
				vsOUT = copy.deepcopy(vsIN)
				   						
				newNM = curNM + 1
				x.nmantle = newNM
						
				if newNM > nmax:
					print ('!!! exceeded the maximum ' +
					       'number of layers allowed!!!\n')
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'
						
				# Get the current model's interface depths
				intfIN = np.zeros(curNM+2)
				intfIN = oldx.radius - oldx.mantleR
				#F = np.zeros(curNL-1)
				#F[:] = copy.deepcopy(oldx.intf[:])
						
				# Generate an interface at a random depth, but 
				# have check measures in place so that the 
				# interface cannot be within hmin of any of 
				# the existing interfaces
				(addI,intfOUT,vsOUT,WARN_BOUNDS,
				 BDi,delv2) = randINTF(vmin,vmax,chmin,hmin,
						       maxz,intfIN,vsIN,thetaV2)
			        
				x.mantleR = x.radius - np.array(intfOUT)    
				x.crustThick = intfOUT[0]
				x.crustVs = vsOUT[0]
				x.mantleVs = []
				x.mantleVs = vsOUT[1:]
				
			# Delete a layer
			elif pDRAW == 5:
						
				# REMOVE a layer!!!
				print DRAW[pDRAW]
						
				newNM = curNM - 1
				x.nmantle = newNM
						
				if newNM < nmin:
					print ('!!! dropped below the ' + 
					       'minimum number of layers ' +
					       'allowed!!!')
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'

				# Get the current model's interface depths
				intfIN = np.zeros(curNM+2)
				intfIN = oldx.radius - oldx.mantleR
						
				# Randomly select an interface to remove
				# except crust or cmb
				kill_I = randint(1,(curNM))
				BDi = kill_I
				intfOUT = np.delete(intfIN, kill_I)
				print "removing interface["+str(kill_I)+"]"
				delv2 = vsIN[kill_I+1]-vsIN[kill_I-1]
						
				# Transfer over velocity values.  We remove the
				# velocity associated with the removed 
				# interface, meaning the new velocity profile
				# simply linearly interpolates over the region
				# of the removed interface
				VSlen = len(vsIN)
				vsOUT = np.zeros(VSlen-1)
				vsOUT[0:kill_I] = vsIN[0:kill_I]
				vsOUT[kill_I:(VSlen-1)] = vsIN[kill_I+1:(VSlen)]
						
				x.mantleR = []
				x.mantleR = x.radius - np.array(intfOUT)    
				x.crustThick = intfOUT[0]
				x.crustVs = vsOUT[0]
				x.mantleVs = []
				x.mantleVs = vsOUT[1:]
						
			# Change Hyper-parameter
			elif pDRAW == 6:	
						
				# Change the estimate of data error
				print DRAW[pDRAW]
						
				# Determine which hyperparameter to change
				ihyp = randint(0,len(oldx.sighyp)-1)

				# Generate random perturbation value
				wHYP = np.random.normal(0,thetaHYP)
						
				# Change the hyper-parameter value
				curhyp = copy.deepcopy(oldx.sighyp)
				newHYP = copy.deepcopy(curhyp)
				newHYP[ihyp] = curhyp[ihyp] + wHYP
				print ('Changing hyperparameter ' + str(ihyp) +
				       ' from ' + str(curhyp[ihyp]) + ' to ' +
				       str(newHYP[ihyp]))
				x.sighyp = newHYP
						
				# if new hyperparameter is outside of range - 
				# REJECT
				if ((newHYP[ihyp] < hypmin[ihyp]) or 
				    (newHYP[ihyp] > hypmax[ihyp])):
					print '!!! Outside the range allowed!!!'
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'
						
				vsOUT = copy.deepcopy(vsIN)
				BDi = 0
				delv2 = 0

			x.crustVp = x.PS_scale * x.crustVs
			x.crustRho = x.RS_scale * x.crustVs
			x.mantleVp = x.PS_scale * x.mantleVs
			x.mantleRho = x.RS_scale * x.mantleVs
				
			newflag =  any(value<0 for value in x.mantleVs)
			if newflag == True:
				WARN_BOUNDS == 'ON'
						
			# Continue as long as proposed model is not out of 
			# bounds:
			if WARN_BOUNDS == 'ON':
				print (' == ! == ! == !  REJECT NEW MODEL  ' +
				       '! == ! == ! ==  ')
				del x
				numreject = numreject + 1
				drawnumreject[pDRAW] = drawnumreject[pDRAW] + 1
				continue
			        # re-do iteration -- DO NOT increment (k)

			print 'test3'
			print x.epiDistkm
			print x.epiTime
			print x.nmantle
			print x.mantleR
			print x.crustVs, x.mantleVs
                        # Create swm input file
			try:
				x.create_swm_file(swm_nlay, 
						  create_nd_file=True)
			except ValueError:
				print 'WARNING: Unable to create model files'
				print (' == ! == ! == !  REJECT NEW MODEL  ' +
				       '! == ! == ! ==  ')
				del x
				continue

                        # Run sw model
			(modearray,nmodes)=runmodel(x,eps,npow,dt,fnyquist,
						    nbran,cmin,cmax,maxlyr)

		        # Confirm that rayleigh run covers desired frequency 
			# band
			parray = 1./modearray[2,:nmodes]
			gvarray = modearray[4,:nmodes]
			pmin = parray.min()
			pmax = parray.max()
			if (pmin > cpmin or pmax < cpmax):
				print pmin, cpmin, pmax, cpmax
				print ('Model did not calculate ' +
				       'correctly')
				print 'Redo iteration'
				del x
				continue
		
		        # Interpolate predicted gv to cp of data
			fgv = interp1d(parray, gvarray)
			gv_pre = []

			for i in range(0,nevts):
				gv_pre.append(fgv(cp[i]))
				dpre_sw[i][:] =  (x.epiTime[i] + 
						  (x.epiDistkm[i]/gv_pre[i]))	

			# Run bw model
			try:
				dpre_bw = runmodel_bw(x, phases) 
			except UserWarning:
				print ('Body wave phases not calculated ' +
				       'correctly')
				print 'Redo iteration'
				del x
				continue
			except:
				print 'taup threw an exception'
				print 'Redo iteration'
				del x
				continue

			# Merge into single array
			dpre[:,k+1] = np.concatenate([np.concatenate(dpre_sw), 
					       np.concatenate(dpre_bw)])
			if (not (len(dpre) == ndata)):
				print 'Problem with dpre'
				raise ValueError('Inconsistent ndata')
			x.dpre = copy.deepcopy(dpre[:,k+1])

			# Calculate error of the new model:
			misfit,newmis,PHI,x,diagCE = finderror(k,x,ndsub,dpre,
							       dobs,misfit,
							       newmis,wsig,PHI,
							       diagCE,
							       weight_opt)
							
			pac,q = accept_reject(PHI,k,pDRAW,WARN_BOUNDS,delv,
					      delv2,thetaV2,diagCE,vsIN,vsOUT,
					      BDi)
	
			if pac < q:
				print (' == ! == ! == !  REJECT NEW MODEL  ' +
				       '! == ! == ! ==  ')
				del x
				PHI[k+1] = PHI[k]
									
				numreject = numreject + 1
				drawnumreject[pDRAW] = drawnumreject[pDRAW] + 1
				# re-do iteration -- DO NOT increment (k)
			else: 
				print ('******   ******  ACCEPT NEW MODEL  ' +
				       '******   ******')
									
				# Calculate the depths and velocities for model 
				# (for plotting purposes to be used later)
				#npts = ((x.nl)*2)-1
				#F = copy.deepcopy(x.intf)
				#VS = copy.deepcopy(x.vs)
				#depth = np.zeros(npts+1)
				#depth[npts]=maxz
				#vels = np.zeros(npts+1)
				#adj = 0.001
				#ii=1
				#jj=0
				#ll=0
									
				#ii=1
				#jj=0
				#while (ii < npts):
				#	depth[ii]=F[jj]-adj
				#	depth[ii+1]=F[jj]+adj
				#	jj = jj + 1
				#	ii = ii + 2
				#ii=0
			       	#jj=0
				#while (ii < npts):
				#	vels[ii]=VS[jj]
				#	vels[ii+1]=VS[jj]
				#	jj = jj + 1
				#	ii = ii + 2
				#x.depths = depth
				#x.vels = vels
								
				# Retain MODEL 
				ITMODS.append(x)
					
				numaccept = numaccept + 1
				drawnumaccept[pDRAW] = drawnumaccept[pDRAW] + 1

				if k == MMM[keep_cnt]:
					print ('Adding model #' + str(k + 1) +
					       ' to the CHAIN ensemble')
					CHMODS.append(x)
					modl = x.filename
					curlocation = MAIN + '/' + modl
					newfilename = ('M' + str(ii) + '_' + 
						       abc[run] + '_' + 
						       str(chain)+ '_' + modl)
					newlocation = SAVEF + '/' + newfilename
					shutil.copy2(curlocation, newlocation)
					# Try to also save a binary version of the
					# MODEL class object
					classOutName = ('M' + str(ii) + '_' + 
							abc[run] + '_' + 
							str(chain) + '_' + modl +
							'.pkl')
					newlocation = SAVEF + '/' + classOutName
					with open(newlocation, 'wb') as output:
						pickle.dump(sample, output, -1)
					keep_cnt = keep_cnt + 1

			        #### Remove all models from current chain ####
				for filename in glob(MAIN+"/*.swm"):
					os.remove(filename) 
				for filename in glob(MAIN+"/*.tvel"):
					os.remove(filename) 
				for filename in glob(MAIN+"/*.npz"):
					os.remove(filename) 
				
			        # move to next iteration
				k = k + 1
	
		# Calculate the acceptance rate for the chain:
		numgen = numreject + numaccept
		drawnumgen = drawnumreject + drawnumaccept
		acc_rate[chain] = (numaccept/numgen)*100
		draw_acc_rate[:,chain] = (drawnumaccept[:]/drawnumgen[:])*100.0
		print draw_acc_rate

		if errorflag1 == 'on':
			print " "
			print " error occurred on first model generation, "
			print " no disp.out file found, try to re-do start of chain "
			print " "
			errorflag1 = 'off'
		else:
			print PHI
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
			# 	sample = copy.deepcopy(ITMODS[ii])
			# 	print ('Adding model #  [ '+str(sample.number)+
			# 	       ' ]  to the CHAIN ensemble')
			# 	CHMODS.append(sample)
			# 	modl = sample.filename
			# 	curlocation = MAIN + '/' + modl
			# 	newfilename = ('M'+str(ii)+'_'+abc[run]+ '_' 
			# 		       + str(chain)+ '_' + modl)
			# 	newlocation = SAVEF + '/' + newfilename
			# 	shutil.copy2(curlocation, newlocation)
			# 	# Try to also save a binary version of the
			# 	# MODEL class object
			# 	classOutName = ('M' + str(ii) + '_' + abc[run] +
			# 			'_' + str(chain) + '_' + modl +
			# 			'.pkl')
			# 	newlocation = SAVEF + '/' + classOutName
			# 	with open(newlocation, 'wb') as output:
			# 		pickle.dump(sample, output, -1)
			# 	ii = ii + M
					
			# #### Remove all models from current chain ####
		       	# for filename in glob(MAIN+"/*.swm"):
			# 	os.remove(filename) 
		       	# for filename in glob(MAIN+"/*.tvel"):
			# 	os.remove(filename) 
		       	# for filename in glob(MAIN+"/*.npz"):
			# 	os.remove(filename) 
				
			#### Plot the error at each iterations ####
			errorfig(PHI, BURN, chain, abc, run, maxz, SAVEF)
								
			# # Keep lowest error models from posterior distribution
			# realmin = np.argsort(PHI)
			# jj=0
			# while (jj < keep):				
			# 	ind=realmin[jj]
			# 	sample = copy.deepcopy(ITMODS[ind])
			# 	BEST_CHMODS.append(sample)
			# 	jj = jj + 1	

		       	#### Advance to next chain ####
			chain = chain + 1
	
	#### Plot acceptance rate ####
	accratefig(totch, acc_rate, draw_acc_rate, abc, run, SAVEF)
	
	keptPHI = []
	nummods = len(CHMODS)		
	INTF=[]	
	NM=[]	
	SIGH=[]
	jj = 0
	while (jj < nummods):
		sample = copy.deepcopy(CHMODS[jj])
		RUNMODS.append(sample)
		
		curPHI = sample.PHI
		keptPHI = np.append(keptPHI, curPHI)
		
		#newcol = 1000*(sample.intf)
		newcol = np.array(sample.mantleR)
		INTF = np.append(INTF, newcol)	
		newnm = sample.nmantle
		NM = np.append(NM, newnm)	
		newsighyp = sample.sighyp
		if (jj == 0):
			SIGH = copy.deepcopy(newsighyp)
		else:
			SIGH=np.vstack((SIGH, newsighyp))
		jj = jj + 1
		
	runINTF = np.append(runINTF, INTF)
	runNM = np.append(runNM, NM)
	if (run == 0):
		runSIGH = copy.deepcopy(SIGH)
	else:
		runSIGH = np.vstack((runSIGH, SIGH))
	runPHI = np.append(runPHI, keptPHI)
	
	PHIind = np.argsort(keptPHI)
	Ult_ind = PHIind[0]
	revPHIind = PHIind[::-1]
				
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
	# 	 dobs,instpd,Ult_ind,weight_opt,wsig,pmin,pmax,vmin,vmax,CS3,
	# 	 maxz_m,abc,run,SAVEF)
	
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
		# 	 scalarMap,dobs,instpd,rUlt_ind,weight_opt,wsig,pmin,
		# 	 pmax,vmin,vmax,CS3,maxz_m,abc,run,SAVEF)

		#### Plot histogram of ALL interface depths ####
		# intffig(rep_cnt,repeat,runINTF,maxz_m,abc,run,SAVEF)

		# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
		# = # = # = # = # = # PROBABILITY DENSITY FUNCTIONS # = # = # = # = # = # = # 
		# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
		# vh,maxline,maxlineDISP,newpin,newzin,newvin,cutpin,newvinD,normvh,normph=pdfdiscrtze(maxz_m,vmax,instpd,rnummods,RUNMODS,pdf_connect,pmin,pmax)

		#### Create the pdf figures for disperions curves and models ####
		# setpdfcmaps(pdfcmap,rep_cnt,repeat,weight_opt,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax)
"""					
                rep_cnt = rep_cnt + 1
"""
Elog.close()

"""

