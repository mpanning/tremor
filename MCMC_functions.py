"""
The functions necessary to run the MCMC inversion
"""
import numpy as np
import math
import tremor
import random
import matplotlib.pyplot as plt
import pylab as P
from matplotlib.colors import LinearSegmentedColormap
import copy
from tqdm import tqdm
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
# import rayleigh_python
# from obspy.taup import TauPyModel,taup_create

# Global variables for use in MCMC_functions
ifVerbose = False
def set_global(verbflag=None):
    global ifVerbose
    if verbflag is not None:
        ifVerbose = verbflag

# ***************************** DEFINE FUNCTIONS *****************************
def mintwo (number1, number2):
    if number1 < number2:
            comparmin = number1
    else:
            comparmin = number2
    return comparmin
# ----------------------------------------------------------------------------
def maxtwo (number1, number2):
    if number1 < number2:
            comparmax = number2
    else:
            comparmax = number1
    return comparmax
# ----------------------------------------------------------------------------
def startmodel(totch, Lmin, Lmax, etamin, etamax, prmin, prmax, wlmin, wlmax,
               h0min, h0max):
    stL = np.zeros(totch)
    steta = np.zeros(totch)
    stpratio = np.zeros(totch)
    stwl = np.zeros(totch)
    sth0 = np.zeros(totch)

    for cnt in range(totch):
        stL[cnt] = random.uniform(Lmin, Lmax)
        steta[cnt] = random.uniform(etamin, etamax)
        stpratio[cnt] = random.uniform(prmin, prmax)
        stwl[cnt] = random.uniform(wlmin, wlmax)
        sth0[cnt] = random.uniform(h0min, h0max)

    return (stL, steta, stpratio, stwl, sth0)
# ----------------------------------------------------------------------------
#def sobs_set (k):
#    nextmodel = k+1
#    modl = 'modl%s.in' % nextmodel
#    try:
#        os.remove('sobs.d')
#        os.remove('log.txt')
#        os.remove('disp.out')
#    except:
#        pass
#    sobs = open('sobs.d','w')
#    sobs.write('0.005 0.005 0.0 0.005 0.0'+'\n')
#    sobs.write('1 0 0 0 0 0 1 0 1 0'+'\n')
#    sobs.write(modl+'\n')
#    sobs.write('disp.d')
#    sobs.close()
#    return (modl)
# ----------------------------------------------------------------------------
def finderror (k,x,ndata,dpre,dobs,misfit,newmis,wsig,PHI,diagCE,weight_opt):
    for i in range(ndata):
        misfit[i] = dpre[i, k+1] - dobs[i]
        sigEST = wsig[i]
        # When adding in data variance hyper parameter
        # sigEST = math.sqrt(((wsig[count]**2)+(curhyp[i]**2)))
        diagCE[i] = sigEST**2
        newmis[i] = misfit[i]*(1.0/sigEST)

    x.misfit = misfit
    x.w_misfit = newmis
   
    e_sqd = []
    if weight_opt == 'OFF':
        e_sqd = (x.misfit[:])**2
    elif weight_opt == 'ON':
        e_sqd = (x.w_misfit[:])**2

    if (k == -1):
        PHI[0] = (sum(e_sqd))
        PHIold = 'nan'
        PHInew = PHI[0]
    else:
        PHI[k+1] = (sum(e_sqd))
        PHIold = PHI[k]
        PHInew = PHI[k+1]
    x.PHI = sum(e_sqd)
    x.diagCE = diagCE[:]
    # print(misfit)
    # print(newmis)
    if ifVerbose:
        print(e_sqd)
        print('PHIold = ' + ' ' + str(PHIold) + '    ' + 'PHInew = ' + ' '
              + str(PHInew))
    return (misfit,newmis,PHI,x,diagCE)    
# ----------------------------------------------------------------------------
def accept_reject (PHI,k,pDRAW,WARN_BOUNDS):
    pi = math.pi
    if WARN_BOUNDS:
        pac = 0
    else:
        # All current options 
        if (pDRAW >= 0) and (pDRAW <= 4):
            try:
                misck = math.exp(-(PHI[k+1]-PHI[k])/2)
            except OverflowError:
                misck = 1
            if ifVerbose:
                print('PHIs: '+str(PHI[k])+', '+str(PHI[k+1]))
                print('misck is:   '+str(misck))
            pac = mintwo(1,misck)
            
    if ifVerbose:
        print(' ')        
        print('pac = min[ 1 , prior ratio x likelihood ratio x proposal ' +
              'ratio ] :   ' + str(pac))
        print(' ')
    q = random.uniform(0,1)
    if ifVerbose:
        print('random q:   '+str(q))
    return (pac,q)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def startchain(Lmin, Lmax, etamin, etamax, prmin, prmax, wlmin, wlmax, h0min,
               h0max):
    L = random.uniform(Lmin, Lmax)
    eta = random.uniform(etamin, etamax)
    pratio = random.uniform(prmin, prmax)
    wl = random.uniform(wlmin, wlmax)
    h0 = random.uniform(h0min, h0max)
    return L, eta, pratio, wl, h0
# ----------------------------------------------------------------------------    
# ----------------------------------------------------------------------------
def errorfig(PHI, BURN, chain, abc, run, SAVEF):
    plt.close('all')
    fig, ax1 = plt.subplots()

    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.63, 0.6, 0.25, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])

    ax1.plot(PHI[BURN:], color='red')
    
    #ax1.set_ylabel('E(m)', fontweight='bold', fontsize=14)
    ax1.set_ylabel(r'$\phi$(m)', fontsize=20)
    ax1.set_xlabel('Iteration',  fontweight='bold', fontsize=14)
    figtitle = 'Evolution of model error '+ r'$\phi$(m)'
    ax1.set_title(figtitle, fontweight='bold', fontsize=14)

    ax2.plot(PHI[0:BURN], color='red')
    ax2.set_ylabel(r'$\phi$(m)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Burn - in Period', fontsize=12)

    Efig = SAVEF + '/' + 'Error_chain_' + str(chain)+'_'+abc[run]+'.png'
    P.savefig(Efig)
# ----------------------------------------------------------------------------
def accratefig(totch, acc_rate, draw_acc_rate, abc, run, SAVEF):
    plt.close('all')
    pltx = range(totch)
    plt.plot(pltx, acc_rate, 'ko')
    plt.plot(pltx, acc_rate, 'k:')
    plt.xlabel('Chain')
    plt.ylabel('Acceptance Percentage (%)')
    plt.axis([0, (totch-1), 0, 100])
    plt.title('Generated Model Acceptance Rate', fontweight='bold', fontsize=14)
    picname='Acceptance_rate_'+abc[run]+'.png'
    accfig = SAVEF + '/' + picname
    P.savefig(accfig)
    plt.close('all')
    for i in range(0, draw_acc_rate.shape[0]):
        plt.plot(pltx, draw_acc_rate[i], 'ko')
        plt.plot(pltx, draw_acc_rate[i], 'k:')
        plt.xlabel('Chain')
        plt.ylabel('Acceptance Percentage (%)')
        plt.axis([0, (totch-1), 0, 100])
        plt.title('Generated Model Acceptance Rate ' + str(i), 
                  fontweight='bold', fontsize=14)
        picname='Acceptance_rate_'+abc[run]+'_'+str(i)+'.png'
        accfig = SAVEF + '/' + picname
        P.savefig(accfig)
        plt.close('all')
# ----------------------------------------------------------------------------
def fhist(dir, letter, fmin, fmax, nfbins, freqs):
    plt.close('all')
    P.figure
    weights = np.ones_like(freqs)/float(len(freqs))
    P.hist(freqs, bins= np.linspace(fmin, fmax, nfbins), weights=weights)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Probability density')
    figname = dir + letter + '_fhist.png'
    P.savefig(figname)

# ----------------------------------------------------------------------------
def pdhist(dir, letter, pmin, pmax, npbins, pds):
    plt.close('all')
    P.figure
    weights = np.ones_like(pds)/float(len(pds))
    P.hist(pds, bins= np.linspace(pmin, pmax, npbins), weights=weights)
    plt.xlabel('Period (s)')
    plt.ylabel('Probability density')
    figname = dir + letter + '_pdhist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def amphist(dir, letter, amin, amax, nabins, amps):
    plt.close('all')
    P.figure
    weights = np.ones_like(amps)/float(len(amps))
    P.hist(amps, bins= np.linspace(amin, amax, nabins), weights=weights)
    plt.xlabel(r'Amplitude (m/s$^2$)')
    plt.ylabel('Probability density')
    figname = dir + letter + '_amphist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def Rhist(dir, letter, Rmin, Rmax, nRbins, Rs):
    plt.close('all')
    P.figure
    weights = np.ones_like(Rs)/float(len(Rs))
    P.hist(Rs, bins= np.linspace(Rmin, Rmax, nRbins), weights=weights)
    plt.xlabel(r'R value')
    plt.ylabel('Probability density')
    figname = dir + letter + '_Rhist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def Lhist(dir, letter, Lmin, Lmax, nLbins, Ls, ylims=None):
    plt.close('all')
    P.figure
    weights = np.ones_like(Ls)/float(len(Ls))
    P.hist(Ls, bins= np.linspace(Lmin, Lmax, nLbins), weights=weights)
    plt.xlabel(r'Channel length (m)')
    plt.ylabel('Probability density')
    if (ylims is not None):
        plt.ylim(ylims)
    figname = dir + letter + '_Lhist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def etahist(dir, letter, etamin, etamax, netabins, etas, ylims=None):
    plt.close('all')
    P.figure
    weights = np.ones_like(etas)/float(len(etas))
    P.hist(etas, bins=np.logspace(math.log10(etamin), math.log10(etamax),
                                  netabins), weights=weights)
    plt.xlabel(r'Fluid viscosity (Pa s)')
    plt.xscale('log')
    plt.ylabel('Probability density')
    if (ylims is not None):
        plt.ylim(ylims)
    figname = dir + letter + '_etahist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def prhist(dir, letter, prmin, prmax, nprbins, prs):
    plt.close('all')
    P.figure
    weights = np.ones_like(prs)/float(len(prs))
    P.hist(prs, bins= np.linspace(prmin, prmax, nprbins), weights=weights)
    plt.xlabel(r'Driving pressure ratio')
    plt.ylabel('Probability density')
    figname = dir + letter + '_prhist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def wlhist(dir, letter, wlmin, wlmax, nwlbins, wls):
    plt.close('all')
    P.figure
    weights = np.ones_like(wls)/float(len(wls))
    P.hist(wls, bins= np.linspace(wlmin, wlmax, nwlbins), weights=weights)
    plt.xlabel(r'Aspect ratio')
    plt.ylabel('Probability density')
    figname = dir + letter + '_wlhist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def h0hist(dir, letter, h0min, h0max, nh0bins, h0s, ylims=None):
    plt.close('all')
    P.figure
    weights = np.ones_like(h0s)/float(len(h0s))
    P.hist(h0s, bins= np.linspace(h0min, h0max, nh0bins), weights=weights)
    plt.xlabel(r'h$_0$ fraction')
    plt.ylabel('Probability density')
    if (ylims is not None):
        plt.ylim(ylims)
    figname = dir + letter + '_h0hist.png'
    P.savefig(figname)
# ----------------------------------------------------------------------------
def fluxhist(dir, letter, fluxmin, fluxmax, nfluxbins, fluxs, ylims=None):
    plt.close('all')
    P.figure
    weights = np.ones_like(fluxs)/float(len(fluxs))
    P.hist(fluxs, bins= np.logspace(math.log10(fluxmin), math.log10(fluxmax),
                                    nfluxbins), weights=weights)
    plt.xlabel(r'Volume flux (m$^3$/s)')
    plt.xscale('log')
    plt.ylabel('Probability density')
    if (ylims is not None):
        plt.ylim(ylims)
    figname = dir + letter + '_fluxhist.png'
    P.savefig(figname)

    
# ----------------------------------------------------------------------------
def pdfdiscrtze(nummods, CHMODS, Lmin, Lmax, etamin, etamax, prmin, prmax,
                wlmin, wlmax, fluxmin, fluxmax, fmin, fmax, amin, amax, Rmin,
                Rmax, h0min, h0max):
    # Discretize L, eta, pressure and aspect ratio bins
    Lints = 100
    Lin = np.linspace(Lmin, Lmax, Lints)
    # etaints = 100
    # etain = np.linspace(etamin, etamax, etaints)
    etaints = 100
    etain = np.logspace(math.log10(etamin), math.log10(etamax), etaints)
    prints = 100
    prin = np.linspace(prmin, prmax, prints)
    wlints = 100
    wlin = np.linspace(wlmin, wlmax, wlints)
    fluxints = 100
    fluxin = np.logspace(math.log10(fluxmin), math.log10(fluxmax), fluxints)
    h0ints = 100
    h0in = np.linspace(h0min, h0max, h0ints)

    # Discretize predicted observation bins
    freqints = 100
    freqin = np.linspace(fmin, fmax, freqints)
    ampints = 100
    ampin = np.linspace(amin, amax, ampints)
    Rints = 100
    Rin = np.linspace(Rmin, Rmax, Rints)
    
    # Establish 2D arrays for all comparison hit counts
    Leta = np.zeros((Lints, etaints), dtype=np.dtype(int))
    Lpr = np.zeros((Lints, prints), dtype=np.dtype(int))
    Lwl = np.zeros((Lints, wlints), dtype=np.dtype(int))
    etapr = np.zeros((etaints, prints), dtype=np.dtype(int))
    etawl = np.zeros((etaints, wlints), dtype=np.dtype(int))
    prwl = np.zeros((prints, wlints), dtype=np.dtype(int))
    Lflux = np.zeros((Lints, fluxints), dtype=np.dtype(int))
    etaflux = np.zeros((etaints, fluxints), dtype=np.dtype(int))
    prflux = np.zeros((prints, fluxints), dtype=np.dtype(int))
    wlflux = np.zeros((wlints, fluxints), dtype=np.dtype(int))
    freqamp = np.zeros((freqints, ampints), dtype=np.dtype(int))
    freqR = np.zeros((freqints, Rints), dtype=np.dtype(int))
    ampR = np.zeros((ampints, Rints), dtype=np.dtype(int))
    h0eta = np.zeros((h0ints, etaints), dtype=np.dtype(int))
    h0pr = np.zeros((h0ints, prints), dtype=np.dtype(int))
    h0wl = np.zeros((h0ints, wlints), dtype=np.dtype(int))
    h0flux = np.zeros((h0ints, fluxints), dtype=np.dtype(int))



    # Loop through all kept models, dealing with one model at a time
    for cmod in tqdm(range(nummods)):    
        sample = copy.deepcopy(CHMODS[cmod])
        cdpre = sample.dpre

        # Find indices
        dL = Lin[1] - Lin[0]
        iL = int((sample.L - Lmin)/dL)
        if (iL >= Lints) or (iL < 0):
            print("Warning: L outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.L, Lmin, Lmax))
            if (iL > Lints - 1):
                iL = Lints - 1
            elif (iL < 0):
                iL = 0                  
        # deta = etain[1] - etain[0]
        # ieta = int((sample.eta[0] - etamin)/deta)
        # if (ieta >= etaints) or (ieta < 0):
        #     print("Warning: eta outside bounds: setting to limit")
        #     print("{} {} {} (val min max)".format(sample.eta[0], etamin,
        #                                           etamax))
        #     if (ieta > etaints - 1):
        #         ieta = etaints - 1
        #     elif (ieta < 0):
        #         ieta = 0
        try:
            ieta = np.where(etain < sample.eta[0])[0][-1]
        except IndexError: # if below etamin, set ieta = 0
            ieta = 0
        if (ieta >= etaints) or (ieta < 0):
            print("Warning: eta outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.eta[0], etamin,
                                                  etamax))
            if (ieta > etaints - 1):
                ieta = etaints - 1
            elif (ieta < 0):
                ieta = 0                  
        dpr = prin[1] - prin[0]
        ipr = int((sample.pratio - prmin)/dpr)
        if (ipr >= prints) or (ipr < 0):
            print("Warning: pressure ratio outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.pratio, prmin, prmax))
            if (ipr > prints - 1):
                ipr = prints - 1
            elif (ipr < 0):
                ipr = 0                  
        dwl = wlin[1] - wlin[0]
        iwl = int((sample.wl - wlmin)/dwl)
        if (iwl >= wlints) or (iwl < 0):
            print("Warning: aspect ratio outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.wl, wlmin, wlmax))
            if (iwl > wlints - 1):
                iwl = wlints - 1
            elif (iwl < 0):
                iwl = 0                  
        # dflux = fluxin[1]-fluxin[0]
        # iflux = int((sample.flux[0] - fluxmin)/dflux)
        try:
            iflux = np.where(fluxin < sample.flux[0])[0][-1]
        except IndexError: # if below etamin, set ieta = 0
            iflux = 0
        if (iflux >= fluxints) or (iflux < 0):
            print("Warning: aspect ratio outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.flux[0], fluxmin,
                                                  fluxmax))
            if (iflux > fluxints - 1):
                iflux = fluxints - 1
            elif (iflux < 0):
                iflux = 0                  
        dfreq = freqin[1]-freqin[0]
        ifreq = int((sample.f[0] - fmin)/dfreq)
        if (ifreq >= freqints) or (ifreq < 0):
            print("Warning: frequency outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.f[0], fmin, fmax))
            if (ifreq > freqints - 1):
                ifreq = freqints - 1
            elif (ifreq < 0):
                ifreq = 0                  
        damp = ampin[1]-ampin[0]
        iamp = int((sample.dpre[1] - amin)/damp)
        if (iamp >= ampints) or (iamp < 0):
            print("Warning: amplitude outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.dpre[1], amin, amax))
            if (iamp > ampints - 1):
                iamp = ampints - 1
            elif (iamp < 0):
                iamp = 0                  
        dR = Rin[1]-Rin[0]
        iR = int((sample.dpre[2] - Rmin)/dR)
        if (iR >= Rints) or (iR < 0):
            print("Warning: R parameter outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.dpre[2], Rmin, Rmax))
            if (iR > Rints - 1):
                iR = Rints - 1
            elif (iR < 0):
                iR = 0                  
        dh0 = h0in[1]-h0in[0]
        ih0 = int((sample.h0_frac - h0min)/dh0)
        if (ih0 >= h0ints) or (ih0 < 0):
            print("Warning: R parameter outside bounds: setting to limit")
            print("{} {} {} (val min max)".format(sample.dpre[2], Rmin, Rmax))
            if (ih0 > h0ints - 1):
                ih0 = h0ints - 1
            elif (ih0 < 0):
                ih0 = 0                  

        # Increment all the 2D arrays
        Leta[iL, ieta] += 1
        Lpr[iL, ipr] += 1
        Lwl[iL, iwl] += 1
        etapr[ieta, ipr] += 1
        etawl[ieta, iwl] += 1
        prwl[ipr, iwl] += 1
        Lflux[iL, iflux] += 1
        etaflux[ieta, iflux] += 1
        prflux[ipr, iflux] += 1
        wlflux[iwl, iflux] += 1
        freqamp[ifreq, iamp] += 1
        freqR[ifreq, iR] += 1
        ampR[iamp, iR] += 1
        h0eta[ih0, ieta] += 1
        h0pr[ih0, ipr] += 1
        h0wl[ih0, iwl] += 1
        h0flux[ih0, iflux] += 1

    # Normalize to get pdf
    normLeta = np.array(Leta/float(nummods))
    normLpr = np.array(Lpr/float(nummods))
    normLwl = np.array(Lwl/float(nummods))
    normetapr = np.array(etapr/float(nummods))
    normetawl = np.array(etawl/float(nummods))
    normprwl = np.array(prwl/float(nummods))
    normLflux = np.array(Lflux/float(nummods))
    normetaflux = np.array(etaflux/float(nummods))
    normprflux = np.array(prflux/float(nummods))
    normwlflux = np.array(wlflux/float(nummods))
    normfreqamp = np.array(freqamp/float(nummods))
    normfreqR = np.array(freqR/float(nummods))
    normampR = np.array(ampR/float(nummods))
    normh0eta = np.array(h0eta/float(nummods))
    normh0pr = np.array(h0pr/float(nummods))
    normh0wl = np.array(h0wl/float(nummods))
    normh0flux = np.array(h0flux/float(nummods))

    # Return 1d and 2d arrays
    return (Lin, etain, prin, wlin, fluxin, freqin, ampin, Rin, h0in, Leta,
            Lpr, Lwl, etapr, etawl, prwl, Lflux, etaflux, prflux, wlflux,
            freqamp, freqR, ampR, h0eta, h0pr, h0wl, h0flux, normLeta, normLpr,
            normLwl, normetapr, normetawl, normprwl, normLflux, normetaflux,
            normprflux, normwlflux, normfreqamp, normfreqR, normampR, normh0eta,
            normh0pr, normh0wl, normh0flux)
            
# ----------------------------------------------------------------------------
# def setpdfcmaps(pdfcmap,rep_cnt,repeat,weight_opt,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax):
# STILL REWORKING THIS TO USE ARRAYS TO PASS TO MAKEPDFFIG
def setpdfcmaps(model_dir, pdfcmap, letter, Lin, etain, prin, wlin, fluxin,
                freqin, ampin, Rin, h0in, normLeta, normLpr, normLwl, normetapr,
                normetawl, normprwl, normLflux, normetaflux, normprflux,
                normwlflux, normfreqamp, normfreqR, normampR, normh0eta,
                normh0pr, normh0wl, normh0flux, Lmin, Lmax,
                etamin, etamax, prmin, prmax, wlmin, wlmax, fluxmin, fluxmax,
                fmin, fmax, amin, amax, Rmin, Rmax, h0min, h0max):
    
    totmaxhist = 50
    itotmaxhist = totmaxhist + 0.0
    levels = []
    Z = []
    name = []
    xlabels = []
    ylabels = []
    normvals = []
    ix = []
    iy = []
    logx = []
    logy = []

    inmin = []
    inmax = []
    invals = []
    inmin.append(Lmin)
    inmax.append(Lmax)
    invals.append(Lin)
    inmin.append(etamin)
    inmax.append(etamax)
    invals.append(etain)
    inmin.append(prmin)
    inmax.append(prmax)
    invals.append(prin)
    inmin.append(wlmin)
    inmax.append(wlmax)
    invals.append(wlin)
    inmin.append(fluxmin)
    inmax.append(fluxmax)
    invals.append(fluxin)
    inmin.append(fmin)
    inmax.append(fmax)
    invals.append(freqin)
    inmin.append(amin)
    inmax.append(amax)
    invals.append(ampin)
    inmin.append(Rmin)
    inmax.append(Rmax)
    invals.append(Rin)
    inmin.append(h0min)
    inmax.append(h0max)
    invals.append(h0in)

    # Leta
    minv = 0
    maxv = normLeta.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('Leta')
    xlabels.append('L (m)')
    ylabels.append(r'$\eta$ (Pa s)')
    ix.append(0)
    logx.append(False)
    iy.append(1)
    logy.append(True)
    normvals.append(normLeta)
    

    # Lpr
    minv = 0
    maxv = normLpr.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('Lpr')
    xlabels.append('L (m)')
    ylabels.append('Driving pressure ratio')
    ix.append(0)
    logx.append(False)
    iy.append(2)
    logy.append(False)
    normvals.append(normLpr)

    # Lwl
    minv = 0
    maxv = normLwl.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('Lwl')
    xlabels.append('L (m)')
    ylabels.append('Aspect ratio')
    ix.append(0)
    logx.append(False)
    iy.append(3)
    logy.append(False)
    normvals.append(normLwl)

    # etapr
    minv = 0
    maxv = normetapr.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('etapr')
    xlabels.append(r'$\eta$ (Pa s)')
    ylabels.append('Driving pressure ratio')
    ix.append(1)
    logx.append(True)
    iy.append(2)
    logy.append(False)
    normvals.append(normetapr)

    # etawl
    minv = 0
    maxv = normetawl.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('etawl')
    xlabels.append(r'$\eta$ (Pa s)')
    ylabels.append('Aspect ratio')
    ix.append(1)
    logx.append(True)
    iy.append(3)
    logy.append(False)
    normvals.append(normetawl)

    # prwl
    minv = 0
    maxv = normprwl.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('prwl')
    xlabels.append('Driving pressure ratio')
    ylabels.append('Aspect ratio')
    ix.append(2)
    logx.append(False)
    iy.append(3)
    logy.append(False)
    normvals.append(normprwl)

    # Lflux
    minv = 0
    maxv = normLflux.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('Lflux')
    xlabels.append('L (m)')
    ylabels.append(r'Volume flux (m$^3$/s)')
    ix.append(0)
    logx.append(False)
    iy.append(4)
    logy.append(True)
    normvals.append(normLflux)
    
    # etaflux
    minv = 0
    maxv = normetaflux.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('etaflux')
    xlabels.append(r'$\eta$ (Pa s)')
    ylabels.append(r'Volume flux (m$^3$/s)')
    ix.append(1)
    logx.append(True)
    iy.append(4)
    logy.append(True)
    normvals.append(normetaflux)
    
    # prflux
    minv = 0
    maxv = normprflux.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('prflux')
    xlabels.append('Driving pressure ratio')
    ylabels.append(r'Volume flux (m$^3$/s)')
    ix.append(2)
    logx.append(False)
    iy.append(4)
    logy.append(True)
    normvals.append(normprflux)
    
    # wlflux
    minv = 0
    maxv = normwlflux.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('wlflux')
    xlabels.append('Aspect ratio')
    ylabels.append(r'Volume flux (m$^3$/s)')
    ix.append(3)
    logx.append(False)
    iy.append(4)
    logy.append(True)
    normvals.append(normwlflux)
    
    # freqamp
    minv = 0
    maxv = normfreqamp.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('freqamp')
    xlabels.append('Frequency (Hz)')
    ylabels.append(r'Seismic amplitude (m/s$^2$)')
    ix.append(5)
    logx.append(False)
    iy.append(6)
    logy.append(False)
    normvals.append(normfreqamp)
    
    # freqR
    minv = 0
    maxv = normfreqR.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('freqR')
    xlabels.append('Frequency (Hz)')
    ylabels.append('R value')
    ix.append(5)
    logx.append(False)
    iy.append(7)
    logy.append(False)
    normvals.append(normfreqR)
    
    # ampR
    minv = 0
    maxv = normampR.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('ampR')
    xlabels.append(r'Seismic amplitude (m/s$^2$)')
    ylabels.append('R value')
    ix.append(6)
    logx.append(False)
    iy.append(7)
    logy.append(False)
    normvals.append(normampR)
    
    # h0eta
    minv = 0
    maxv = normh0eta.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('h0eta')
    xlabels.append(r'h$_0$ fraction')
    ylabels.append('Viscosity (Pa s)')
    ix.append(8)
    logx.append(False)
    iy.append(1)
    logy.append(True)
    normvals.append(normh0eta)
    
    # h0pr
    minv = 0
    maxv = normh0pr.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('h0pr')
    xlabels.append(r'h$_0$ fraction')
    ylabels.append('Driving pressure ratio')
    ix.append(8)
    logx.append(False)
    iy.append(2)
    logy.append(False)
    normvals.append(normh0pr)
    
    # h0ewl
    minv = 0
    maxv = normh0wl.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('h0wl')
    xlabels.append(r'h$_0$ fraction')
    ylabels.append('Aspect ratio')
    ix.append(8)
    logx.append(False)
    iy.append(3)
    logy.append(False)
    normvals.append(normh0wl)
    
    # h0flux
    minv = 0
    maxv = normh0flux.max()
    levels.append(np.linspace(minv, maxv, totmaxhist+1))
    Z.append(np.array([[0,0],[0,0],[0,0]]))
    name.append('h0flux')
    xlabels.append(r'h$_0$ fraction')
    ylabels.append(r'Volume flux (m$^3$/s)')
    ix.append(8)
    logx.append(False)
    iy.append(4)
    logy.append(True)
    normvals.append(normh0flux)
    

    for cmap in pdfcmap:
        cmap2 = []
        CS = []
        linopt = []
        for i, level in enumerate(levels):
            if cmap == 'GREYS':
                CCC = [plt.cm.Greys(chosen/itotmaxhist)
                       for chosen in range(totmaxhist)]
                cmap2.append(LinearSegmentedColormap.from_list('Greys', CCC,
                                                               N=totmaxhist,
                                                               gamma=1.0))
                CS.append(plt.contourf(Z[i], level, cmap=cmap2[i]))
                linopt.append(['r-.','r--','black','ko'])
            
            elif cmap == 'GREYS_rev':
                CCC1 = [plt.cm.Greys(chosen/itotmaxhist)
                        for chosen in range(totmaxhist)]
                CCC=CCC1[::-1]
                cmap2.append(LinearSegmentedColormap.from_list('Greys', CCC,
                                                               N=totmaxhist,
                                                               gamma=1.0))
                CS.append(plt.contourf(Z[i], level, cmap=cmap2[i]))
                linopt.append(['r-.','r--','black','wo'])

            elif cmap == 'HOT':
                CCC = [plt.cm.hot(chosen/itotmaxhist)
                       for chosen in range(totmaxhist)]
                cmap2.append(LinearSegmentedColormap.from_list('HOT', CCC,
                                                               N=totmaxhist,
                                                               gamma=1.0))
                CS.append(plt.contourf(Z[i], level, cmap=cmap2[i]))
                linopt.append(['w-.','w--','black','wo'])

            elif cmap == 'HOT_rev':
                CCC1=[plt.cm.hot(chosen/itotmaxhist)
                      for chosen in range(totmaxhist)]
                CCC=CCC1[::-1]
                cmap2.append(LinearSegmentedColormap.from_list('HOT', CCC,
                                                               N=totmaxhist,
                                                               gamma=1.0))
                CS.append(plt.contourf(Z[i], level, cmap=cmap2[i]))
                linopt.append(['k-.','k--','black','ko'])
        
        mkpdffigs(model_dir, letter, cmap, linopt, cmap2, CS, ix, iy, inmin,
                  inmax, invals, normvals, xlabels, ylabels, name, logx, logy)

# ----------------------------------------------------------------------------
# def mkpdffigs(rep_cnt,repeat,weight_opt,curcmap,linopt,cmap2,CS33,CS22,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax):
def mkpdffigs(model_dir, letter, cmap, linopt, cmap2, CS, ix, iy, inmin,
              inmax, invals, normvals, xlabels, ylabels, name, logx, logy):

    for i, cm2 in enumerate(cmap2):
        print("Working on figure {} using cmap {}: {}".format(i, cmap, name[i]))
        plt.close("all")
        plt.contourf(invals[ix[i]], invals[iy[i]], np.transpose(normvals[i]),
                     cmap=cm2)
        cb = plt.colorbar(CS[i])
        plt.axis([inmin[ix[i]], inmax[ix[i]], inmin[iy[i]], inmax[iy[i]]])
        plt.xlabel(xlabels[i])
        if logx[i]:
            plt.xscale('log')
        plt.ylabel(ylabels[i])
        if logy[i]:
            plt.yscale('log')

        Mfig = model_dir + letter + '_' + cmap + '_' + name[i] + '.png'
        P.savefig(Mfig)

    # # Dispersion Curve PDFs
    # plt.close("all")
    # plt.contourf(newpin, newvinD, normph, cmap=cmap2)
    # cb=plt.colorbar(CS22)    
    # plt.ylabel('VS (km/s)')
    # plt.xlabel('Period (s)')
    # plt.axis([pmin, pmax, 0, vmax,])
    # obs, = plt.plot(instpd, dobs, linopt[3], zorder=100, markersize=7, 
    #                 label='measured group velocities')
    # if weight_opt == 'ON':
    #     obserr = plt.errorbar(instpd, dobs, yerr=wsig, zorder=20, linestyle="None", linewidth=2.0, ecolor='k')
    # plt.legend(handles=[obs], fontsize=10)
    # if rep_cnt == (repeat-1):
    #     figname = 'PDF_REP_DISP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
    # else:
    #     figname = 'PDF_DISP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
    # figtitle = 'Probability Density Function \n Transdimensional, '+str(maxz_m)+' m total depth'
    # plt.title(figtitle, fontweight='bold', fontsize=14)
    # Dfig = SAVEF + '/' + figname
    # P.savefig(Dfig)
    # meanD, = plt.plot(cutpin, maxlineDISP, linopt[0], linewidth=4.0, zorder=3, label='MEAN')
    # meanD.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='black'),
    #                PathEffects.Normal()])
    # plt.legend(handles=[meanD, obs], fontsize=10)
    # figname = 'PDF_DISP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_meanline_'+abc[run]+'.png'
    # Dfig = SAVEF + '/' + figname
    # P.savefig(Dfig)
    
    
    # # Velocity Model PDFs
    # plt.close("all")
    # plt.contourf(newvin, newzin, normvh, cmap=cmap2)
    # cb=plt.colorbar(CS33)    
    # plt.axis([vmin, vmax, 0, maxz_m])
    # plt.gca().invert_yaxis()
    # plt.xlabel('VS (km/s)')
    # plt.ylabel('Depth (m)')
    # if rep_cnt == (repeat-1):
    #     figname = 'PDF_REP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
    # else:
    #     figname = 'PDF_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
    # figtitle = 'Probability Density Function \n Transdimensional, '+str(maxz_m)+' m total depth'
    # plt.title(figtitle, fontweight='bold', fontsize=14)
    # Mfig = SAVEF + '/' + figname
    # P.savefig(Mfig)
    # meanline = plt.plot(maxline, newzin, linopt[1], linewidth=4.0)
    # figname = 'PDF_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_meanline_'+abc[run]+'meanline.png'
    # Mfig = SAVEF + '/' + figname
    # P.savefig(Mfig)    

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------








