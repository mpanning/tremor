"""
Read in models from saved-models from 2 previous MCMC run and make combined plots
"""
import math
import numpy as np
import pickle
from tqdm import tqdm
import tremor
import os
import string
from MCMC_functions import (fhist_combined, pdhist_combined, amphist_combined,
                            Rhist_combined, Lhist_combined, etahist_combined,
                            prhist_combined, wlhist_combined, fluxhist_combined,
                            h0hist_combined)

model_dir1 = "/Users/panning/work_local/Insight/tremor/MCMC/gattaca/EH45Tcold1/saved_models/"
model_dir2 = "/Users/panning/work_local/Insight/tremor/MCMC/gattaca/TAYAK1/saved_models/"
out_dir = "./" # where the figures go
# model_dir = "/Users/panning/work_local/Insight/tremor/MCMC/gattaca/TAYAK1/04_07_2020_16_09_0033/saved_models/"
fmin = 0.0
fmax = 1.0
nfbins = 25
pmin = 0.5
pmax = 25.5
npbins = 25
amin = 0.
amax = 8.5e-9
nabins = 25
Rmin = 0.5
Rmax = 1.3
nRbins = 25
Lmin = 1.0
Lmax = 1000.0
nLbins = 100
etamin = 8.9e-4
etamax = 1000.0
netabins = 100
prmin = 1.0
prmax = 1.2
nprbins = 20
wlmin = 5.0
wlmax = 100.0
nwlbins = 50
h0min = 0.90
h0max = 1.00
nh0bins = 20

# Set downselect parameters if desired
ifDownselect = True
ifFreqselect = True
fselectmin = 0.2
fselectmax = 0.6
ifRselect = True
Rselectmin = 0.8
Rselectmax = 1.1

model_pkls1 = []
for root, dirs, files in os.walk(model_dir1):
    for file in files:
        if file.endswith('.pkl'):
            model_pkls1.append(file)
model_pkls2 = []
for root, dirs, files in os.walk(model_dir2):
    for file in files:
        if file.endswith('.pkl'):
            model_pkls2.append(file)

# Split list into models separated by run letter
all_letters = list(string.ascii_lowercase)
max_ind = 0
# Currently assumes same number of letters (i.e. depths) for both runs
for pklfile in model_pkls1:
    letter = pklfile.split(".")[0].split("_")[1]
    ind = all_letters.index(letter)
    if ind > max_ind:
        max_ind = ind

nletters = max_ind + 1
pkl_separate1 = [[] for i in range(nletters)]
pkl_separate2 = [[] for i in range(nletters)]
for pklfile in model_pkls1:
    ind  = all_letters.index(pklfile.split(".")[0].split("_")[1])
    pkl_separate1[ind].append(os.path.join(model_dir1, pklfile))
for pklfile in model_pkls2:
    ind  = all_letters.index(pklfile.split(".")[0].split("_")[1])
    pkl_separate2[ind].append(os.path.join(model_dir2, pklfile))

# print("Number of model families: {}".format(nletters))
# for i in range(nletters):
#     print("Models in family {}".format(all_letters[i]))
#     print(pkl_separate[i])

for i in range(nletters):
    print("Working on model set {} for run 1".format(all_letters[i]))
    models1 = []
    for pklfile in tqdm(pkl_separate1[i]):
        with open(pklfile, 'rb') as file:
            models1.append(pickle.load(file))
    models1 = np.array(models1)

    # Down select models by desired criteria
    nmodels1 = len(models1)
    if ifDownselect:
        if ifFreqselect:
            freqs = np.array([model.f[0] for model in models1])
            models1 = models1[np.logical_and(freqs > fselectmin,
                                             freqs < fselectmax)]
        if ifRselect:
            Rs = np.array([model.dpre[2] for model in models1])
            models1 = models1[np.logical_and(Rs > Rselectmin, Rs < Rselectmax)]
        print("Models downselected from {} to {}".format(nmodels1, len(models1)))
    nmodels1 = len(models1)

    print("Working on model set {} for run 2".format(all_letters[i]))
    models2 = []
    for pklfile in tqdm(pkl_separate2[i]):
        with open(pklfile, 'rb') as file:
            models2.append(pickle.load(file))
    models2 = np.array(models2)

    # Down select models by desired criteria
    nmodels2 = len(models2)
    if ifDownselect:
        if ifFreqselect:
            freqs = np.array([model.f[0] for model in models2])
            models2 = models2[np.logical_and(freqs > fselectmin,
                                             freqs < fselectmax)]
        if ifRselect:
            Rs = np.array([model.dpre[2] for model in models2])
            models2 = models2[np.logical_and(Rs > Rselectmin, Rs < Rselectmax)]
        print("Models downselected from {} to {}".format(nmodels2, len(models2)))
    nmodels2 = len(models2)

    # First plot up predicted observations
    freqs1 = np.array([model.f[0] for model in models1])
    freqs2 = np.array([model.f[0] for model in models2])
    # print(freqs)
    fhist_combined(out_dir, all_letters[i], fmin, fmax, nfbins, freqs1, freqs2)
    
    pds1 = 1.0/freqs1
    pds2 = 1.0/freqs2
    pdhist_combined(out_dir, all_letters[i], pmin, pmax, npbins, pds1, pds2)
    amps1 = np.array([model.dpre[1] for model in models1])
    amps2 = np.array([model.dpre[1] for model in models2])
    amphist_combined(out_dir, all_letters[i], amin, amax, nabins, amps1, amps2)
    Rs1 = np.array([model.dpre[2] for model in models1])
    Rs2 = np.array([model.dpre[2] for model in models2])
    Rhist_combined(out_dir, all_letters[i], Rmin, Rmax, nRbins, Rs1, Rs2)

    # Now do the raw model parameters
    Ls1 = np.array([model.L for model in models1])
    Ls2 = np.array([model.L for model in models2])
    etas1 = np.array([model.eta[0] for model in models1])
    etas2 = np.array([model.eta[0] for model in models2])
    prs1 = np.array([model.pratio for model in models1])
    prs2 = np.array([model.pratio for model in models2])
    wls1 = np.array([model.wl for model in models1])
    wls2 = np.array([model.wl for model in models2])
    h0s1 = np.array([model.h0_frac for model in models1])
    h0s2 = np.array([model.h0_frac for model in models2])

    # Set the ylims for some plots if desired
    ifylims = True
    if (ifylims):
        Lylims = (0.0, 0.3)
        etaylims = (0.0, 0.06)
        h0ylims = (0.0, 0.35)
        fluxylims = (0.0, 0.09)
    else:
        Lylims = None
        etaylims = None
        h0ylims = None
        fluxylims = None

    Lhist_combined(out_dir, all_letters[i], Lmin, Lmax, nLbins, Ls1, Ls2,
                   ylims=Lylims)
    etahist_combined(out_dir, all_letters[i], etamin, etamax, netabins, etas1,
                     etas2, ylims=etaylims)
    prhist_combined(out_dir, all_letters[i], prmin, prmax, nprbins, prs1, prs2)
    wlhist_combined(out_dir, all_letters[i], wlmin, wlmax, nwlbins, wls1, wls2)
    h0hist_combined(out_dir, all_letters[i], h0min, h0max, nh0bins, h0s1,
                    h0s2, ylims=h0ylims)

    # Add in some flux estimates
    for model in models1:
        model.calc_flux()
    fluxs1 = np.array([model.flux[0] for model in models1])
    for model in models2:
        model.calc_flux()
    fluxs2 = np.array([model.flux[0] for model in models2])
    fluxmax = np.amax(np.array([np.amax(fluxs1), np.amax(fluxs2)]))
    fluxmin = np.amin(np.array([np.amin(fluxs1), np.amax(fluxs2)]))
    nfluxbins = 50
    fluxhist_combined(out_dir, all_letters[i], fluxmin, fluxmax, nfluxbins,
                      fluxs1, fluxs2, ylims=fluxylims)
    
    # # Make 2D pdf plots
    # pdfcmaps = ('GREYS', 'HOT')
    # (Lin, etain, prin, wlin, fluxin, freqin, ampin, Rin, h0in, Leta,
    #  Lpr, Lwl, etapr, etawl, prwl,
    #  Lflux, etaflux, prflux, wlflux, freqamp, freqR, ampR,
    #  h0eta, h0pr, h0wl, h0flux,
    #  normLeta, normLpr, normLwl,
    #  normetapr, normetawl, normprwl, normLflux, normetaflux,
    #  normprflux, normwlflux, normfreqamp,
    #  normfreqR, normampR, normh0eta, normh0pr,
    #  normh0wl, normh0flux) = pdfdiscrtze(nmodels, models, Lmin, Lmax,
    #                                       etamin, etamax, prmin, prmax,
    #                                       wlmin, wlmax, fluxmin, fluxmax, fmin,
    #                                       fmax, amin, amax, Rmin, Rmax, h0min,
    #                                       h0max)
    # setpdfcmaps(model_dir, pdfcmaps, all_letters[i], Lin, etain, prin, wlin,
    #             fluxin, freqin, ampin, Rin, h0in, normLeta, normLpr, normLwl,
    #             normetapr, normetawl, normprwl, normLflux, normetaflux,
    #             normprflux, normwlflux, normfreqamp, normfreqR, normampR,
    #             normh0eta, normh0pr, normh0wl, normh0flux,
    #             Lmin, Lmax, etamin, etamax, prmin, prmax, wlmin, wlmax, fluxmin,
    #             fluxmax, fmin, fmax, amin, amax, Rmin, Rmax, h0min, h0max)
