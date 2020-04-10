"""
Read in models from saved-models from a previous MCMC run and make plots
"""
import math
import numpy as np
import pickle
from tqdm import tqdm
import tremor
import os
import string
from MCMC_functions import (fhist, pdhist, amphist, Rhist, Lhist, etahist,
                            prhist, wlhist, pdfdiscrtze, setpdfcmaps, fluxhist,
                            h0hist)

model_dir = "/Users/panning/work_local/Insight/tremor/MCMC/gattaca/TAYAK1/saved_models/"
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

model_pkls = []
for root, dirs, files in os.walk(model_dir):
    for file in files:
        if file.endswith('.pkl'):
            model_pkls.append(file)

# Split list into models separated by run letter
all_letters = list(string.ascii_lowercase)
max_ind = 0
for pklfile in model_pkls:
    letter = pklfile.split(".")[0].split("_")[1]
    ind = all_letters.index(letter)
    if ind > max_ind:
        max_ind = ind

nletters = max_ind + 1
pkl_separate = [[] for i in range(nletters)]
for pklfile in model_pkls:
    ind  = all_letters.index(pklfile.split(".")[0].split("_")[1])
    pkl_separate[ind].append(os.path.join(model_dir, pklfile))

# print("Number of model families: {}".format(nletters))
# for i in range(nletters):
#     print("Models in family {}".format(all_letters[i]))
#     print(pkl_separate[i])

for i in range(nletters):
    print("Working on model set {}".format(all_letters[i]))
    models = []
    for pklfile in tqdm(pkl_separate[i]):
        with open(pklfile, 'rb') as file:
            models.append(pickle.load(file))
    models = np.array(models)

    # Down select models by desired criteria
    nmodels = len(models)
    if ifDownselect:
        if ifFreqselect:
            freqs = np.array([model.f[0] for model in models])
            models = models[np.logical_and(freqs > fselectmin,
                                           freqs < fselectmax)]
        if ifRselect:
            Rs = np.array([model.dpre[2] for model in models])
            models = models[np.logical_and(Rs > Rselectmin, Rs < Rselectmax)]
        print("Models downselected from {} to {}".format(nmodels, len(models)))
    nmodels = len(models)

    # First plot up predicted observations
    freqs = np.array([model.f[0] for model in models])
    # print(freqs)
    fhist(model_dir, all_letters[i], fmin, fmax, nfbins, freqs)
    pds = 1.0/freqs
    pdhist(model_dir, all_letters[i], pmin, pmax, npbins, pds)
    amps = np.array([model.dpre[1] for model in models])
    amphist(model_dir, all_letters[i], amin, amax, nabins, amps)
    Rs = np.array([model.dpre[2] for model in models])
    Rhist(model_dir, all_letters[i], Rmin, Rmax, nRbins, Rs)

    # Now do the raw model parameters
    Ls = np.array([model.L for model in models])
    etas = np.array([model.eta[0] for model in models])
    prs = np.array([model.pratio for model in models])
    wls = np.array([model.wl for model in models])
    h0s = np.array([model.h0_frac for model in models])

    # Set the ylims for some plots if desired
    ifylims = False
    if (ifylims):
        Lylims = (0.0, 0.28)
        etaylims = (0.0, 0.055)
        h0ylims = (0.0, 0.35)
        fluxylims = (0.0, 0.07)
    else:
        Lylims = None
        etaylims = None
        h0ylims = None
        fluxylims = None

    Lhist(model_dir, all_letters[i], Lmin, Lmax, nLbins, Ls, ylims=Lylims)
    etahist(model_dir, all_letters[i], etamin, etamax, netabins, etas,
            ylims=etaylims)
    prhist(model_dir, all_letters[i], prmin, prmax, nprbins, prs)
    wlhist(model_dir, all_letters[i], wlmin, wlmax, nwlbins, wls)
    h0hist(model_dir, all_letters[i], h0min, h0max, nh0bins, h0s, ylims=h0ylims)

    # Add in some flux estimates
    for model in models:
        model.calc_flux()
    fluxs = np.array([model.flux[0] for model in models])
    fluxmax = np.amax(fluxs)
    fluxmin = np.amin(fluxs)
    nfluxbins = 50
    fluxhist(model_dir, all_letters[i], fluxmin, fluxmax, nfluxbins, fluxs,
             ylims=fluxylims)
    
    # Make 2D pdf plots
    pdfcmaps = ('GREYS', 'HOT')
    (Lin, etain, prin, wlin, fluxin, freqin, ampin, Rin, h0in, Leta,
     Lpr, Lwl, etapr, etawl, prwl,
     Lflux, etaflux, prflux, wlflux, freqamp, freqR, ampR,
     h0eta, h0pr, h0wl, h0flux,
     normLeta, normLpr, normLwl,
     normetapr, normetawl, normprwl, normLflux, normetaflux,
     normprflux, normwlflux, normfreqamp,
     normfreqR, normampR, normh0eta, normh0pr,
     normh0wl, normh0flux) = pdfdiscrtze(nmodels, models, Lmin, Lmax,
                                          etamin, etamax, prmin, prmax,
                                          wlmin, wlmax, fluxmin, fluxmax, fmin,
                                          fmax, amin, amax, Rmin, Rmax, h0min,
                                          h0max)
    setpdfcmaps(model_dir, pdfcmaps, all_letters[i], Lin, etain, prin, wlin,
                fluxin, freqin, ampin, Rin, h0in, normLeta, normLpr, normLwl,
                normetapr, normetawl, normprwl, normLflux, normetaflux,
                normprflux, normwlflux, normfreqamp, normfreqR, normampR,
                normh0eta, normh0pr, normh0wl, normh0flux,
                Lmin, Lmax, etamin, etamax, prmin, prmax, wlmin, wlmax, fluxmin,
                fluxmax, fmin, fmax, amin, amax, Rmin, Rmax, h0min, h0max)
