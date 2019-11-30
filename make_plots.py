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
                            prhist, wlhist, pdfdiscrtze, setpdfcmaps, fluxhist)

model_dir = "/Users/panning/work_local/Insight/tremor/MCMC/halo/run4/saved_models/"
fmin = 0.0
fmax = 1.0
nfbins = 25
pmin = 0.5
pmax = 25.5
npbins = 25
amin = 0.5e-9
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
nwlbins = 47


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

    Lhist(model_dir, all_letters[i], Lmin, Lmax, nLbins, Ls)
    etahist(model_dir, all_letters[i], etamin, etamax, netabins, etas)
    prhist(model_dir, all_letters[i], prmin, prmax, nprbins, prs)
    wlhist(model_dir, all_letters[i], wlmin, wlmax, nwlbins, wls)

    # Add in some flux estimates
    for model in models:
        model.calc_flux()
    fluxs = np.array([model.flux[0] for model in models])
    fluxmax = np.amax(fluxs)
    fluxmin = np.amin(fluxs)
    nfluxbins = 50
    fluxhist(model_dir, all_letters[i], fluxmin, fluxmax, nfluxbins, fluxs)
    
    # Make 2D pdf plots
    pdfcmaps = ('GREYS', 'HOT')
    (Lin, etain, prin, wlin, fluxin, freqin, ampin, Rin, Leta,
     Lpr, Lwl, etapr, etawl, prwl,
     Lflux, etaflux, prflux, wlflux, freqamp, freqR, ampR,
     normLeta, normLpr, normLwl,
     normetapr, normetawl, normprwl, normLflux, normetaflux,
     normprflux, normwlflux, normfreqamp,
     normfreqR, normampR) = pdfdiscrtze(nmodels, models, Lmin, Lmax,
                                        etamin, etamax, prmin, prmax,
                                        wlmin, wlmax, fluxmin, fluxmax, fmin,
                                        fmax, amin, amax, Rmin, Rmax)
    setpdfcmaps(model_dir, pdfcmaps, all_letters[i], Lin, etain, prin, wlin,
                fluxin, freqin, ampin, Rin, normLeta, normLpr, normLwl,
                normetapr, normetawl, normprwl, normLflux, normetaflux,
                normprflux, normwlflux, normfreqamp, normfreqR, normampR,
                Lmin, Lmax, etamin, etamax, prmin, prmax, wlmin, wlmax, fluxmin,
                fluxmax, fmin, fmax, amin, amax, Rmin, Rmax)
