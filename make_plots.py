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
from MCMC_functions import fhist, pdhist, amphist, Rhist

model_dir = "/Users/panning/work_local/Insight/tremor/MCMC/09_20_2019_14:22/saved_models/"
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
    freqs = np.array([model.f[0] for model in models])
    # print(freqs)
    fhist(model_dir, all_letters[i], fmin, fmax, nfbins, freqs)
    pds = 1.0/freqs
    pdhist(model_dir, all_letters[i], pmin, pmax, npbins, pds)
    amps = np.array([model.dpre[1] for model in models])
    amphist(model_dir, all_letters[i], amin, amax, nabins, amps)
    Rs = np.array([model.dpre[2] for model in models])
    Rhist(model_dir, all_letters[i], Rmin, Rmax, nRbins, Rs)

