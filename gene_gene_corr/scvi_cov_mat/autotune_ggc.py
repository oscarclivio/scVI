import numpy as np
import torch


import json
import argparse
from hyperopt import fmin, tpe, hp, Trials, rand
import scvi
from scvi.dataset import CortexDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE
from typing import Tuple
from functools import partial

import datasets_ggc
from synth_data_cov import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--max_evals', type=int)
parser.add_argument('--vae', type=str, default='vae')
parser.add_argument('--force_autotune', default=False,
                    type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()

dataset_name = args.dataset
mode = args.mode
vae_name = args.vae
force_autotune = args.force_autotune
max_evals = args.max_evals


VAE_DICT = {'vae': VAE}
vae_model = VAE_DICT[vae_name]

dataset = datasets_ggc.DATASETS_GGC[dataset_name]()

print("Now autotuning....")
best_trainer, trials = autotune_fixed_loss(dataset,
                                           dataset_name=dataset_name,
                                           vae_model=vae_model,
                                           vae_name = vae_name,
                                           reconstruction_loss=mode,
                                           force_autotune=force_autotune,
                                           max_evals=max_evals)

print("Autotune completed")

print(trials.results)

