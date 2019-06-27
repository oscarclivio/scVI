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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--vae', type=str, default='vae')

args = parser.parse_args()

dataset_name = args.dataset
mode = args.mode
vae_name = args.vae


VAE_DICT = {'vae': VAE}
vae_model = VAE_DICT[vae_name]

print("Now autotuning....")
best_trainer, trials = autotune_fixed_loss(dataset,
                                           dataset_name='dataset',
                                           vae_model=VAE,
                                           vae_name = 'scvi',
                                           reconstruction_loss='nb',
                                           force_autotune=False)

print("Autotune completed")

