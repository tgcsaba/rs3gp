import os
import sys

if len(sys.argv) > 1:
  os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
results_dir = str(sys.argv[2]) if len(sys.argv) > 2 else './results/'

import numpy as np
import pandas as pd
import torch
import yaml

from gluonts.dataset.repository import get_dataset
from time import time
from torch import Tensor
from tqdm.auto import tqdm
from typing import Optional, Sequence

from src._eval import compute_metrics


def load_dataset(name):
  datasets_dict = {
    "solar": {'name': 'solar_nips', 'context_len': 336, 'prediction_len': 24},
    "electricity": {'name': 'electricity_nips', 'context_len': 336, 'prediction_len': 24},
    "traffic": {'name': 'traffic_nips', 'context_len': 336, 'prediction_len': 24},
    "exchange": {'name': 'exchange_rate_nips', 'context_len': 360, 'prediction_len': 30},
    "m4": {'name': 'm4_hourly', 'context_len': 312, 'prediction_len': 48},
    "uberTLC": {'name': "uber_tlc_hourly", 'context_len': 312, 'prediction_len': 48},
    "KDDCup": {'name': "kdd_cup_2018_without_missing", 'context_len': 336, 'prediction_len': 24},
    "wikipedia": {'name':  "wiki2000_nips",'context_len': 360, 'prediction_len': 30},
  }
  if name in datasets_dict.keys():
    metadata = datasets_dict[name]
    dataset = get_dataset(metadata["name"])
    return dataset, metadata
  else:
    raise ValueError(
      "Dataset name should be from the following list: " + str(list(datasets_dict.keys())))

def model_selector(model_name: str, X_train: Optional[Sequence[Tensor]], Y_train: Sequence[Tensor],
                   context_len: int, prediction_len: int, device: str):
  if model_name == "svgp":
    from src._svgp import SparseVariationalGP
    model = SparseVariationalGP(X_train, Y_train, context_len, prediction_len, device=device)
  if model_name == "dklgp":
    from src._dklgp import DeepKernelLearningGP
    model = DeepKernelLearningGP(X_train, Y_train, context_len, prediction_len, device=device)
  if model_name == "smkgp":
    from src._smkgp import SpectralMixtureVariationalGP
    model = SpectralMixtureVariationalGP(
      X_train, Y_train, context_len, prediction_len, device=device)
  elif model_name == "rs3gp":
    from src._rs3gp import RecurrentSparseSpectrumSignatureGP
    model = RecurrentSparseSpectrumSignatureGP(
      X_train, Y_train, prediction_len, learn_variational=False, device=device)
  elif model_name == "vrs3gp":
    from src._rs3gp import RecurrentSparseSpectrumSignatureGP
    model = RecurrentSparseSpectrumSignatureGP(X_train, Y_train, prediction_len, device=device)
  return model

def run_experiment(dataset_name, model_name):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  dataset, metadata = load_dataset(dataset_name)

  Y_train = [torch.Tensor(x['target']) for x in dataset.train]
  Y_test = [torch.Tensor(x['target']) for x in dataset.test]

  context_len = metadata['context_len']
  prediction_len = metadata['prediction_len']

  model = model_selector(model_name, None, Y_train, context_len, prediction_len, device)

  results = {'results': {}, 'time': {}}

  start_train = time()
  try:
    loss = model.train()
  except KeyboardInterrupt:
    loss = np.inf
  results['time']['train'] = time() - start_train

  start_test = time()
  pred_mean, pred_std = model.predict(None, Y_test)
  results['time']['test'] = time() - start_test

  _pred_mean = torch.stack([x[-prediction_len:] for x in pred_mean], axis=0)
  _pred_std = torch.stack([x[-prediction_len:] for x in pred_std], axis=0)
  _test_y = torch.stack([x[-prediction_len:] for x in Y_test], axis=0)

  alphas = []
  alphas_grid = np.linspace(0.1, 2., 20)
  for i in tqdm(range(len(Y_test))):
    best_alpha = float('nan')
    best_score = float('inf')
    start = Y_test[i].size(0) - pred_mean[i].size(0)
    for alpha in alphas_grid:
      score = compute_metrics(
        Y_test[i][start:-prediction_len],
        pred_mean[i][:-prediction_len],
        pred_std[i][:-prediction_len] * alpha
      )['mean_abs_quantileLoss']
      if score < best_score:
        best_score = score
        best_alpha = alpha
    alphas.append(best_alpha)
  alphas = torch.Tensor(alphas)

  results['results'] = compute_metrics(
    _test_y, _pred_mean, _pred_std * alphas[:, None].to(_pred_mean))
  results['results']['train_loss'] = loss
  return results

if __name__ == '__main__':
  num_runs = 3
  datasets = [
    'solar',
    'electricity',
    'traffic',
    'exchange',
    'm4',
    'uberTLC',
    'KDDCup',
    'wikipedia'
  ]
  models = ['vrs3gp', 'rs3gp', 'svgp', 'dklgp', 'smkgp']
  if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
  for i in range(num_runs):
    for model_name in models:
      for dataset_name in datasets:
        experiment_name = f'{dataset_name}_{model_name}_{i}'
        save_path = os.path.join(results_dir, experiment_name + '.yml')
        if os.path.exists(save_path):
          print(f'Skipping {experiment_name}...')
          continue
        with open(save_path, 'w') as f:
          pass
        print(f'Running experiment {experiment_name}...')
        results = run_experiment(dataset_name, model_name)
        if results == -1:
          continue
        with open(save_path, 'w') as f:
          yaml.dump(results, f)
