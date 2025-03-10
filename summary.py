import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd

if len(sys.argv) > 1:
  results_dir = str(sys.argv[1])
else:
  results_dir = './results/'

result_files = [
  fp for fp in glob.glob(os.path.join(results_dir, '*.yml')) if os.stat(fp).st_size > 0
]
result_dict = {'train': {}, 'test': {}, 'time': {}}
for fp in result_files:
  dataset, model = os.path.basename(fp).rstrip('.yml').split('_')[:2]
  if model not in result_dict['train']:
    result_dict['train'][model] = {}
  if model not in result_dict['test']:
    result_dict['test'][model] = {}
  if model not in result_dict['time']:
    result_dict['time'][model] = {}
  with open(fp, 'r') as f:
    result = yaml.load(f, Loader=yaml.SafeLoader)
  if result == '':
    continue
  if dataset not in result_dict['train'][model]:
    result_dict['train'][model][dataset] = []
  if dataset not in result_dict['test'][model]:
    result_dict['test'][model][dataset] = []
  if dataset not in result_dict['time'][model]:
    result_dict['time'][model][dataset] = []

  if 'results' in result:
    if 'train_loss' in result['results']:
      result_dict['train'][model][dataset].append(result['results']['train_loss'])
    result_dict['test'][model][dataset].append(result['results']['mean_wQuantileLoss'])
    result_dict['time'][model][dataset].append(result['time']['train'] / 3600)
  else:
    result_dict['test'][model][dataset].append(result['calibrated']['mean_wQuantileLoss'])

result_dict_mean = {
  key: {model: {
    dataset: np.mean(result_dict[key][model][dataset]) for dataset in result_dict[key][model]
  }
  for model in result_dict[key]} for key in result_dict
}

result_dict_str = {
  key: {model: {
    dataset: f'{np.mean(result_dict[key][model][dataset]):.3f} +- {np.std(result_dict[key][model][dataset]):.3f}' for dataset in result_dict[key][model]
  }
  for model in result_dict[key]} for key in result_dict
}

# for model in result_dict_str:
#   for ds_key in ds_keys:
#     if ds_key not in result_dict_str[model]:
#       result_dict_str[model][ds_key] = '-'
datasets = ['solar', 'electricity', 'traffic', 'exchange', 'm4', 'uberTLC', 'KDDCup', 'wikipedia']
print('Train')
df = pd.DataFrame.from_dict(result_dict_str['train']).transpose().sort_index()
print(df[datasets])
print('Test')
df = pd.DataFrame.from_dict(result_dict_str['test']).transpose().sort_index()
print(df[datasets])
print('Time')
df = pd.DataFrame.from_dict(result_dict_str['time']).transpose().sort_index()
print(df[datasets])
