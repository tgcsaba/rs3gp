import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Optional, Sequence, Tuple


def draw_inducing_points(train_x, n_inducing_points=500):
  """
  Drawing inducing points via sparcifying the train_x

  Args:
    - train_x: torch.tensor, 2d tensor of input, N x d
    - n_inducing_points: int, number of inducing points, default is 500.

  Return:
    - inducing_points: torch.tensor, 2d tensor of the sparse train_x
  """
  random_samples = torch.quasirandom.SobolEngine(train_x.shape[-1], scramble=True).draw(n_inducing_points).to(train_x)
  bound_min = train_x.min(axis=0).values.unsqueeze(0)
  bound_max = train_x.max(axis=0).values.unsqueeze(0)
  inducing_points = bound_min + (bound_max - bound_min) * random_samples
  #idx_init = math.floor(len(train_x) / n_inducing_points)
  #indices = torch.linspace(idx_init, idx_init * n_inducing_points, n_inducing_points).long()
  #inducing_points = train_x[indices]
  return inducing_points
#
def draw_autoregressive_inducing_points(train_x, train_y, context_len, prediction_len, num_inducing=500):
  """
  Drawing inducing points via sparcifying the train_x

  Args:
    - train_x: torch.Tensor, 3d tensor of inputs, N x L x d
    - train_y: torch.Tensor, 2d tensor of targets, N x L
    - context_len: positive int, context length
    - prediction_len: positive int, prediction length
    - n_inducing_points: int, number of inducing points, default is 500.

  Return:
    - inducing_points: torch.Tensor, 2d tensor of initial inducing points.
  """
  loader = set_autoregressive_dataset(
    train_x, train_y, context_len, prediction_len, rescale=True, batch_size=num_inducing)
  return next(iter(loader))[0]


def set_dataset(train_x, train_y, shuffle=True, batch_size=1024):
  """
  Set the dataset and data loader

  Args:
    - train_x: torch.tensor, 2d tensor of input, N x d
    - train_y: torch.tensor, 1d tensor of target value, N
    - shuffle: bool, shuffle training data if true, otherwise not.

  Return:
    - train_loader: torch.utils.data.DataLoader, data loader
  """
  train_dataset = TensorDataset(train_x, train_y)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
  return train_loader

class AutoregressiveDataset(Dataset):
  def __init__(self, inputs: Optional[Sequence[Tensor]], targets: Sequence[Tensor],
               context_len: int, prediction_len: int, rescale: bool = False):
    """Define autoregressive dataset from a time series."""
    if inputs is not None:
      assert len(inputs) == len(targets)
      for i in range(len(targets)):
        assert inputs[i].size(0) == targets[i].size(0)
    self.inputs = inputs
    self.targets = targets
    self.context_len = context_len
    self.prediction_len = prediction_len
    self.rescale = rescale
    self.cache = {}

  @property
  def lens(self):
    return [ts.size(0)-self.prediction_len-self.context_len+1 for ts in self.targets]

  @property
  def num_data(self):
    return sum(self.lens)

  def __len__(self):
    return self.num_data

  def __getitem__(self, idx):
    if idx in self.cache:
      return self.cache[idx]
    border_idx = np.cumsum([0] + self.lens)
    series_idx = np.where(idx < border_idx)[0][0] - 1
    step_idx = idx - border_idx[series_idx]
    x = self.targets[series_idx][step_idx:step_idx+self.context_len]
    if self.rescale:
      scale = x.abs().mean().maximum(torch.tensor(1e-6).to(x))
      x = x / scale
    if self.inputs is not None:
      x_ = self.inputs[series_idx][step_idx+max(self.context_len-1, 0)]
      x = torch.cat((x_, x), axis=0)
    y = self.targets[series_idx][step_idx+self.prediction_len+max(self.context_len-1, 0)]
    if self.rescale:
      y = y / scale
      self.cache[idx] = (x, y, scale)
      return x, y, scale
    else:
      self.cache[idx] = (x, y)
      return x, y

def set_autoregressive_dataset(train_x: Optional[Sequence[Tensor]], train_y: Sequence[Tensor],
                               context_len: int, prediction_len: int, rescale: bool = False,
                               batch_size: int = 128, shuffle: bool = True):
  """Set autoregressive dataset and corresponding loader"""
  dataset = AutoregressiveDataset(train_x, train_y, context_len, prediction_len, rescale=rescale)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return loader

class TimeSeriesDataset(Dataset):
  def __init__(self, inputs: Optional[Sequence[Tensor]], targets: Sequence[Tensor],
               prediction_len: int, multioutput: bool = False, rescale: bool = False):
    if inputs is not None:
      assert len(inputs) == len(targets)
      for i in range(len(targets)):
        assert inputs[i].size(0) == targets[i].size(0)
    self.inputs = inputs
    self.targets = targets
    self.prediction_len = prediction_len
    self.multioutput = multioutput
    self.rescale = rescale
    self.cache = {}

  def __len__(self) -> int:
    return len(self.targets)

  @property
  def lens(self) -> List[int]:
    return [ts.size(0) - self.prediction_len for ts in self.targets]

  @property
  def num_data(self) -> int:
    return sum(self.lens)

  def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
    if idx in self.cache:
      return self.cache[idx]
    x = self.targets[idx][:-self.prediction_len, None]
    if self.rescale:
      scale = x.abs().mean().maximum(torch.tensor(1e-6).to(x))
      x = x / scale
    if self.inputs is not None:
      x = torch.cat((self.inputs[idx][:-self.prediction_len], x), axis=-1)
    x = x.view(-1)
    if self.multioutput:
      y = self.targets[idx][1:].unfold(0, self.prediction_len, 1).transpose(-2, -1)
    else:
      y = self.targets[idx][self.prediction_len:]
    if self.rescale:
      y = y / scale
      self.cache[idx] = (x, y, scale)
      return x, y, scale
    else:
      self.cache[idx] = (x, y)
      return x, y

# def collate_fn(data: List[Tuple]):
#   data = tuple(zip(*data))
#   inputs, targets = data[0], data[1]
#   max_len = max([x.size(0) for x in inputs])
#   inputs = torch.stack(
#     [torch.nn.functional.pad(x, (0, 0, max_len-x.size(0), 0)) for x in inputs])
#   targets = torch.stack(
#     [torch.nn.functional.pad(y, (max_len-y.size(0), 0)) for y in targets])
#   print(inputs.shape, targets.shape)
#   if len(data) == 3:
#     scales = torch.stack(data[2])
#     return inputs, targets, scales
#   else:
#     return inputs, targets

def set_timeseries_dataset(train_x: Tensor, train_y: Tensor, prediction_len: int,
                           multioutput: bool = False, rescale: bool = False, shuffle: bool = True):
  dataset = TimeSeriesDataset(train_x, train_y, prediction_len, rescale=rescale,
                              multioutput=multioutput)
  loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
  return loader