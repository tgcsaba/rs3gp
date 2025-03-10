import gpytorch
import numpy as np
import torch

from abc import ABC, abstractmethod
from itertools import chain
from linear_operator.settings import cholesky_jitter
from torch import Tensor
from tqdm.auto import tqdm
from typing import Optional, Sequence, Tuple

from ._utils import set_autoregressive_dataset


class TemplateGP(ABC):
  @abstractmethod
  def train(self, train_x, train_y):
    pass

  @abstractmethod
  def predict(self, test_x):
    pass

class AutoregressiveGP(TemplateGP):
  """Base class for autoregressive Gaussian process."""
  def __init__(self, train_x, train_y, context_len: int, prediction_len: int, rescale: bool = True,
               max_tries: int = 10, device: str = 'cuda'):
    self.train_x = train_x
    self.train_y = train_y
    self.context_len = context_len
    self.prediction_len = prediction_len
    self.rescale = rescale
    self.max_tries = max_tries
    self.device = device

  @abstractmethod
  def parameters(self):
    pass

  @abstractmethod
  def mll(self, lik: gpytorch.likelihoods.Likelihood, model: gpytorch.Module, num_data: int):
    pass

  def train(self, num_epochs: int = 200, lr: float = 1e-3, batch_size: int = 32,
            min_steps: int = 20000, max_steps: int = 200000) -> float:
    # setting training conditions
    train_loader = set_autoregressive_dataset(
      self.train_x, self.train_y, self.context_len, self.prediction_len, self.rescale, batch_size)
    num_epochs = max(num_epochs, int(np.ceil(min_steps / len(train_loader))))
    num_epochs = min(num_epochs, int(np.ceil(max_steps / len(train_loader))))
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    opt = torch.optim.Adam([{'params': param_group} for param_group in self.parameters()], lr=lr)

    self.model.train()
    self.likelihood.train()

    mll = self.mll(self.likelihood, self.model, train_loader.dataset.num_data)

    minibatch_iter = tqdm(total=len(train_loader))
    for i in epochs_iter:
      minibatch_iter.reset()
      avg_loss = 0.
      for x_batch, y_batch, *s_batch in train_loader:
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        opt.zero_grad()
        with cholesky_jitter(1e-3):
          output = self.model(x_batch, diag=True)
        loss = -mll(output, y_batch).mean()
        avg_loss += loss.item()
        loss.backward()
        opt.step()
        minibatch_iter.update()
        minibatch_iter.set_postfix(loss=loss.item())
        # sched.step()
      avg_loss /= len(train_loader)
      epochs_iter.set_postfix(loss=avg_loss)

    self.model.eval()
    self.likelihood.eval()

    return avg_loss

  def predict(self, test_x: Optional[Sequence[Tensor]], test_y: Sequence[Tensor],
              batch_size: int = 32) -> Tuple[Tensor, Tensor]:
    means = []
    stdevs = []
    test_loader = set_autoregressive_dataset(
      test_x, test_y, self.context_len, self.prediction_len, self.rescale, batch_size,
      shuffle=False)
    minibatch_iter = tqdm(test_loader, desc="Minibatch", leave=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_max_tries(self.max_tries):
      for x_batch, _, *s_batch in minibatch_iter:
        x_batch = x_batch.to(self.device)
        with cholesky_jitter(1e-3):
          preds = self.likelihood(self.model(x_batch, diag=True))
        preds_mean = preds.mean
        preds_var = preds.variance
        if self.rescale:
          s_batch = s_batch[0].to(self.device)
          preds_mean *= s_batch
          preds_var *= s_batch**2
        means.append(preds_mean.cpu())
        stdevs.append(preds_var.abs().sqrt().cpu())
    means = torch.cat(means, axis=0)
    stdevs = torch.cat(stdevs, axis=0)
    border_idx = np.cumsum([0] + test_loader.dataset.lens)
    means = [means[border_idx[i]:border_idx[i+1]] for i in range(len(test_y))]
    stdevs = [stdevs[border_idx[i]:border_idx[i+1]] for i in range(len(test_y))]
    return means, stdevs
