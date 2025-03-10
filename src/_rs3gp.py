import gpytorch
import numpy as np
import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import settings
from torch import Tensor
from tqdm.auto import tqdm
from typing import Optional, Sequence, Tuple

from ._s3gp_model._feature import RandomFourierSignatureFeatures
from ._s3gp_model._model import VariationalFeatureGPModel
from ._base_gp import TemplateGP
from ._utils import TimeSeriesDataset, set_timeseries_dataset


class RecurrentSparseSpectrumSignatureGP(TemplateGP):
  def __init__(self, train_x: Optional[Sequence[Tensor]], train_y: Sequence[Tensor],
               prediction_len: int, multioutput: bool = False, rescale: bool = True,
               variance_penalty: Optional[float] = 0.5, max_tries: int = 3,
               device: str= 'cpu', **kernel_hparams):
    self.train_x = [x.to(device) for x in train_x] if train_x is not None else None
    self.train_y = [y.to(device) for y in train_y]
    self.prediction_len = prediction_len
    self.multioutput = multioutput
    self.rescale = rescale
    self.device = device
    self.dtype = train_y[0].dtype
    num_dims = train_x[0].size(-1) + 1 if train_x is not None else 1
    train_dataset = TimeSeriesDataset(train_x, train_y, self.prediction_len,
                                      multioutput=multioutput, rescale=rescale)
    num_data = train_dataset.num_data
    feat_map = RandomFourierSignatureFeatures(
      num_dims, num_data=num_data, return_sequences=True, **kernel_hparams).to(device, self.dtype)
    self.model = VariationalFeatureGPModel(
      feat_map, num_outputs=self.prediction_len if multioutput else 1,
      variance_penalty=variance_penalty, learn_prior=True).to(device, self.dtype)
    self.likelihood = GaussianLikelihood().to(device, self.dtype)
    self.max_tries = max_tries

  def train(self, num_epochs: int = 200, lr: float = 1e-3, min_steps = 20000):
    # setting training conditions
    train_loader = set_timeseries_dataset(
      self.train_x, self.train_y, self.prediction_len, self.multioutput, self.rescale)
    num_epochs = max(num_epochs, int(np.ceil(min_steps / len(train_loader))))
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    opt = torch.optim.Adam([
      {'params': self.model.parameters()},
      {'params': self.likelihood.parameters()}
    ], lr=lr)
    mll = gpytorch.mlls.PredictiveLogLikelihood(
      self.likelihood, self.model, num_data=train_loader.dataset.num_data)

    self.model.train()
    self.likelihood.train()
    minibatch_iter = tqdm(total=len(train_loader))
    for i in epochs_iter:
      minibatch_iter.reset()
      avg_loss = 0.
      for x_batch, y_batch, *s_batch in train_loader:
        opt.zero_grad()
        output = self.model(x_batch, diag=True)
        loss = -mll(output, y_batch.squeeze(axis=1)).mean()
        avg_loss += loss.item()
        loss.backward()
        opt.step()
        minibatch_iter.update()
        minibatch_iter.set_postfix(loss=loss.item())
      avg_loss /= len(train_loader)
      epochs_iter.set_postfix(loss=avg_loss)

    self.model.eval()
    self.likelihood.eval()

    return avg_loss

  def predict(self, test_x: Optional[Sequence[Tensor]], test_y: Sequence[Tensor]) \
              -> Tuple[Tensor, Tensor]:
    means = []
    stdevs = []
    test_loader = set_timeseries_dataset(
      test_x, test_y, self.prediction_len, self.multioutput, self.rescale, shuffle=False)
    minibatch_iter = tqdm(test_loader, desc="Minibatch", leave=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_max_tries(self.max_tries):
      for x_batch, _, *s_batch in minibatch_iter:
        x_batch = x_batch.to(self.device)
        preds = self.likelihood(self.model(x_batch, diag=True))
        preds_mean = preds.mean
        preds_var = preds.variance
        if self.rescale:
          s_batch = s_batch[0].to(self.device)
          preds_mean *= s_batch
          preds_var *= s_batch**2
        means.append(preds_mean.cpu())
        stdevs.append(preds_var.abs().sqrt().cpu())
    return means, stdevs
