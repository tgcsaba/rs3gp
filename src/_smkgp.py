import gpytorch
import numpy as np
import torch

from ._base_gp import AutoregressiveGP
from ._utils import set_autoregressive_dataset, draw_autoregressive_inducing_points

from torch import Tensor
from typing import Optional, Sequence

class ApproximateGPModel(gpytorch.models.ApproximateGP):
  def __init__(self, input_dim: int, inducing_points: Tensor, num_mixtures: int = 4):
    variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
      inducing_points.size(0))
    variational_strategy = gpytorch.variational.VariationalStrategy(
      self, inducing_points, variational_distribution, learn_inducing_locations=True)
    super().__init__(variational_strategy)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=input_dim)

  def forward(self, x, diag: bool = False) -> gpytorch.distributions.Distribution:
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SpectralMixtureVariationalGP(AutoregressiveGP):
  def __init__(self, train_x: Optional[Sequence[Tensor]], train_y: Sequence[Tensor],
               context_len: int, prediction_len: int, rescale: bool = True,
               num_inducing: int = 500, num_mixtures: int = 4, max_tries: int = 10,
               device: str = 'cuda'):
    super().__init__(train_x, train_y, context_len, prediction_len, rescale, max_tries, device)
    self.dtype = train_y[0].dtype
    inducing_points = draw_autoregressive_inducing_points(
      train_x, train_y, context_len, prediction_len, num_inducing)
    input_dim = (train_x[0].size(-1) + 1) * context_len if train_x is not None else context_len
    self.model = ApproximateGPModel(input_dim, inducing_points, num_mixtures=num_mixtures).to(
      device, self.dtype)
    train_loader = set_autoregressive_dataset(
      train_x, train_y, self.context_len, self.prediction_len)
    train_batch = next(iter(train_loader))
    self.model.covar_module.initialize_from_data_empspect(
      train_batch[0].to(device, self.dtype), train_batch[1].to(device, self.dtype))
    self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device, self.dtype)

  def mll(self, lik: gpytorch.likelihoods.Likelihood, model: gpytorch.Module, num_data: int):
    return gpytorch.mlls.PredictiveLogLikelihood(lik, model, num_data=num_data)

  def parameters(self) -> Sequence[Tensor]:
    return [self.model.parameters(), self.likelihood.parameters()]
