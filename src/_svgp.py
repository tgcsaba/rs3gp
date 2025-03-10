import gpytorch
import numpy as np
import torch

from ._base_gp import AutoregressiveGP
from ._utils import draw_autoregressive_inducing_points

from torch import Tensor
from typing import Optional, Sequence


class ApproximateGPModel(gpytorch.models.ApproximateGP):
  def __init__(self, input_dim: int, inducing_points: Tensor):
    variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
      inducing_points.size(0))
    variational_strategy = gpytorch.variational.VariationalStrategy(
      self, inducing_points, variational_distribution, learn_inducing_locations=True)
    super().__init__(variational_strategy)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
      gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

  def forward(self, x: Tensor, diag: bool = False) -> gpytorch.distributions.Distribution:
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SparseVariationalGP(AutoregressiveGP):
  def __init__(self, train_x: Optional[Sequence[Tensor]], train_y: Sequence[Tensor],
               context_len: int, prediction_len: int, rescale: bool = True,
               num_inducing: int = 500, max_tries: int = 10, device: str = 'cuda'):
    super().__init__(train_x, train_y, context_len, prediction_len, rescale, max_tries, device)
    self.dtype = train_y[0].dtype
    inducing_points = draw_autoregressive_inducing_points(
      train_x, train_y, context_len, prediction_len, num_inducing)
    input_dim = (train_x[0].size(-1) + 1) * context_len if train_x is not None else context_len
    self.model = ApproximateGPModel(input_dim, inducing_points).to(device, self.dtype)
    self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device, self.dtype)

  def mll(self, lik: gpytorch.likelihoods.Likelihood, model: gpytorch.Module, num_data: int):
    return gpytorch.mlls.PredictiveLogLikelihood(lik, model, num_data=num_data)

  def parameters(self) -> Sequence[Tensor]:
    return [self.model.parameters(), self.likelihood.parameters()]
