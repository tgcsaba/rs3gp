import gpytorch
import torch

from torch import Tensor
from torch.nn import Module
from typing import Optional, Sequence

from ._base_gp import AutoregressiveGP
from ._svgp import ApproximateGPModel
from ._utils import set_autoregressive_dataset, draw_autoregressive_inducing_points

class FeatureExtractor(torch.nn.Sequential):
  def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()
    self.add_module('linear1', torch.nn.Linear(input_dim, hidden_dim))
    self.add_module('relu1', torch.nn.ReLU())
    self.add_module('bn1', torch.nn.BatchNorm1d(hidden_dim))
    self.add_module('linear2', torch.nn.Linear(hidden_dim, hidden_dim))
    self.add_module('relu2', torch.nn.ReLU())
    self.add_module('bn2', torch.nn.BatchNorm1d(hidden_dim))

class DKLModel(gpytorch.Module):
  def __init__(self, hidden_dim: int, feature_extractor: Module, inducing_points: int):
    super().__init__()
    self.feature_extractor = feature_extractor
    self.gp_layer = ApproximateGPModel(hidden_dim, inducing_points)

  def forward(self, x: Tensor, diag: bool = False):
    features = self.feature_extractor(x)
    res = self.gp_layer(features)
    return res

class DeepKernelLearningGP(AutoregressiveGP):
  def __init__(self, train_x: Optional[Sequence[Tensor]], train_y: Sequence[Tensor],
               context_len: int, prediction_len: int, rescale: bool = True, hidden_dim: int = 64,
               num_inducing: int = 500, max_tries: int = 10, device: str = 'cuda'):
    super().__init__(train_x, train_y, context_len, prediction_len, rescale, max_tries, device)
    self.dtype = train_y[0].dtype
    input_dim = (train_x[0].size(-1) + 1) * context_len if train_x is not None else context_len
    feature_extractor = FeatureExtractor(input_dim, hidden_dim).to(device, self.dtype)
    with torch.no_grad():
      inducing_points = feature_extractor(draw_autoregressive_inducing_points(
        train_x, train_y, context_len, prediction_len, num_inducing).to(device, self.dtype))
    self.model = DKLModel(hidden_dim, feature_extractor, inducing_points).to(device, self.dtype)
    self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device, self.dtype)

  def mll(self, lik: gpytorch.likelihoods.Likelihood, model: gpytorch.Module, num_data: int) \
          -> gpytorch.mlls.MarginalLogLikelihood:
    return gpytorch.mlls.PredictiveLogLikelihood(lik, model.gp_layer, num_data=num_data)

  def parameters(self) -> Sequence[Tensor]:
    return [self.model.parameters(), self.likelihood.parameters()]

  # def train(self, train_x, train_y, num_epochs=2, lr=0.01, batch_size=1024):
  #   # setting training conditions
  #   train_loader = set_dataset(train_x, train_y, batch_size=batch_size)
  #   epochs_iter = tqdm(range(num_epochs), desc="Epoch")

  #   # Use the adam optimizer
  #   optimizer = torch.optim.Adam([
  #     {'params': self.model.hyperparameters()},
  #     {'params': self.model.variational_parameters()},
  #     {'params': self.likelihood.parameters()},
  #   ], lr=lr)

  #   self.model.train()
  #   self.likelihood.train()
  #   mll = gpytorch.mlls.VariationalELBO(
  #     self.likelihood,
  #     self.model,
  #     num_data=len(train_loader.dataset),
  #   )
  #   with (
  #     gpytorch.settings.cholesky_max_tries(self.max_tries),
  #     gpytorch.settings.max_cholesky_size(self.max_cholesky_size),
  #     gpytorch.settings.cholesky_jitter(1e-1),
  #     gpytorch.settings.use_toeplitz(False),
  #   ):
  #     for i in epochs_iter:
  #       minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
  #       for x_batch, y_batch in minibatch_iter:
  #         ### Perform NGD step to optimize variational parameters
  #         optimizer.zero_grad()
  #         output = self.model(x_batch)
  #         loss = -mll(output, y_batch)
  #         minibatch_iter.set_postfix(loss=loss.item())
  #         loss.backward()
  #         optimizer.step()
  #   self.model.eval()
  #   self.likelihood.eval()

  # def predict(self, test_x):
  #   means = torch.tensor([0.])
  #   variances = torch.tensor([0.])
  #   test_loader = set_dataset(test_x, torch.zeros(len(test_x)), shuffle=False)
  #   minibatch_iter = tqdm(test_loader, desc="Minibatch", leave=False)
  #   with (
  #     torch.no_grad(),
  #     gpytorch.settings.cholesky_max_tries(self.max_tries),
  #     gpytorch.settings.max_cholesky_size(self.max_cholesky_size),
  #     gpytorch.settings.cholesky_jitter(1e-1),
  #     gpytorch.settings.use_toeplitz(False),
  #   ):
  #     for x_batch, y_batch in minibatch_iter:
  #       preds = self.likelihood(self.model(x_batch))
  #       means = torch.cat([means, preds.mean.cpu()])
  #       variances = torch.cat([variances, preds.variance.cpu()])

  #   return means[1:], variances[1:].abs().sqrt()
