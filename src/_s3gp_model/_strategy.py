import torch

from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import AddedLossTerm
from gpytorch.models import ApproximateGP
from gpytorch.module import Module
from gpytorch.variational import (
  _VariationalDistribution,
  _VariationalStrategy,
  VariationalStrategy
)
from linear_operator.operators import (
  DiagLinearOperator,
  LinearOperator,
  MatmulLinearOperator,
)
from torch import Tensor
from typing import Optional

# ------------------------------------------------------------------------------

class VariancePenalty(AddedLossTerm):
  def __init__(self, strength: float, variances: Tensor):
    self.strength = strength
    self.variances = variances

  def loss(self) -> Tensor:
    return self.strength * self.variances.mean()

class VariationalFeatureStrategy(_VariationalStrategy):

  def __init__(self, model: ApproximateGP, variational_distribution: _VariationalDistribution,
               variance_penalty: Optional[float] = None, learn_prior: bool = False):
    Module.__init__(self)
    # Model.
    object.__setattr__(self, "model", model)
    # Variational distribution.
    self._variational_distribution = variational_distribution
    self.register_buffer("variational_params_initialized", torch.tensor(False))
    self.learn_prior = learn_prior
    if self.learn_prior:
      self.register_parameter(
        name="prior_mean",
        parameter=torch.nn.Parameter(torch.zeros(variational_distribution.shape())))
      self.register_parameter(
        name="raw_prior_var",
        parameter=torch.nn.Parameter(0.5413 * torch.ones(variational_distribution.shape())))
      self.register_constraint(param_name="raw_prior_var", constraint=Positive())

    self.variance_penalty = variance_penalty
    if variance_penalty is not None:
      self.register_added_loss_term("variance_penalty")

  @property
  def prior_var(self) -> Tensor:
    return self.raw_prior_var_constraint.transform(self.raw_prior_var)

  @property
  def prior_distribution(self) -> MultivariateNormal:
    if self.learn_prior:
      res = MultivariateNormal(self.prior_mean, DiagLinearOperator(self.prior_var))
    else:
      zeros = torch.zeros(self._variational_distribution.shape(),
                          dtype=self._variational_distribution.dtype,
                          device=self._variational_distribution.device)
      ones = torch.ones_like(zeros)
      res = MultivariateNormal(zeros, DiagLinearOperator(ones))
    return res

  def forward(self, x: Tensor, variational_mean: Tensor, variational_covar: LinearOperator,
              diag: bool = False, **kwargs) -> MultivariateNormal:
    # Compute features. [N, (L,) num_features]
    features = self.model.covar_module.forward(x, None, feature=True, **kwargs)
    # Squeeze if `return_sequences` mode. [N/L, num_features]
    features = features.squeeze(axis=0)
    # Compute predictive mean. [(num_outputs,) N/L]
    if len(self._variational_distribution.batch_shape) > 0:
      pred_mean = torch.einsum('np,qp->qn', features, variational_mean)
    else:
      pred_mean = torch.einsum('np,p->n', features, variational_mean)

    # Compute predictive variances/covariances.
    L = variational_covar.cholesky().to_dense()  # [(num_outputs,) num_features, num_features]
    feat_mul_L = features @ L  # [(num_outputs,) N/L, num_features]
    if diag:
      pred_var = feat_mul_L.square().sum(axis=-1)  # [(num_outputs,) N/L)
      pred_cov = DiagLinearOperator(pred_var)  # [(num_outputs,), N/L, N/L]
    else:
      # [(num_outputs,) N/L, N/L]
      pred_cov = feat_mul_L @ feat_mul_L.transpose(-2, -1)
      pred_var = pred_cov.diag()
    if self.training and self.variance_penalty is not None:
      self.update_added_loss_term(
        "variance_penalty", VariancePenalty(self.variance_penalty, pred_var))
    # Return the distribution.  [(num_outputs,) N/L]
    return MultivariateNormal(pred_mean, pred_cov)

  def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
    # If we're in prior mode, then we're done!
    if prior:
      return self.model.forward(x, **kwargs)

    # Delete previously cached items from the training distribution.
    if self.training:
      self._clear_cache()
    # (Maybe) initialize variational distribution.
    if not self.variational_params_initialized.item():
      prior_dist = self.prior_distribution
      self._variational_distribution.initialize_variational_distribution(
        prior_dist)
      self.variational_params_initialized.fill_(True)

    variational_mean = self.variational_distribution.mean
    variational_covar = self.variational_distribution.lazy_covariance_matrix

    # Get q(f).
    if isinstance(self.variational_distribution, MultivariateNormal):
      return Module.__call__(self, x, variational_mean=variational_mean,
                             variational_covar=variational_covar, **kwargs)
    else:
      raise RuntimeError(
        "Invalid variational distribution: "
        f"({type(self.variational_distribution)}). "
        "Expected a multivariate normal distribution."
      )

# ------------------------------------------------------------------------------