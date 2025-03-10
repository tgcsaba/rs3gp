
from ._strategy import VariationalFeatureStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from linear_operator.operators import DiagLinearOperator
from torch import Tensor
from typing import Optional

# ------------------------------------------------------------------------------

class VariationalFeatureGPModel(ApproximateGP):
  def __init__(self, gp_kernel: Kernel, num_outputs: int = 1,
               variance_penalty: Optional[float] = None, learn_prior: bool = False):
    """Variational GP model for weight-space (feature-based) inference.

    Args:
      gp_kernel: GPyTorch kernel function with an explicit feature map.
      num_outputs: Number of outputs to the model.
    """
    if num_outputs > 1:
      variational_distribution = CholeskyVariationalDistribution(
        gp_kernel.num_features, batch_shape=[num_outputs])
    else:
      variational_distribution = CholeskyVariationalDistribution(gp_kernel.num_features)
    variational_strategy = VariationalFeatureStrategy(self, variational_distribution,
                                                      variance_penalty=variance_penalty,
                                                      learn_prior=learn_prior)
    super().__init__(variational_strategy)
    self.mean_module = ZeroMean()
    self.covar_module = gp_kernel

  def forward(self, x: Tensor, diag: bool = False) -> MultivariateNormal:
    """Computes the prior distribution of the inputs.

    Args:
      x: The inputs we wish to compute the prior for.

    Return:
      dist: The prior distribution of the inputs.
    """
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x, diag=diag)
    if diag:
      covar_x = DiagLinearOperator(covar_x)
    return MultivariateNormal(mean_x, covar_x)

# ------------------------------------------------------------------------------