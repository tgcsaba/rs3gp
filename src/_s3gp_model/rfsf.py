import math
import torch
import sys

sys.path.append('..')

from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.module import Module
from torch import Size, Tensor
from typing import List, Union

# ------------------------------------------------------------------------------

class NormalSampler(Module):
  def __init__(self, shape: Size, resample: bool = True):
    super().__init__()
    self.shape = shape
    self.resample = resample
    self.register_buffer(name="buffer", tensor=torch.randn(self.shape))

  def sample(self, mean: Tensor, stdev: Tensor):
    if self.resample and self.training:
      self.buffer = torch.randn(self.shape).to(stdev)
    return mean + stdev * self.buffer

# ------------------------------------------------------------------------------

class UniformSampler(Module):
  def __init__(self, shape: Size, resample: bool = True):
    super().__init__()
    self.shape = shape
    self.resample = resample
    self.register_buffer(name="buffer", tensor=torch.rand(self.shape))

  def sample(self):
    if self.resample and self.training:
      self.buffer = torch.rand(self.shape).to(self.buffer)
    return self.buffer

# ------------------------------------------------------------------------------

class RandomRandomFourierSignatureFeatures(Kernel):
  has_lengthscale = False

  def __init__(self, num_features_per_level: int, state_dim: int,
               num_levels: int = 4, normalize: bool = True,
               add_time: bool = True, add_basepoint: bool = True,
               resample: bool = True, **kwargs):
    """Initializes the RandomFourierSignatureFeatures kernel.

    This method implements a method called .feature, which computes the feature
    map of the input, with dimension 1 + num_features * num_levels.

    Args:
      num_features_per_level: Number of features per signature level, 1 <=.
      state_dim: The state-space dimension, 1 <=.
      num_levels: Number of signature levels, 1 <=.
      normalize: Whether to normalize the signature levels.
      add_time: Whether to included time parametrization as a coordinate.
      add_basepoint: Whether to include a basepoint to the remove
        translations invariance of signatures.
      resample: Whether to resample the reparametrized random weights and
        phases or keep the underlying random outcomes as fixed.
      active_dims: List of data dimensions to operate on.
    """
    self.num_features_per_level = num_features_per_level
    self.state_dim = state_dim
    self.num_levels = num_levels
    self.normalize = normalize
    self.add_time = add_time
    self.add_basepoint = add_basepoint
    self.resample = resample
    self.state_dim_ = self.state_dim + int(self.add_time)
    super().__init__(**kwargs)

    # Register variance hyperparameters that multiply each sig-level.
    init_vars = torch.zeros(num_levels+1)
    self.register_parameter(
      name="raw_variances", parameter=torch.nn.Parameter(init_vars))
    self.register_constraint(param_name="raw_variances", constraint=Positive())
    # Frequencies. [state_dim_, num_levels, num_features_per_level]
    freq = torch.zeros(
      self.state_dim_, self.num_levels, self.num_features_per_level)
    self.register_buffer(name="freq", tensor=freq)
    # Phases. [num_levels, num_features_per_level]
    phase = torch.zeros(self.num_levels, self.num_features_per_level)
    self.register_buffer(name="phase", tensor=phase)
    # Initialize prior and variational distributions.
    self._initialize_distributions()
    # Sample initial freq and prior values.
    with torch.no_grad():
      self._sample_random_variables()

  @property
  def num_features(self) -> int:
    return self.num_features_per_level * self.num_levels + 1

  @property
  def variances(self) -> Tensor:
    return self.raw_variances_constraint.transform(self.raw_variances)

  @variances.setter
  def variances(self, value: Union[float, Tensor]):
    self._set_variances(value)

  def _set_variances(self, value: Union[float, Tensor]):
    if not torch.is_tensor(value):
      value = torch.as_tensor(value).to(self.raw_variances)
    self.initialize(
      raw_variances=
      self.raw_variances_constraint.inverse_transform(value))

  @property
  def freq_prior_stdev(self) -> Tensor:
    return self.raw_freq_prior_stdev_constraint.transform(
      self.raw_freq_prior_stdev)

  def _initialize_distributions(self):
    """Initializes prior and variational distributions over freq and phase."""
    # Frequency prior.
    self.register_parameter(
      name="raw_freq_prior_stdev",
      parameter=torch.nn.Parameter(torch.randn(self.freq.shape)))
    self.register_constraint(
      param_name="raw_freq_prior_stdev", constraint=Positive())

    # Frequency sampler.
    self.freq_sampler = NormalSampler(self.freq.shape, resample=self.resample)
    # Phase sampler.
    self.phase_sampler = UniformSampler(
      self.phase.shape, resample=self.resample)

  def _sample_random_variables(self):
    """Samples frequencies and phases."""
    # Frequency prior samples.
    self.freq = self.freq_sampler.sample(
      torch.tensor(0., device=self.device), self.freq_prior_stdev)
    # Phase prior samples.
    self.phase = 2 * math.pi * self.phase_sampler.sample()
    # Return added loss terms.

  def _preprocess(self, x: Tensor) -> Tensor:
    """Takes as input a flattened sequence tensor and preprocesses it.

    Args:
      x: Tensor of shape [..., N, L * state_dim]

    Returns:
      Tensor of shape [..., N, L_, state_dim_], where L_ is equal to L+1
        if add_basepoint otherwise to L, while state_dim_ is equal to
        state_dim+1 if add_time else to state_dim.
    """
    # Unflatten the sequence axis.
    x = x.view(*x.shape[:-1], -1, self.state_dim)
    # Add basepoint. [..., N, 1, state_dim]
    x_0 = x[..., :1, :]  # Rescaled first step.
    if self.add_basepoint:
      x_0 = torch.zeros_like(x[..., :1, :])  # Zero start.
    else:
      x_0 = x[..., :1, :]  # First step warped.
    # [..., N, L + 1, state_dim]
    x = torch.cat((x_0, x), axis=-2)
    # Add time parametrization.
    if self.add_time:
      # Equispaced time-coordinate between 0-1, [L].
      t = torch.arange(x.size(-2), dtype=x.dtype) / (x.size(1) - 1)
      # Broadcast shape, [..., N, L, 1]
      t = torch.broadcast_to(t[:, None], x.shape[:-1] + (1,)).to(x)
      # Concatenate time channel, [..., N, L, state_dim + 1]
      x = torch.cat((t, x), axis=-1)
    return x

  def _feature(self, x: Tensor) -> List[Tensor]:
    """Computes the feature map.

    Args:
      x: Tensor of shape [..., N, L, state_dim_].

    Returns:
      P: List of features per signature
        [..., N, num_features_per_level, num_levels+1]
    """
    L = x.size(-2) - 1
    # Pre-activations. [..., N, L, num_levels, num_features_per_level]
    proj = torch.einsum('...d,dmp->...mp', x, self.freq) + self.phase
    # Random Fourier Features. [..., N, L, num_levels, num_features_per_level].
    U = proj.cos()
    # Take increments. [..., N, L-1, num_levels, num_features_per_level].
    U = torch.diff(U, axis=-3)
    # Initialize level-1 features. [..., N, num_features_per_level]
    P = [U[..., 0, :].sum(axis=-2)]
    # Iterate to get higher levels. [..., N, L-1, num_features_per_level]
    R = U[..., 0, :]
    for m in range(1, self.num_levels):
      # Perform exclusive cumsum. [..., N, L-1, num_features_per_level]
      R = R.cumsum(axis=-2).roll(1, -2)
      R[..., 0, :] = 0.
      R *= U[..., m, :]
      # Collapse into level-m features. [..., N, num_features_per_level]
      P.append(R.sum(axis=-2))
    if self.normalize:
      norms = [p.norm(dim=-1, keepdim=True) + 1e-12 for p in P]
      P = [p / norms[i] for i, p in enumerate(P)]
    # [..., N, num_levels, num_features_per_level]
    return torch.stack(P, axis=-2)

  def feature(self, x: Tensor, **params) -> Tensor:
    """Computes the feature map of the input.

    Args:
      x: Tensor of shape [..., N, L * state_dim].

    Returns:
      x_feat: Features of shape [..., N, num_features].
    """
    # Path augmentations. [..., N, L_, state_dim_]
    x = self._preprocess(x)
    # Compute features. [..., N, num_levels, num_features_per_level]
    x_feat = self.variances[1:, None].sqrt() * self._feature(x)
    # [..., N, num_levels * num_features_per_level]
    x_feat = x_feat.view(*x_feat.shape[:-2], -1)
    # Add constant term. # [..., N, num_features]
    x_feat = torch.cat((
      self.variances[0].sqrt() * torch.ones_like(x_feat[..., :1]),
      x_feat), axis=-1)
    return x_feat

  def forward(self, x1: Tensor, x2: Tensor, feature: bool = False,
              diag: bool = False, **params) -> Tensor:
    """Computes the kernel matrix of the inputs.

    Args:
      x1: Tensor of shape [..., N1, L1 * state_dim].
      x2: Tensor of shape [..., N2, L2 * state_dim].
      feature: Whether to return only the features of the first input.
      diag: Whether to compute only the diagonals.

    Returns:
      Feature matrix of shape [..., N1, num_features] if feature else
        kernel matrix of shape [..., N1] if diag else of shape [..., N1, N2].
    """
    if self.training:
      self._sample_random_variables()

    if feature:  # Ignore second input argument.
      # [..., N1, num_features]
      return self.feature(x1)

    # Handled normalized diagonal case.
    if self.normalize and diag:
      # Constant array of shape [..., N1]
      return self.variances.sum() * torch.ones(*x1.shape[:-1]).to(x1)

    is_symmetric = x1.size() == x2.size() and torch.equal(x1, x2)
    # [..., N1, num_features]
    x1_feat = self.feature(x1)
    if is_symmetric or diag:
      x2_feat = x1_feat
    else:
      # [..., N2, num_features]
      x2_feat = self._feature(x2)

    if diag:
      # [..., N1]
      K = torch.einsum('...kp->...kp', x1_feat, x2_feat)
    else:
      # [..., N1, N2]
      K = torch.einsum('...kp,...np->...kn', x1_feat, x2_feat)
    return K

# ------------------------------------------------------------------------------