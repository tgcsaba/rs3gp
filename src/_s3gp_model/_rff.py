import math
import torch
import sys

sys.path.append('..')

from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from torch import Tensor
from torch.distributions import Beta, Normal, Uniform
from typing import Optional, Tuple

from ._hputils import (
    AddedKLLossTerm,
    BetaSampler,
    NormalSampler,
    UniformSampler
)

# ------------------------------------------------------------------------------

class RandomFourierFeatures(Kernel):
  has_lengthscale = False

  def __init__(self, input_dim: int, num_features: int = 1000, learn_variational: bool = True,
               num_data: Optional[int] = None, resample_steps: int = None,  **kwargs):
    """Initializes the RandomFourierFeatures kernel.

    This method implements a method called .feature, which computes the feature
    map of the input, with dimension 1 + num_features * num_levels.

    Args:
      input_dim: The input dimension, 1 <=.
      num_features: Number of features, 1 <=.
      learn_variational: Learn a variational approxiation to freqs and phases.
      num_data: Provide this if learn_variational. The approximate mll divides
        all terms through by num_data, hence the variational KL divergences
        added to the loss by this module should also be divided by it.
      resample_steps: Number of steps to resample the random parameters,
        set to None to keep as fixed throughout.
      active_dims: List of data dimensions to operate on.
    """
    self.input_dim = input_dim
    self.num_features = num_features
    self.learn_variational = learn_variational
    self.resample_steps = resample_steps
    self.num_data = num_data
    super().__init__(**kwargs)

    # Frequencies. [input_dim, num_features]
    freq = torch.zeros(
      self.input_dim, self.num_features)
    self.register_buffer(name="freq", tensor=freq)
    # Phases. [num_features]
    phase = torch.zeros(self.num_features)
    self.register_buffer(name="phase", tensor=phase)

    self._initialize_distributions()

    self.register_buffer(name="steps_since_resample", tensor=torch.tensor(0))
    # Sample initial freq and prior values.
    with torch.no_grad():
      self._sample_random_variables()

  @property
  def freq_prior_stdev(self) -> Tensor:
    return self.raw_freq_prior_stdev_constraint.transform(
      self.raw_freq_prior_stdev)

  @property
  def freq_variational_stdev(self)  -> Tensor:
    return self.raw_freq_variational_stdev_constraint.transform(
      self.raw_freq_variational_stdev)

  @property
  def phase_variational_alpha(self)  -> Tensor:
    return self.raw_phase_variational_alpha_constraint.transform(
      self.raw_phase_variational_alpha)

  @property
  def phase_variational_beta(self)  -> Tensor:
    return self.raw_phase_variational_beta_constraint.transform(
      self.raw_phase_variational_beta)

  def _initialize_distributions(self):
    """Initializes prior and variational distributions over freq and phase."""
    # Frequency prior.
    self.register_parameter(
      name="raw_freq_prior_stdev",
      parameter=torch.nn.Parameter(0.5413 * torch.ones(self.freq.shape)))
    self.register_constraint(
      param_name="raw_freq_prior_stdev", constraint=Positive())

    # Frequency sampler is always Gaussian.
    self.freq_sampler = NormalSampler(self.freq.shape)
    if self.learn_variational:  # Initialize variational distributions to prior.
      # Frequency hyperparameters.
      ## Variational mean.
      self.register_parameter(
        name="freq_variational_mean",
        parameter=torch.nn.Parameter(torch.zeros(self.freq.shape)))
      ## Variational stdev.
      self.register_parameter(
        name="raw_freq_variational_stdev",
        parameter=torch.nn.Parameter(0.5413 * torch.ones(self.freq.shape)))
      ## Variational stdev constraint.
      self.register_constraint(
        param_name="raw_freq_variational_stdev", constraint=Positive())
      ## Variational freq KL loss term.
      self.register_added_loss_term("freq_kl")

      # Phase hyperparameters.
      ## Variational phase alpha.
      self.register_parameter(
        name="raw_phase_variational_alpha",
        parameter=torch.nn.Parameter(0.5413 * torch.ones(self.phase.shape)))
      ## Variational phase alpha constraint.
      self.register_constraint(
        param_name="raw_phase_variational_alpha", constraint=Positive())
      ## Variational phase beta.
      self.register_parameter(
        name="raw_phase_variational_beta",
        parameter=torch.nn.Parameter(0.5413 * torch.ones(self.phase.shape)))
      ## Variational phase beta constraint.
      self.register_constraint(
        param_name="raw_phase_variational_beta", constraint=Positive())
      ## Variational phase sampler.
      self.phase_sampler = BetaSampler(self.phase.shape)
      # Variational phase KL loss term.
      self.register_added_loss_term("phase_kl")
    else:
      # Prior phase sampler is uniform.
      self.phase_sampler = UniformSampler(self.phase.shape)

  def _sample_random_variables(self) -> Tuple[Tensor]:
    """Samples freq and phase and returns any added loss terms.

    Returns:
      freq_kl, phase_kl: Added KL divergence terms added to the ELBO for each if
        set to variational otherwise 0.
    """
    if self.training:
      # Resample parameters if exceeded resample_steps.
      if (self.resample_steps is not None and
          self.steps_since_resample >= self.resample_steps):
        self.freq_sampler.fit()
        self.phase_sampler.fit()
        self.steps_since_resample.copy_(1)
      else:
        self.steps_since_resample.add_(1)
    if self.learn_variational:
      # Frequency.
      ## Prior distribution.
      freq_prior = Normal(
        torch.zeros(self.freq.shape, device=self.device), self.freq_prior_stdev)
      ## Variational distribution.
      freq_variational = Normal(
        self.freq_variational_mean, self.freq_variational_stdev)
      ## Variational samples.
      freq_sample = self.freq_sampler.sample(
        self.freq_variational_mean, self.freq_variational_stdev)
      ## KL divergence.
      freq_kl = AddedKLLossTerm(freq_variational, freq_prior, self.num_data)
      # Phase.
      ## Prior distribution.
      phase_prior = Uniform(torch.zeros(self.phase.shape, device=self.device),
                            torch.ones(self.phase.shape, device=self.device))
      ## Variational distribution.
      phase_variational = Beta(
        self.phase_variational_alpha, self.phase_variational_beta)
      ## Variational samples.
      phase_sample = 2. * math.pi * self.phase_sampler.sample(
        self.phase_variational_alpha, self.phase_variational_beta)
      ## KL divergence
      phase_kl = AddedKLLossTerm(phase_variational, phase_prior, self.num_data)
    else:
      # Frequency.
      ## Prior samples.
      freq_sample = self.freq_sampler.sample(0., self.freq_prior_stdev)
      freq_kl = 0.
      # Phase.
      ## Prior samples.
      phase_sample = 2 * math.pi * self.phase_sampler.sample()
      phase_kl = 0.

    # Store sampled values.
    self.freq = freq_sample
    self.phase = phase_sample
    # Return added loss terms.
    return freq_kl, phase_kl

  def feature(self, x: Tensor, **params) -> Tensor:
    """Computes the feature map of the input.

    Args:
      x: Tensor of shape [..., N, input_dim].

    Returns:
      x_feat: Features of shape [..., N, num_features].
    """
    x_feat = math.sqrt(2. / self.num_features) * (x @ self.freq + self.phase).cos()
    return x_feat

  def forward(self, x1: Tensor, x2: Tensor, feature: bool = False, **params) -> Tensor:
    """Computes the kernel matrix or features.

    Args:
      x1: Tensor of shape [..., N1, input_dim].
      x2: Tensor of shape [..., N2, input_dim].
      feature: Whether to return only the features of the first input.

    Returns:
      Feature matrix of shape [..., N1, num_features] if feature else [..., N1, N2].
    """
    if self.training:
      freq_kl, phase_kl = self._sample_random_variables()
      if self.learn_variational:
        self.update_added_loss_term("freq_kl", freq_kl)
        self.update_added_loss_term("phase_kl", phase_kl)

    # [..., N1, num_features]
    x1_feat = self.feature(x1)
    if feature:
      return x1_feat
    is_symmetric = x1.size() == x2.size() and torch.equal(x1, x2)
    if is_symmetric:
      x2_feat = x1_feat
    else:
      # [..., N2, num_features]
      x2_feat = self._feature(x2)
    # [..., N1, N2]
    K = torch.einsum('...kp,...np->...kn', x1_feat, x2_feat)
    return K