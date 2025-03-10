import math
import torch
import sys

sys.path.append('..')

from accelerated_scan.warp import scan
from gpytorch.constraints import Interval, Positive
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

class RandomFourierSignatureFeatures(Kernel):
  has_lengthscale = False

  def __init__(self, state_dim: int, num_features_per_level: int = 200,
               num_levels: int = 5, normalize: bool = True, first_order: bool = False,
               window: int = 10, num_lags: int = 9,  learn_variational: bool = True,
               num_data: Optional[int] = None, resample_steps: int = None,
               return_sequences: bool = False, **kwargs):
    """Initializes the RandomFourierSignatureFeatures kernel.

    This method implements a method called .feature, which computes the feature
    map of the input, with dimension 1 + num_features * num_levels.

    Args:
      state_dim: The state-space dimension, 1 <=.
      num_features_per_level: Number of features per signature level, 1 <=.
      num_levels: Number of signature levels, 1 <=.
      normalize: Whether to normalize the signature levels.
      first_order: Whether to compute the first order approximation.
      window: Window size for fractional differencing.
      num_lags: Number of lagged coordinates to include.
      learn_variational: Learn a variational approxiation to freqs and phases.
      num_data: Provide this if learn_variational. The approximate mll divides
        all terms through by num_data, hence the variational KL divergences
        added to the loss by this module should also be divided by it.
      resample_steps: Number of steps to resample the random parameters,
        set to None to keep as fixed throughout.
      return_sequences: Return the feature/kernel over expanding window.
        Warning: This option is not compatible with the __call__ method.
      active_dims: List of data dimensions to operate on.
    """
    self.num_features_per_level = num_features_per_level
    self.state_dim = state_dim
    self.num_levels = num_levels
    self.first_order = first_order
    self.normalize = normalize
    self.window = window
    self.num_lags = num_lags
    self.learn_variational = learn_variational
    self.resample_steps = resample_steps
    self.num_data = num_data
    self.return_sequences = return_sequences
    self.state_dim_ = (self.num_lags + 1) * self.state_dim
    super().__init__(**kwargs)

    # Register multiplier for each feature coordinate.
    self.register_parameter(
      name="raw_sigma", parameter=torch.nn.Parameter(0.5413 * torch.ones(self.num_levels + 1)))
    self.register_constraint(param_name="raw_sigma", constraint=Positive())

    if self.window > 1:  # Register fractional differencing orders.
      self.register_parameter(
        name="raw_diff", parameter=torch.nn.Parameter(torch.zeros(self.num_features_per_level)))
      self.register_constraint(param_name="raw_diff", constraint=Interval(0., 1.))
      with torch.no_grad():  # Initialize diff to uniform distribution.
        self.raw_diff.copy_(self.raw_diff_constraint.inverse_transform(
          torch.rand(self.num_features_per_level)))

    if self.return_sequences:  # Register decay hyperparameters.
      self.register_parameter(
        name="raw_decay",
        parameter=torch.nn.Parameter(torch.zeros(self.num_features_per_level)))
      self.register_constraint(
        param_name="raw_decay", constraint=Interval(0., 1.))
      with torch.no_grad():  # Initialize decay to uniform distribution.
        self.raw_decay.copy_(self.raw_decay_constraint.inverse_transform(
          torch.rand(self.num_features_per_level)))

    # Frequencies. [state_dim_, num_levels, num_features_per_level]
    freq = torch.zeros(
      self.num_levels * self.num_features_per_level, self.state_dim, self.num_lags + 1)
    self.register_buffer(name="freq", tensor=freq)
    # Phases. [num_levels, num_features_per_level]
    phase = torch.zeros(self.num_levels * self.num_features_per_level)
    self.register_buffer(name="phase", tensor=phase)

    self._initialize_distributions()

    self.register_buffer(name="steps_since_resample", tensor=torch.tensor(0))
    # Sample initial freq and prior values.
    with torch.no_grad():
      self._sample_random_variables()

  @property
  def num_features(self) -> int:
    return self.num_features_per_level * self.num_levels + 1

  @property
  def sigma(self) -> Tensor:
    return self.raw_sigma_constraint.transform(self.raw_sigma)

  @property
  def diff(self) -> Tensor:
    return self.raw_diff_constraint.transform(self.raw_diff)

  @property
  def basepoint(self) -> Tensor:
    return self.raw_basepoint_constraint.transform(self.raw_basepoint)

  @property
  def decay(self) -> Tensor:
    return self.raw_decay_constraint.transform(self.raw_decay)

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

  @property
  def _fracdiff_filter(self) -> Tensor:
    """Builds a convolutional filter for fractional differencing.

    Returns:
      A tensor of shape [num_features_per_level, num_features_per_level, window].
    """
    window = torch.arange(self.window).flip(0).to(self.diff)
    # [num_features_per_level, window]
    coef = (
      (self.diff[:, None] + 1).lgamma()
      - (window[None, :] + 1).lgamma()
      - (self.diff[:, None] - window[None, :] + 1).lgamma()
    ).exp()
    filt = torch.eye(self.num_features_per_level).to(coef)[..., None] * coef[None]
    return filt

  def _transform(self, x: Tensor) -> Tensor:
    """Preprocesses the sequence and transforms it into RFF increments per level.

    Args:
      x: Tensor of shape [..., N, L * state_dim_]

    Returns:
      U: RFF increments, [..., N, num_levels, num_features per level, L].
    """
    # Unflatten the sequence axis and put it last. [..., N, state_dim, L]
    x = x.view(*x.shape[:-1], -1, self.state_dim).transpose(-1, -2)

    # if self.num_lags > 0:  # Add lags. [..., N, L, state_dim_]
    #   x = torch.cat(
    #     (torch.zeros((*x.shape[:-2], self.num_lags, self.state_dim)).to(x), x), axis=-2)
    #   x = x.unfold(-2, self.num_lags+1, 1).reshape(*x.shape[:-2], -1, self.state_dim_)
    # # Pre-activations. [..., N, L, num_levels, num_features_per_level]
    # U = torch.einsum('...d,dmp->...mp', x, self.freq)

    # Pad before convolving with freq. [..., N, state_dim, L + num_lags]
    # U = torch.cat((torch.zeros_like(x[..., :self.num_lags]), x), axis=-1)
    U = torch.nn.functional.pad(x, (self.num_lags, 0))
    # Convolve with freq. [..., N, num_levels * num_features_per_level, L]
    U = torch.nn.functional.conv1d(
      U.view(-1, *U.shape[-2:]), self.freq, self.phase).view(*x.shape[:-2], -1, x.shape[-1])
    # Random Fourier Features. [..., N, num_levels, num_features_per_level, L].
    U = math.sqrt(2.) * U.cos().view(
      *x.shape[:-2], self.num_levels, self.num_features_per_level, x.shape[-1])
    if self.window > 1:  # Fractional differencing.
      # Pad with zeros. [..., N, num_levels, num_features_per_level, L + window-1]
      # U = torch.cat((torch.zeros_like(U[..., :self.window-1]), U), axis=-1)
      U = torch.nn.functional.pad(U, (self.window-1, 0))
      # Compute fractional differences. [..., N, num_levels, num_features_per_level, L]
      # U = (U.unfold(-3, self.window, 1) * self._fracdiff_coef).sum(axis=-1)
      U = torch.nn.functional.conv1d(
        U.view(-1, self.num_features_per_level, U.shape[-1]), self._fracdiff_filter).view(
          *x.shape[:-2], self.num_levels, self.num_features_per_level, x.shape[-1])
    return U

  def _feature_first_order(self, U: Tensor) -> Tensor:
    """Computes the first-order approximation to the feature map.

    Args:
      U: RFF increments, [..., N, num_levels, num_features_per_level, L].

    Returns:
      P: List of features per signature level [..., N, num_features_per_level, num_levels].
    """
    # Initialize level-1 features. [..., N, num_features_per_level]
    P = [U[..., 0, :, :].sum(axis=-1)]
    # Iterate to get higher levels. [..., N, num_features_per_level, L]
    R = U[..., 0, :, :]
    for m in range(1, self.num_levels):
      # Propagate features. [..., N, num_features_per_level, L]
      R = R.cumsum(axis=-1).roll(1, -1)
      R[..., 0] = 0.
      R *= U[..., m, :, :]
      # Collapse into level-m features. [..., N, num_features_per_level]
      P.append(R.sum(axis=-1))
    # [..., N, num_levels, num_features_per_level]
    return torch.stack(P, axis=-2)

  def _feature_higher_order(self, U: Tensor) -> Tensor:
    """Computes the full higher-order feature map.

    Args:
      U: RFF increments, [..., N, num_levels, num_features_per_level, L].

    Returns:
      P: List of features per signature level [..., N, num_features_per_level, num_levels].
    """
    # Initialize level-1 features. [..., N, num_features_per_level]
    P = [U[..., 0, :, :].sum(axis=-1)]
    # Iterate to get higher levels. [..., N, num_features_per_level, L]
    R = [U[..., 0, :, :]]
    Q = sum(R)
    for m in range(1, self.num_levels):
      # Propagate features. [..., N, num_features_per_level, L]
      Q = Q.cumsum(axis=-1).roll(1, -1)
      Q[..., 0] = 0.
      Q *= U[..., m, :, :]
      R_next = [Q]
      for d in range(1, m+1):
        R_next.append(1. / (d+1.) * R[d-1] * U[..., m, :, :])
      R = R_next
      Q = sum(R)
      # Collapse into level-m features. [..., N, num_features_per_level]
      P.append(Q.sum(axis=-1))
    # [..., N, num_levels, num_features_per_level]
    return torch.stack(P, axis=-2)

  def _geometric_decay_scan(self, U: Tensor, decay: Tensor) -> Tensor:
    """Computes parallel prefix sum with geometric decay.

    Args:
      U: Tensor of shape [..., N, num_features_per_level, L]
      decay: Decay factors of shape [num_features_per_level].

    Returns:
      Geometrically decayed tensor U of shape [..., N, num_features_per_level, L].
    """
    # Pad U to power of 2 length.
    curr_len = U.size(-1)
    if curr_len > 65536:
      raise ValueError('Max seqlen is 65536')
    bit_len = curr_len.bit_length()
    target_len = 2**bit_len
    # [..., N, num_features_per_level, target_len]
    Y = torch.cat(
      (U, torch.zeros_like(U[..., :target_len-curr_len])), axis=-1)
    # Put Y in required shape, [-1, num_features_per_level, target_len]
    Y = Y.view(-1, *Y.shape[-2:]).contiguous()
    # Broadcast gates. [-1, num_features_per_level, target_len]
    gates = decay[:, None].broadcast_to(Y.shape).contiguous()
    # Scan and format to original shape, [..., N, num_features_per_level, L].
    Y = scan(gates, Y)[..., :curr_len].view(*U.shape)
    return Y

  def _feature_seq_first_order(self, U: Tensor) -> Tensor:
    """Computes the decayed feature map by means of a linear scan.

    Args:
      U: RFF increments, [..., N, num_levels, num_features_per_level, L].

    Returns:
      P: Features per signature level of shape [..., N, L, num_levels, num_features_per_level].
    """
    feat = []
    for m in range(self.num_levels):
      if m == 0:
        # [..., N, num_features_per_level, L]
        P = self._geometric_decay_scan(U[..., 0, :, :], self.decay)
      else:
        # [..., N, num_features_per_level, L]
        P = P.roll(1, -1)
        P[..., 0] = 0.
        P = self.decay[:, None]**m * self._geometric_decay_scan(
          P * U[..., m, :, :], self.decay**(m+1))
      feat.append(P.transpose(-1, -2))  # [..., N, L, num_features_per_level]
    # [..., N, L, num_levels, num_features_per_level]
    return torch.stack(feat, axis=-2)

  def _feature_seq_higher_order(self, U: Tensor) -> Tensor:
    """Computes the decayed feature map by means of a linear scan.

    Args:
      U: RFF increments, [..., N, num_levels, num_features_per_level, L].

    Returns:
      P: Features per signature level of shape [..., N, L, num_levels, num_features_per_level].
    """
    feat = []
    # [..., N, num_features_per_level, L]
    R = [U[..., 0, :, :]]
    for m in range(self.num_levels):
      if m == 0:
        # [..., N, num_features_per_level, L]
        P = self._geometric_decay_scan(R[0], self.decay)
      else:
        P = P.roll(1, -1)
        P[..., 0] = 0.
        R_next = []
        R_next.append(self.decay[:, None]**m * P * U[..., m, :, :])
        for i in range(m):
          R_next.append(1./(i+2) * R[i] * U[..., m, :, :])
        P = self._geometric_decay_scan(sum(R_next), self.decay**(m+1))
        R = R_next
      feat.append(P.transpose(-1, -2))  # [..., N, L, num_features_per_level]
    # [..., N, L, num_levels, num_features_per_level]
    return torch.stack(feat, axis=-2)

  def _normalize(self, P: Tensor) -> Tensor:
    """Normalizes the features to unit norm per tensor level.

    Args:
      P: RFSF of shape [..., num_levels, num_features_per_level]

    Returns:
      Normalized RFSF of the same shape.
    """
    norm = P.norm(dim=-1, keepdim=True) + 1e-12
    P = P / norm
    return P

  def feature(self, x: Tensor, **params) -> Tensor:
    """Computes the feature map of the input.

    Args:
      x: Tensor of shape [..., N, L * state_dim].

    Returns:
      x_feat: Features of shape [..., N, num_features] or [..., N, L, num_features].
    """
    # Preprocess and compute RFF increments. [..., N, num_levels, num_features_per_level, L]
    x_rff = self._transform(x)
    # Compute features.
    if self.return_sequences:
      # [..., N, L, num_levels, num_features_per_level]
      x_feat = (self._feature_seq_first_order(x_rff) if self.first_order
                else self._feature_seq_higher_order(x_rff))
    else:
      # [..., N, num_levels, num_features_per_level]
      x_feat = (self._feature_first_order(x_rff) if self.first_order
                else self._feature_higher_order(x_rff))
    # Normalize by number of features.
    x_feat /= (self.num_levels * self.num_features_per_level)
    if self.normalize:  # Normalize to unit norm.
      # [..., N, (L,) num_levels, num_features_per_level]
      x_feat = self._normalize(x_feat)
    # Rescale by sigma. [..., N, (L,) num_levels, num_features_per_level]
    x_feat = x_feat * self.sigma[1:, None]
    # [..., N, (L,) num_levels * num_features_per_level]
    x_feat = x_feat.view(*x_feat.shape[:-2], -1)
    # Add constant term. # [..., N, (L,) num_features]
    x_feat = torch.cat((self.sigma[0] * torch.ones_like(x_feat[..., :1]), x_feat), axis=-1)
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
      freq_kl, phase_kl = self._sample_random_variables()
      if self.learn_variational:
        self.update_added_loss_term("freq_kl", freq_kl)
        self.update_added_loss_term("phase_kl", phase_kl)

    # [..., N1, (L1,) num_features]
    x1_feat = self.feature(x1)
    if feature:
      return x1_feat
    if diag:
      # [..., N1, (L1)]
      return self.x1_feat.square().sum(axis=-1)
    is_symmetric = x1.size() == x2.size() and torch.equal(x1, x2)
    if is_symmetric:
      x2_feat = x1_feat
    else:
      # [..., N2, (L2,) num_features]
      x2_feat = self._feature(x2)
    if self.return_sequences:
      # [..., N1, L1, N2,, L2]
      K = torch.einsum('...mkp,...nlp->...mknl', x1_feat, x2_feat)
    else:
      # [..., N1, N2]
      K = torch.einsum('...kp,...np->...kn', x1_feat, x2_feat)
    return K

# ------------------------------------------------------------------------------
# GRAVEYARD

  # def _geometric_prog_pad_dft(self, L: int, decay: Tensor):
  #   """Computes the DFT of padded geometric progression.

  #   Args:
  #     L: The length of the sequences to be convolved with.
  #     decay: Decay factors of shape [num_features_per_level].

  #   Returns:
  #     DFT of padded geometric decay, shape [2 * L - 1, num_features_per_level].
  #   """
  #   L_ = 2 * L - 1
  #   steps = torch.arange(L_).to(decay)
  #   return (
  #     (decay[None]**L * (-2.*math.pi*1j*steps[:, None] * L / L_).exp() - 1)
  #     / (decay[None] * (-2.*math.pi*1j*steps[:, None] / L_).exp() - 1)
  #   )

  # def _geometric_decay_scan(self, U: Tensor, decay: Tensor):
  #   """Computes non-circular convolution with geometric decay sequence.

  #   Args:
  #     U: Tensor of shape [..., N, L, num_features_per_level]
  #     decay: Decay factors of shape [num_features_per_level].

  #   Returns:
  #     Geometrically decayed tensor U.
  #   """
  #   L = U.size(-2)
  #   U_pad = torch.cat((U, torch.zeros_like(U[..., :-1, :])), axis=-2)
  #   U_pad_fft = torch.fft.fft(U_pad, axis=-2)
  #   geom_decay_pad_dft = self._geometric_prog_pad_dft(L, decay)
  #   conv = torch.fft.ifft(U_pad_fft * geom_decay_pad_dft, axis=-2)
  #   return conv[..., :L, :].real

  # def _geometric_decay_scan(self, U: Tensor, decay: Tensor):
  #   """Computes parallel prefix sum with geometric decay.

  #   Args:
  #     U: Tensor of shape [..., N, L, num_features_per_level]
  #     decay: Decay factors of shape [num_features_per_level].

  #   Returns:
  #     Geometrically decayed tensor U.
  #   """
  #   # Pad U to power of 2 length.
  #   curr_len = U.size(-2)
  #   bit_len = curr_len.bit_length()
  #   target_len = 2**bit_len
  #   Y = torch.cat(
  #     (U, torch.zeros_like(U[..., :target_len-curr_len, :])), axis=-2)
  #   # Perform up-sweep phase.
  #   for p in range(bit_len):
  #     left = [..., slice(2**p-1, target_len, 2**(p+1)), slice(None)]
  #     right = [..., slice(2**(p+1)-1, target_len, 2**(p+1)), slice(None)]
  #     Y[right] = decay**(2**p) * Y[left].clone() + Y[right].clone()
  #   # Perform down-sweep phase.
  #   Y[..., -1, :] = 0.
  #   for p in reversed(range(bit_len)):
  #       left = [..., slice(2**p-1, target_len, 2**(p+1)), slice(None)]
  #       right = [..., slice(2**(p+1)-1, target_len, 2**(p+1)), slice(None)]
  #       t = Y[left].clone()
  #       Y[left] = Y[right]
  #       Y[right] = t + decay**(2**p) * Y[right].clone()
  #   Y = Y.roll(-1, -2)
  #   Y[..., -1, :] = decay * Y[..., -2, :].clone() + U[..., -1, :]
  #   return Y[..., :curr_len, :]