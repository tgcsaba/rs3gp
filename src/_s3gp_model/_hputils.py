"""Utilities for variational treatment of kernel/feature hyperparameters."""
import torch
import sys

sys.path.append('..')

from gpytorch.mlls import AddedLossTerm
from gpytorch.module import Module
from torch import Size, Tensor
from torch.distributions import (kl_divergence, Beta, Distribution, Gamma)
from typing import Optional

# ------------------------------------------------------------------------------

class AddedKLLossTerm(AddedLossTerm):
  def __init__(self, q: Distribution, p: Distribution,
               num_data: Optional[int] = None):
    """Added KL loss term for variational treatment of hyperparameters.

    Args:
      q: Variational hyperparameter distribution.
      p: Prior hyperparameter distribution.
      num_data: If given, normalizes the KL divergence by dividing it.
    """
    self.q = q
    self.p = p
    self.num_data = num_data

  def loss(self) -> Tensor:
    kl_loss = kl_divergence(self.q, self.p).sum()
    if self.num_data is not None:
      kl_loss /= self.num_data
    return kl_loss

# ------------------------------------------------------------------------------

class NormalSampler(Module):
  def __init__(self, shape: Size):
    super().__init__()
    self.shape = shape
    self.register_buffer(name="buffer", tensor=torch.zeros(self.shape))
    self.fit()

  def fit(self):
    self.buffer = torch.randn(self.shape).to(self.buffer)

  def sample(self, mean: Tensor, stdev: Tensor):
    return mean + stdev * self.buffer

# ------------------------------------------------------------------------------

class UniformSampler(Module):
  def __init__(self, shape: Size):
    super().__init__()
    self.shape = shape
    self.register_buffer(name="buffer", tensor=torch.zeros(self.shape))
    self.fit()

  def fit(self):
    self.buffer = torch.rand(self.shape).to(self.buffer)

  def sample(self):
    return self.buffer

# ------------------------------------------------------------------------------

class StandardGammaSampler(Module):
  def __init__(self, shape: Size, B: int = 10):
    super().__init__()
    self.shape = shape
    self.B = B
    self.register_buffer(name="buffer", tensor=torch.zeros(self.shape))
    self.register_buffer(name="buffer2",
                         tensor=torch.zeros(self.shape + (self.B,)))
    self.fit()

  def h_func(self, alpha: Tensor, eps: Tensor) -> Tensor:
    z = (alpha - 1./3.) * (1 + eps / torch.sqrt(9. * alpha - 3.))**3
    return z

  def inv_h_func(self, alpha: Tensor, z: Tensor) -> Tensor:
    eps = torch.sqrt(9. * alpha - 3.) * ((z / (alpha - 1./3.))**(1. / 3.) - 1.)
    return eps

  def fit(self):
    B = torch.tensor(float(self.B)).broadcast_to(self.shape)
    z = Gamma(B, 1.).rsample()
    self.buffer = self.inv_h_func(B, z).to(self.buffer)
    self.buffer2 = torch.rand(self.shape + (self.B,)).to(self.buffer2)

  def sample(self, alpha: Tensor):
    alpha = alpha.broadcast_to(self.shape)
    # Distributed as Gamma(alpha + B, 1).
    z = self.h_func(alpha + self.B, self.buffer)
    B_range = torch.arange(self.B).to(self.buffer)
    # Distributed as Gamma(alpha, 1).
    z *= (self.buffer2**(1./(alpha[..., None] + B_range))).prod(axis=-1)
    return z

# ------------------------------------------------------------------------------

class BetaSampler(Module):
  def __init__(self, shape: Size, B: int = 10):
    super().__init__()
    self.shape = shape
    self.B = B
    self.gamma_sampler1 = StandardGammaSampler(self.shape, B=self.B)
    self.gamma_sampler2 = StandardGammaSampler(self.shape, B=self.B)

  def fit(self):
    self.gamma_sampler1.fit()
    self.gamma_sampler2.fit()

  def sample(self, alpha: Tensor, beta: Tensor):
    alpha = alpha.broadcast_to(self.shape)
    beta = beta.broadcast_to(self.shape)
    # Distributed as Gamma(alpha, 1).
    gamma1 = self.gamma_sampler1.sample(alpha)
    # Distributed as Gamma(beta, 1).
    gamma2 = self.gamma_sampler2.sample(beta)
    # Distributed as Beta(alpha, beta).
    return gamma1 / (gamma1 + gamma2 + 1e-12)

# ------------------------------------------------------------------------------