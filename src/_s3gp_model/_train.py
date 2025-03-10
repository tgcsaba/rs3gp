import gpytorch
import torch

from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.settings import (
    cholesky_max_tries, max_cholesky_size, fast_pred_var)
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from typing import List, Tuple, Union

TensorPair = Tuple[Tensor, Tensor]
ListOrTensor = Union[List, Tensor]

# ------------------------------------------------------------------------------

def train_model(model: ApproximateGP, likelihood: Likelihood,
                mll: MarginalLogLikelihood,
                dataset: Union[Tuple[List, List], DataLoader],
                num_epochs: int = 100, max_lr: float = 0.01,
                max_tries: int = 3
               ) -> Tuple[ApproximateGP, Likelihood]:
  """Train the model using variational ELBO and by SGD with Adam.

  Args:
    model: Approximate GP model as a GPyTorch module.
    likelihood: Likelihood as a GPyTorch module.
    mll: Marginal log-likelihood used as loss to train the model.
    dataset: Training data either as Dataset or (X, Y) tuple of lists.
    num_epochs: Number of epochs; larger = slower, but closer to optimum.
    max_lr: Max learning rate for OneCycleLR scheduler.
    max_tries: Number of tries of cholesky computations with adding jitter to
      the diagonal elements larger is more robust but less accurate.
      GPyTorch default is 3. Increase this only if continuous failure.

  Returns:
    (model, likelihood): The trained model and likelihood objects.
  """
  # Training setup.
  model.train()
  likelihood.train()
  epochs_iter = tqdm(range(num_epochs), desc="Epoch", leave=False)
  if isinstance(dataset, tuple):  # Create variable time series dataset.
    dataset = TimeSeriesDataset(*dataset)
    loader = DataLoader(dataset, shuffle=True)
  elif isinstance(dataset, DataLoader):
    loader = dataset
  else:
    raise ValueError('Either provide input-output data or a DataLoader.')

  opt = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
  ], lr=max_lr)

  sched = OneCycleLR(
    opt, max_lr, epochs=num_epochs, steps_per_epoch=len(loader))

  with cholesky_max_tries(max_tries), max_cholesky_size(float('inf')):
    for i in epochs_iter:
      minibatch_iter = tqdm(loader, desc="Minibatch", leave=False)
      for x_batch, y_batch in minibatch_iter:
        # Perform Adam step to update all parameters.
        opt.zero_grad()
        output = model(x_batch, diag=True)
        loss = -mll(output, y_batch).mean()
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        opt.step()
        sched.step()
      minibatch_iter.close()


  return model, likelihood

# ------------------------------------------------------------------------------

def predict(model: ApproximateGP, likelihood: Likelihood,
            dataset: Union[Tuple[List], DataLoader],
            max_tries: int = 3) -> Tuple[Tensor, Tensor]:
    """Compute predictions for the trained model.

    Args:
        model: Approximate GP model as a GPyTorch module.
        likelihood: Likelihood as a GPyTorch module
        max_tries: Number of tries of cholesky computations with adding jitter
          to the diagonal elements. larger is more robust but less accurate.
          GPyTorch default is 3. Increase this only if continuous failure.

    Returns:
      (means, variances): Posterior predictive distribution mean and variances.
    """
    # Evaluation setup.
    model.eval()
    likelihood.eval()
    if isinstance(dataset, tuple):  # Create variable time series dataset.
      dataset = TimeSeriesDataset(*dataset)
      loader = DataLoader(dataset, shuffle=True)
    elif isinstance(dataset, DataLoader):
      loader = dataset
    else:
      raise ValueError('Either provide input-output data or a DataLoader.')
    means = []
    variances = []
    minibatch_iter = tqdm(loader, desc="Minibatch", leave=False)
    with torch.no_grad(), fast_pred_var(), cholesky_max_tries(max_tries):
      for x_batch in minibatch_iter:
        if isinstance(x_batch, list):
          x_batch = x_batch[0]
        preds = likelihood(model(x_batch, diag=True))
        means.append(preds.mean)
        variances.append(preds.variance)
    minibatch_iter.close()
    means = torch.cat(means, axis=0)
    variances = torch.cat(variances, axis=0)
    return means, variances

# ------------------------------------------------------------------------------