import math
import numpy as np
import torch


_DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_metrics(targets, preds_mean, preds_stddev, quantiles = _DEFAULT_QUANTILES):
    metrics = {}
    # RMSE.
    metrics['rmse'] = float((targets - preds_mean).pow(2).mean().sqrt())
    # CRPS.
    dist = torch.distributions.Normal(0, 1)
    z = (targets - preds_mean) / preds_stddev.maximum(torch.tensor(1e-6))
    metrics['crps'] = float((preds_stddev * (z * (2 * dist.cdf(z) - 1) + 2 * dist.log_prob(z).exp() -
                       1 / math.sqrt(math.pi))).mean())
    # Coverage.
    deviation = dist.icdf(torch.tensor(0.975))
    lower = preds_mean - deviation * preds_stddev
    upper = preds_mean + deviation * preds_stddev
    fraction = ((targets > lower) * (targets < upper)).to(float).mean()
    metrics['coverage'] = float(torch.abs(fraction - 0.95).item())
    # Mean Normalized QuantileLoss.
    quantile_losses = {}
    abs_target_sum = targets.abs().sum()
    for q in quantiles:
        preds_quant = preds_mean + preds_stddev * dist.icdf(torch.tensor(q))
        quantile_losses[q] = 2 * (
            (preds_quant - targets) * ((targets <= preds_quant).to(targets.dtype) - q)
        ).abs().sum()
    metrics['mean_abs_quantileLoss'] = float(np.mean([x.item() for x in quantile_losses.values()]))
    metrics['mean_wQuantileLoss'] = float(np.mean([
        (x / abs_target_sum).item() for x in quantile_losses.values()]))
    return metrics