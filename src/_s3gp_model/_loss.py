
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood


class PredictiveELBO(_ApproximateMarginalLogLikelihood):
    def __init__(self, likelihood, model, num_data, alpha=0.1, beta=1.0, combine_terms=True):
        self.alpha = alpha
        super().__init__(likelihood, model, num_data, beta, combine_terms)

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        return (
            self.alpha *  self.likelihood.expected_log_prob(
                target, approximate_dist_f, **kwargs).sum(-1)
            + (1. - self.alpha) * self.likelihood.log_marginal(
                target, approximate_dist_f, **kwargs).sum(-1)
        )

    def forward(self, approximate_dist_f, target, **kwargs):
        return super().forward(approximate_dist_f, target, **kwargs)
