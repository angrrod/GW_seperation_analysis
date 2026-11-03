"""CPU regression checks for reusing each Dingo marginal's latent output.

Run: python -m unittest discover -s vector_copula_vi_v2/tests -p test_dingo_single_pass.py -v
"""

import copy
import unittest

import torch
from dingo.core.nn.nsf import create_nsf_wrapped
from vector_copula_vi.VectorCopulaFlow_V2 import (
    DingoVectorCopulaFlow,
    GaussianVectorCopulaFlow,
    ZukoVectorCopulaFlow,
)


class LegacyDingoVectorCopulaFlow(DingoVectorCopulaFlow):
    # Exercise the original two calls while sharing all other copula code.
    _marginal_log_prob_and_base = GaussianVectorCopulaFlow._marginal_log_prob_and_base


class TestDingoSinglePass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_flows(self, training, context_dim=3):
        torch.manual_seed(123)
        flows = torch.nn.ModuleList([
            create_nsf_wrapped(
                input_dim=features, context_dim=context_dim, num_flow_steps=2,
                base_transform_kwargs=dict(
                    hidden_dim=12, num_transform_blocks=1, activation='elu',
                    dropout_probability=0.0, norm='BatchNorm', num_bins=4,
                    base_transform_type='rq-coupling',
                ),
            ) for features in (3, 2)
        ]).double()
        flows.train(training)
        return flows

    def test_density_and_all_gradients_match_two_pass_reference(self):
        # Cover training batches, several samples for one context, and no context.
        cases = ((4, False, False), (1, True, False), (4, False, True))
        for training in (False, True):
            for models, sample_axis, unconditional in cases:
                with self.subTest(training=training, models=models,
                                  sample_axis=sample_axis, unconditional=unconditional):
                    old_flows = self.make_flows(training, None if unconditional else 3)
                    new_flows = copy.deepcopy(old_flows)
                    value = 0.2 * torch.randn(4, 5, dtype=torch.float64)
                    if sample_axis:
                        value = value.unsqueeze(1)
                    context = None if unconditional else torch.randn(models, 3, dtype=torch.float64)
                    factor = 0.1 * torch.randn(models, 5, 2, dtype=torch.float64)
                    z = torch.randn(models, dtype=torch.float64)
                    results = []
                    for cls, flows in ((LegacyDingoVectorCopulaFlow, old_flows),
                                       (DingoVectorCopulaFlow, new_flows)):
                        v = value.clone().requires_grad_()
                        c = None if context is None else context.clone().requires_grad_()
                        b = factor.clone().requires_grad_()
                        zz = z.clone().requires_grad_()
                        dist = cls(flows, b, zz, c)
                        components = dist.log_prob_components(v)
                        inputs = [v, b, zz] + ([] if c is None else [c]) + list(flows.parameters())
                        grads = torch.autograd.grad(
                            -components['log_prob_total'].mean(), inputs,
                            allow_unused=True,
                        )
                        results.append((components, grads))
                    old, new = results
                    for key in old[0]:
                        a, b = old[0][key], new[0][key]
                        if not isinstance(a, list):
                            a, b = [a], [b]
                        for before, after in zip(a, b):
                            torch.testing.assert_close(before, after, rtol=1e-9, atol=1e-10)
                    for before, after in zip(old[1], new[1]):
                        self.assertEqual(before is None, after is None)
                        if before is not None:
                            torch.testing.assert_close(before, after, rtol=1e-8, atol=1e-9)
                    self.assertEqual(old_flows.state_dict().keys(), new_flows.state_dict().keys())

    def test_each_distinct_marginal_runs_once_per_call(self):
        flows = self.make_flows(False)
        self.assertTrue(set(map(id, flows[0].parameters())).isdisjoint(
            set(map(id, flows[1].parameters()))))
        calls = [0, 0]
        def counter(index):
            def hook(module, inputs, result):
                calls[index] += 1
            return hook
        handles = [flow.flow._transform.register_forward_hook(counter(i))
                   for i, flow in enumerate(flows)]
        try:
            dist = DingoVectorCopulaFlow(
                flows, 0.1*torch.randn(4, 5, 2, dtype=torch.float64),
                torch.zeros(4, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64))
            dist.log_prob(torch.randn(4, 5, dtype=torch.float64))
            self.assertEqual(calls, [1, 1])
            # A fresh evaluation must execute both transforms again.
            dist.log_prob(torch.randn(4, 5, dtype=torch.float64))
            self.assertEqual(calls, [2, 2])
        finally:
            for handle in handles:
                handle.remove()

    def test_batchnorm_updates_once_instead_of_twice(self):
        flows = self.make_flows(True)
        old_flows = copy.deepcopy(flows)
        context = torch.randn(4, 3, dtype=torch.float64)
        factor = 0.1*torch.randn(4, 5, 2, dtype=torch.float64)
        z = torch.zeros(4, dtype=torch.float64)
        value = torch.randn(4, 5, dtype=torch.float64)
        DingoVectorCopulaFlow(flows, factor, z, context).log_prob(value)
        LegacyDingoVectorCopulaFlow(old_flows, factor, z, context).log_prob(value)
        for modules, expected in ((flows, 1), (old_flows, 2)):
            norms = [m for m in modules.modules() if isinstance(m, torch.nn.BatchNorm1d)]
            self.assertTrue(norms)
            self.assertTrue(all(m.num_batches_tracked.item() == expected for m in norms))

    def test_extra_context_embedding_retains_fallback_behavior(self):
        flows = self.make_flows(False)
        flows[0].flow._embedding_net = torch.nn.Linear(3, 3).double()
        dist = DingoVectorCopulaFlow(
            flows, torch.zeros(4, 5, 2, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64))
        value = torch.randn(4, 3, dtype=torch.float64)
        expected = GaussianVectorCopulaFlow._marginal_log_prob_and_base(dist, 0, value, True)
        actual = dist._marginal_log_prob_and_base(0, value, True)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after)

    def test_per_example_shapes_and_gaussian_reference(self):
        for batch in (1, 2, 5):
            with self.subTest(batch=batch):
                flows = self.make_flows(False)
                value = (0.2*torch.randn(batch, 5, dtype=torch.float64)).requires_grad_()
                context = torch.randn(batch, 3, dtype=torch.float64, requires_grad=True)
                factor = (0.2*torch.randn(batch, 5, 2, dtype=torch.float64)).requires_grad_()
                z = torch.zeros(batch, dtype=torch.float64, requires_grad=True)
                dist = DingoVectorCopulaFlow(flows, factor, z, context)
                parts = dist.log_prob_components(value)
                for key in ('log_prob_total', 'logp_marg_total', 'log_copula_total',
                            'log_det_term', 'log_copula_quad'):
                    self.assertEqual(tuple(parts[key].shape), (batch,), key)
                self.assertTrue(all(tuple(p.shape) == (batch,)
                                    for p in parts['logp_marginals']))

                # Independent density reference: joint Gaussian in latent space
                # plus the two flow Jacobians, rather than a broadcasted sum.
                transformed = [flow.flow._transform(v, context=context)
                               for flow, v in zip(flows, value.split((3, 2), dim=-1))]
                q = torch.cat([pair[0] for pair in transformed], dim=-1)
                jacobian = torch.stack([pair[1] for pair in transformed]).sum(0)
                normal = torch.distributions.MultivariateNormal(
                    torch.zeros(5, dtype=torch.float64), covariance_matrix=dist.Omega())
                expected = normal.log_prob(q) + jacobian
                torch.testing.assert_close(parts['log_prob_total'], expected,
                                           rtol=1e-9, atol=1e-10)

                # The old cross-example matrix has the same uniform mean, even
                # though its off-diagonal entries are not per-example densities.
                old_matrix = parts['logp_marg_total'][:, None] + parts['log_copula_total'][None, :]
                inputs = [value, context, factor, z] + list(flows.parameters())
                actual_grad = torch.autograd.grad(-parts['log_prob_total'].mean(), inputs,
                                                 allow_unused=True, retain_graph=True)
                old_grad = torch.autograd.grad(-old_matrix.mean(), inputs, allow_unused=True)
                for before, after in zip(old_grad, actual_grad):
                    self.assertEqual(before is None, after is None)
                    if before is not None:
                        torch.testing.assert_close(before, after, rtol=1e-8, atol=1e-9)

    def test_sample_axis_matches_separate_per_sample_evaluations(self):
        flows = self.make_flows(False)
        dist = DingoVectorCopulaFlow(
            flows, 0.2*torch.randn(1, 5, 2, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), torch.randn(1, 3, dtype=torch.float64))
        samples = 0.2*torch.randn(4, 1, 5, dtype=torch.float64)
        result = dist.log_prob(samples)
        self.assertEqual(tuple(result.shape), (4, 1))
        reference = torch.stack([dist.log_prob(row) for row in samples])
        torch.testing.assert_close(result, reference)

    def test_mismatched_density_shapes_raise_instead_of_broadcasting(self):
        class WrongCopulaShape(DingoVectorCopulaFlow):
            def _logProbCopula(self, q):
                density, determinant, quadratic = super()._logProbCopula(q)
                return density.unsqueeze(-1), determinant, quadratic
        dist = WrongCopulaShape(
            self.make_flows(False), torch.zeros(2, 5, 2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'density shapes must match'):
            dist.log_prob(torch.randn(2, 5, dtype=torch.float64))

    def test_zuko_shapes_with_multiple_samples_and_contexts(self):
        import zuko
        flows = torch.nn.ModuleList([
            zuko.flows.NSF(features=d, context=3, transforms=1, hidden_features=(8,))
            for d in (3, 2)
        ]).double()
        context = torch.randn(2, 3, dtype=torch.float64)
        dist = ZukoVectorCopulaFlow(
            flows, 0.2*torch.randn(2, 5, 2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            distribs=[flow(context) for flow in flows])
        samples = 0.2*torch.randn(3, 2, 5, dtype=torch.float64)
        result = dist.log_prob(samples)
        self.assertEqual(tuple(result.shape), (3, 2))
        reference = torch.stack([dist.log_prob(row) for row in samples])
        torch.testing.assert_close(result, reference)


if __name__ == '__main__':
    unittest.main()
