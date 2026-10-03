import torch

from reap.router_stability import RouterStabilityCollector, collect_router_observation


class _ToyGLMRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 6))
        self.n_routed_experts = 8
        self.n_group = 2
        self.topk_group = 2
        self.top_k = 2
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.register_buffer("e_score_correction_bias", torch.zeros(8))


def test_router_observation_has_original_and_four_perturbations():
    router = _ToyGLMRouter()
    hidden = torch.randn(7, 6)
    observation = collect_router_observation(
        router,
        hidden,
        [0.25, 0.5, 1.0, 2.0],
        torch.Generator().manual_seed(7),
        top_k=2,
        boundary=4,
    )

    assert observation.top12_ids.shape == (5, 7, 6)
    assert observation.top12_ranking_values.shape == (5, 7, 6)
    assert observation.top8_applied_weights.shape == (5, 7, 2)
    assert observation.perturbed_scores.shape == (4, 7, 8)


def test_router_stability_accumulators_update_without_persisting_scores(tmp_path):
    router = _ToyGLMRouter()
    hidden = torch.randn(5, 6)
    observation = collect_router_observation(
        router,
        hidden,
        [0.25, 0.5, 1.0, 2.0],
        torch.Generator().manual_seed(3),
        top_k=2,
        boundary=4,
    )
    collector = RouterStabilityCollector(
        tmp_path,
        [0.25, 0.5, 1.0, 2.0],
        seed=3,
        max_tokens=3,
    )
    collector.observe(2, 4, observation)

    assert collector.total_tokens == 3
    assert collector.accumulators["abs_change_sum"].shape == (4, 64, 8)
    assert torch.all(collector.accumulators["abs_change_sum"][:, 2] >= 0)
