import torch

from reap.seap import SeapCollector, collect_seap_observation


class ToyRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(4))
        self.n_routed_experts = 4
        self.n_group = 1
        self.topk_group = 1
        self.top_k = 2
        self.register_buffer("e_score_correction_bias", torch.zeros(4))
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0


def test_seap_observation_has_full_selected_ranks():
    router = ToyRouter()
    observation = collect_seap_observation(
        router, torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        torch.Generator().manual_seed(1), top_k=2,
    )
    assert observation.original_selected.shape == (1, 2)
    assert observation.original_ranks.shape == (1, 2)
    assert observation.perturbed_ranks.shape == (1, 2)
    assert torch.all(observation.original_ranks >= 1)


def test_seap_scores_combine_stability_and_reap():
    collector = SeapCollector("/tmp/reap-seap-test", deltas=(2.0, 4.0), seed=1)
    collector._ensure(0, 2)
    collector.stability_sum[0][0] = torch.tensor([2.0, 1.0])
    collector.stability_sum[0][1] = torch.tensor([2.0, 2.0])
    collector.original_selected_count[0] = torch.tensor([2, 2])
    state = {0: {"reap": torch.tensor([2.0, 1.0])}}
    result = collector.add_scores_to_state(state, delta=2.0, lambda_=0.5)
    assert torch.allclose(result[0]["stability_score"], torch.tensor([1.0, 0.5]))
    assert torch.allclose(result[0]["seap_score"], torch.tensor([1.0, 0.5]))
