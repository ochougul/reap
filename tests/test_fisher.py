import pytest
import torch

from reap.fisher import FisherCollector, assistant_label_mask


class ToyGate(torch.nn.Module):
    def __init__(self, hidden, experts, top_k):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(experts, hidden, dtype=torch.float64))
        self.top_k = top_k

    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])
        logits = x @ self.weight.T
        scores = logits.sigmoid()
        idx = scores.topk(self.top_k, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return idx, w, logits


class ToyMoE(torch.nn.Module):
    def __init__(self, hidden=6, experts=5, top_k=2):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden, bias=False).double() for _ in range(experts)]
        )
        self.gate = ToyGate(hidden, experts, top_k)
        self.shared = torch.nn.Linear(hidden, hidden, bias=False).double()
        with torch.no_grad():  # small expert outputs keep the first-order approximation valid
            for expert in self.experts:
                expert.weight.mul_(0.02)
        self.dropped = None  # expert whose contribution is skipped (no re-routing)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        idx, w, logits = self.gate(flat)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            if e == self.dropped:
                continue
            tok, slot = torch.where(idx == e)
            if tok.numel():
                out.index_add_(0, tok, expert(flat[tok]) * w[tok, slot].unsqueeze(-1))
        return (out + self.shared(flat)).reshape(x.shape), logits


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].mlp = ToyMoE()
        self.embed = torch.nn.Embedding(11, 6).double()
        self.head = torch.nn.Linear(6, 11, bias=False).double()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, ids, labels):
        h = self.embed(ids).requires_grad_(True)
        h, _ = self.model.layers[0].mlp(h)
        return torch.nn.functional.cross_entropy(self.head(h).reshape(-1, 11), labels.reshape(-1), ignore_index=-100)


def _run(mode, mask=None):
    torch.manual_seed(0)
    model = ToyModel()
    ids = torch.randint(0, 11, (2, 8))
    labels = ids.clone()
    collector = FisherCollector(mode=mode)
    collector.attach(model)
    collector.set_token_mask(mask if mask is not None else torch.ones_like(ids))
    model(ids, labels).backward()
    return model, ids, labels, collector.state()


def test_router_grad_matches_recompute():
    _, _, _, fast = _run("router_grad")
    _, _, _, ref = _run("recompute")
    for key in ("fisher_score", "fisher_signed"):
        assert torch.allclose(fast[0][key], ref[0][key], rtol=1e-4, atol=1e-12), key
    assert torch.equal(fast[0]["expert_frequency"], ref[0]["expert_frequency"])
    assert fast[0]["total_tokens"] == ref[0]["total_tokens"] == 16


def test_signed_score_predicts_loss_change_when_expert_dropped():
    model, ids, labels, state = _run("router_grad")
    base = float(model(ids, labels))
    moe = model.model.layers[0].mlp
    for expert in range(5):
        moe.dropped = expert
        delta = float(model(ids, labels)) - base
        moe.dropped = None
        predicted = -float(state[0]["fisher_signed"][expert]) * state[0]["total_tokens"]
        assert delta == pytest.approx(predicted, rel=0.1, abs=1e-6)


def test_padding_tokens_are_excluded():
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, 4:] = 0
    _, _, _, state = _run("router_grad", mask)
    assert state[0]["total_tokens"] == 12
    assert int(state[0]["expert_frequency"].sum()) == 12 * 2


def test_assistant_label_mask_uses_last_marker():
    ids = torch.tensor([[1, 9, 2, 9, 3, 4]])
    mask = assistant_label_mask(ids, torch.ones_like(ids), 9)
    assert mask.tolist() == [[False, False, False, False, True, True]]
    assert assistant_label_mask(ids, None, None).all()


def test_sorted_moe_matches_stock_forward_and_gradients():
    import importlib.util
    import sys
    from pathlib import Path

    from transformers import Glm4MoeConfig

    from reap.fast_moe import patch_sorted_moe

    path = Path(__file__).parents[1] / "src/reap/models/glm/modeling_glm4_moe.py"
    spec = importlib.util.spec_from_file_location("local_glm4_moe_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = Glm4MoeConfig(
        hidden_size=16, intermediate_size=32, moe_intermediate_size=8, n_routed_experts=8,
        num_experts_per_tok=3, n_shared_experts=1, n_group=1, topk_group=1, norm_topk_prob=True,
    )
    torch.manual_seed(0)
    moe = module.Glm4MoeMoE(config)
    moe.gate.weight.data.normal_()
    x1 = torch.randn(2, 9, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out1, _ = moe(x1)
    out1.square().sum().backward()
    assert patch_sorted_moe(moe) == 1
    out2, _ = moe(x2)
    out2.square().sum().backward()
    assert torch.allclose(out1, out2, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)


def test_joint_collector_matches_reference_reap_stability_and_fisher(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    from transformers import Glm4MoeConfig

    from reap.joint_calibration import JointCollector
    from reap.pruning_metrics import initialize_pruning_state, update_pruning_state
    from reap.router_stability import _route_once
    from reap.seap import SeapCollector, collect_seap_observation

    path = Path(__file__).parents[1] / "src/reap/models/glm/modeling_glm4_moe.py"
    spec = importlib.util.spec_from_file_location("local_glm4_moe_for_joint_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = Glm4MoeConfig(
        hidden_size=16, intermediate_size=32, moe_intermediate_size=8, n_routed_experts=8,
        num_experts_per_tok=3, n_shared_experts=1, n_group=1, topk_group=1, norm_topk_prob=True,
    )
    torch.manual_seed(0)
    moe = module.Glm4MoeMoE(config)
    moe.gate.weight.data.normal_()
    for p in moe.parameters():
        p.requires_grad_(False)

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = moe
    x = torch.randn(2, 9, 16, requires_grad=True)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 6:] = False

    joint = JointCollector(tmp_path / "joint", seed=7)
    joint.attach(model)
    joint.set_token_mask(mask)
    out, _ = moe(x)
    out.square().sum().backward()
    joint.detach()
    got = joint.state()[0]

    # Reference: dense stack + legacy REAP update, legacy stability observation.
    hidden = x.detach().reshape(-1, 16)[mask.reshape(-1)]
    route = _route_once(moe.gate, hidden, moe.gate.top_k, boundary=0)
    ref = initialize_pruning_state(8, device="cpu")
    update_pruning_state(
        ref, activations=torch.stack([e(hidden) for e in moe.experts]),
        selected_experts=route[4], router_logits=route[0], num_experts=8,
        renormalize_router_weights=True,
    )
    seap = SeapCollector(tmp_path / "ref", seed=7)
    obs = collect_seap_observation(moe.gate, hidden, seap.generator(hidden.device, 0),
                                   variance_multiplier=2.0, top_k=moe.gate.top_k)
    seap.observe(0, obs, 8)
    ref_state = seap.add_scores_to_state({0: ref})[0]

    assert torch.allclose(got["reap_score"], ref_state["reap_score"], rtol=1e-4, atol=1e-6)
    assert torch.equal(got["expert_frequency"], ref["expert_frequency"])
    assert got["total_tokens"] == int(ref["total_tokens"])
    assert torch.allclose(got["stability_score"], ref_state["stability_score"])
    assert torch.allclose(got["seap_score"], ref_state["seap_score"], rtol=1e-4, atol=1e-6)

    only_fisher = FisherCollector()
    only_fisher.attach(model)
    only_fisher.set_token_mask(mask)
    out, _ = moe(x.detach().clone().requires_grad_(True))
    out.square().sum().backward()
    assert torch.allclose(got["fisher_score"], only_fisher.state()[0]["fisher_score"], rtol=1e-5)
