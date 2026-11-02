# Tests for lora.py: adapters on a frozen model, checkpoint state, merge into plain weights.
# Run with `pytest test_functions` or `python test_functions/test_lora.py`.

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

import lora
from network_cond import ConditionedAudioDiT

B, N, TOKEN_DIM = 2, 64, 72


def build(kind="S", seed=0, adaln="shared"):
    torch.manual_seed(seed)
    model = ConditionedAudioDiT(kind=kind, token_dim=TOKEN_DIM, frame_cond_dims={"energy": 1},
                                frame_cond_out_dims={"energy": 16}, frame_reinject_every=1,
                                adaln=adaln)
    for p in model.parameters():
        if p.dim() >= 2 and p.abs().sum() == 0:
            nn.init.normal_(p, std=0.02)
    return model.eval()


def run(model):
    torch.manual_seed(1)
    x, t = torch.randn(B, N, TOKEN_DIM), torch.rand(B)
    return model(x, t, frame_conditions={"energy": torch.randn(B, N, 1)})


def adapted(tmp, adapter, targets="blocks", rank=4, adaln="shared"):
    base = build(adaln=adaln)
    path = str(Path(tmp) / f"base_{adapter}_{targets}.pt")
    torch.save({"model_state_dict": base.state_dict(), "t_features": "high_freq"}, path)
    model = build(seed=1, adaln=adaln)
    cfg = OmegaConf.create({"paths": {"init_from": path},
                            "model": {"t_features": "high_freq"},
                            "lora": {"adapter": adapter, "rank": rank, "targets": targets}})
    lora.attach(model, cfg)
    return base, model, path, cfg


def train_steps(model, steps=3):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(run(model), torch.ones(B, N, TOKEN_DIM)).backward()
        opt.step()


def test_start_is_the_base():
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        base, model, _, _ = adapted(tmp, "lora")
        assert torch.equal(run(model), run(base))
        base, model, _, _ = adapted(tmp, "dora-rows")
        assert (run(model) - run(base)).abs().max() < 1e-2


def test_only_the_adapters_train():
    with tempfile.TemporaryDirectory() as tmp:
        for adapter in lora.ADAPTERS:
            base, model, _, _ = adapted(tmp, adapter)
            before = {k: v.clone() for k, v in lora.adapter_state(model).items()}
            train_steps(model)
            state = model.state_dict()
            for k, v in base.state_dict().items():
                k = k.replace(".weight", ".parametrizations.weight.original") if k not in state else k
                assert torch.equal(state[k], v), k
            assert any(not torch.equal(v, before[k]) for k, v in lora.adapter_state(model).items())


def test_targets():
    per_block = {"attention": 2, "blocks": 4}
    with tempfile.TemporaryDirectory() as tmp:
        for targets, n in per_block.items():
            _, model, _, _ = adapted(tmp, "lora", targets=targets)
            layers = {k.split(".parametrizations.")[0] for k in lora.adapter_state(model)}
            assert len(layers) == n * len(model.blocks), (targets, len(layers))
            assert not [k for k in layers if "adaLN" in k or "frame" in k or "inpaint" in k]

        _, model, _, _ = adapted(tmp, "lora", targets="all")
        layers = {k.split(".parametrizations.")[0] for k in lora.adapter_state(model)}
        want = {n for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv1d))}
        assert layers == want, sorted(layers ^ want)
        for name in ("pre_conv", "post_conv", "input_proj", "output_proj", "adaLN.0",
                     "t_embedder.mlp.0", "blocks.0.mlp.proj"):
            assert name in layers, name


def test_every_linear_and_conv_with_all():
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for adapter in lora.ADAPTERS:
            base, model, path, _ = adapted(tmp, adapter, targets="all")
            with torch.enable_grad():
                train_steps(model)
            ckpt = {"model_state_dict": lora.adapter_state(model), "step": 3}
            plain = build(seed=2)
            plain.load_state_dict(lora.merge_checkpoint(ckpt, path)["model_state_dict"])
            assert torch.equal(run(plain), run(model))
            assert not torch.equal(run(plain), run(base))
            assert model.pre_conv.parametrizations.weight[0].lora_A.shape == (4, model.pre_conv.in_channels)


def test_per_block_adaln():
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for targets, n in {"attention": 2, "blocks": 4}.items():
            _, model, _, _ = adapted(tmp, "lora", targets=targets, adaln="per_block")
            layers = {k.split(".parametrizations.")[0] for k in lora.adapter_state(model)}
            assert len(layers) == n * len(model.blocks), (targets, len(layers))
            assert not [k for k in layers if "adaLN" in k or "final_layer" in k]
        for adapter in lora.ADAPTERS:
            base, model, path, _ = adapted(tmp, adapter, targets="all", adaln="per_block")
            layers = {k.split(".parametrizations.")[0] for k in lora.adapter_state(model)}
            for name in ("blocks.0.adaLN.1", "final_layer.adaLN.1", "final_layer.linear"):
                assert name in layers, name
            with torch.enable_grad():
                train_steps(model)
            ckpt = {"model_state_dict": lora.adapter_state(model), "step": 3}
            plain = build(seed=2, adaln="per_block")
            plain.load_state_dict(lora.merge_checkpoint(ckpt, path)["model_state_dict"])
            assert torch.equal(run(plain), run(model))
            assert not torch.equal(run(plain), run(base))


def test_merge_is_the_adapted_model():
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for adapter in lora.ADAPTERS:
            base, model, path, _ = adapted(tmp, adapter)
            with torch.enable_grad():
                train_steps(model)
            ckpt = {"model_state_dict": lora.adapter_state(model), "step": 3}
            plain = build(seed=2)
            merged = lora.merge_checkpoint(ckpt, path)
            assert merged["step"] == 3
            plain.load_state_dict(merged["model_state_dict"])
            del merged
            assert torch.equal(run(plain), run(model))
            plain.load_state_dict(lora.merge_checkpoint(ckpt, path, 0.0)["model_state_dict"])
            assert torch.equal(run(plain), run(base))


def test_resume_loads_the_adapters():
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for adapter in lora.ADAPTERS:
            _, model, _, cfg = adapted(tmp, adapter)
            with torch.enable_grad():
                train_steps(model)
            state = lora.adapter_state(model)
            again = build(seed=3)
            lora.attach(again, cfg)
            lora.load_adapters(again, state)
            assert torch.equal(run(again), run(model))


def test_a_base_that_does_not_fit_stops():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, cfg = adapted(tmp, "lora")
        try:
            lora.attach(build(kind="B"), cfg)
        except SystemExit as e:
            assert "does not fit" in str(e)
        else:
            raise AssertionError("a B model took the weights of an S model")


def test_a_base_with_other_time_features_stops():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, cfg = adapted(tmp, "lora")
        cfg.model.t_features = "low_freq"
        try:
            lora.attach(build(), cfg)
        except SystemExit as e:
            assert "t_features" in str(e)
        else:
            raise AssertionError("a base with other time features was accepted")


def test_normalizer_of_another_dataset_stops():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = str(Path(tmp) / "a.pt"), str(Path(tmp) / "b.pt")
        torch.save({"mean": torch.zeros(3), "std": torch.ones(3)}, src)
        lora.install_normalizer(src, dst)
        lora.install_normalizer(src, dst)
        torch.save({"mean": torch.ones(3), "std": torch.ones(3)}, dst)
        try:
            lora.install_normalizer(src, dst)
        except SystemExit as e:
            assert "not the normalizer" in str(e)
        else:
            raise AssertionError("another normalizer was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  OK\n")
