# Tests for training_cond.py: learning-rate schedules, AdamW parameter groups,
# optimizer restore on resume, the new config keys.
# Run with `pytest test_functions` or `python test_functions/test_training_cond.py`.

import sys
import math
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

import latent_codec as lc
import training_cond as tc
from network_cond import ConditionedAudioDiT

CFG_PATH = str(ROOT / "configs" / "training_cond_default.yaml")
NEW_TRAINING_KEYS = ("lr_schedule", "lr_inv_gamma", "lr_power",
                     "weight_decay_matrices_only", "adam_betas", "adam_eps")


def _old_cosine(num_steps, warmup_steps, decay_start_frac):
    decay_start = int(num_steps * decay_start_frac)

    def f(step):
        if step < warmup_steps:
            return step / warmup_steps
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / (num_steps - decay_start)
        return 0.5 * (1 + math.cos(progress * math.pi))
    return f


def _train_cfg(**kw):
    base = dict(lr=1e-4, num_steps=1000001, warmup_steps=5000,
                decay_start_frac=0.8, weight_decay=0.0)
    base.update(kw)
    return OmegaConf.create(base)


def _model():
    return ConditionedAudioDiT(token_dim=lc.active().latent_dim, kind='S',
                               frame_cond_dims={}, frame_cond_out_dims={},
                               global_cond_configs={"text": {"dim": 512}},
                               text_cross_every=2, text_ctx_dim=768,
                               attention="differential", qk_norm=True)


def _fake_step(optimizer, model):
    torch.manual_seed(0)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-3
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_lr_schedules():
    f = tc.make_lr_lambda(1000001, 5000, 0.8, schedule="inverse_power",
                          inv_gamma=1.0e6, power=0.5)
    assert f(0) == 0.0
    assert math.isclose(f(2500), 0.5 * (1 + 2500 / 1e6) ** -0.5, rel_tol=1e-12)
    for step, want in ((5000, 1.005 ** -0.5), (100000, 1.1 ** -0.5),
                       (380000, 1.38 ** -0.5), (1000000, 2 ** -0.5)):
        assert math.isclose(f(step), want, rel_tol=1e-12), (step, f(step), want)
    after = [f(s) for s in range(5000, 1000001, 5000)]
    assert all(a > b for a, b in zip(after, after[1:]))
    print(f"  inverse_power: x{f(100000):.3f} at 100k, x{f(380000):.3f} at 380k, "
          f"x{f(1000000):.3f} at 1M")

    old = _old_cosine(1000001, 5000, 0.8)
    for g in (tc.make_lr_lambda(1000001, 5000, 0.8),
              tc.make_lr_lambda(1000001, 5000, 0.8, schedule="cosine"),
              tc.lr_lambda_from_cfg(_train_cfg())):
        for s in (0, 2500, 5000, 400000, 800000, 900000, 1000000):
            assert math.isclose(g(s), old(s), rel_tol=1e-6, abs_tol=1e-6), s
    print("  cosine: identical to the schedule of every earlier run, and the "
          "default when the key is absent")

    g = tc.lr_lambda_from_cfg(_train_cfg(lr_schedule="inverse_power",
                                         lr_inv_gamma=2.0e5, lr_power=1.0))
    assert math.isclose(g(200000), 0.5, rel_tol=1e-12)
    try:
        tc.make_lr_lambda(10, 1, 0.8, schedule="linear")
        raise AssertionError("an unknown schedule should have raised")
    except ValueError:
        pass


def test_param_groups():
    model = _model()
    named = dict(model.named_parameters())

    assert tc.optimizer_layout(_train_cfg()) == "single"
    assert tc.optimizer_layout(_train_cfg(weight_decay=0.01)) == "single"
    assert tc.optimizer_layout(_train_cfg(weight_decay=0.01,
                                          weight_decay_matrices_only=True)) == "matrices"
    assert tc.optimizer_layout(_train_cfg(weight_decay=0.0,
                                          weight_decay_matrices_only=True)) == "single"

    opt = tc.build_optimizer(model, _train_cfg(weight_decay=0.01), "single")
    assert len(opt.param_groups) == 1
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in model.parameters()]
    g = opt.param_groups[0]
    assert g["weight_decay"] == 0.01 and g["betas"] == (0.9, 0.999) and g["eps"] == 1e-8
    print("  single: one group, the parameters in model.parameters() order "
          "(the layout of every earlier checkpoint), torch-default betas/eps")

    cfg = _train_cfg(weight_decay=0.01, weight_decay_matrices_only=True,
                     adam_betas=[0.9, 0.95], adam_eps=1e-7)
    opt = tc.build_optimizer(model, cfg, tc.optimizer_layout(cfg))
    decay, keep = opt.param_groups
    assert decay["weight_decay"] == 0.01 and keep["weight_decay"] == 0.0
    assert decay["betas"] == (0.9, 0.95) and decay["eps"] == 1e-7
    id_decay = {id(p) for p in decay["params"]}
    id_keep = {id(p) for p in keep["params"]}
    assert not id_decay & id_keep
    assert id_decay | id_keep == {id(p) for p in model.parameters()}
    names_keep = sorted(n for n, p in named.items() if id(p) in id_keep)
    assert all(named[n].ndim >= 2 and not n.endswith("text_null")
               for n, p in named.items() if id(p) in id_decay)
    for must in ("q_norm.weight", "k_norm.weight", "subln.weight", "lambda_q1",
                 "lambda_k2", "text_null", ".bias"):
        assert any(must in n for n in names_keep), must
    n_dec = sum(p.numel() for p in decay["params"])
    n_keep = sum(p.numel() for p in keep["params"])
    print(f"  matrices: {len(decay['params'])} tensors decayed ({n_dec / 1e6:.2f}M), "
          f"{len(keep['params'])} not ({n_keep / 1e3:.1f}k: biases, norm gains, "
          f"lambda vectors, text null)")

    try:
        tc.build_optimizer(model, cfg, "groups")
        raise AssertionError("an unknown layout should have raised")
    except ValueError:
        pass


def test_restore_optimizer_from_old_checkpoint():
    model = _model()
    old_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    old_sch = torch.optim.lr_scheduler.LambdaLR(old_opt, _old_cosine(1000001, 5000, 0.8))
    _fake_step(old_opt, model)
    old_sch.step()
    ckpt = {"optimizer_state_dict": old_opt.state_dict(),
            "scheduler_state_dict": old_sch.state_dict()}
    want = {id(p): old_opt.state[p]["exp_avg_sq"].clone() for p in model.parameters()}

    cfg = _train_cfg(weight_decay=0.01, weight_decay_matrices_only=True,
                     adam_betas=[0.9, 0.95], lr_schedule="inverse_power")
    lam = tc.lr_lambda_from_cfg(cfg)
    opt = tc.build_optimizer(model, cfg, tc.optimizer_layout(cfg))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    assert len(opt.param_groups) == 2
    opt, sch, rebuilt, ignored = tc.restore_optimizer(model, cfg, opt, sch, lam, ckpt)
    assert rebuilt == "single" and len(opt.param_groups) == 1
    assert all(torch.equal(opt.state[p]["exp_avg_sq"], want[id(p)])
               for p in model.parameters())
    assert sch.last_epoch == 1
    assert any(s.startswith("betas=") for s in ignored)
    assert any(s.startswith("weight_decay=") for s in ignored)
    assert not any(s.startswith("lr=") for s in ignored)
    print(f"  old single-group state restored into a matrices config: rebuilt as "
          f"'{rebuilt}', moments intact; noted: {ignored}")

    opt2 = tc.build_optimizer(model, cfg, "matrices")
    sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lam)
    _fake_step(opt2, model)
    sch2.step()
    ckpt2 = {"optimizer_state_dict": opt2.state_dict(),
             "scheduler_state_dict": sch2.state_dict()}
    opt3 = tc.build_optimizer(model, cfg, "matrices")
    sch3 = torch.optim.lr_scheduler.LambdaLR(opt3, lam)
    opt3, sch3, rebuilt, ignored = tc.restore_optimizer(model, cfg, opt3, sch3, lam, ckpt2)
    assert rebuilt is None and ignored == []
    assert [g["weight_decay"] for g in opt3.param_groups] == [0.01, 0.0]
    assert all(torch.equal(opt3.state[p]["exp_avg"], opt2.state[p]["exp_avg"])
               for p in model.parameters())
    print("  matrices state restored into a matrices config: no rebuild, nothing noted")


def _load_config(argv):
    saved = sys.argv
    sys.argv = ["training_cond.py", "--config", CFG_PATH, "--run_name", "t"] + argv
    try:
        cfg, _ = tc.load_config()
    finally:
        sys.argv = saved
    return cfg


def test_config_keys_new_run_and_resume():
    cfg = _load_config([])
    assert cfg.model.qk_norm is True
    assert cfg.training.lr_schedule == "inverse_power"
    assert cfg.training.lr_inv_gamma == 1.0e6 and cfg.training.lr_power == 0.5
    assert list(cfg.training.adam_betas) == [0.9, 0.999]
    assert cfg.training.weight_decay == 0.0 and cfg.training.grad_clip == 0.0
    print("  new run: qk_norm on, inverse_power schedule; betas, weight decay "
          "and clipping as before")

    old = OmegaConf.to_container(OmegaConf.load(CFG_PATH))
    del old["model"]["qk_norm"]
    for k in NEW_TRAINING_KEYS:
        del old["training"][k]
    with tempfile.TemporaryDirectory() as d:
        p_old = str(Path(d) / "old.pt")
        torch.save({"config": old,
                    "model_state_dict": {"blocks.0.attn.qkv.weight": torch.zeros(1)}},
                   p_old)
        cfg = _load_config(["--resume", p_old])
        assert cfg.model.qk_norm is False
        assert cfg.training.lr_schedule == "cosine"
        assert list(cfg.training.adam_betas) == [0.9, 0.999]
        assert cfg.training.adam_eps == 1e-8
        assert cfg.training.weight_decay_matrices_only is False
        cfg = _load_config(["--resume", p_old, "training.lr_schedule=inverse_power"])
        assert cfg.training.lr_schedule == "inverse_power"
        print("  resume of a checkpoint that predates the keys: qk_norm off, "
              "cosine, torch betas, one group; the command line still wins")

        new = OmegaConf.to_container(OmegaConf.load(CFG_PATH))
        new["training"]["adam_betas"] = [0.9, 0.95]
        p_new = str(Path(d) / "new.pt")
        torch.save({"config": new, "qk_norm": True,
                    "model_state_dict": {"blocks.0.attn.q_norm.weight": torch.ones(1)}},
                   p_new)
        cfg = _load_config(["--resume", p_new])
        assert cfg.model.qk_norm is True
        assert cfg.training.lr_schedule == "inverse_power"
        assert list(cfg.training.adam_betas) == [0.9, 0.95]
        print("  resume of a new checkpoint: its own values")

    for bad in (["training.lr_schedule=linear"], ["training.adam_betas=[0.9]"],
                ["training.adam_betas=[0.9,1.0]"], ["training.adam_eps=0"],
                ["training.weight_decay=-0.1"], ["training.lr_inv_gamma=0"]):
        try:
            _load_config(bad)
            raise AssertionError(f"{bad} should have been refused")
        except SystemExit:
            pass
    print("  invalid schedule / betas / eps / weight decay refused before anything runs")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  OK\n")
