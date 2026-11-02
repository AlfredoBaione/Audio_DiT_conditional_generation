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
                decay_start_frac=0.8, weight_decay=0.0,
                adam_betas=[0.9, 0.999], adam_eps=1e-8)
    base.update(kw)
    return OmegaConf.create(base)


def _model():
    return ConditionedAudioDiT(token_dim=lc.active().latent_dim, kind='S',
                               frame_cond_dims={}, frame_cond_out_dims={},
                               global_cond_configs={"text": {"dim": 512}},
                               text_conditioning="all", text_ctx_dim=768,
                               attention="differential", qk_norm=True)


def _fake_step(optimizer, model):
    torch.manual_seed(0)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-3
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_flow_direction():
    class Recorder(torch.nn.Module):
        def forward(self, xt, t, **kw):
            self.xt = xt
            return torch.zeros_like(xt)

    torch.manual_seed(0)
    x1 = torch.randn(2, 8, lc.active().latent_dim)
    x0 = torch.randn_like(x1)
    t = torch.tensor([0.25, 0.75])
    model = Recorder()
    loss = tc.compute_loss(model, (x1, {}, None, None, None, None), "cpu", use_amp=False,
                           global_configs={}, p_drop_all=0.0, p_drop_frame=0.0,
                           p_drop_global=0.0, training=False, x0=x0, t=t)
    want = (1 - t).view(2, 1, 1) * x1 + t.view(2, 1, 1) * x0
    assert torch.allclose(model.xt, want)
    assert math.isclose(loss.item(), (x0 - x1).pow(2).mean().item(), rel_tol=1e-6)
    print("  x_t = (1 - t) data + t noise, target = noise - data")


def test_euler_grid():
    assert tc.euler_grid(4, "uniform") == [(1.0, -0.25), (0.75, -0.25), (0.5, -0.25), (0.25, -0.25)]
    for steps in (1, 8, 50, 100):
        for schedule in ("uniform", "logsnr_uniform"):
            g = tc.euler_grid(steps, schedule)
            assert len(g) == steps and g[0][0] == 1.0
            assert all(dt < 0 for _, dt in g)
            assert math.isclose(sum(dt for _, dt in g), -1.0, abs_tol=1e-12)
    g = tc.euler_grid(50, "logsnr_uniform")
    last = 1.0 / (1.0 + math.exp(2.0 - (1 / 50) * 8.2))
    assert math.isclose(g[-1][0], last, rel_tol=1e-12)
    e = tc.euler_grid(50, "logsnr_uniform", t_start=0.3)
    assert e[0][0] == 0.3 and math.isclose(sum(dt for _, dt in e), -0.3, abs_tol=1e-12)
    print(f"  from t=1 to t=0; log-SNR grid last evaluated at t={g[-1][0]:.3f}")


def test_guided_velocity():
    torch.manual_seed(0)
    x, v_c, v_u = torch.randn(2, 8, 4), torch.randn(2, 8, 4), torch.randn(2, 8, 4)
    t, g = 0.4, 3.0
    plain = tc.guided_velocity(x, t, v_c, v_u, g, "cfg")
    assert torch.allclose(plain, v_u + g * (v_c - v_u), atol=1e-5)
    apg = tc.guided_velocity(x, t, v_c, v_u, g, "apg")
    den_c = x - t * v_c
    part = (x - t * apg) - den_c
    inner = (part * den_c).sum(dim=(-2, -1))
    assert inner.abs().max().item() < 1e-3, inner
    half = tc.guided_velocity(x, t, v_c, v_u, g, "mix")
    assert torch.allclose(half, 0.5 * plain + 0.5 * apg, atol=1e-5)
    print("  cfg = v_u + g (v_c - v_u); apg = the correction is orthogonal to x - t v_c; "
          "mix = half and half")


def test_t_samplers():
    torch.manual_seed(0)
    for sampler, draw in (
            ("truncated_logit", lambda n: tc.sample_t_truncated_logit(n, "cpu", seq_len=431)),
            ("uniform", lambda n: tc.sample_t_uniform(n, "cpu")),
            ("logit_normal", lambda n: tc.sample_logit_normal(n, "cpu"))):
        t = draw(1000000)
        assert t.min() >= 0.0 and t.max() <= 1.0
        for x in (0.1, 0.2, 0.5, 0.9):
            got = (t < x).float().mean().item()
            want = tc.t_sampler_cdf(sampler, x, 431)
            assert abs(got - want) < 2e-3, (sampler, x, got, want)
    print("  draws match t_sampler_cdf for all three samplers")


def test_lr_warmup_exponential():
    f = tc.make_lr_lambda(1000001, 5000, 0.8, schedule="inverse_power", inv_gamma=1.0e6,
                          power=0.5, warmup_factor=0.995)
    for step in (0, 1, 100, 920, 5000, 500000):
        want = (1 - 0.995 ** (step + 1)) * (1 + step / 1.0e6) ** -0.5
        assert math.isclose(f(step), want, rel_tol=1e-12), step
    assert f(920) > 0.98
    g = tc.lr_lambda_from_cfg(_train_cfg(lr_schedule="inverse_power", lr_warmup=0.995))
    assert math.isclose(g(100), f(100), rel_tol=1e-12)
    print(f"  x{f(0):.3f} at step 0, x{f(920):.3f} at step 920")


def test_ema_decay_at():
    cfg = _train_cfg(ema_decay=0.9995, ema_power=0.75)
    assert math.isclose(tc.ema_decay_at(cfg, 1), 1 - 2 ** -0.75, rel_tol=1e-12)
    assert math.isclose(tc.ema_decay_at(cfg, 10000), 1 - 10001 ** -0.75, rel_tol=1e-12)
    assert tc.ema_decay_at(cfg, 10 ** 6) == 0.9995
    assert tc.ema_decay_at(_train_cfg(ema_decay=0.9999), 7) == 0.9999
    print(f"  {tc.ema_decay_at(cfg, 1000):.5f} after 1000 updates, "
          f"{tc.ema_decay_at(cfg, 10 ** 6)} at the end")


def test_ot_assign_noise():
    torch.manual_seed(0)
    x1 = torch.randn(8, 16, 4)
    noise = torch.randn(8, 16, 4)
    out = tc.ot_assign_noise(x1, noise)
    assert sorted(out.sum(dim=(1, 2)).tolist()) == sorted(noise.sum(dim=(1, 2)).tolist())
    gain = []
    for _ in range(50):
        x1 = torch.randn(8, 16, 4) + torch.randn(8, 1, 1)
        noise = torch.randn(8, 16, 4)
        gain.append((x1 - noise).pow(2).sum().item()
                    - (x1 - tc.ot_assign_noise(x1, noise)).pow(2).sum().item())
    assert sum(gain) > 0
    print(f"  a permutation of the noise; mean squared path shorter by "
          f"{sum(gain) / len(gain):.1f} on 50 batches of 8")


def test_muon_optimizer():
    model = _model()
    named = dict(model.named_parameters())
    cfg = _train_cfg(optimizer="muon", muon_lr=1e-3, muon_momentum=0.95, lr=5e-5,
                     adam_betas=[0.9, 0.95], weight_decay=0.01)
    assert tc.optimizer_layout(cfg) == "muon"
    opt = tc.build_optimizer(model, cfg, "muon")
    g_m, g_a = opt.param_groups
    assert g_m["lr"] == 1e-3 and g_m["momentum"] == 0.95
    assert g_a["lr"] == 5e-5 and g_a["betas"] == (0.9, 0.95) and g_a["weight_decay"] == 0.01
    ids_m = {id(p) for p in g_m["params"]}
    ids_a = {id(p) for p in g_a["params"]}
    assert not ids_m & ids_a and ids_m | ids_a == {id(p) for p in model.parameters()}
    assert all((p.ndim >= 2) == (id(p) in ids_m) for p in named.values())
    fused = {n for n, p in named.items() if p in opt._fused}
    assert any(".attn.qkv." in n for n in fused) and any(".mlp.proj." in n for n in fused)
    assert any(".cross_attn.kv." in n for n in fused)
    assert not any(".proj.bias" in n or ".attn.proj." in n for n in fused)

    lam = tc.lr_lambda_from_cfg(_train_cfg(lr_schedule="inverse_power", lr_warmup=0.995))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    before = {n: p.detach().clone() for n, p in named.items()}
    _fake_step(opt, model)
    sch.step()
    assert all(not torch.equal(before[n], p) for n, p in named.items())
    assert all("momentum_buffer" in opt.state[p] for p in g_m["params"])
    assert all("exp_avg_sq" in opt.state[p] for p in g_a["params"])
    assert sch.base_lrs == [1e-3, 5e-5]

    ckpt = {"optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sch.state_dict()}
    opt2 = tc.build_optimizer(model, cfg, "muon")
    sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lam)
    opt2, sch2, rebuilt, ignored = tc.restore_optimizer(model, cfg, opt2, sch2, lam, ckpt)
    assert rebuilt is None and ignored == []
    assert all(torch.equal(opt2.state[p]["momentum_buffer"], opt.state[p]["momentum_buffer"])
               for p in g_m["params"])

    other = _train_cfg(optimizer="muon", muon_lr=2e-3, muon_momentum=0.9, lr=1e-4,
                       adam_betas=[0.9, 0.95], weight_decay=0.01)
    opt3 = tc.build_optimizer(model, other, "muon")
    sch3 = torch.optim.lr_scheduler.LambdaLR(opt3, lam)
    _, _, _, ignored = tc.restore_optimizer(model, other, opt3, sch3, lam, ckpt)
    assert any(s.startswith("lr=") for s in ignored)
    assert any(s.startswith("muon_lr=") for s in ignored)
    assert any(s.startswith("muon_momentum=") for s in ignored)

    adam_cfg = _train_cfg(weight_decay=0.01, adam_betas=[0.9, 0.95])
    opt4 = tc.build_optimizer(model, adam_cfg, tc.optimizer_layout(adam_cfg))
    sch4 = torch.optim.lr_scheduler.LambdaLR(opt4, lam)
    opt4, _, rebuilt, _ = tc.restore_optimizer(model, adam_cfg, opt4, sch4, lam, ckpt)
    assert rebuilt == "muon" and opt4.param_groups[0]["use_muon"]
    print(f"  {len(g_m['params'])} tensors with Muon, {len(g_a['params'])} with AdamW; "
          f"{len(fused)} fused; state restored; a different optimizer keeps the checkpoint's")


def test_lora_recipe_in_the_config():
    with tempfile.TemporaryDirectory() as d:
        base, norm = str(Path(d) / "base.pt"), str(Path(d) / "normalizer.pt")
        Path(base).touch()
        Path(norm).touch()
        cfg = _load_config(["lora.adapter=dora-rows", f"paths.init_from={base}",
                            f"paths.normalizer_from={norm}"])
    assert cfg.training.optimizer == "adamw" and cfg.training.use_ema is False
    assert cfg.training.lr == 1.0e-4 and cfg.training.lr_schedule == "constant"
    assert tc.optimizer_layout(cfg.training) == "single"
    assert list(cfg.training.adam_betas) == [0.9, 0.95] and cfg.training.weight_decay == 0.01
    assert cfg.lora.rank == 16 and cfg.lora.targets == "all"
    assert tc.lr_lambda_from_cfg(cfg.training)(12345) == 1.0
    print("  AdamW 1e-4, betas 0.9/0.95, weight decay 0.01, constant lr, no EMA, every layer")


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
    assert all(named[n].ndim >= 2 for n, p in named.items() if id(p) in id_decay)
    for must in ("q_norm.weight", "k_norm.weight", "text_pad", ".bias"):
        assert any(must in n for n in names_keep), must
    n_dec = sum(p.numel() for p in decay["params"])
    n_keep = sum(p.numel() for p in keep["params"])
    print(f"  matrices: {len(decay['params'])} tensors decayed ({n_dec / 1e6:.2f}M), "
          f"{len(keep['params'])} not ({n_keep / 1e3:.1f}k: biases, norm gains, "
          f"text padding)")

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
    assert cfg.training.optimizer == "muon"
    assert cfg.training.lr == 5.0e-5 and cfg.training.muon_lr == 1.0e-3
    assert cfg.training.lr_warmup == 0.995 and cfg.training.ot_coupling is True
    assert cfg.training.ema_decay == 0.9995 and cfg.training.ema_power == 0.75
    assert list(cfg.training.adam_betas) == [0.9, 0.95]
    assert cfg.training.weight_decay == 0.01 and cfg.training.grad_clip == 0.0
    print("  new run: Muon + AdamW, inverse_power schedule, exponential warmup, "
          "OT coupling, EMA 0.9995 with power 0.75")

    new = OmegaConf.to_container(OmegaConf.load(CFG_PATH))
    new["training"]["adam_betas"] = [0.9, 0.999]
    with tempfile.TemporaryDirectory() as d:
        p_new = str(Path(d) / "new.pt")
        torch.save({"config": new}, p_new)
        cfg = _load_config(["--resume", p_new])
        assert cfg.model.qk_norm is True
        assert cfg.training.lr_schedule == "inverse_power"
        assert list(cfg.training.adam_betas) == [0.9, 0.999]
        print("  resume: the checkpoint's own values")

    for bad in (["training.lr_schedule=linear"], ["training.adam_betas=[0.9]"],
                ["training.adam_betas=[0.9,1.0]"], ["training.adam_eps=0"],
                ["training.weight_decay=-0.1"], ["training.lr_inv_gamma=0"],
                ["training.optimizer=sgd"], ["training.muon_lr=0"],
                ["training.muon_momentum=1.0"], ["training.lr_warmup=1.0"],
                ["training.ema_power=-1"], ["sampling.guidance_type=none"]):
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
