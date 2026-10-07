# Tests for network_cond.py: conditioning paths, frame re-injection, differential attention.
# Run with `pytest test_functions` or `python test_functions/test_network_cond.py`.

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import latent_codec as lc
from network_cond import (ConditionedAudioDiT, DifferentialSelfAttention,
                          apply_rotary_pos_emb, ckpt_attention)

B, N = 2, 430
TOKEN_DIM = lc.active().latent_dim   # 72 con DAC (default), 128 con EnCodec


def _inputs():
    return torch.randn(B, N, TOKEN_DIM), torch.rand(B)


def test_frame_f0_only():
    x, t = _inputs()
    model = ConditionedAudioDiT(token_dim=TOKEN_DIM,
        kind='S',
        frame_cond_dims={"f0": 2},
        frame_cond_out_dims={"f0": 16},
        global_cond_configs={},
    )
    out = model(x, t, frame_conditions={"f0": torch.randn(B, N, 2)})
    print(f"  input {x.shape} -> output {out.shape}")
    assert out.shape == x.shape
    assert out.shape == model(x, t).shape


def test_frame_and_global():
    x, t = _inputs()
    model2 = ConditionedAudioDiT(
        kind='S',
        frame_cond_dims={"f0": 2, "chroma": 12, "rhythm": 2},
        frame_cond_out_dims={"f0": 16, "chroma": 64, "rhythm": 32},
        global_cond_configs={"text": {"dim": 512}, "image": {"dim": 512}},
    )
    out2 = model2(
        x, t,
        frame_conditions={
            "f0": torch.randn(B, N, 2),
            "chroma": torch.randn(B, N, 12),
            "rhythm": torch.randn(B, N, 2),
        },
        global_conditions={"text": torch.randn(B, 512), "image": torch.randn(B, 512)},
    )
    print(f"  full conditioned -> {out2.shape}")
    assert out2.shape == x.shape


def test_global_only():
    x, t = _inputs()
    model3 = ConditionedAudioDiT(token_dim=TOKEN_DIM,
        kind='S',
        frame_cond_dims={},
        frame_cond_out_dims={},
        global_cond_configs={"text": {"dim": 512}},
    )
    out3 = model3(x, t, global_conditions={"text": torch.randn(B, 512)})
    print(f"  text only -> {out3.shape}")
    assert out3.shape == x.shape


def test_frame_reinjection():
    x, t = _inputs()
    cond_kw = dict(token_dim=TOKEN_DIM,
                   kind='S',
                   frame_cond_dims={"f0": 2, "chroma": 12},
                   frame_cond_out_dims={"f0": 16, "chroma": 64},
                   global_cond_configs={})
    model4_off = ConditionedAudioDiT(**cond_kw, frame_reinject_every=0)
    model4     = ConditionedAudioDiT(**cond_kw, frame_reinject_every=1)

    assert model4_off.reinject_layers == []
    assert model4.reinject_layers == [1, 2, 3, 4, 5], model4.reinject_layers

    f0c = {"f0": torch.randn(B, N, 2), "chroma": torch.randn(B, N, 12)}
    out4 = model4(x, t, frame_conditions=f0c)
    print(f"  re-injected -> {out4.shape}")
    assert out4.shape == x.shape

    with torch.no_grad():
        for blk in model4_off.blocks:
            nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
            nn.init.normal_(blk.adaLN_modulation[-1].bias,   std=0.02)
        nn.init.normal_(model4_off.final_layer.linear.weight, std=0.02)
        nn.init.normal_(model4_off.final_layer.adaLN_modulation[-1].weight, std=0.02)

    missing, unexpected = model4.load_state_dict(model4_off.state_dict(),
                                                 strict=False)
    assert unexpected == [], unexpected
    assert missing and all(k.startswith("frame_reinject.") or
                           k.startswith("frame_reinject_gate.")
                           for k in missing), missing
    print(f"  shared weights copied; only {len(missing)} re-injection tensors "
          f"are new (projections + per-condition gates)")

    with torch.no_grad():
        base = model4_off(x, t, frame_conditions=f0c)
        assert base.abs().max().item() > 0, "baseline is still degenerate"
        d = (model4(x, t, frame_conditions=f0c) - base).abs().max().item()
    print(f"  zero-init check: max |reinject - baseline| = {d:.3e}")
    assert d == 0.0, "re-injection is not zero at init"

    with torch.no_grad():
        for lin in model4.frame_reinject.values():
            nn.init.normal_(lin.weight, std=0.02)
        d2 = (model4(x, t, frame_conditions=f0c) - base).abs().max().item()
    print(f"  wired check:     max |reinject - baseline| = {d2:.3e}")
    assert d2 > 0.0, "re-injection has no effect on the output"

    with torch.no_grad():
        f0c_b = {"f0": torch.randn(B, N, 2), "chroma": torch.randn(B, N, 12)}
        d3 = (model4(x, t, frame_conditions=f0c)
              - model4(x, t, frame_conditions=f0c_b)).abs().max().item()
    print(f"  sensitivity:     max |cond_A - cond_B| = {d3:.3e}")
    assert d3 > 0.0

    model4.zero_grad(set_to_none=True)
    model4(x, t, frame_conditions=f0c).pow(2).mean().backward()
    for k, lin in model4.frame_reinject.items():
        g = lin.weight.grad
        assert g is not None and g.abs().sum().item() > 0, f"no grad on block {k}"
    print(f"  grad check:      all {len(model4.frame_reinject)} projections "
          f"receive gradient")


def test_reinjection_stride_and_inert_cases():
    x, t = _inputs()
    model6 = ConditionedAudioDiT(token_dim=TOKEN_DIM,
        kind='S', frame_cond_dims={"f0": 2}, frame_cond_out_dims={"f0": 16},
        global_cond_configs={}, frame_reinject_every=2,
    )
    assert model6.reinject_layers == [2, 4], model6.reinject_layers
    assert model6(x, t, frame_conditions={"f0": torch.randn(B, N, 2)}).shape == x.shape

    model7 = ConditionedAudioDiT(token_dim=TOKEN_DIM,
        kind='S', frame_cond_dims={}, frame_cond_out_dims={},
        global_cond_configs={"text": {"dim": 512}}, frame_reinject_every=1,
    )
    assert model7.reinject_layers == []
    assert not hasattr(model7, "frame_reinject")

    model8 = ConditionedAudioDiT(token_dim=TOKEN_DIM,
        kind='S', frame_cond_dims={"f0": 2}, frame_cond_out_dims={"f0": 16},
        global_cond_configs={}, frame_reinject_every=99,
    )
    assert model8.reinject_layers == []

    try:
        ConditionedAudioDiT(kind='S', frame_cond_dims={"f0": 2},
                            frame_cond_out_dims={"f0": 16},
                            global_cond_configs={}, frame_reinject_every=-1)
        raise AssertionError("negative stride should have raised")
    except ValueError as e:
        print(f"  negative stride rejected: {e}")


def test_differential_attention():
    x, t = _inputs()
    plain_kw = dict(token_dim=TOKEN_DIM,
                    kind='S', frame_cond_dims={}, frame_cond_out_dims={},
                    global_cond_configs={})
    m_std = ConditionedAudioDiT(**plain_kw)
    m_dif = ConditionedAudioDiT(**plain_kw, attention="differential")
    hid, nh, nl = 512, 8, 6
    hd = hid // nh
    sd_s, sd_d = m_std.state_dict(), m_dif.state_dict()
    extra = sorted(set(sd_d) - set(sd_s))
    assert set(sd_s) <= set(sd_d)
    assert len(extra) == 5 * nl and all(".attn.lambda_" in k or ".attn.subln." in k
                                        for k in extra), extra
    assert all(sd_s[k].shape == sd_d[k].shape for k in sd_s)
    n_s = sum(p.numel() for p in m_std.parameters())
    n_d = sum(p.numel() for p in m_dif.parameters())
    assert n_d - n_s == nl * 6 * hd, (n_d - n_s)
    print(f"  parameters: {n_s} standard, {n_d} differential (+{n_d - n_s} = "
          f"{nl} blocks x 6 x {hd})")
    for i, blk in enumerate(m_dif.blocks):
        assert abs(blk.attn.lambda_init - (0.8 - 0.6 * math.exp(-0.3 * i))) < 1e-12

    att = m_dif.blocks[3].attn
    xa = torch.randn(B, N, hid)
    with torch.no_grad():
        got = att(xa)
        q_, k_, v_ = att.qkv(xa).split(hid, dim=-1)
        q_ = q_.reshape(B, N, nh, hd).transpose(1, 2)
        k_ = k_.reshape(B, N, nh, hd).transpose(1, 2)
        v_ = v_.reshape(B, N, nh // 2, 2 * hd).transpose(1, 2)
        cs, sn = att._cos_sin(N, xa.device, xa.dtype)
        q_, k_ = apply_rotary_pos_emb(q_, k_, cs, sn)
        A1 = torch.softmax(q_[:, 0::2] @ k_[:, 0::2].transpose(-1, -2) / math.sqrt(hd), -1)
        A2 = torch.softmax(q_[:, 1::2] @ k_[:, 1::2].transpose(-1, -2) / math.sqrt(hd), -1)
        lam = (torch.exp((att.lambda_q1 * att.lambda_k1).sum())
               - torch.exp((att.lambda_q2 * att.lambda_k2).sum()) + att.lambda_init)
        o_ = (A1 - lam * A2) @ v_
        o_ = o_ * torch.rsqrt(o_.pow(2).mean(-1, keepdim=True) + 1e-5) * att.subln.weight
        o_ = o_ * (1 - att.lambda_init)
        ref = att.proj(o_.transpose(1, 2).reshape(B, N, hid))
        dd = (got - ref).abs().max().item()
    print(f"  formula check:   max |module - explicit (A1 - lambda A2) v| = {dd:.3e}")
    assert dd < 1e-5, dd
    rs = (A1 - lam * A2).sum(-1)
    assert (rs - (1 - lam)).abs().max().item() < 1e-5

    with torch.no_grad():
        for blk in m_dif.blocks:
            nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
            nn.init.normal_(blk.adaLN_modulation[-1].bias,   std=0.02)
        nn.init.normal_(m_dif.final_layer.linear.weight, std=0.02)
        nn.init.normal_(m_dif.final_layer.adaLN_modulation[-1].weight, std=0.02)
    m_dif.zero_grad(set_to_none=True)
    out_d = m_dif(x, t)
    assert out_d.shape == x.shape and torch.isfinite(out_d).all()
    out_d.pow(2).mean().backward()
    for i, blk in enumerate(m_dif.blocks):
        for nm in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            g = getattr(blk.attn, nm).grad
            assert g is not None and g.abs().sum().item() > 0, f"no grad, block {i} {nm}"
        g = blk.attn.subln.weight.grad
        assert g is not None and g.abs().sum().item() > 0, f"no grad, block {i} subln"
    print(f"  grad check:      lambda vectors and RMSNorm gains of all "
          f"{nl} blocks receive gradient")

    assert ckpt_attention({"model_state_dict": sd_s}) == "standard"
    assert ckpt_attention({"model_state_dict": sd_d}) == "differential"
    assert ckpt_attention({"ema_state_dict": sd_d, "attention": "differential"}) == "differential"
    assert ckpt_attention({}) == "standard"
    try:
        ckpt_attention({"model_state_dict": sd_s, "attention": "differential"})
        raise AssertionError("a field contradicting the weights should raise")
    except RuntimeError:
        pass
    print("  ckpt_attention:  standard / differential read off the weights; "
          "a contradicting field is refused")

    for bad in ("diff", "Differential", ""):
        try:
            ConditionedAudioDiT(**plain_kw, attention=bad)
            raise AssertionError(f"attention={bad!r} should have raised")
        except ValueError:
            pass
    try:
        DifferentialSelfAttention(96, 3, layer_idx=0)
        raise AssertionError("an odd number of heads should have raised")
    except ValueError as e:
        print(f"  odd head count rejected: {e}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  OK\n")