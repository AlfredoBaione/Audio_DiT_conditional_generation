# Tests for network_cond.py: conditioning paths, frame re-injection, differential attention,
# QK-norm.
# Run with `pytest test_functions` or `python test_functions/test_network_cond.py`.

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import latent_codec as lc
from conditions import GlobalConditionEncoder
from network_cond import (ConditionedAudioDiT, SelfAttention, TimestepEmbedder,
                          apply_rotary_pos_emb, rope_dim, QK_NORM_EPS)

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
    model2 = ConditionedAudioDiT(token_dim=TOKEN_DIM,
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

    _wake_up(model4_off)

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
        ConditionedAudioDiT(token_dim=TOKEN_DIM, kind='S', frame_cond_dims={"f0": 2},
                            frame_cond_out_dims={"f0": 16},
                            global_cond_configs={}, frame_reinject_every=-1)
        raise AssertionError("negative stride should have raised")
    except ValueError as e:
        print(f"  negative stride rejected: {e}")


def _differential_reference(att, xa, norm=None):
    hid, nh, hd = xa.shape[-1], att.n_heads, att.head_dim
    q1, k1, v, q2, k2 = att.qkv(xa).reshape(B, N, 5, nh, hd).permute(2, 0, 3, 1, 4)
    if norm:
        q1, q2 = norm(q1, att.q_norm.weight), norm(q2, att.q_norm.weight)
        k1, k2 = norm(k1, att.k_norm.weight), norm(k2, att.k_norm.weight)
    cs, sn = att._cos_sin(N, xa.device, xa.dtype)
    q1, k1 = apply_rotary_pos_emb(q1, cs, sn), apply_rotary_pos_emb(k1, cs, sn)
    q2, k2 = apply_rotary_pos_emb(q2, cs, sn), apply_rotary_pos_emb(k2, cs, sn)
    A1 = torch.softmax(q1 @ k1.transpose(-1, -2) / math.sqrt(hd), -1)
    A2 = torch.softmax(q2 @ k2.transpose(-1, -2) / math.sqrt(hd), -1)
    return att.proj(((A1 - A2) @ v).transpose(1, 2).reshape(B, N, hid)), A1, A2


def test_differential_attention():
    x, t = _inputs()
    plain_kw = dict(token_dim=TOKEN_DIM,
                    kind='S', frame_cond_dims={}, frame_cond_out_dims={},
                    global_cond_configs={})
    m_std = ConditionedAudioDiT(**plain_kw)
    m_dif = ConditionedAudioDiT(**plain_kw, attention="differential")
    hid, nl = 512, 6
    sd_s, sd_d = m_std.state_dict(), m_dif.state_dict()
    assert set(sd_s) == set(sd_d)
    n_s = sum(p.numel() for p in m_std.parameters())
    n_d = sum(p.numel() for p in m_dif.parameters())
    assert n_d - n_s == nl * 2 * hid * hid, (n_d - n_s)
    print(f"  parameters: {n_s} standard, {n_d} differential (+{n_d - n_s} = "
          f"{nl} blocks x 2 x {hid}^2)")

    _wake_up(m_dif)
    att = m_dif.blocks[3].attn
    xa = torch.randn(B, N, hid)
    with torch.no_grad():
        got = att(xa)
        ref, A1, A2 = _differential_reference(att, xa)
        dd = (got - ref).abs().max().item()
    print(f"  formula check:   max |module - explicit (A1 - A2) v| = {dd:.3e}")
    assert dd < 1e-5, dd
    assert (A1 - A2).sum(-1).abs().max().item() < 1e-5

    m_dif.zero_grad(set_to_none=True)
    out_d = m_dif(x, t)
    assert out_d.shape == x.shape and torch.isfinite(out_d).all()
    out_d.pow(2).mean().backward()
    for i, blk in enumerate(m_dif.blocks):
        g = blk.attn.qkv.weight.grad
        assert g is not None and g[3 * hid:].abs().sum().item() > 0, f"no grad, block {i}"
    print(f"  grad check:      q2, k2 of all {nl} blocks receive gradient")

    for bad in ("diff", "Differential", ""):
        try:
            ConditionedAudioDiT(**plain_kw, attention=bad)
            raise AssertionError(f"attention={bad!r} should have raised")
        except ValueError:
            pass


def test_block_modulation():
    torch.manual_seed(0)
    hid = 512
    m = ConditionedAudioDiT(token_dim=TOKEN_DIM, kind='S', frame_cond_dims={},
                            frame_cond_out_dims={}, global_cond_configs={}, qk_norm=True)
    blk = m.blocks[1]
    xa, mod = torch.randn(B, N, hid), torch.randn(B, 6 * hid)
    with torch.no_grad():
        assert torch.equal(blk(xa, mod), xa)
    _wake_up(m)
    with torch.no_grad():
        blk.norm1.weight.uniform_(0.5, 1.5)
        blk.norm2.weight.uniform_(0.5, 1.5)
        got = blk(xa, mod)
        s_a, b_a, g_a, s_f, b_f, g_f = (blk.adaLN_bias + mod).unsqueeze(1).chunk(6, dim=-1)
        h = xa + torch.sigmoid(1 - g_a) * blk.attn(_rms(xa, blk.norm1.weight, 1e-5) * (1 + s_a) + b_a)
        ref = h + torch.sigmoid(1 - g_f) * blk.mlp(_rms(h, blk.norm2.weight, 1e-5) * (1 + s_f) + b_f)
        dd = (got - ref).abs().max().item()
    print(f"  block is the identity at init; max |block - explicit| = {dd:.3e}")
    assert dd < 1e-5, dd
    assert blk.mlp.proj.weight.shape == (2 * 4 * hid, hid)
    assert blk.mlp.out.weight.shape == (hid, 4 * hid)
    assert m.adaLN[-1].out_features == 6 * hid and blk.adaLN_bias.shape == (6 * hid,)


def test_timestep_features():
    emb = TimestepEmbedder(64)
    assert emb.freqs.shape == (128,)
    assert math.isclose(emb.freqs[0].item(), 0.5, rel_tol=1e-5)
    assert math.isclose(emb.freqs[-1].item(), 10000.0, rel_tol=1e-5)
    t = torch.tensor([0.0, 0.25, 1.0])
    args = t[:, None] * emb.freqs * 2 * math.pi
    want = emb.mlp(torch.cat([args.cos(), args.sin()], dim=-1))
    assert torch.equal(emb(t), want)
    print("  features: 128 cos + 128 sin, cycles from 0.5 to 10000 on t in [0, 1]")
    low = TimestepEmbedder(64, "low_freq")
    radians = low.freqs * 2 * math.pi
    assert math.isclose(radians[0].item(), 1e-4, rel_tol=1e-5)
    assert math.isclose(radians[-1].item(), 1.0, rel_tol=1e-5)
    args = t[:, None] * low.freqs * 2 * math.pi
    assert torch.equal(low(t), low.mlp(torch.cat([args.cos(), args.sin()], dim=-1)))
    print("  low_freq: 2 pi f from 1e-4 to 1 radian on t in [0, 1]")


def test_adaln_per_block():
    torch.manual_seed(0)
    hid = 512
    x, t = _inputs()
    m = ConditionedAudioDiT(token_dim=TOKEN_DIM, kind='S', frame_cond_dims={},
                            frame_cond_out_dims={}, global_cond_configs={}, qk_norm=True,
                            memory_tokens=64, adaln="per_block")
    assert m.adaLN is None
    assert not hasattr(m, "output_proj") and not hasattr(m, "post_conv")
    blk, last = m.blocks[1], m.final_layer
    assert blk.adaLN_bias is None and blk.adaLN[-1].out_features == 6 * hid
    assert blk.attn.proj.weight.abs().sum() > 0 and blk.mlp.out.weight.abs().sum() > 0
    xa, c = torch.randn(B, N, hid), torch.randn(B, hid)
    with torch.no_grad():
        assert torch.equal(blk(xa, c), xa)
        assert torch.equal(m(x, t), torch.zeros_like(x))
        for w in (blk.adaLN[-1].weight, last.adaLN[-1].weight, last.linear.weight):
            nn.init.normal_(w, std=0.02)
        got = blk(xa, c)
        s_a, b_a, g_a, s_f, b_f, g_f = blk.adaLN(c).unsqueeze(1).chunk(6, dim=-1)
        h = xa + g_a * blk.attn(_rms(xa, blk.norm1.weight, 1e-5) * (1 + s_a) + b_a)
        ref = h + g_f * blk.mlp(_rms(h, blk.norm2.weight, 1e-5) * (1 + s_f) + b_f)
        assert (got - ref).abs().max().item() < 1e-5
        s, b = last.adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        norm = ((xa - xa.mean(-1, keepdim=True))
                / (xa.var(-1, unbiased=False, keepdim=True) + 1e-6).sqrt())
        assert (last(xa, c) - last.linear(norm * (1 + s) + b)).abs().max().item() < 1e-5
    m.zero_grad(set_to_none=True)
    m(x, t).pow(2).mean().backward()
    for nm, p_ in (("block", blk.adaLN[-1].weight), ("final AdaLN", last.adaLN[-1].weight),
                   ("final Linear", last.linear.weight)):
        assert p_.grad is not None and p_.grad.abs().sum().item() > 0, nm
    print("  blocks start as the identity, the model output starts at zero")
    for bad in ({"adaln": "bad"}, {"t_features": "bad"}):
        try:
            ConditionedAudioDiT(token_dim=TOKEN_DIM, kind='S', **bad)
            raise AssertionError(f"{bad} should have raised")
        except ValueError:
            pass


def test_input_output_and_memory_tokens():
    torch.manual_seed(0)
    x, t = _inputs()
    fc = {"f0": torch.randn(B, N, 2)}
    ctx = torch.randn(B, N, TOKEN_DIM + 1)
    kw = dict(token_dim=TOKEN_DIM, kind='S', frame_cond_dims={"f0": 2},
              frame_cond_out_dims={"f0": 16}, global_cond_configs={},
              frame_reinject_every=1, inpainting=True)
    off = ConditionedAudioDiT(**kw)
    assert off.memory_tokens is None and off.n_memory == 0
    m = ConditionedAudioDiT(**kw, memory_tokens=64)
    assert m.memory_tokens.shape == (64, 512)
    assert m.input_proj.bias is None and m.output_proj.bias is None
    assert m.pre_conv.weight.abs().sum() == 0 and m.post_conv.weight.abs().sum() == 0
    assert m.pre_conv.in_channels == TOKEN_DIM + 16
    with torch.no_grad():
        out = m(x, t, frame_conditions=fc, inpaint_context=ctx)
    assert out.shape == x.shape and out.abs().max().item() > 0

    _wake_up(m)
    with torch.no_grad():
        for lin in m.frame_reinject.values():
            nn.init.normal_(lin.weight, std=0.02)
        for blk in m.blocks:
            nn.init.normal_(blk.inpaint_embed[-1].weight, std=0.02)
    m.zero_grad(set_to_none=True)
    m(x, t, frame_conditions=fc, inpaint_context=ctx).pow(2).mean().backward()
    for nm, p_ in (("memory_tokens", m.memory_tokens), ("pre_conv", m.pre_conv.weight),
                   ("post_conv", m.post_conv.weight)):
        assert p_.grad is not None and p_.grad.abs().sum().item() > 0, nm
    print("  64 memory tokens, re-injection and context skip them; "
          "pre_conv, post_conv and tokens receive gradient")


def test_global_encoder_null_is_zero():
    enc = GlobalConditionEncoder({"text": {"dim": 512}, "image": {"dim": 768}}, 64)
    zeros = {"text": torch.zeros(B, 512), "image": torch.zeros(B, 768)}
    assert torch.equal(enc(zeros), torch.zeros(B, 64))
    full = {"text": torch.randn(B, 512), "image": torch.randn(B, 768)}
    assert enc(full).abs().max().item() > 0
    assert sum(p.numel() for p in enc.parameters()) == 1280 * 64 + 64 * 64


def test_rope_partial():
    torch.manual_seed(0)
    for kind in ("S", "B", "G", "L", "XL"):
        cfg = ConditionedAudioDiT.CONFIGS[kind]
        assert cfg["hidden_size"] % cfg["n_heads"] == 0, kind
        assert cfg["hidden_size"] // cfg["n_heads"] == 64, kind
    att = SelfAttention(512, 8)
    r = rope_dim(att.head_dim)
    assert r == 32 and att.inv_freq.shape == (16,)
    cs, sn = att._cos_sin(N, torch.device("cpu"), torch.float32)
    x = torch.randn(B, 8, N, 64)
    y = apply_rotary_pos_emb(x, cs, sn)
    assert torch.equal(y[..., r:], x[..., r:])
    assert (y[:, :, 1:, :r] - x[:, :, 1:, :r]).abs().max().item() > 1e-3
    assert torch.allclose(y[:, :, 0, :r], x[:, :, 0, :r])
    print("  every size has heads of 64; RoPE rotates 32 values, the other 32 are untouched")


def _rms(x, w, eps=QK_NORM_EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _wake_up(model):
    with torch.no_grad():
        for blk in model.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02)
            nn.init.normal_(blk.mlp.out.weight, std=0.02)
            if blk.cross_attn is not None:
                nn.init.normal_(blk.cross_attn.proj.weight, std=0.02)


def test_qk_norm_weights_and_checkpoints():
    plain_kw = dict(token_dim=TOKEN_DIM,
                    kind='S', frame_cond_dims={}, frame_cond_out_dims={},
                    global_cond_configs={})
    nl, hd = 6, 512 // 8
    for attention in ("standard", "differential"):
        m_off = ConditionedAudioDiT(**plain_kw, attention=attention)
        m_on  = ConditionedAudioDiT(**plain_kw, attention=attention, qk_norm=True)
        assert not m_off.qk_norm and m_on.qk_norm
        sd_off, sd_on = m_off.state_dict(), m_on.state_dict()
        extra = sorted(set(sd_on) - set(sd_off))
        assert set(sd_off) <= set(sd_on)
        assert len(extra) == 2 * nl and all(
            k.endswith(".attn.q_norm.weight") or k.endswith(".attn.k_norm.weight")
            for k in extra), extra
        assert all(sd_on[k].shape == (hd,) and torch.all(sd_on[k] == 1)
                   for k in extra)
        print(f"  {attention}: +{len(extra)} gain vectors of {hd} "
              f"(q_norm/k_norm in {nl} blocks), initialised to 1")

        m_off.load_state_dict(sd_off)
        missing, unexpected = m_on.load_state_dict(sd_off, strict=False)
        assert unexpected == [] and sorted(missing) == extra
        try:
            m_on.load_state_dict(sd_off)
            raise AssertionError("an old state_dict must not load strictly "
                                 "into a QK-norm model")
        except RuntimeError:
            pass

    assert not ConditionedAudioDiT(**plain_kw).qk_norm


def test_qk_norm_self_attention():
    torch.manual_seed(0)
    plain_kw = dict(token_dim=TOKEN_DIM,
                    kind='S', frame_cond_dims={}, frame_cond_out_dims={},
                    global_cond_configs={})
    hid, nh = 512, 8
    hd = hid // nh
    m_on = ConditionedAudioDiT(**plain_kw, qk_norm=True)
    _wake_up(m_on)
    att = m_on.blocks[2].attn
    with torch.no_grad():
        att.q_norm.weight.uniform_(0.5, 1.5)
        att.k_norm.weight.uniform_(0.5, 1.5)
    xa = torch.randn(B, N, hid)
    with torch.no_grad():
        got = att(xa)
        q_, k_, v_ = att.qkv(xa).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q_, k_ = _rms(q_, att.q_norm.weight), _rms(k_, att.k_norm.weight)
        cs, sn = att._cos_sin(N, xa.device, xa.dtype)
        q_, k_ = apply_rotary_pos_emb(q_, cs, sn), apply_rotary_pos_emb(k_, cs, sn)
        A = torch.softmax(q_ @ k_.transpose(-1, -2) / math.sqrt(hd), -1)
        ref = att.proj((A @ v_).transpose(1, 2).reshape(B, N, hid))
        dd = (got - ref).abs().max().item()
    print(f"  formula check:   max |module - explicit RMSNorm(q), RMSNorm(k)| = {dd:.3e}")
    assert dd < 1e-5, dd

    m_off = ConditionedAudioDiT(**plain_kw)
    _wake_up(m_off)
    att_off = m_off.blocks[2].attn
    with torch.no_grad():
        for a in (att, att_off):
            base = a(xa)
            a.qkv.weight[:2 * hid] *= 50.0
            moved = (a(xa) - base).abs().max().item()
            a.qkv.weight[:2 * hid] /= 50.0
            if a is att:
                assert moved < 1e-3, moved
                print(f"  q/k weights x50: output moves {moved:.2e} with QK-norm")
            else:
                assert moved > 1e-2, moved
                print(f"  q/k weights x50: output moves {moved:.2e} without")

    x, t = _inputs()
    m_on.zero_grad(set_to_none=True)
    out = m_on(x, t)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.pow(2).mean().backward()
    for i, blk in enumerate(m_on.blocks):
        for nm in ("q_norm", "k_norm"):
            g = getattr(blk.attn, nm).weight.grad
            assert g is not None and g.abs().sum().item() > 0, f"no grad, block {i} {nm}"
    print(f"  grad check:      q_norm/k_norm gains of all {len(m_on.blocks)} "
          f"blocks receive gradient")


def test_qk_norm_differential_attention():
    torch.manual_seed(0)
    plain_kw = dict(token_dim=TOKEN_DIM,
                    kind='S', frame_cond_dims={}, frame_cond_out_dims={},
                    global_cond_configs={})
    hid = 512
    m = ConditionedAudioDiT(**plain_kw, attention="differential", qk_norm=True)
    _wake_up(m)
    att = m.blocks[3].attn
    with torch.no_grad():
        att.q_norm.weight.uniform_(0.5, 1.5)
        att.k_norm.weight.uniform_(0.5, 1.5)
    xa = torch.randn(B, N, hid)
    with torch.no_grad():
        got = att(xa)
        ref = _differential_reference(att, xa, norm=_rms)[0]
        dd = (got - ref).abs().max().item()
    print(f"  formula check:   max |module - explicit, both maps normalised| = {dd:.3e}")
    assert dd < 1e-5, dd

    x, t = _inputs()
    m.zero_grad(set_to_none=True)
    m(x, t).pow(2).mean().backward()
    for i, blk in enumerate(m.blocks):
        for nm in ("q_norm", "k_norm"):
            g = getattr(blk.attn, nm).weight.grad
            assert g is not None and g.abs().sum().item() > 0, f"no grad, block {i} {nm}"
    print(f"  grad check:      q_norm/k_norm gains of all {len(m.blocks)} blocks "
          f"receive gradient")


def test_qk_norm_cross_attention():
    torch.manual_seed(0)
    kw = dict(token_dim=TOKEN_DIM, kind='S', frame_cond_dims={},
              frame_cond_out_dims={}, global_cond_configs={"text": {"dim": 512}},
              text_conditioning="cross_att", text_ctx_dim=768)
    m_off = ConditionedAudioDiT(**kw)
    m_on  = ConditionedAudioDiT(**kw, qk_norm=True)
    extra = sorted(set(m_on.state_dict()) - set(m_off.state_dict()))
    n_cross = len(m_on.blocks)
    assert len(extra) == 2 * (6 + n_cross), extra
    assert sum(".cross_attn." in k for k in extra) == 2 * n_cross

    hid, nh, L = 512, 8, 5
    hd = hid // nh
    ca = m_on.blocks[0].cross_attn
    assert ca.q.bias is None and ca.kv.bias is None and ca.proj.bias is None
    with torch.no_grad():
        nn.init.normal_(ca.proj.weight, std=0.02)
        ca.q_norm.weight.uniform_(0.5, 1.5)
        ca.k_norm.weight.uniform_(0.5, 1.5)
        xa, ctx = torch.randn(B, N, hid), torch.randn(B, L, hid)
        got = ca(xa, ctx)
        q_ = ca.q(xa).view(B, N, nh, hd).transpose(1, 2)
        kv = ca.kv(ctx).view(B, L, 2, nh, hd)
        k_, v_ = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        def _attend(q, k):
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), -1)
            return ca.proj((A @ v_).transpose(1, 2).reshape(B, N, hid))

        ref = _attend(_rms(q_, ca.q_norm.weight), _rms(k_, ca.k_norm.weight))
        dd = (got - ref).abs().max().item()
        d_raw = (got - _attend(q_, k_)).abs().max().item()
    print(f"  formula check:   max |cross-attention - explicit| = {dd:.3e} "
          f"(vs {d_raw:.3e} without the norm)")
    assert dd < 1e-5, dd
    assert d_raw > 1e-3, d_raw

    x, t = _inputs()
    _wake_up(m_on)
    m_on.zero_grad(set_to_none=True)
    out = m_on(x, t, text_context=torch.randn(B, L, 768),
               text_context_mask=torch.ones(B, L, dtype=torch.bool))
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.pow(2).mean().backward()
    for i, blk in enumerate(m_on.blocks):
        for nm in ("q_norm", "k_norm"):
            g = getattr(blk.cross_attn, nm).weight.grad
            assert g is not None and g.abs().sum().item() > 0, f"no grad, cross {i} {nm}"
    print(f"  {n_cross} cross-attention layers carry q_norm/k_norm and receive gradient")


def test_text_conditioning_modes():
    torch.manual_seed(0)
    x, t = _inputs()
    L = 5
    vec, tok = torch.randn(B, 512), torch.randn(B, L, 768)
    mask = torch.ones(B, L, dtype=torch.bool)
    kw = dict(token_dim=TOKEN_DIM, kind='S', frame_cond_dims={}, frame_cond_out_dims={},
              global_cond_configs={"text": {"dim": 512}}, text_ctx_dim=768)
    ada = ConditionedAudioDiT(**kw, text_conditioning="ada")
    cross = ConditionedAudioDiT(**kw, text_conditioning="cross_att")
    both = ConditionedAudioDiT(**kw, text_conditioning="all")
    assert not ada.text_in_cross and ada.has_global
    assert all(b.cross_attn is None for b in ada.blocks) and not hasattr(ada, "text_proj")
    assert cross.text_in_cross and not cross.has_global
    assert all(b.cross_attn is not None for b in cross.blocks)
    assert both.text_in_cross and both.has_global
    _wake_up(cross)
    _wake_up(both)
    with torch.no_grad():
        base = cross(x, t, text_context=tok, text_context_mask=mask)
        assert torch.equal(base, cross(x, t, global_conditions={"text": vec},
                                       text_context=tok, text_context_mask=mask))
        assert not torch.equal(base, cross(x, t, text_context=torch.randn_like(tok),
                                           text_context_mask=mask))
        assert not torch.equal(both(x, t, global_conditions={"text": vec},
                                    text_context=tok, text_context_mask=mask),
                               both(x, t, global_conditions={"text": torch.randn_like(vec)},
                                    text_context=tok, text_context_mask=mask))
        none = cross(x, t)
        empty = cross(x, t, text_context=tok, text_context_mask=torch.zeros_like(mask))
        assert torch.equal(none, empty)
        part = mask.clone()
        part[:, 3:] = False
        tok2 = tok.clone()
        tok2[:, 3:] = torch.randn(B, L - 3, 768)
        assert torch.equal(cross(x, t, text_context=tok, text_context_mask=part),
                           cross(x, t, text_context=tok2, text_context_mask=part))
    print("  ada: vector only; cross_att: tokens only; all: both; no tokens = zero "
          "contribution; padded positions read the learned padding")

    d = ConditionedAudioDiT(**kw, text_conditioning="cross_att", attention="differential")
    ca = d.blocks[0].cross_attn
    assert ca.q.out_features == 2 * 512 and ca.kv.out_features == 3 * 512
    _wake_up(d)
    with torch.no_grad():
        assert d(x, t, text_context=tok, text_context_mask=mask).shape == x.shape

    for bad in ("cross", "", "ADA"):
        try:
            ConditionedAudioDiT(**kw, text_conditioning=bad)
            raise AssertionError(f"text_conditioning={bad!r} should have raised")
        except ValueError:
            pass
    try:
        ConditionedAudioDiT(token_dim=TOKEN_DIM, kind='S', frame_cond_dims={},
                            frame_cond_out_dims={}, global_cond_configs={"text": {"dim": 512}},
                            text_conditioning="all")
        raise AssertionError("tokens without their width should have raised")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  OK\n")