# The conditioned DiT.

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_dataset_npy import DAC_LATENT_DIM, MAX_FRAMES
from conditions import FrameConditionEncoder, GlobalConditionEncoder


TOKEN_DIM = DAC_LATENT_DIM

ATTENTION_KINDS = ("standard", "differential")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_default_rope_parameters(head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, max_seq_len: int = 4096,
                 theta: float = 10000.0):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_size // n_heads

        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        inv_freq = compute_default_rope_parameters(self.head_dim, theta=theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, S: int, device, dtype):
        t = torch.arange(S, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        cos, sin = self._cos_sin(S, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, S, -1)
        return self.proj(x)


def diff_lambda_init(layer_idx: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class DifferentialSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, layer_idx: int,
                 max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        assert hidden_size % n_heads == 0
        if n_heads % 2 != 0:
            raise ValueError(
                f"the differential attention pairs the heads two by two: "
                f"n_heads must be even, got {n_heads}")
        self.n_heads   = n_heads
        self.n_diff    = n_heads // 2
        self.head_dim  = hidden_size // n_heads
        self.layer_idx = int(layer_idx)
        self.lambda_init = diff_lambda_init(self.layer_idx)

        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        inv_freq = compute_default_rope_parameters(self.head_dim, theta=theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        d = self.head_dim
        self.lambda_q1 = nn.Parameter(torch.zeros(d).normal_(mean=0.0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(d).normal_(mean=0.0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(d).normal_(mean=0.0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(d).normal_(mean=0.0, std=0.1))
        self.subln = nn.RMSNorm(2 * d, eps=1e-5, elementwise_affine=True)

    def _cos_sin(self, S: int, device, dtype):
        t = torch.arange(S, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

    def lambda_full(self) -> torch.Tensor:
        l1 = torch.exp(torch.sum(self.lambda_q1.float() * self.lambda_k1.float()))
        l2 = torch.exp(torch.sum(self.lambda_q2.float() * self.lambda_k2.float()))
        return l1 - l2 + self.lambda_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        d, h = self.head_dim, self.n_diff
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.reshape(B, S, 2 * h, d).transpose(1, 2)
        k = k.reshape(B, S, 2 * h, d).transpose(1, 2)
        v = v.reshape(B, S, h, 2 * d).transpose(1, 2)

        cos, sin = self._cos_sin(S, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        o1 = F.scaled_dot_product_attention(q[:, 0::2], k[:, 0::2], v)
        o2 = F.scaled_dot_product_attention(q[:, 1::2], k[:, 1::2], v)
        o = o1 - self.lambda_full().to(o1.dtype) * o2
        with torch.autocast(device_type=o.device.type, enabled=False):
            o = self.subln(o.float())
        o = o.to(o1.dtype) * (1.0 - self.lambda_init)
        o = o.transpose(1, 2).reshape(B, S, C)
        return self.proj(o)


class FFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0,
                 multiple_of: int = 256, dropout: float = 0.0):
        super().__init__()
        inner = int(2 * (mlp_ratio * hidden_size) / 3)
        inner = multiple_of * ((inner + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(hidden_size, inner, bias=False)
        self.w3 = nn.Linear(hidden_size, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden_size, bias=False)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w3(x)
        h = self.drop1(h)
        h = self.w2(h)
        h = self.drop2(h)
        return h


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, ctx_dim: int):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.kv = nn.Linear(ctx_dim, 2 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx.to(x.dtype)).view(B, L, 2, self.n_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask.view(B, 1, 1, L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len=4096,
                 mlp_ratio=4.0, drop=0.0, cross_attn_ctx_dim=0,
                 attention="standard", layer_idx=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attention == "standard":
            self.attn  = SelfAttention(hidden_size, num_heads, max_seq_len=max_seq_len)
        elif attention == "differential":
            self.attn  = DifferentialSelfAttention(hidden_size, num_heads,
                                                   layer_idx=layer_idx,
                                                   max_seq_len=max_seq_len)
        else:
            raise ValueError(
                f"attention must be one of {ATTENTION_KINDS}, got {attention!r}")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = FFN(hidden_size, mlp_ratio=mlp_ratio, dropout=drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.norm_cross = None
        self.cross_attn = None
        if int(cross_attn_ctx_dim or 0) > 0:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False,
                                           eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads,
                                             int(cross_attn_ctx_dim))

    def forward(self, x, c, ctx=None, ctx_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.cross_attn is not None:
            if ctx is None:
                raise ValueError(
                    "this DiTBlock has a cross-attention but no text context "
                    "reached it; ConditionedAudioDiT.forward must pass one "
                    "(its learned null token when there is no text)")
            x = x + self.cross_attn(self.norm_cross(x), ctx, ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConditionedAudioDiT(nn.Module):
    CONFIGS = {
        'S':  dict(n_layers=6,  hidden_size=512,  n_heads=8),
        'B':  dict(n_layers=12, hidden_size=768,  n_heads=12),
        'G':  dict(n_layers=18, hidden_size=1024, n_heads=16),
        'L':  dict(n_layers=24, hidden_size=1024, n_heads=16),
        'XL': dict(n_layers=28, hidden_size=1152, n_heads=16),
    }

    def __init__(
        self,
        token_dim:           int   = TOKEN_DIM,
        max_seq_len:         int   = MAX_FRAMES + 16,
        kind:                str   = 'L',
        mlp_ratio:           float = 4.0,
        drop:                float = 0.0,
        frame_cond_dims:     Optional[Dict[str, int]]  = None,
        frame_cond_out_dims: Optional[Dict[str, int]]  = None,
        global_cond_configs: Optional[Dict[str, dict]] = None,
        frame_reinject_every: int = 0,
        text_cross_every:     int = 0,
        text_ctx_dim:         int = 0,
        attention:            str = "standard",
    ):
        super().__init__()
        cfg = self.CONFIGS[kind]
        self.kind        = kind
        self.token_dim   = token_dim
        self.max_seq_len = max_seq_len
        hidden_size      = cfg['hidden_size']
        n_layers         = cfg['n_layers']
        n_heads          = cfg['n_heads']
        if attention not in ATTENTION_KINDS:
            raise ValueError(
                f"attention must be one of {ATTENTION_KINDS}, got {attention!r}")
        self.attention   = attention

        self.frame_cond_dims     = dict(frame_cond_dims)     if frame_cond_dims     else {}
        self.frame_cond_out_dims = dict(frame_cond_out_dims) if frame_cond_out_dims else {}
        self.global_cond_configs = dict(global_cond_configs) if global_cond_configs else {}
        self.has_frame  = len(self.frame_cond_dims)     > 0
        self.has_global = len(self.global_cond_configs) > 0

        if self.has_frame:
            missing = set(self.frame_cond_dims) - set(self.frame_cond_out_dims)
            if missing:
                raise ValueError(
                    f"frame_cond_out_dims is missing the out_dim for {sorted(missing)}. "
                    f"Every frame condition must declare both raw_dim and out_dim."
                )

        frame_extra = 0
        if self.has_frame:
            self.frame_encoder = FrameConditionEncoder(
                self.frame_cond_dims, self.frame_cond_out_dims,
            )
            frame_extra = self.frame_encoder.total_out_dim

        self.input_proj = nn.Linear(token_dim + frame_extra, hidden_size, bias=True)

        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.has_global:
            self.global_encoder = GlobalConditionEncoder(
                self.global_cond_configs, hidden_size,
            )

        self.text_cross_every = int(text_cross_every or 0)
        if self.text_cross_every < 0:
            raise ValueError(
                f"text_cross_every must be >= 0 (0 = off), got {text_cross_every}.")
        self.text_ctx_dim = int(text_ctx_dim or 0)
        self.text_cross_layers = []
        if self.text_cross_every > 0 and "text" in self.global_cond_configs:
            if self.text_ctx_dim <= 0:
                raise ValueError(
                    "text_cross_every > 0 needs text_ctx_dim: the width of ONE "
                    "token state of the text encoder (768 for CLAP "
                    "clap-htsat-unfused). It is not the pooled dim (512) -- see "
                    "CLAPTextCondition.encode_tokens.")
            self.text_cross_layers = [
                i for i in range(n_layers) if i % self.text_cross_every == 0
            ]

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, max_seq_len=max_seq_len,
                     mlp_ratio=mlp_ratio, drop=drop,
                     cross_attn_ctx_dim=(self.text_ctx_dim
                                         if i in self.text_cross_layers else 0),
                     attention=attention, layer_idx=i)
            for i in range(n_layers)
        ])

        self.text_null = None
        if self.text_cross_layers:
            self.text_null = nn.Parameter(torch.randn(1, 1, self.text_ctx_dim) * 0.02)

        self.frame_reinject_every = int(frame_reinject_every or 0)
        if self.frame_reinject_every < 0:
            raise ValueError(
                f"frame_reinject_every must be >= 0 (0 = off), got "
                f"{frame_reinject_every}."
            )
        self.reinject_layers = []
        if self.frame_reinject_every > 0 and self.has_frame:
            self.reinject_layers = [
                i for i in range(1, n_layers)
                if i % self.frame_reinject_every == 0
            ]
        if self.reinject_layers:
            self.frame_reinject = nn.ModuleDict({
                str(i): nn.Linear(frame_extra, hidden_size, bias=False)
                for i in self.reinject_layers
            })

            self.frame_reinject_gate = nn.ModuleDict({
                str(i): nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, len(self.frame_encoder.names)),
                )
                for i in self.reinject_layers
            })

            self.register_buffer(
                "frame_slot_repeats",
                torch.tensor(
                    [self.frame_cond_out_dims[n]
                     for n in self.frame_encoder.names],
                    dtype=torch.long,
                ),
                persistent=False,
            )

        self.final_layer = FinalLayer(hidden_size, token_dim)

        self.initialize_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ConditionedAudioDiT-{kind}] {n_params/1e6:.1f}M params | "
              f"hidden={hidden_size} | layers={n_layers} | heads={n_heads}")
        print(f"  Frame conditions (concat): "
              f"{self.frame_cond_dims if self.has_frame else 'NONE'}"
              + (f" -> out {self.frame_cond_out_dims} "
                 f"(+{frame_extra} input channels)" if self.has_frame else ""))
        print(f"  Global conditions (AdaLN): "
              f"{list(self.global_cond_configs.keys()) if self.has_global else 'NONE'}")

        if self.reinject_layers:
            n_re = len(self.reinject_layers)
            n_fc = len(self.frame_encoder.names)
            re_params = n_re * frame_extra * hidden_size
            gate_params = n_re * (hidden_size + 1) * n_fc
            print(f"  Frame re-injection: every {self.frame_reinject_every} "
                  f"block(s) -> {n_re} of {n_layers} blocks "
                  f"{self.reinject_layers} | zero-init, bias-free | "
                  f"gate on c, {n_fc} per block, identity-init | "
                  f"+{(re_params + gate_params)/1e6:.2f}M params")
        elif self.frame_reinject_every > 0 and not self.has_frame:
            print(f"  Frame re-injection: REQUESTED (every "
                  f"{self.frame_reinject_every}) but INACTIVE -- this model has "
                  f"no frame conditions to re-inject.")
        elif self.frame_reinject_every > n_layers - 1 and self.has_frame:
            print(f"  Frame re-injection: REQUESTED (every "
                  f"{self.frame_reinject_every}) but INACTIVE -- the stride "
                  f"exceeds the {n_layers - 1} eligible blocks (1..{n_layers-1}).")
        else:
            print(f"  Frame re-injection: OFF (input concat only, plain JASCO)")

        if self.text_cross_layers:
            n_x = len(self.text_cross_layers)
            x_params = n_x * ((hidden_size * hidden_size + hidden_size)
                              + (self.text_ctx_dim * 2 * hidden_size + 2 * hidden_size)
                              + (hidden_size * hidden_size + hidden_size))
            print(f"  Text cross-attention: every {self.text_cross_every} "
                  f"block(s) -> {n_x} of {n_layers} blocks | "
                  f"ctx_dim={self.text_ctx_dim} | W_o zero-init | "
                  f"learned null token | +{x_params/1e6:.2f}M params")
        elif self.text_cross_every > 0 and "text" not in self.global_cond_configs:
            print(f"  Text cross-attention: REQUESTED (every "
                  f"{self.text_cross_every}) but INACTIVE -- this model has no "
                  f"'text' global condition to attend to.")
        else:
            print(f"  Text cross-attention: OFF (the text reaches the blocks "
                  f"only as a pooled vector through AdaLN)")

        _d = hidden_size // n_heads
        if attention == "differential":
            print(f"  Self-attention: DIFFERENTIAL (Ye et al. 2024) | "
                  f"{n_heads // 2} heads, q/k {_d} x 2 maps, v {2 * _d} | "
                  f"lambda_init {diff_lambda_init(0):.2f} (block 0) -> "
                  f"{diff_lambda_init(n_layers - 1):.2f} (block {n_layers - 1}) | "
                  f"per-head RMSNorm | +{n_layers * 6 * _d} params")
        else:
            print(f"  Self-attention: standard ({n_heads} heads x {_d})")

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.reinject_layers:
            for lin in self.frame_reinject.values():
                nn.init.constant_(lin.weight, 0)

            for gate in self.frame_reinject_gate.values():
                nn.init.constant_(gate[-1].weight, 0)
                nn.init.constant_(gate[-1].bias, 1.0)

        for block in self.blocks:
            if block.cross_attn is not None:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

    def _gather_frame_conditions(
        self,
        frame_conditions: Optional[Dict[str, torch.Tensor]],
        B: int, T: int, device, dtype,
    ) -> Dict[str, torch.Tensor]:
        out = {}
        fc = frame_conditions or {}
        for name, raw_dim in self.frame_cond_dims.items():
            c = fc.get(name, None)
            if c is None:
                c = torch.zeros(B, T, raw_dim, device=device, dtype=dtype)
            else:
                c = c.to(device=device, dtype=dtype)
            out[name] = c
        return out

    def _gather_global_conditions(
        self,
        global_conditions: Optional[Dict[str, torch.Tensor]],
        B: int, device, dtype,
    ) -> Dict[str, torch.Tensor]:
        out = {}
        gc = global_conditions or {}
        for name, cfg in self.global_cond_configs.items():
            c = gc.get(name, None)
            if c is None:
                c = torch.zeros(B, int(cfg["dim"]), device=device, dtype=dtype)
            else:
                c = c.to(device=device, dtype=dtype)
            out[name] = c
        return out

    def _gather_text_context(self, ctx, mask, B, device):
        null = self.text_null.to(device=device, dtype=torch.float32)
        if ctx is None:
            return (null.expand(B, 1, -1),
                    torch.ones(B, 1, dtype=torch.bool, device=device))
        ctx = ctx.to(device=device, dtype=torch.float32)
        if ctx.dim() != 3 or ctx.shape[0] != B:
            raise ValueError(
                f"text_context must be (B, L, {self.text_ctx_dim}) with B={B}, "
                f"got {tuple(ctx.shape)}")
        if ctx.shape[2] != self.text_ctx_dim:
            raise ValueError(
                f"text_context has width {ctx.shape[2]}, this model was built "
                f"for {self.text_ctx_dim}. The context is the TOKEN-level state "
                f"of the text encoder (768 for CLAP), not the pooled embedding "
                f"(512) that goes into the AdaLN slot.")
        if mask is None:
            mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device).bool()
            if mask.shape != ctx.shape[:2]:
                raise ValueError(
                    f"text_context_mask {tuple(mask.shape)} does not match the "
                    f"context {tuple(ctx.shape[:2])}")
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            ctx, mask = ctx.clone(), mask.clone()
            ctx[empty] = 0.0
            ctx[empty, 0] = null[0, 0]
            mask[empty] = False
            mask[empty, 0] = True
        return ctx, mask

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        frame_conditions:  Optional[Dict[str, torch.Tensor]] = None,
        global_conditions: Optional[Dict[str, torch.Tensor]] = None,
        text_context:      Optional[torch.Tensor] = None,
        text_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()
        B, T, _ = x.shape

        frame_proj = None
        if self.has_frame:
            fc = self._gather_frame_conditions(frame_conditions, B, T, x.device, x.dtype)
            frame_proj = self.frame_encoder(fc)
            x = torch.cat([x, frame_proj], dim=-1)

        x = self.input_proj(x)

        c = self.t_embedder(t)
        if self.has_global:
            gc = self._gather_global_conditions(global_conditions, B,
                                                x.device, torch.float32)
            g = self.global_encoder(gc)
            if g is not None:
                c = c + g

        ctx = ctx_mask = None
        if self.text_cross_layers:
            ctx, ctx_mask = self._gather_text_context(
                text_context, text_context_mask, B, x.device)

        reinject = self.frame_reinject if (self.reinject_layers and
                                           frame_proj is not None) else None
        for i, block in enumerate(self.blocks):
            if reinject is not None and str(i) in reinject:
                gate = self.frame_reinject_gate[str(i)](c)
                gate = torch.repeat_interleave(
                    gate, self.frame_slot_repeats, dim=-1).unsqueeze(1)
                x = x + reinject[str(i)](frame_proj * gate)
            x = block(x, c, ctx, ctx_mask)

        x = self.final_layer(x, c)
        return x


def ckpt_frame_reinject_every(ckpt: dict) -> int:
    v = ckpt.get("frame_reinject_every", None)
    if v is None:
        cfg = ckpt.get("config", None)
        if isinstance(cfg, dict):
            v = (cfg.get("model", None) or {}).get("frame_reinject_every", None)
    return int(v or 0)


def ckpt_text_cross_every(ckpt: dict) -> int:
    v = ckpt.get("text_cross_every", None)
    if v is None:
        cfg = ckpt.get("config", None)
        if isinstance(cfg, dict):
            v = (cfg.get("model", None) or {}).get("text_cross_every", None)
    return int(v or 0)


def ckpt_text_ctx_dim(ckpt: dict) -> int:
    sd = ckpt.get("model_state_dict", None)
    if not isinstance(sd, dict) or not sd:
        sd = ckpt.get("ema_state_dict", None)
    if not isinstance(sd, dict):
        return 0
    for k, v in sd.items():
        if k.endswith("cross_attn.kv.weight") and hasattr(v, "shape"):
            return int(v.shape[1])
    return 0


def ckpt_attention(ckpt: dict) -> str:
    sd = ckpt.get("model_state_dict", None)
    if not isinstance(sd, dict) or not sd:
        sd = ckpt.get("ema_state_dict", None)
    from_weights = None
    if isinstance(sd, dict) and sd:
        from_weights = ("differential"
                        if any(k.endswith("attn.lambda_q1") for k in sd)
                        else "standard")
    field = ckpt.get("attention", None)
    if field is None:
        cfg = ckpt.get("config", None)
        if isinstance(cfg, dict):
            field = (cfg.get("model", None) or {}).get("attention", None)
    if from_weights is None:
        return str(field or "standard")
    if field is not None and str(field) != from_weights:
        raise RuntimeError(
            f"this checkpoint says attention={field!r} but its weights are "
            f"those of the {from_weights!r} attention "
            f"({'with' if from_weights == 'differential' else 'without'} "
            f"the lambda vectors). The weights are what gets loaded; the field "
            f"is wrong -- the checkpoint was edited or assembled by hand.")
    return from_weights


def check_ckpt_reinject_gate(ckpt: dict, where: str = "this checkpoint") -> None:
    if ckpt_frame_reinject_every(ckpt) <= 0:
        return
    sd = ckpt.get("model_state_dict", None)
    if not isinstance(sd, dict):
        sd = ckpt.get("ema_state_dict", None)
    if not isinstance(sd, dict) or not sd:
        return
    if any(k.startswith("frame_reinject_gate.") for k in sd):
        return
    if not any(k.startswith("frame_reinject.") for k in sd):
        return
    raise RuntimeError(
        f"{where} was trained with model.frame_reinject_every="
        f"{ckpt_frame_reinject_every(ckpt)} BEFORE the per-condition gate on "
        f"the re-injection existed, so it has the per-block projections but "
        f"none of the `frame_reinject_gate.*` weights this code builds. It "
        f"cannot be loaded or resumed as-is: the two are different "
        f"architectures that happen to agree on every other field.\n"
        f"  - to keep using that checkpoint, check out the code from before "
        f"the gate (network_cond.py.pre-reinject-gate.bak);\n"
        f"  - to use this code, start a NEW run -- a gate initialised to 1 "
        f"makes step 0 identical to the ungated model, so nothing is lost "
        f"except the steps already spent."
    )


if __name__ == "__main__":
    B, N = 2, 430

    x = torch.randn(B, N, TOKEN_DIM)
    t = torch.rand(B)

    print("=== Frame: f0 only ===")
    model = ConditionedAudioDiT(
        kind='S',
        frame_cond_dims={"f0": 2},
        frame_cond_out_dims={"f0": 16},
        global_cond_configs={},
    )
    out = model(x, t, frame_conditions={"f0": torch.randn(B, N, 2)})
    print(f"  input {x.shape} -> output {out.shape}")
    assert out.shape == x.shape
    assert out.shape == model(x, t).shape

    print("\n=== Frame: f0+chroma+rhythm | Global: text+image ===")
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

    print("\n=== Global only (text) ===")
    model3 = ConditionedAudioDiT(
        kind='S',
        frame_cond_dims={},
        frame_cond_out_dims={},
        global_cond_configs={"text": {"dim": 512}},
    )
    out3 = model3(x, t, global_conditions={"text": torch.randn(B, 512)})
    print(f"  text only -> {out3.shape}")
    assert out3.shape == x.shape

    print("\n=== Frame re-injection (every block) ===")
    cond_kw = dict(kind='S',
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

    print("\n=== Re-injection: stride 2 / inert cases ===")
    model6 = ConditionedAudioDiT(
        kind='S', frame_cond_dims={"f0": 2}, frame_cond_out_dims={"f0": 16},
        global_cond_configs={}, frame_reinject_every=2,
    )
    assert model6.reinject_layers == [2, 4], model6.reinject_layers
    assert model6(x, t, frame_conditions={"f0": torch.randn(B, N, 2)}).shape == x.shape

    model7 = ConditionedAudioDiT(
        kind='S', frame_cond_dims={}, frame_cond_out_dims={},
        global_cond_configs={"text": {"dim": 512}}, frame_reinject_every=1,
    )
    assert model7.reinject_layers == []
    assert not hasattr(model7, "frame_reinject")

    model8 = ConditionedAudioDiT(
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

    print("\n=== Self-attention: differential ===")
    plain_kw = dict(kind='S', frame_cond_dims={}, frame_cond_out_dims={},
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

    print("\nTest passed!")
