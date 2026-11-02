# Muon on the parameters with ndim >= 2, AdamW on the others, as one optimizer.

import fnmatch

import torch
from torch.optim import Optimizer


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonAdamW(Optimizer):
    def __init__(self, named_params, muon_lr=1e-3, muon_momentum=0.95, ns_steps=5,
                 adam_lr=5e-5, adam_betas=(0.9, 0.95), adam_eps=1e-8,
                 adam_weight_decay=0.01, fused_patterns=()):
        named_params = list(named_params)
        self._fused = {p for name, p in named_params
                       if p.ndim >= 2 and any(fnmatch.fnmatch(name, pat)
                                              for pat in fused_patterns)}
        muon = [p for _, p in named_params if p.ndim >= 2]
        adam = [p for _, p in named_params if p.ndim < 2]
        super().__init__(
            [{"params": muon, "lr": muon_lr, "momentum": muon_momentum,
              "ns_steps": ns_steps, "use_muon": True},
             {"params": adam, "lr": adam_lr, "betas": tuple(adam_betas), "eps": adam_eps,
              "weight_decay": adam_weight_decay, "use_muon": False}],
            {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    (self._muon if group["use_muon"] else self._adamw)(p, group)
        return loss

    def _muon(self, p, group):
        grad = p.grad
        state = self.state[p]
        if not state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        beta = group["momentum"]
        state["momentum_buffer"].lerp_(grad, 1 - beta)
        update = grad.lerp_(state["momentum_buffer"], beta)
        shape = update.shape
        if p in self._fused and update.size(0) > update.size(1):
            chunks = update.split(update.size(1), dim=0)
            update = torch.cat([newton_schulz(c, group["ns_steps"]) for c in chunks], dim=0)
        else:
            update = newton_schulz(update, group["ns_steps"])
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
        p.add_(update.reshape(shape).to(p.dtype), alpha=-group["lr"])

    def _adamw(self, p, group):
        grad = p.grad
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["step"] += 1
        lr, wd = group["lr"], group["weight_decay"]
        beta1, beta2 = group["betas"]
        if wd > 0:
            p.mul_(1 - lr * wd)
        state["exp_avg"].lerp_(grad, 1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        step_size = lr / (1 - beta1 ** state["step"])
        denom = (state["exp_avg_sq"].sqrt()
                 / (1 - beta2 ** state["step"]) ** 0.5).add_(group["eps"])
        p.addcdiv_(state["exp_avg"], denom, value=-step_size)
