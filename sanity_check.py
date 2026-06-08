# sanity_check.py
#
# End-to-end sanity check for the CONDITIONED Audio DiT pipeline, BEFORE
# committing to a long training run. It validates the full chain
#
#     conditions (.npz) -> conditioned dataset -> model -> gradient
#
# on a tiny amount of data, in a few seconds/minutes, and reports PASS/FAIL
# per stage. The most informative stage is the single-batch overfit: if the
# model cannot drive the rectified-flow loss down on ONE batch, the
# conditioning path is not carrying gradient and a long run is pointless.
#
# It does NOT load DAC/Encodec (no audio decoding, no FAD): it checks the
# learnable core only, so it stays light and fast.
#
# Usage (melody only, matching the first-test cond_default.yaml):
#   python sanity_check.py \
#       --latent_root    ./dataset_ready_cond/latents \
#       --condition_root ./dataset_ready_cond/conditions \
#       --enabled_frame  melody \
#       --kind S --batch_size 4 --overfit_steps 300
#
#   # all three frame conditions:
#   python sanity_check.py ... --enabled_frame melody,chroma,rhythm
#
#   # add a global condition (needs text/image embeddings available):
#   python sanity_check.py ... --enabled_frame melody --enabled_global text

import argparse
import sys

import torch
import torch.nn.functional as F

from conditions import ConditionRegistry
from network_cond import ConditionedAudioDiT, TOKEN_DIM
from audio_dataset_cond import build_conditioned_datasets, collate_conditioned


# ------------------------------------------------------------
# small helpers
# ------------------------------------------------------------
def _ok(msg):   print(f"  [PASS] {msg}")
def _info(msg): print(f"  [INFO] {msg}")
def _fail(msg):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def sample_logit_normal(batch_size, device, t_min, t_max, mean=0.0, std=1.0):
    """Identical to training_cond.sample_logit_normal (rectified-flow t)."""
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u).clamp(t_min, t_max)


def rectified_flow_loss(model, frames, fc, gc, device, t_min, t_max):
    """
    Same rectified-flow objective as training_cond.compute_loss, but with NO
    CFG dropout (we WANT the conditions active so the overfit test exercises
    the conditioning path).
    """
    x1 = frames.to(device).float()
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = sample_logit_normal(B, device, t_min, t_max)
    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0
    pred = model(xt, t, frame_conditions=fc, global_conditions=gc)
    return F.mse_loss(pred, target)


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sanity check for the conditioned Audio DiT.")
    ap.add_argument("--latent_root",     type=str, required=True)
    ap.add_argument("--condition_root",  type=str, required=True)
    ap.add_argument("--image_root",      type=str, default=None)
    ap.add_argument("--normalizer_path", type=str, default=None,
                    help="Optional cached normalizer.pt (else computed from data).")
    ap.add_argument("--duration_s",      type=float, default=5.0)
    ap.add_argument("--kind",            type=str, default="S")
    ap.add_argument("--enabled_frame",   type=str, default="melody",
                    help="Comma-separated, e.g. melody,chroma,rhythm")
    ap.add_argument("--enabled_global",  type=str, default="",
                    help="Comma-separated, e.g. text,image (default: none)")
    ap.add_argument("--batch_size",      type=int, default=4)
    ap.add_argument("--overfit_steps",   type=int, default=300)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--t_min",           type=float, default=0.001)
    ap.add_argument("--t_max",           type=float, default=0.999)
    ap.add_argument("--device",          type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    enabled_f = _csv(args.enabled_frame)
    enabled_g = _csv(args.enabled_global)

    print("=" * 64)
    print("CONDITIONED AUDIO DiT - SANITY CHECK")
    print("=" * 64)
    print(f"  device={device} | kind={args.kind} | batch_size={args.batch_size}")
    print(f"  enabled_frame={enabled_f} | enabled_global={enabled_g}")
    print(f"  overfit_steps={args.overfit_steps} | lr={args.lr}\n")

    # ========================================================
    # STAGE 1 - REGISTRY
    # ========================================================
    print("STAGE 1 - Condition registry")
    registry = ConditionRegistry(enabled_frame=enabled_f, enabled_global=enabled_g)
    frame_dims     = registry.frame_cond_dims
    frame_out_dims = registry.frame_cond_out_dims
    global_configs = registry.global_cond_configs
    _info(f"frame_cond_dims     = {frame_dims}")
    _info(f"frame_cond_out_dims = {frame_out_dims}")
    _info(f"global_cond_configs = {global_configs}")
    if enabled_f and not frame_dims:
        _fail("enabled_frame requested but registry built no frame extractors.")
    _ok("registry built")

    # ========================================================
    # STAGE 2 - CONDITIONED DATASET + ONE BATCH
    # ========================================================
    print("\nSTAGE 2 - Conditioned dataset + one batch")
    train_ds, val_ds, normalizer, label_map = build_conditioned_datasets(
        latent_root=args.latent_root,
        condition_root=args.condition_root,
        image_root=args.image_root,
        duration_s=args.duration_s,
        normalizer_path=args.normalizer_path,
        registry=registry,
        preload=False,
    )
    if len(train_ds) == 0:
        _fail("train dataset is empty - check latent_root / duration_s.")
    _info(f"train samples={len(train_ds)} | val samples={len(val_ds)} | n_frames={train_ds.n_frames}")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=collate_conditioned,
    )
    frames, frame_cond, labels, text_embs, image_embs = next(iter(loader))
    n_frames = train_ds.n_frames

    # ---- shape + finiteness checks ----
    if tuple(frames.shape) != (args.batch_size, n_frames, TOKEN_DIM):
        _fail(f"frames shape {tuple(frames.shape)} != "
              f"{(args.batch_size, n_frames, TOKEN_DIM)}")
    if not torch.isfinite(frames).all():
        _fail("frames contain NaN/Inf.")
    _ok(f"frames {tuple(frames.shape)} finite")

    for name, dim in frame_dims.items():
        if name not in frame_cond:
            _fail(f"frame condition '{name}' missing from the batch.")
        c = frame_cond[name]
        if tuple(c.shape) != (args.batch_size, n_frames, dim):
            _fail(f"'{name}' shape {tuple(c.shape)} != "
                  f"{(args.batch_size, n_frames, dim)}")
        if not torch.isfinite(c).all():
            _fail(f"'{name}' contains NaN/Inf.")
        nz = (c.abs().sum() > 0).item()
        _ok(f"'{name}' {tuple(c.shape)} finite | non-zero={nz}")
        if not nz:
            _info(f"     (note: '{name}' is all-zero in this batch - possible if "
                  f"the .npz lacks it or the chunks are silent; verify extraction)")

    # ---- build conditioning dicts for the forward ----
    fc = {k: v.to(device).float() for k, v in frame_cond.items()}
    gc = {}
    if "text" in global_configs:
        gc["text"] = text_embs.to(device)
    if "image" in global_configs:
        gc["image"] = image_embs.to(device)

    # ========================================================
    # STAGE 3 - MODEL BUILD + FORWARD/BACKWARD
    # ========================================================
    print("\nSTAGE 3 - Model build + forward/backward")
    model = ConditionedAudioDiT(
        kind=args.kind,
        frame_cond_dims=frame_dims,
        frame_cond_out_dims=frame_out_dims,
        global_cond_configs=global_configs,
    ).to(device)

    loss0 = rectified_flow_loss(model, frames, fc, gc, device, args.t_min, args.t_max)
    if not torch.isfinite(loss0):
        _fail(f"initial loss is not finite: {loss0.item()}")
    loss0.backward()
    n_with_grad = sum(1 for p in model.parameters()
                      if p.grad is not None and torch.isfinite(p.grad).all())
    n_params    = sum(1 for _ in model.parameters())
    _info(f"initial loss = {loss0.item():.4f}")
    _ok(f"backward OK | params with finite grad: {n_with_grad}/{n_params}")

    # NOTE on the conditioning path at INIT: by design (adaLN-Zero + zeroed
    # final layer, exactly as in the official DiT and in network.py), the model
    # starts as a zero map, so at step 0 the gradient does NOT yet reach the
    # frame encoder / input projection - this is EXPECTED, not a bug. The path
    # "turns on" after a few optimiser steps; we verify it carries gradient
    # post-warmup in Stage 4.
    if frame_dims:
        fe_g0 = sum(p.grad.abs().sum().item()
                    for p in model.frame_encoder.parameters() if p.grad is not None)
        _info(f"frame encoder |grad| at init = {fe_g0:.3e} "
              f"(0 is expected here; re-checked after warmup)")

    # ========================================================
    # STAGE 4 - SINGLE-BATCH OVERFIT (the real test)
    # ========================================================
    print(f"\nSTAGE 4 - Single-batch overfit ({args.overfit_steps} steps)")
    print("  A healthy model memorises one batch: the loss should fall sharply.")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    init_loss = None
    last_loss = None
    for step in range(args.overfit_steps):
        opt.zero_grad()
        loss = rectified_flow_loss(model, frames, fc, gc, device, args.t_min, args.t_max)
        if not torch.isfinite(loss):
            _fail(f"loss became non-finite at step {step}: {loss.item()}")
        loss.backward()
        opt.step()
        if init_loss is None:
            init_loss = loss.item()
        last_loss = loss.item()
        if step % max(1, args.overfit_steps // 10) == 0 or step == args.overfit_steps - 1:
            print(f"    step {step:4d} | loss {loss.item():.4f}")

    drop = (init_loss - last_loss) / max(init_loss, 1e-8)
    _info(f"overfit: {init_loss:.4f} -> {last_loss:.4f}  ({100*drop:.1f}% drop)")
    # NB: with random t each step the loss is noisy and never reaches 0, but a
    # learning model should still cut it substantially on a single batch.
    if drop < 0.30:
        _fail("loss dropped < 30% on a single batch - the model is not "
              "learning (check conditioning wiring, lr, or data).")
    _ok("single-batch overfit: loss decreased substantially")

    # Post-warmup connectivity check: now that the final layer is no longer
    # zero, the frame-conditioning path MUST carry gradient. (At init it is
    # zero by design - see the note in Stage 3.) A still-zero gradient here
    # means the conditions are not wired into the model.
    if frame_dims:
        model.zero_grad(set_to_none=True)
        loss_w = rectified_flow_loss(model, frames, fc, gc, device, args.t_min, args.t_max)
        loss_w.backward()
        fe_g = sum(p.grad.abs().sum().item()
                   for p in model.frame_encoder.parameters() if p.grad is not None)
        if fe_g == 0.0:
            _fail("frame-condition encoder receives ZERO gradient even after "
                  "warmup - the conditioning path is disconnected.")
        _ok(f"frame encoder gradient flows post-warmup (|grad| sum = {fe_g:.3e})")

    # ========================================================
    # SUMMARY
    # ========================================================
    print("\n" + "=" * 64)
    print("SANITY CHECK PASSED")
    print("=" * 64)
    print("  The conditioned pipeline is wired correctly end to end:")
    print("  registry -> dataset -> batch shapes -> model -> gradient -> learning.")
    print("  You can launch the short real-training smoke test next (see notes).")


if __name__ == "__main__":
    main()
