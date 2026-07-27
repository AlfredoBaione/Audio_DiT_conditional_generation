"""
launch_training_cond.py

GPU-lock wrapper for IRCAM servers, dedicated to the CONDITIONED training.

Locks GPU(s) BEFORE importing torch, then runs training_cond.py forwarding
all the remaining CLI arguments to it. Mirrors the unconditional launcher
in the sister repository.

Note:
    This launcher relies on the IRCAM-internal `manage_gpus` package and on
    POSIX `fcntl`. It is therefore meant for the IRCAM Linux servers only.
    On other systems (e.g. the Windows VM at CAIS LAB) launch the training
    script directly:
        python training_cond.py [--config ...] [--run_name ...] [overrides]

Use:
    # All defaults (1 GPU, configs/cond_default.yaml)
    python launch_training_cond.py

    # Custom config
    python launch_training_cond.py --config configs/my_cond_run.yaml

    # CLI overrides (passed straight to training_cond.py)
    python launch_training_cond.py --run_name "cond_B_lr5e5" \\
        model.kind=B training.lr=5e-5

    # Resume
    python launch_training_cond.py \\
        --resume runs/cond_old/checkpoints/checkpoint_step50000.pt

    # Multi-GPU (only --num-gpus is interpreted by the launcher;
    #  everything else goes to training_cond.py)
    python launch_training_cond.py --num-gpus 2 training.lr=2e-4

The only arguments the launcher consumes are --num-gpus and --script.
Everything else is passed verbatim to training_cond.py.
"""

import os
import sys
import fcntl
import argparse
import platform


# ============================================================
# PARALLEL LOCK
# ============================================================
class ParallelLock:
    def __init__(self, path=None):
        if path is None:
            path = os.path.expanduser("~/.gpu_setup.lock")
        self.path = path
        self.fd = None

    def __enter__(self):
        self.fd = open(self.path, "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()


# ============================================================
# GPU LOCKING
# ============================================================
def acquire_gpu_locks(num_devices=1):
    """Locks num_devices GPUs using IRCAM lock system."""
    if 'torch' in sys.modules:
        raise RuntimeError(
            "torch has been imported BEFORE locking the GPU. "
            "Lock the GPUs before every torch import."
        )

    try:
        import manage_gpus as mgp
    except ImportError:
        raise RuntimeError(
            "did not find manage_gpus. "
            "Available only on IRCAM servers."
        )

    if num_devices > 4:
        raise ValueError(
            f"At most 4 GPU supported per node (asked {num_devices})."
        )

    with ParallelLock():
        devices = mgp.retrieve_my_gpu_locks()

        if not devices:
            gpu_ids = mgp.board_ids()
            if gpu_ids is None or len(gpu_ids) == 0:
                raise RuntimeError(
                    f"No GPU available on {platform.node()}."
                )

            for _ in range(num_devices):
                locked_gpu_id = mgp.get_gpu_lock()
                if locked_gpu_id >= 0:
                    devices.append(locked_gpu_id)

            if not devices:
                raise RuntimeError(
                    f"Impossible to obtain a GPU on {platform.node()}."
                )

            if len(devices) < num_devices:
                print(
                    f"[WARN] asked {num_devices} GPU, obtained {len(devices)}.",
                    file=sys.stderr,
                )

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES not set after locking."
            )
        cuda_visible = sorted(cuda_visible.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible)

        return devices


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Parse only --num-gpus and --script; everything else is forwarded
    # to training_cond.py
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to lock (default: 1)")
    parser.add_argument("--script", type=str, default="training_cond.py",
                        help="Training script to run "
                              "(default: training_cond.py)")
    parser.add_argument("-h", "--help", action="store_true")

    args, forwarded_args = parser.parse_known_args()

    if args.help:
        print(__doc__)
        sys.exit(0)

    print(f"[launcher] Locking {args.num_gpus} GPU(s)...")
    devices = acquire_gpu_locks(num_devices=args.num_gpus)
    print(f"[launcher] GPU locked: {devices}")
    print(f"[launcher] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not os.path.exists(args.script):
        raise FileNotFoundError(f"Training script not found: {args.script}")

    print(f"[launcher] Running: {args.script} {' '.join(forwarded_args)}\n")

    # Replace sys.argv so the training script sees its own args (no --num-gpus)
    sys.argv = [args.script] + forwarded_args

    import runpy
    runpy.run_path(args.script, run_name="__main__")
