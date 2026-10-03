"""
launch_test_cond.py

GPU-lock wrapper for IRCAM servers, dedicated to the TEST of a conditioned
checkpoint. The twin of launch_training_cond.py.

Locks the GPU BEFORE importing torch, then runs test_cond.py forwarding
all the remaining CLI arguments to it. The test runs on one device, so the
launcher always locks ONE GPU (there is no --num-gpus).

Note:
    This launcher relies on the IRCAM-internal `manage_gpus` package and on
    POSIX `fcntl`. It is therefore meant for the IRCAM Linux servers only.
    On other systems (e.g. the Windows VM at CAIS LAB) launch the test
    script directly:
        python test_cond.py --ckpt <ckpt> [--config ...] [overrides]

Use:
    # All defaults (configs/test_cond.yaml)
    python launch_test_cond.py \\
        --ckpt runs/cond_A/checkpoints/checkpoint_step50000.pt

    # Custom test config
    python launch_test_cond.py --ckpt <ckpt> --config configs/my_test.yaml

    # Metrics on the whole test split / no metrics (panels only)
    python launch_test_cond.py --ckpt <ckpt> --metrics_samples all
    python launch_test_cond.py --ckpt <ckpt> --metrics_samples 0

    # CLI overrides (passed straight to test_cond.py)
    python launch_test_cond.py --ckpt <ckpt> \\
        sampling.n_test_panels=4 metrics.dac_device=cuda

The only argument the launcher consumes is --script.
Everything else is passed verbatim to test_cond.py.
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
    # Parse only --script; everything else is forwarded to test_cond.py.
    # No --num-gpus: the test runs on one device, so one GPU is locked.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--script", type=str, default="test_cond.py",
                        help="Test script to run "
                              "(default: test_cond.py)")
    parser.add_argument("-h", "--help", action="store_true")

    args, forwarded_args = parser.parse_known_args()

    if args.help:
        print(__doc__)
        sys.exit(0)

    print("[launcher] Locking 1 GPU...")
    devices = acquire_gpu_locks(num_devices=1)
    print(f"[launcher] GPU locked: {devices}")
    print(f"[launcher] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not os.path.exists(args.script):
        raise FileNotFoundError(f"Test script not found: {args.script}")

    print(f"[launcher] Running: {args.script} {' '.join(forwarded_args)}\n")

    # Replace sys.argv so the test script sees its own args (no --script)
    sys.argv = [args.script] + forwarded_args

    import runpy
    runpy.run_path(args.script, run_name="__main__")
