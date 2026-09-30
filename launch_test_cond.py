"""
launch_test_cond.py

GPU-lock wrapper for IRCAM servers, dedicated to the TEST of a conditioned
checkpoint (test_cond.py). The training has its own wrapper,
launch_training_cond.py; the two are kept separate on purpose.

Locks ONE GPU (the test runs on a single device) BEFORE importing torch, then
runs test_cond.py with all the CLI arguments given here, unchanged.

Note:
    This launcher relies on the IRCAM-internal `manage_gpus` package and on
    POSIX `fcntl`, so it is meant for the IRCAM Linux servers only. Elsewhere
    (the laptop, the Windows VM) run the test directly:
        python test_cond.py --ckpt <ckpt> [--metrics_samples N|all] [overrides]

Use:
    # generation only, for listening (as test_cond.py alone)
    python launch_test_cond.py --ckpt runs/<run>/checkpoints/best_model_step<N>.pt

    # + the metrics on the whole test set
    python launch_test_cond.py --ckpt <ckpt> --metrics_samples all

    # + a dotlist override, passed through like any other argument
    python launch_test_cond.py --ckpt <ckpt> --metrics_samples all \\
        metrics.fad_reference=decoded

The launcher consumes no argument of its own. -h / --help prints this text
without locking anything; for the test's options: python test_cond.py --help
"""

import os
import sys
import platform


# ============================================================
# PARALLEL LOCK  (same mechanism as launch_training_cond.py)
# ============================================================
class ParallelLock:
    """Serializes the lock negotiation between processes starting at the
    same time on the same machine."""

    def __init__(self, path=None):
        if path is None:
            path = os.path.expanduser("~/.gpu_setup.lock")
        self.path = path
        self.fd = None

    def __enter__(self):
        import fcntl
        self.fd = open(self.path, "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import fcntl
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()


# ============================================================
# GPU LOCK (one device)
# ============================================================
def acquire_gpu_lock():
    """Lock one GPU with the IRCAM lock system; returns the locked ids."""
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch has been imported BEFORE locking the GPU. "
            "Lock the GPU before every torch import.")

    try:
        import manage_gpus as mgp
    except ImportError:
        raise RuntimeError(
            "did not find manage_gpus. Available only on IRCAM servers; "
            "elsewhere run: python test_cond.py ...")

    with ParallelLock():
        # A lock this user already holds (e.g. taken by a shell) is reused.
        devices = mgp.retrieve_my_gpu_locks()

        if not devices:
            gpu_ids = mgp.board_ids()
            if gpu_ids is None or len(gpu_ids) == 0:
                raise RuntimeError(f"No GPU available on {platform.node()}.")
            locked_gpu_id = mgp.get_gpu_lock()
            if locked_gpu_id >= 0:
                devices.append(locked_gpu_id)
            if not devices:
                raise RuntimeError(
                    f"Impossible to obtain a GPU on {platform.node()}.")

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            raise RuntimeError("CUDA_VISIBLE_DEVICES not set after locking.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            sorted(cuda_visible.split(",")))
        return devices


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        sys.exit(0)

    print("[launcher] Locking 1 GPU for the test...")
    devices = acquire_gpu_lock()
    print(f"[launcher] GPU locked: {devices}")
    print(f"[launcher] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_cond.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"test script not found: {script}")
    print(f"[launcher] Running: test_cond.py {' '.join(sys.argv[1:])}\n")

    # test_cond.py sees exactly the arguments given to this launcher.
    sys.argv = [script] + sys.argv[1:]

    import runpy
    runpy.run_path(script, run_name="__main__")
