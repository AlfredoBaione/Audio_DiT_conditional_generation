# IRCAM launcher of training_cond.py: GPU locks (manage_gpus), --num-gpus N.
"""
launch_training_cond.py

GPU-lock wrapper for IRCAM servers, dedicated to the CONDITIONED training.

Locks GPU(s) BEFORE importing torch, then runs training_cond.py forwarding
all the remaining CLI arguments to it. Mirrors the unconditional launcher
in the sister repository.

Use:
    # All defaults (1 GPU, configs/training_cond_default.yaml)
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

"""

import os
import sys
import time
import fcntl
import socket
import signal
import argparse
import platform
import subprocess


CHILD_RANK_ENV = "LAUNCH_TRAINING_COND_RANK"


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


def acquire_gpu_locks(num_devices=1):
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


def retrieve_parent_gpu_locks(rank):
    if 'torch' in sys.modules:
        raise RuntimeError(
            f"[rank {rank}] torch was imported before the GPU locks were "
            f"retrieved from the launcher.")
    import manage_gpus as mgp
    devices = mgp.retrieve_my_gpu_locks()
    if not devices:
        raise RuntimeError(
            f"[rank {rank}] no GPU lock inherited from the launcher "
            f"(pid {os.getppid()}): this process would run on a GPU it does "
            f"not hold.")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        raise RuntimeError(
            f"[rank {rank}] CUDA_VISIBLE_DEVICES not set after retrieving "
            f"the locks.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        sorted(cuda_visible.split(",")))
    return devices


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


INTERRUPT_WAIT_S = 30 * 60


def _child_setup():
    os.setpgrp()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:
        pass


def _stop(procs, grace_s=30.0):
    for p in procs:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
            except OSError:
                pass
    t_end = time.time() + grace_s
    for p in procs:
        while p.poll() is None and time.time() < t_end:
            time.sleep(0.5)
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass


def run_distributed(script, forwarded_args, world_size):
    port = _free_port()
    procs = []
    for rank in range(world_size):
        env = dict(os.environ)
        env.update({
            CHILD_RANK_ENV: str(rank),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        })
        cmd = [sys.executable, os.path.abspath(__file__), "--script", script] \
            + list(forwarded_args)
        procs.append(subprocess.Popen(
            cmd, env=env,
            preexec_fn=_child_setup if os.name == "posix" else None))
    print(f"[launcher] {world_size} training processes started "
          f"(pids {[p.pid for p in procs]}, rendezvous 127.0.0.1:{port})",
          flush=True)

    interrupt = {"count": 0, "since": None}

    def _on_sigint(_signum, _frame):
        interrupt["count"] += 1
        if interrupt["count"] == 1:
            interrupt["since"] = time.time()
            print("[launcher] Ctrl+C: rank 0 writes the last checkpoint, then "
                  "every process stops (Ctrl+C again to stop at once)",
                  file=sys.stderr, flush=True)
            if procs[0].poll() is None:
                try:
                    procs[0].send_signal(signal.SIGINT)
                except OSError:
                    pass
        else:
            print("[launcher] second Ctrl+C: stopping every process now",
                  file=sys.stderr, flush=True)
            _stop(procs, grace_s=5.0)
            sys.exit(130)

    def _on_sigterm(signum, _frame):
        print(f"[launcher] signal {signum}: stopping the training processes",
              file=sys.stderr, flush=True)
        _stop(procs)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigterm)

    rc = 0
    while True:
        alive = [p for p in procs if p.poll() is None]
        if interrupt["count"]:
            if (procs[0].poll() is not None
                    or time.time() - interrupt["since"] > INTERRUPT_WAIT_S):
                if procs[0].poll() is None:
                    print(f"[launcher] rank 0 still running after "
                          f"{INTERRUPT_WAIT_S // 60} min: stopping it",
                          file=sys.stderr, flush=True)
                _stop(procs)
                rc = 130
                break
        else:
            failed = [(r, p.returncode) for r, p in enumerate(procs)
                      if p.poll() is not None and p.returncode != 0]
            if failed:
                r, code = failed[0]
                print(f"[launcher] rank {r} exited with code {code}: stopping "
                      f"the others", file=sys.stderr, flush=True)
                _stop(procs)
                rc = code
                break
        if not alive:
            break
        time.sleep(1.0)
    return rc


if __name__ == "__main__":
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

    if CHILD_RANK_ENV in os.environ:
        _rank = int(os.environ[CHILD_RANK_ENV])
        retrieve_parent_gpu_locks(_rank)
        sys.argv = [args.script] + forwarded_args
        import runpy
        runpy.run_path(args.script, run_name="__main__")
        sys.exit(0)

    print(f"[launcher] Locking {args.num_gpus} GPU(s)...", flush=True)
    devices = acquire_gpu_locks(num_devices=args.num_gpus)
    print(f"[launcher] GPU locked: {devices}", flush=True)
    print(f"[launcher] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}",
          flush=True)

    if not os.path.exists(args.script):
        raise FileNotFoundError(f"Training script not found: {args.script}")

    print(f"[launcher] Running: {args.script} {' '.join(forwarded_args)}\n",
          flush=True)

    if len(devices) > 1:
        sys.exit(run_distributed(args.script, forwarded_args, len(devices)))

    sys.argv = [args.script] + forwarded_args

    import runpy
    runpy.run_path(args.script, run_name="__main__")
