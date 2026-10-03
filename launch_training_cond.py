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

ONE GPU (--num-gpus 1, the default): the training runs INSIDE this process,
exactly as it always has.

SEVERAL GPUs (--num-gpus N, N > 1): this process locks the N GPUs and then
starts N training processes, one per GPU (PyTorch DistributedDataParallel),
and stays alive as their parent for the whole run: the locks belong to it.
Each child takes the parent's locks over with retrieve_my_gpu_locks() BEFORE
importing torch (LAB rules, "Sharing GPU locks with sub-processes"), then runs
training_cond.py with RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR /
MASTER_PORT in its environment -- the same variables torchrun sets, which is
how training_cond.py knows it is one of N. If one process dies, the others are
stopped: a distributed training cannot continue with a missing member, and a
survivor would otherwise sit on its GPU waiting for it.
  * Only rank 0 writes: TensorBoard, checkpoints, the dumps, the console. The
    other ranks write their console to runs/<run>/rank<R>.log.
  * data.train_batch_size is PER GPU: the batch of one optimizer step is
    train_batch_size x grad_accum x N.
  * If fewer GPUs than asked are obtained, the run goes on with those (a
    warning says how many); with one, it is the ordinary single-GPU run.
  * From the second GPU on the locks are SOFT (LAB rules, "GPU lock levels"):
    a user with hard-lock authority can interrupt them, and the whole run then
    stops (the last periodic checkpoint is the one to resume from).
  * Ctrl+C: rank 0 writes checkpoint_last, as a single-GPU run does, then the
    other processes are stopped; a second Ctrl+C stops everything at once.
    A kill (SIGTERM) stops every process at once, without checkpoint_last,
    as it does a single-GPU run.
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


# Set by the parent on every training process it starts: their rank. Its
# presence is what tells this file it is running as one of those processes.
CHILD_RANK_ENV = "LAUNCH_TRAINING_COND_RANK"


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


def retrieve_parent_gpu_locks(rank):
    """A training process started by this launcher: take over the parent's
    locks BEFORE torch is imported, and order CUDA_VISIBLE_DEVICES exactly as
    the parent did, so that cuda:<LOCAL_RANK> names the same physical GPU in
    every process and no two ranks share one."""
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


# ============================================================
# MULTI-GPU: one training process per GPU
# ============================================================
def _free_port():
    """A free TCP port on this machine for the rendezvous of the ranks."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# After a Ctrl+C, how long the launcher waits for rank 0 to write the last
# checkpoint and exit before it stops every process anyway.
INTERRUPT_WAIT_S = 30 * 60


def _child_setup():
    """Linux, in each training process just before it starts:
      * its own process group, so the Ctrl+C of the terminal reaches the
        launcher alone, which hands it to rank 0 only (run_distributed);
      * SIGTERM when the launcher dies (PR_SET_PDEATHSIG), so no training
        process is ever left running on a GPU whose lock is gone."""
    os.setpgrp()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _stop(procs, grace_s=30.0):
    """SIGTERM every running process, SIGKILL whatever is still alive after
    `grace_s` (a process that does not die on SIGTERM)."""
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
    """Start `world_size` training processes and supervise them. Returns the
    exit code of the run: 0 if every rank ended cleanly, else the first
    failure's."""
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

    # CTRL+C. Rank 0 alone gets it, and does what a single-GPU run does: it
    # leaves the training loop and writes checkpoint_last. If it is waiting
    # inside a collective at that moment, the collective still completes,
    # because the other ranks did not get the signal and keep going. They are
    # stopped once rank 0 has exited: they have nothing to save. Handing the
    # signal to every rank, as the terminal would, can leave rank 0 waiting in
    # a collective for ranks that have already left -- and never saving. A
    # second Ctrl+C stops every process at once (the last checkpoint can then
    # be lost).
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

    # SIGTERM (kill): every process stops now, as a single-GPU run killed the
    # same way does -- no last checkpoint, the run resumes from the last
    # periodic one.
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
            # After a Ctrl+C only rank 0 matters: wait for it, then stop the
            # others (they may be waiting for it inside a collective).
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

    # ---- one of the N training processes started by run_distributed ----
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

    # Replace sys.argv so the training script sees its own args (no --num-gpus)
    sys.argv = [args.script] + forwarded_args

    import runpy
    runpy.run_path(args.script, run_name="__main__")
