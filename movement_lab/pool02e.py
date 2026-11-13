"""Bounded, memory-guarded worker pools for 02E batteries (operational only: no result changes).

Why this exists (see results/02e/ATTEMPTS.md, "Worker memory"): NumPy 2.4.1's ``np.unique``
retains one input-sized buffer per call and never releases it. ``physics02b.support_polygon``
calls it on every control step when an evaluation records support margins, so a
``diag02e.evaluate`` worker grows by about 11.5 MB per 6 s trial. Over 150 trials per worker
that reached about 1.6 GB per worker, and six of them exhausted WSL. Learning rollouts
(no margins) and 200 Hz recording do not grow.

The evaluation sources are pinned by every battery identity, so they cannot change. This
module fixes it from outside them:

* bounded worker lifetime: each batch of about ``workers x tasks_per_worker`` tasks runs in a
  freshly forked pool that is shut down afterwards, so leaked memory goes back to the OS;
* a memory guard, polled from the main thread: if any worker exceeds ``max_worker_mb``, the
  driver plus all workers exceed ``max_total_mb``, or WSL ``MemAvailable`` falls below
  ``min_available_mb``, every worker is terminated and the run stops. Batteries write a configuration only once all of its rows exist, and learning
  writes generations atomically, so a stop loses only work in progress;
* explicit cleanup: workers get PR_SET_PDEATHSIG, so they die with their driver. SIGTERM
  and SIGINT terminate and reap the workers, and every pool is shut down.

The same functions are called on the same tasks and results come back in task order, so
rows are identical to an unbounded run (verified against published rows).

    python -m movement_lab.pool02e --workers 1 diag02e characterize
    python -m movement_lab.pool02e --workers 2 --tasks-per-worker 20 diag02e pilot

``study02e`` (not pinned) uses the same pools and guard directly.
"""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
import contextlib
import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

MAX_WORKERS = 4              # hard cap for this machine (8 GB WSL); see ATTEMPTS.md for the budget
TASKS_PER_WORKER = 20        # about 20 x 11.5 MB of retained growth before a worker is replaced
MAX_WORKER_MB = 900          # expected peak about 420 MB; anything near this is unexpected growth
MAX_TOTAL_MB = 2000          # driver + every worker; WSL memory is backed by a 16 GB host with other work on it
MIN_AVAILABLE_MB = 1500      # stop before the host starts to swap
PR_SET_PDEATHSIG = 1


class MemoryGuardError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- /proc helpers
def meminfo_mb():
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            key, rest = line.split(':', 1)
            info[key] = int(rest.split()[0]) // 1024
    return dict(available=info['MemAvailable'], total=info['MemTotal'], swap_used=info['SwapTotal'] - info['SwapFree'])


def rss_mb(pid):
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def descendants(pid):
    """All live descendant pids of ``pid`` (any depth), from /proc."""
    children = {}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat') as f:
                stat = f.read()
        except (FileNotFoundError, ProcessLookupError):
            continue
        ppid = int(stat.rsplit(')', 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    out, stack = [], [pid]
    while stack:
        for c in children.get(stack.pop(), ()):
            out.append(c); stack.append(c)
    return out


def terminate_tree(pids, grace_s=5.0):
    """SIGTERM, then SIGKILL whatever is still alive after ``grace_s``."""
    for p in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(p, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(rss_mb(p) for p in pids):
        time.sleep(.1)
    for p in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(p, signal.SIGKILL)
    for c in mp.active_children():    # reap multiprocessing children
        c.join(1)


# ----------------------------------------------------------------------------- guard
class Guard:
    """Checks worker RSS and host memory; logs a JSON line every ``log_every_s``."""

    def __init__(self, max_worker_mb=MAX_WORKER_MB, min_available_mb=MIN_AVAILABLE_MB, log=None,
                 poll_s=1.0, log_every_s=5.0, max_total_mb=MAX_TOTAL_MB):
        self.max_worker_mb, self.min_available_mb, self.max_total_mb = max_worker_mb, min_available_mb, max_total_mb
        self.poll_s, self.log_every_s = poll_s, log_every_s
        self.log = Path(log) if log else None
        self.peak_worker_mb = self.peak_total_mb = 0
        self.min_available_seen = None
        self._last = 0.

    def check(self, pids, **context):
        mem = meminfo_mb()
        workers = {p: rss_mb(p) for p in pids}
        live = [v for v in workers.values() if v]
        driver = rss_mb(os.getpid())
        total = driver + sum(live)
        self.peak_worker_mb = max([self.peak_worker_mb, *live])
        self.peak_total_mb = max(self.peak_total_mb, total)
        self.min_available_seen = mem['available'] if self.min_available_seen is None else min(self.min_available_seen, mem['available'])
        now = time.time()
        breach = None
        if live and max(live) > self.max_worker_mb:
            breach = f'worker RSS {max(live)} MB > cap {self.max_worker_mb} MB'
        elif total > self.max_total_mb:
            breach = f'driver + workers RSS {total} MB > cap {self.max_total_mb} MB'
        elif mem['available'] < self.min_available_mb:
            breach = f'host MemAvailable {mem["available"]} MB < floor {self.min_available_mb} MB'
        if self.log and (breach or now - self._last >= self.log_every_s):
            self._last = now
            row = dict(t=round(now, 1), available_mb=mem['available'], swap_used_mb=mem['swap_used'], total_mb=total,
                       driver_mb=driver, workers_mb=sorted(live, reverse=True), breach=breach, **context)
            with self.log.open('a') as f:
                f.write(json.dumps(row) + '\n')
        if breach:
            raise MemoryGuardError(breach)

    def summary(self):
        return dict(peak_worker_mb=self.peak_worker_mb, peak_total_mb=self.peak_total_mb, min_available_mb=self.min_available_seen)


# ----------------------------------------------------------------------------- executor
def _worker_init(initializer, driver):
    # Forked workers inherit the driver's handlers; restore plain termination (SIGTERM kills,
    # Ctrl-C is the driver's to handle) so a stop is never turned into a per-task exception.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Die with the driver: a killed or crashed driver must not leave workers simulating.
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() != driver:          # the driver died before prctl took effect
        os._exit(1)
    if initializer is not None:
        initializer()


def _stop_workers(grace_s=5.0):
    terminate_tree([c.pid for c in mp.active_children()], grace_s)


class RecyclingExecutor:
    """Drop-in for ``with ProcessPoolExecutor(n, initializer=f) as pool: list(pool.map(fn, tasks))``.

    Tasks run in batches of ``workers * tasks_per_worker``, each in a freshly forked pool that
    is shut down before the next batch starts. Results are returned in task order. The guard
    is polled from the main thread, so no extra thread is alive when the next pool forks.
    """

    def __init__(self, max_workers, initializer=None, *, tasks_per_worker=TASKS_PER_WORKER, guard=None):
        if not 1 <= max_workers <= MAX_WORKERS:
            raise ValueError(f'workers must be 1..{MAX_WORKERS}')
        self.workers, self.initializer = max_workers, initializer
        self.tasks_per_worker = max(1, int(tasks_per_worker))
        self.guard = guard or Guard()
        self.batches = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _stop_workers(grace_s=2.)   # every batch pool is already shut down; make sure nothing remains
        return False

    def map(self, fn, iterable, chunksize=1):   # chunksize accepted for compatibility; tasks are submitted singly
        tasks = list(iterable)
        out = []
        size = self.workers * self.tasks_per_worker
        for start in range(0, len(tasks), size):
            out.extend(self._batch(fn, tasks[start:start + size], start, len(tasks)))
        return iter(out)

    def _batch(self, fn, tasks, offset, total):
        ex = ProcessPoolExecutor(min(self.workers, len(tasks)), mp_context=mp.get_context('fork'),
                                 initializer=_worker_init, initargs=(self.initializer, os.getpid()))
        self.batches += 1
        try:
            futures = [ex.submit(fn, t) for t in tasks]
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=self.guard.poll_s, return_when=FIRST_EXCEPTION)
                for f in done:
                    if f.exception() is not None:
                        raise f.exception()
                self.guard.check([c.pid for c in mp.active_children()], batch=self.batches,
                                 done=offset + len(tasks) - len(pending), total=total)
            results = [f.result() for f in futures]
        except BaseException:
            _stop_workers()
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
        return results


@contextlib.contextmanager
def bounded(module, tasks_per_worker=TASKS_PER_WORKER, guard=None):
    """Within the block, ``module.ProcessPoolExecutor(n, initializer=f)`` builds a RecyclingExecutor."""
    original = module.ProcessPoolExecutor
    guard = guard or Guard()

    def factory(max_workers, initializer=None):
        return RecyclingExecutor(max_workers, initializer, tasks_per_worker=tasks_per_worker, guard=guard)

    module.ProcessPoolExecutor = factory
    try:
        yield guard
    finally:
        module.ProcessPoolExecutor = original


@contextlib.contextmanager
def cleanup_on_signals():
    """SIGTERM/SIGINT raise in the main thread, so ``finally`` blocks terminate and reap every descendant."""
    def handler(signum, frame):
        raise SystemExit(f'stopped by signal {signum}')
    old = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        terminate_tree(descendants(os.getpid()), grace_s=2.)


def add_arguments(p, workers_default=1):
    p.add_argument('--workers', type=int, default=workers_default, help=f'1..{MAX_WORKERS} (default {workers_default})')
    p.add_argument('--tasks-per-worker', type=int, default=TASKS_PER_WORKER, help='battery tasks before a worker is replaced')
    p.add_argument('--max-worker-mb', type=int, default=MAX_WORKER_MB, help='stop if any process exceeds this RSS')
    p.add_argument('--max-total-mb', type=int, default=MAX_TOTAL_MB, help='stop if driver + all workers exceed this RSS')
    p.add_argument('--min-available-mb', type=int, default=MIN_AVAILABLE_MB, help='stop if host MemAvailable falls below')
    p.add_argument('--memory-log', type=Path, help='JSON-lines memory samples (default results/02e/<step>-memory.log)')


def guard_from(a, step):
    if not 1 <= a.workers <= MAX_WORKERS:
        raise SystemExit(f'--workers must be 1..{MAX_WORKERS} on this machine (see ATTEMPTS.md, "Worker memory")')
    return Guard(a.max_worker_mb, a.min_available_mb, a.memory_log or Path('results/02e') / f'{step}-memory.log',
                 max_total_mb=a.max_total_mb)


def main(argv=None):
    """Run a pinned ``diag02e`` step through bounded, guarded pools (diag02e itself cannot change)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('module', choices=('diag02e',))
    p.add_argument('step')
    p.add_argument('rest', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if '--workers' in a.rest:
        p.error('pass --workers to pool02e, not to the wrapped module')
    guard = guard_from(a, f'{a.module}-{a.step}')
    from movement_lab import diag02e as target
    started = time.time()
    with cleanup_on_signals(), bounded(target, a.tasks_per_worker, guard):
        target.main([a.step, *a.rest, '--workers', str(a.workers)])
    print(json.dumps(dict(memory=guard.summary(), wall_s=round(time.time() - started))), flush=True)


if __name__ == '__main__':
    main()
