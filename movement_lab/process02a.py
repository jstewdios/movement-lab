"""Linux subprocess-group deadline, including adopted descendant reaping."""
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time


def run_until(cmd, deadline, **kwargs):
    """Run under one absolute monotonic deadline; kill/reap the group on every exit.

    Linux subreaping adopts grandchildren when the group leader is killed, allowing
    us to wait for them rather than leaving zombies to an external init process.
    Nested simulator/validator processes inherit this dedicated session/group.
    Cleanup may extend past the deadline; no child work is allowed to continue.
    """
    if sys.platform != 'linux':
        raise RuntimeError('02A process-tree reaping requires Linux')
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(cmd, 0)
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot read child-subreaper state')
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot enable child subreaping')
    proc = None
    try:
        capture = kwargs.pop('capture_output', False)
        if capture:
            if 'stdout' in kwargs or 'stderr' in kwargs:
                raise ValueError('capture_output conflicts with stdout/stderr')
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=max(0., deadline-time.monotonic()))
        except BaseException:
            _kill_group(proc.pid)
            stdout, stderr = proc.communicate()  # drains pipes and reaps direct child
            # Preserve partial diagnostics for interrupted validator attempts.
            exc = sys.exc_info()[1]
            if isinstance(exc, subprocess.TimeoutExpired):
                exc.output, exc.stderr = stdout, stderr
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        try:
            if proc is not None:
                _kill_group(proc.pid)
                proc.wait()
                while True:
                    try:
                        os.waitpid(-proc.pid, 0)  # only adopted members of this group
                    except InterruptedError:
                        continue
                    except ChildProcessError:
                        break
        finally:
            if libc.prctl(36, previous.value, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), 'cannot restore child-subreaper state')


def _kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError as exc:
        if exc.errno != errno.ESRCH: raise
