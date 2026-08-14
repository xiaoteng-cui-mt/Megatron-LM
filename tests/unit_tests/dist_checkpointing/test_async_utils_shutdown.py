"""Regression tests for abnormal async-checkpoint helper shutdown."""

import signal
from unittest import mock

from megatron.core.dist_checkpointing.strategies import async_utils, filesystem_async
from megatron.core.dist_checkpointing.strategies.async_utils import PersistentAsyncCaller


def _caller_with_mock_process(is_alive: bool):
    caller = object.__new__(PersistentAsyncCaller)
    caller.process = mock.Mock()
    caller.process.is_alive.return_value = is_alive
    caller.queue = mock.Mock()
    return caller


def test_abort_close_kills_and_joins_live_worker():
    caller = _caller_with_mock_process(is_alive=True)
    process = caller.process
    queue = caller.queue

    caller.close(abort=True)

    process.kill.assert_called_once_with()
    process.join.assert_called_once_with()
    queue.put.assert_not_called()
    queue.join.assert_not_called()
    assert caller.process is None


def test_graceful_close_does_not_wait_on_dead_worker_queue():
    caller = _caller_with_mock_process(is_alive=False)
    process = caller.process
    queue = caller.queue

    caller.close()

    process.kill.assert_not_called()
    process.join.assert_called_once_with()
    queue.put.assert_not_called()
    queue.join.assert_not_called()
    assert caller.process is None


def test_destructor_uses_abort_close():
    caller = object.__new__(PersistentAsyncCaller)
    caller.close = mock.Mock()

    PersistentAsyncCaller.__del__(caller)

    caller.close.assert_called_once_with(abort=True)


def test_parent_death_signal_is_installed_on_linux():
    parent_pid = 1234
    libc = mock.Mock()
    libc.prctl.return_value = 0

    with (
        mock.patch.object(async_utils.sys, "platform", "linux"),
        mock.patch.object(async_utils.os, "getppid", return_value=parent_pid),
        mock.patch.object(async_utils.os, "kill") as kill,
        mock.patch.object(async_utils.ctypes, "CDLL", return_value=libc),
    ):
        async_utils._set_parent_death_signal(parent_pid)

    libc.prctl.assert_called_once_with(1, signal.SIGKILL, 0, 0, 0)
    kill.assert_not_called()


def test_results_manager_installs_parent_death_initializer():
    context = mock.Mock()
    manager = mock.Mock()
    results_queue = mock.Mock()
    manager.Queue.return_value = results_queue

    with (
        mock.patch.object(filesystem_async, "_results_manager", None),
        mock.patch.object(filesystem_async, "_results_queue", None),
        mock.patch.object(filesystem_async.mp, "get_context", return_value=context),
        mock.patch.object(filesystem_async, "SyncManager", return_value=manager) as manager_cls,
        mock.patch.object(filesystem_async.os, "getpid", return_value=5678),
    ):
        assert filesystem_async._get_write_results_queue() is results_queue

    manager_cls.assert_called_once_with(ctx=context)
    manager.start.assert_called_once_with(
        initializer=async_utils._set_parent_death_signal, initargs=(5678,)
    )
    manager.Queue.assert_called_once_with()
