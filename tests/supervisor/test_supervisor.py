from weakref import WeakKeyDictionary
import unittest

try:
    import zmq
except ImportError:
    raise unittest.SkipTest("zmq not installed in test harness")
from supervisor import (Context, HostConnected, HostDisconnected, ProcessStarted,
                        ProcessFailedToStart, get_message_queue, ProcessExited,
                        FunctionCall)
from supervisor.launchers import mast
from unittest.mock import patch, Mock
from contextlib import contextmanager
import supervisor
import time
import subprocess
import os
import pickle
import threading
import supervisor.host
import logging
from socket import gethostname
import signal
import tempfile
from pathlib import Path
import sys
import socket
from collections import deque
from tests.supervisor.methods import Reply, Mapper

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s", level=logging.INFO
)


@contextmanager
def context(*args, **kwargs):
    ctx = Context(*args, **kwargs)
    yield ctx
    ctx.shutdown()


@contextmanager
def mock_process_handling():
    class Interface:
        pass

    lock = threading.Lock()
    all_processes = []

    def killpg(pid, return_code):
        with lock:
            p = all_processes[pid]
            if p.immortal and return_code != signal.SIGKILL:
                return
            if not hasattr(p, "returncode"):
                p.returncode = return_code
                with os.fdopen(p._done_w, "w") as f:
                    f.write("done")

    class MockPopen:
        def __init__(self, *args, fail=False, immortal=False, **kwargs):
            if fail:
                raise RuntimeError("process fail")
            with lock:
                self.args = args
                self.kwargs = kwargs
                self.immortal = immortal
                self.pid = len(all_processes)
                all_processes.append(self)
                self.signals_sent = []
                self._done_r, self._done_w = os.pipe()

        def send_signal(self, sig):
            killpg(self.pid, sig)

        def wait(self):
            return getattr(self, 'returncode', None)

    def mock_pidfdopen(pid):
        with lock:
            return all_processes[pid]._done_r

    with patch.object(subprocess, "Popen", MockPopen), patch.object(
        supervisor.host, "pidfd_open", mock_pidfdopen
    ), patch.object(os, "killpg", killpg):
        yield killpg


@contextmanager
def connected_host():
    context: zmq.Context = zmq.Context(1)
    backend = context.socket(zmq.ROUTER)
    backend.setsockopt(zmq.IPV6, True)
    backend.bind("tcp://*:55555")
    exited: Exception = Exception()
    host = supervisor.host.Host("tcp://127.0.0.1:55555")

    def run_host():
        nonlocal exited
        try:
            host.run_event_loop_forever()
        except ConnectionAbortedError as e:
            exited = e
        except SystemExit:
            pass

    thread = threading.Thread(target=run_host, daemon=True)
    thread.start()
    try:
        yield backend, host
    finally:
        backend.close()
        context.term()
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("thread did not terminate")
    host.context.destroy(linger=500)
    if isinstance(exited, ConnectionAbortedError):
        raise exited


@contextmanager
def host_sockets(N: int):
    context: zmq.Context = zmq.Context(1)

    def create_socket():
        backend = context.socket(zmq.DEALER)
        backend.setsockopt(zmq.IPV6, True)
        backend.connect("tcp://127.0.0.1:55555")
        return backend

    sockets = [create_socket() for i in range(N)]
    try:
        yield sockets
    finally:
        context.destroy(linger=500)


def emulate_mast_launch(args, N: int = 4, connections: int = 4):
    # for testing the mast launcher entrypoint
    def create_host(i):
        env = {**os.environ}
        env["TW_TASK_ID"] = str(i)
        env["MAST_HPC_TASK_GROUP_HOSTNAMES"] = socket.gethostname()
        env["HOSTNAME"] = socket.gethostname() if i == 0 else f"fake_host_{i}"
        env["MAST_HPC_TASK_GROUP_SIZE"] = str(N)
        # fast heartbeat so we do not wait so long to see the timeout
        # in the tests
        env["TORCH_SUPERVISOR_HEARTBEAT_INTERVAL"] = str(0.1)
        env["MAST_HPC_TASK_FAILURE_REPLY_FILE"] = '/tmp/reply_file'
        env["MAST_HPC_JOB_ATTEMPT_INDEX"] = "0"
        return subprocess.Popen(args, env=env)

    hosts = [create_host(i) for i in range(connections)]
    expiry = time.time() + 30
    try:
        r = [h.wait(timeout=max(0, expiry - time.time())) for h in hosts]
        return r
    finally:
        for h in hosts:
            if h.poll() is None:
                h.kill()


class SupervisorUnitTests(unittest.TestCase):
    @patch("subprocess.Popen", "subprocess.")
    def test_host_manager(self):
        with mock_process_handling() as kill, connected_host() as (
            socket,
            host,
        ), patch.object(supervisor.host, "ABORT_INTERVAL", 0.01):
            f, msg = socket.recv_multipart()
            _hostname, _, hostname = pickle.loads(msg)

            def launch(
                proc_id,
                rank=0,
                processes_per_rank=1,
                world_size=1,
                popen=None,
                name="fake",
                simulate=False,
                log_file=None,
            ):
                if popen is None:
                    popen = {"env": None}
                msg = (
                    "launch",
                    proc_id,
                    rank,
                    processes_per_rank,
                    world_size,
                    popen,
                    name,
                    simulate,
                    log_file,
                )
                socket.send_multipart([f, pickle.dumps(msg)])

            def send(msg):
                socket.send_multipart([f, pickle.dumps(msg)])

            def recv():
                return pickle.loads(socket.recv_multipart()[1])

            self.assertEqual(_hostname, "_hostname")
            self.assertEqual(hostname, gethostname())

            launch(1)
            self.assertEqual(recv(), ("_started", 1, 0))
            kill(0, 4)
            self.assertEqual(recv(), ("_exited", 1, 4))

            launch(2)
            self.assertEqual(recv(), ("_started", 2, 1))
            send(("send", 2, pickle.dumps("a message")))
            msg_queue = get_message_queue(2, host.proc_addr)
            self.assertEqual(msg_queue.recv().message, "a message")
            send(("send", 2, pickle.dumps("another message")))
            self.assertEqual(msg_queue.recv().message, "another message")
            msg_queue.send(b"a reply")
            msg_queue.close()
            self.assertEqual(recv(), ("_response", 2, pickle.dumps(b"a reply")))
            send(("signal", 2, 8, True))
            self.assertEqual(recv(), ("_exited", 2, 8))
            launch(3, popen={"env": {"foo": "3"}})
            self.assertEqual(recv(), ("_started", 3, 2))
            send(("signal", 3, 9, False))
            self.assertEqual(recv(), ("_exited", 3, 9))
            launch(4, popen={"fail": True, "env": None})
            _started, _, msg = recv()
            self.assertEqual(_started, "_started")
            self.assertIn("process fail", msg)
            launch(5, simulate=True)
            self.assertEqual(recv(), ("_started", 5, 2))
            self.assertEqual(recv(), ("_exited", 5, 0))
            launch(6)  # leave something open
            launch(7, popen={"immortal": True, "env": None})
            send(("abort", None))
        # test double shutodwn
        host.shutdown()

        with self.assertRaises(ConnectionAbortedError), connected_host() as (socket, _):
            f, msg = socket.recv_multipart()
            socket.send_multipart([f, pickle.dumps(("abort", "An error"))])

    def test_host_timeout_and_heartbeat(self):
        with self.assertRaises(ConnectionAbortedError), \
             patch.object(supervisor.host, "HEARTBEAT_INTERVAL", 0.01), connected_host() as (socket, host):
            f, msg = socket.recv_multipart()
            socket.send_multipart([f, b""])
            time.sleep(0.1)
            f, msg = socket.recv_multipart()
            self.assertEqual(msg, b"")

    def test_supervisor_api(self):
        with context() as ctx:
            h0, h1 = ctx.request_hosts(2)
            proc = ctx.create_process_group([h0], args=["test"])[0]
            proc.send("hello")
            pm = ctx.messagefilter(lambda m: m.sender is proc and m.message == "world")
            with host_sockets(1) as (socket,):
                socket.send_pyobj(("_hostname", None, "host0"))
                self.assertEqual(ctx.recv(timeout=1).message.hostname, "host0")
                self.assertIn("host0", str(h0))
                self.assertEqual(socket.recv(), b"")
                expected = (
                    "launch",
                    0,
                    0,
                    1,
                    1,
                    {"args": ["test"], "env": None, "cwd": None},
                    "pg0",
                    False,
                    None,
                )
                self.assertEqual(socket.recv_pyobj(), expected)
                self.assertEqual(socket.recv_pyobj(), ("send", 0, pickle.dumps("hello")))
                self.assertTrue(pm.recvready(timeout=1) == [])
                socket.send_pyobj(("_started", 0, 7))
                socket.send_pyobj(("_response", 0, pickle.dumps("nope")))
                socket.send_pyobj(("_response", 0, pickle.dumps("world")))
                self.assertEqual(ctx.messagefilter(ProcessStarted).recv(timeout=1).message.pid, 7)
                self.assertEqual(pm.recv(timeout=1).message, "world")
                self.assertEqual(ctx.recv(timeout=1).message, "nope")
                ctx.return_hosts([h0])
                self.assertTrue(isinstance(ctx.messagefilter(ProcessExited).recv(timeout=1).message.result,
                                           ConnectionAbortedError))
                self.assertTrue(ctx.messagefilter(HostDisconnected).recv(timeout=1).sender is h0)
                (p,) = ctx.create_process_group([h0], args=["test3"])
                self.assertTrue(isinstance(ctx.messagefilter(ProcessExited).recv(timeout=1).message.result,
                                           ConnectionAbortedError))

            with host_sockets(1) as (socket,):
                socket.send_pyobj(("_hostname", None, "host1"))
                self.assertEqual(socket.recv(), b"")
                self.assertEqual(ctx.recv(timeout=1).message.hostname, "host1")
                proc = ctx.create_process_group([h1], args=["test2"])[0]
                self.assertIn("rank=0", str(proc))
                expected = (
                    "launch",
                    2,
                    0,
                    1,
                    1,
                    {"args": ["test2"], "env": None, "cwd": None},
                    "pg2",
                    False,
                    None,
                )
                self.assertEqual(socket.recv_pyobj(), expected)
                proc = None
                # test sending a message after a proc timeout
                socket.send_pyobj(("_response", 2, pickle.dumps("old response")))
                socket.send(b"")
                self.assertEqual(socket.recv(), b"")

        # now try with host connecting before the host object exists
        with host_sockets(1) as (socket,):
            socket.send_pyobj(("_hostname", None, "host0"))
            with context() as ctx:
                self.assertEqual(socket.recv(), b"")
                (h,) = ctx.request_hosts(1)
                self.assertEqual(ctx.recv(timeout=1).message.hostname, "host0")

    def test_bad_host_managers(self):
        with context() as ctx, host_sockets(5) as (socket0, socket1, socket2, socket3, socket4):
            socket0.send(b"somegarbage")
            self.assertEqual(
                socket0.recv_pyobj(),
                ("abort", "Connection did not start with a hostname"),
            )
            socket1.send_pyobj(("_hostname", None, 7))
            self.assertEqual(
                socket1.recv_pyobj(),
                ("abort", "Connection did not start with a hostname"),
            )
            socket2.send_pyobj(("_hostname", None, "host0"))
            self.assertEqual(socket2.recv(), b"")
            socket2.send_pyobj(("_started", 0, 7))
            self.assertEqual(
                socket2.recv_pyobj(),
                ("abort", "Host manager sent messages before attached."),
            )
            (h,) = ctx.request_hosts(1)
            socket3.send_pyobj(("_hostname", None, "host3"))

            self.assertEqual(ctx.recv(1).sender.hostname, "host3")  # type: ignore
            socket4.send(b"")
            self.assertEqual(
                socket4.recv_pyobj(),
                ("abort", "Connection did not start with a hostname"),
            )

    def test_host_manager_no_heartbeat(self):
        with patch.object(
            supervisor, "HEARTBEAT_INTERVAL", 0.01
        ), context() as ctx, host_sockets(1) as (socket,):
            socket.send_pyobj(("_hostname", None, "host0"))
            self.assertEqual(socket.recv(), b"")
            socket.send(b"")
            self.assertEqual(socket.recv(), b"")
            (h,) = ctx.request_hosts(1)
            self.assertEqual(socket.recv_pyobj(), ("abort", "Host did not heartbeat"))
            socket.send(b"")
            self.assertEqual(
                socket.recv_pyobj(), ("abort", "Supervisor thought host timed out")
            )
            self.assertTrue(ctx.messagefilter(HostDisconnected).recv(timeout=1).sender is h)

    def test_proc_creation(self):
        with context() as ctx, host_sockets(2) as (socket0, socket1):
            h0, h1 = ctx.request_hosts(2)
            socket0.send_pyobj(("_hostname", None, "host0"))
            self.assertEqual(socket0.recv(), b"")
            socket1.send_pyobj(("_hostname", None, "host1"))
            self.assertEqual(socket1.recv(), b"")
            pg = ctx.create_process_group([h0, h1], args=["test"], processes_per_host=3)
            self.assertEqual(len(pg), 6)
            pg[0].signal(signal.SIGTERM)
            for _i in range(3):
                socket0.recv_pyobj()  # launches
            self.assertEqual(socket0.recv_pyobj(), ("signal", 0, signal.SIGTERM, True))
            socket0.send_pyobj(("_response", 0, pickle.dumps("hello")))
            socket0.send_pyobj(("_response", 0, pickle.dumps("world")))
            r = ctx.messagefilter(str)
            self.assertEqual(
                "world", ctx.messagefilter(lambda x: x.message == 'world').recv(timeout=1).message
            )
            self.assertEqual("hello", r.recv(timeout=1).message)
            socket0.send_pyobj(("_started", 1, 8))
            socket0.send_pyobj(("_exited", 1, 7))

            exit_messages = ctx.messagefilter(ProcessExited)
            exited = exit_messages.recv(timeout=1)
            self.assertEqual(7, exited.message.result)
            self.assertTrue(exited.sender is pg[1])
            socket0.send_pyobj(("_started", 2, "Failed"))

            self.assertTrue(isinstance(exit_messages.recv(timeout=1).message.result, ProcessFailedToStart))

    def test_proc_deletion(self):
        with context() as ctx, host_sockets(1) as (socket,):
            h0, = ctx.request_hosts(1)
            socket.send_pyobj(("_hostname", None, "host0"))
            ctx.recv()
            self.assertEqual(socket.recv(), b"")
            pgs_weak = WeakKeyDictionary()
            for _i in range(3):
                pgs = ctx.create_process_group([h0,], args=["test"], processes_per_host=1000)
                for pg in pgs:
                    pgs_weak[pg] = True
                del pgs
                pg = None
                assert len(pgs_weak) == 1000
                for _i in range(1000):
                    _launch, id, *rest = socket.recv_pyobj()
                    socket.send_pyobj(("_started", id, id))
                    socket.send_pyobj(("_exited", id, 0))
                for _ in range(2*1000):
                    ctx.recv()
                assert len(pgs_weak) == 0

    def test_log_redirect(self):
        m = Mock()
        with tempfile.NamedTemporaryFile(delete=True) as f, patch.object(
            os, "dup2", m
        ), context(log_format=f.name):
            pass
        m.assert_called()

    def test_host_lost_first(self):
        with context() as ctx, host_sockets(1) as (socket0,):
            (h0,) = ctx.request_hosts(1)
            (h1,) = ctx.replace_hosts([h0])
            self.assertTrue(h0 is ctx.messagefilter(HostDisconnected).recv(timeout=1).sender)
            socket0.send_pyobj(("_hostname", None, "host0"))
            self.assertEqual(socket0.recv(), b"")
            self.assertEqual(HostConnected("host0"), ctx.recv(1).message)
            self.assertEqual("host0", h1.hostname)

    def test_host_replace(self):
        with context() as ctx:
            b = ctx.request_hosts(2)
            nh = ctx.replace_hosts(b)
            self.assertEqual(len(b), len(nh))

    def test_pstree(self):
        from supervisor.log_pstree import log_pstree_output

        try:
            log_pstree_output(os.getppid())
        except FileNotFoundError:
            self.skipTest("pstree not installed")

    def test_messaging(self):
        with mock_messages([(0, [0, 1]), (1, [2])]) as ctx:
            self.assertEqual(0, ctx.recv())
            self.assertEqual(1, ctx.recv(timeout=0))
            with self.assertRaises(TimeoutError):
                ctx.recv(timeout=.5)
            self.assertEqual(2, ctx.recv(timeout=.6))
        with mock_messages([(3, [1])]) as ctx:
            with self.assertRaises(TimeoutError):
                ctx.recv(timeout=0)
            time.sleep(3)
            self.assertEqual(1, ctx.recv(timeout=0))

        with mock_messages([(.5, [1, 2]), (1, [3, 4]), (1, [4, 5]), (1.1, [6])]) as ctx:
            self.assertEqual([], ctx.recvready())
            self.assertEqual([1, 2], ctx.recvready(timeout=None))
            self.assertEqual([], ctx.recvready(timeout=0.2))
            self.assertEqual(time.time(), .7)
            self.assertEqual([3, 4, 4, 5], ctx.recvready(timeout=.5))
            self.assertEqual([6], ctx.recvready(timeout=.5))

        with mock_messages([(0, [1, 2]), (0, [3, 4]), (1, [5, 6, 7, 8]), (1, [10, 11])]) as ctx:
            odd = ctx.messagefilter(lambda x: x % 2 != 0)
            even = ctx.messagefilter(lambda x: x % 2 == 0)
            self.assertEqual(2, even.recv())
            self.assertEqual(1, odd.recv())
            self.assertEqual(3, odd.recv())
            self.assertEqual(4, ctx.recv())
            self.assertEqual(6, even.recv())
            self.assertEqual(8, even.recv())
            self.assertEqual(5, odd.recv())
            self.assertEqual(7, odd.recv())
            self.assertEqual(11, odd.recv())
            self.assertEqual(10, even.recv())

        with mock_messages([(0, [1, 1]), (0, [2, 2]), (1, [5, 6, 7, 8]), (1, [10, 11])]) as ctx:
            odd = ctx.messagefilter(lambda x: x % 2 != 0)
            even = ctx.messagefilter(lambda x: x % 2 == 0)
            self.assertEqual(2, even.recv(0))
            self.assertEqual(2, even.recv(0))
            self.assertEqual(1, odd.recv(0))
            self.assertEqual(1, ctx.recv(0))
            with self.assertRaises(TimeoutError):
                self.assertEqual(3, odd.recv(0))
            self.assertEqual(6, even.recv(1))
            self.assertEqual(8, even.recv(0))
            self.assertEqual(5, odd.recv(0))
            self.assertEqual(7, odd.recv(0))
            self.assertEqual(11, odd.recv(0))
            self.assertEqual(10, even.recv(0))

        with mock_messages([(0, [1, 1]), (0, [2, 2]), (1, [5, 6, 7, 8]), (1, [10, 11])]) as ctx:
            odd = ctx.messagefilter(lambda x: x % 2 != 0)
            even = ctx.messagefilter(lambda x: x % 2 == 0)
            self.assertEqual([2, 2], even.recvready())
            self.assertEqual([], even.recvready())
            self.assertEqual([1, 1], odd.recvready())
            self.assertEqual([6, 8, 10], even.recvready(1))
            self.assertEqual([5, 7, 11], odd.recvready(0))

        with mock_messages([(0, [1, 1]), (0, [2, 2]), (1, [5, 6, 7, 8]), (1, [10, 11])]) as ctx:
            even = ctx.messagefilter(lambda x: x % 2 == 0)
            self.assertEqual(2, even.recv())
            self.assertEqual(1, ctx.recv())
            self.assertEqual(1, ctx.recv())
            self.assertEqual(2, ctx.recv())
            self.assertEqual(6, even.recv())
            self.assertEqual(5, ctx.recv())
            self.assertEqual(8, even.recv())
            self.assertEqual(7, ctx.recv())
            self.assertEqual(10, ctx.recv())
            self.assertEqual(11, ctx.recv())

        with mock_messages([(0, [1, 1]), (1, [3, 4]), (1, [5, 6, 7, 8]), (2, [10, 11])]) as ctx:
            even = ctx.messagefilter(lambda x: x % 2 == 0)
            self.assertEqual([4, 6, 8], even.recvready(1))
            self.assertEqual([1, 1, 3, 5, 7], ctx.recvready())
            self.assertEqual([10], even.recvready(1))
            self.assertEqual(11, ctx.recv())


@contextmanager
def mock_messages(messages):
    orig_bind = zmq.Socket.bind

    def bind(self, pattern):
        if pattern.startswith('tcp://'):
            return
        return orig_bind(self, pattern)
    with patch.object(threading.Thread, 'start', new=lambda self: None), patch.object(zmq.Socket, 'bind', new=bind):
        the_time = 0
        m = 0
        context = Context()
        requests_ready = deque()

        def time():
            return the_time

        def sleep(t):
            nonlocal the_time
            the_time += t

        def poll(self, timeout):
            nonlocal the_time, m
            if m >= len(messages):
                if timeout is None:
                    raise RuntimeError("would never return")
                sleep(timeout/1000)
                return []
            t, msg = messages[m]
            if timeout is None or the_time + timeout/1000 >= t:
                m += 1
                the_time = t
                sleep(t - the_time)
                context._delivered_messages.append(msg)
                requests_ready.append(b'')
                return [context._doorbell]
            else:
                sleep(timeout/1000)
                return []

        def recv(self):
            if not requests_ready:
                poll(self, None)
            return requests_ready.popleft()

        with (patch('time.time', new=time),
              patch('time.sleep', new=sleep),
              patch.object(zmq.Poller, 'poll', new=poll),
              patch.object(zmq.Socket, 'recv', new=recv)):
            yield context
            context._backend.close()
            context._requests_ready.close()
            context._doorbell.close()
            context._context.term()


class SupervisorIntegrationTests(unittest.TestCase):
    def launch(
        self,
        health,
        train,
        expect,
        N=4,
        run_fraction: float = 1,
        rank_fraction: float = 1,
        connections=4,
    ):
        test_name = Path(__file__).parent / "supervisor_integration.py"

        config = {
            "N": N,
            "health": health,
            "train": train,
            "run_fraction": run_fraction,
            "rank_fraction": rank_fraction,
        }
        result = emulate_mast_launch(
            [sys.executable, test_name, "--supervise", repr(config)], N, connections
        )
        failed = sum(1 for r in result if r != 0)
        self.assertEqual(expect, failed)
        return result

    def check_supervisor_function(self, name: str, expect: int = 0, N=4, connections=4):
        result = emulate_mast_launch(
            [sys.executable, __file__, "--supervise", name], N, connections
        )
        failed = sum(1 for r in result if r != 0)
        self.assertEqual(expect, failed)

    def test_success(self):
        self.launch(health=[[4, 3, 2, 1]], train=["........"], expect=0)

    def test_fail(self):
        self.launch(
            health=[[4, 3, 2, 1], [4, 3, 2, 1]],
            train=["....F...", "........"],
            expect=0,
        )

    def test_hang(self):
        self.launch(
            health=[[4, 3, "hang", 1], [4, 3, 2, 1]],
            train=["......"],
            rank_fraction=0.75,
            run_fraction=0.75,
            expect=0,
        )

    def test_error_fail(self):
        self.launch(
            health=[[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]],
            train=["...E..", "....F.", "......"],
            rank_fraction=0.75,
            run_fraction=0.75,
            expect=1,
        )

    def test_error_error(self):
        self.launch(
            health=[[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]],
            train=["...E..", ".E....", "......"],
            rank_fraction=0.75,
            run_fraction=0.75,
            connections=5,
            expect=2,
        )

    def test_function_call_process(self):
        self.check_supervisor_function("function_call_process")
        self.check_supervisor_function("function_call_process_module")

    def test_mapreduce(self):
        self.check_supervisor_function("map_reduce")


def function_call_process(ctx, hosts):
    host_connected = ctx.messagefilter(HostConnected)
    for _ in hosts:
        host_connected.recv(timeout=1)
    hosts[0].create_process(FunctionCall('tests.supervisor.methods.reply_hello', 3, 4, x=5))
    r = ctx.messagefilter(Reply).recv(timeout=1)
    assert r.message == Reply(3, 4, x=5)

def function_call_process_module(ctx, hosts):
    for _ in hosts:
        ctx.recv(timeout=1)  # connected
    hosts[0].create_process(FunctionCall('builtins.print', "hello, world"))
    m = ctx.recv(timeout=1)
    assert isinstance(m.message, ProcessStarted)
    assert ctx.recv(timeout=1).message.result == 0
    hosts[0].create_process(FunctionCall('builtins.this_doesnt_work', "hello, world"))
    assert isinstance(ctx.recv(timeout=1).message, ProcessStarted)
    m = ctx.recv(timeout=1)
    assert m.message.result != 0

def map_reduce(ctx, hosts):
    for _ in hosts:
        ctx.recv(timeout=1)  # connected
    from supervisor.mapreduce import mapreduce
    for items, branch in (([*range(17)], 4), ([*range(16)], 2), ([*range(1)], 2)):
        r = mapreduce(hosts, Mapper(), items, branch=branch)
        assert sum(2*i for i in items) == r


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == '--supervise':
        mast(globals()[sys.argv[2]])
    else:
        unittest.main()
