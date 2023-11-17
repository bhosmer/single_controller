import zmq
import sys
import time
import os
import pickle
import subprocess
from typing import NamedTuple, Any
import ctypes
from supervisor import HEARTBEAT_INTERVAL
import signal
ABORT_INTERVAL = 5
__NR_pidfd_open = 434
libc = ctypes.CDLL(None)
syscall = libc.syscall
# older libc do not have this syscall
def pidfd_open(pid):
    return syscall(__NR_pidfd_open, pid, 0)

# objects in this file represent Host/Process
# on the host machine itself.

# main package has Host/Process used by
# the supervisor.

class Process:
    def __init__(self, proc_comm, proc_id, rank, world_size, args, proc_addr):
        self.proc_id = proc_id
        self.proc_comm = proc_comm
        environ = dict(os.environ)
        environ['RANK'] = str(rank)
        environ['WORLD_SIZE'] = str(world_size)
        environ['SUPERVISOR_PIPE'] = proc_addr
        environ['SUPERVISOR_IDENT'] = str(proc_id)
        self.subprocess = subprocess.Popen(args, env=environ)
        self.fd = pidfd_open(self.subprocess.pid)
        self.proc_id_bytes = proc_id.to_bytes(8, byteorder='little')
        self.deferred_sends = []

    def _send(self, msg):
        msg = pickle.dumps(msg)
        if self.deferred_sends is not None:
            self.deferred_sends.append(msg)
        else:
            self.proc_comm.send_multipart([self.proc_id_bytes, msg])

    def _notify_connected(self):
        if self.deferred_sends is not None:
            for msg in self.deferred_sends:
                self.proc_comm.send_multipart([self.proc_id_bytes, msg])
            self.deferred_sends = None



class Host:
    def __init__(self, supervisor_port):
        self.context = zmq.Context(1)
        self.backend = self.context.socket(zmq.DEALER)
        self.backend.connect(supervisor_port)

        # initial heartbeat to tell supervisor we exist
        self.heartbeat()

        self.poller = zmq.Poller()
        self.poller.register(self.backend, zmq.POLLIN)

        self.proc_comm = self.context.socket(zmq.ROUTER)
        self.proc_addr = f'ipc:///tmp/proc_{os.getpid()}'
        self.proc_comm.bind(self.proc_addr)
        self.poller.register(self.proc_comm, zmq.POLLIN)

        self.process_table = {}
        self.fd_to_pid = {}

    def heartbeat(self):
        self.backend.send(b'')

    # TODO: validate these are valid messages to send

    def launch(self, proc_id, rank, world_size, args):
        process = Process(self.proc_comm, proc_id, rank, world_size, args, self.proc_addr)
        self.process_table[process.proc_id_bytes] = process
        self.fd_to_pid[process.fd] = process.proc_id_bytes
        self.poller.register(process.fd, zmq.POLLIN)
        self.backend.send(pickle.dumps(('_started', process.proc_id, process.subprocess.pid)))

    def send(self, proc_id, msg):
        proc_id = proc_id.to_bytes(8, byteorder='little')
        if proc_id in self.process_table:
            process = self.process_table[proc_id]
            process._send(msg)

    def signal(self, proc_id, sig, group):
        proc_id = proc_id.to_bytes(8, byteorder='little')
        if proc_id in self.process_table:
            process = self.process_table[proc_id]
            if group:
                os.killpg(process.subprocess.pid, sig)
            else:
                process.send_signal(sig)

    def _fd_exit(self, fd):
        pid_bytes = self.fd_to_pid.pop(fd)
        process = self.process_table.pop(pid_bytes)
        returncode = process.subprocess.wait()
        self.poller.unregister(fd)
        os.close(fd)
        return process, returncode

    def abort(self, with_error):
        for proc in self.process_table.values():
            os.killpg(proc.subprocess.pid, signal.SIGTERM)
        expiry = time.time() + ABORT_INTERVAL
        ttl = ABORT_INTERVAL
        while ttl > 0 and self.process_table:
            for s, _ in poller.poll(timeout=ttl):
                if isinstance(s, int):
                    self._fd_exit(s)
            ttl = time.time() - expiry
        if self.process_table:
            for proc in self.process_table.values():
                os.killpg(proc.subprocess.pid, signal.SIGKILL)
        if with_error:
            raise ConnectionAbortedError("Supervisor aborted host")
        else:
            sys.exit(0)

    def run_event_loop_forever(self):
        heartbeat_at = time.time() + HEARTBEAT_INTERVAL
        while True:
            for s, _ in self.poller.poll(timeout=HEARTBEAT_INTERVAL):
                if isinstance(s, int):
                    process, returncode = self._fd_exit(s)
                    self.backend.send(pickle.dumps(('_exited', process.proc_id, returncode)))
                elif s is self.backend:
                    cmd, *args = pickle.loads(self.backend.recv())
                    getattr(self, cmd)(*args)
                elif s is self.proc_comm:
                    proc_id_bytes, msg = self.proc_comm.recv_multipart()
                    process = self.process_table[proc_id_bytes]
                    process._notify_connected()
                    if len(msg):
                        self.backend.send(pickle.dumps(('_response', process.proc_id, msg)))

            if time.time() > heartbeat_at:
                heartbeat_at = time.time() + HEARTBEAT_INTERVAL
                self.backend.send(b"")

if __name__ == '__main__':
    manager = Host(sys.argv[1])
    manager.run_event_loop_forever()