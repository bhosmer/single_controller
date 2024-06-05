import traceback
import sys
from supervisor.logging import fix_exception_lines
def custom_excepthook(exc_type, exc_value, exc_traceback):
    tb_lines = fix_exception_lines(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print('\n'.join(tb_lines), file=sys.stderr)

sys.excepthook = custom_excepthook

from collections import defaultdict
from supervisor import Context, HostConnected
from supervisor.host import Host as HostManager
from controller import world_mesh, active_mesh, _Controller, DeviceMesh, remote_function, Future, RemoteException, fetch_shard, Stream, active_stream
from contextlib import contextmanager, ExitStack
from threading import Thread
from functools import cache, partial
from unittest import main, TestCase
from time import sleep
import torch
import signal
import os
from unittest.mock import patch
from weakref import WeakKeyDictionary
from controller._testing import LocalContext

@remote_function('controller.worker.log')
def log(*args):
    pass

@remote_function('builtins.list')
def rlist(elem):
    return elem

@remote_function('controller._testing.do_bogus_tensor_work')
def do_bogus_tensor_work(x, y, fail_rank=None):
    return x + y  # real function actually does x @ y


local = LocalContext()

class TestController(TestCase):

    @classmethod
    def tearDownClass(cls):
        local.close()

    @classmethod
    def local_device_mesh(cls, N, gpu_per_host, activate=True):
        return local.local_device_mesh(N, gpu_per_host, activate)

    def test_hello(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            log(device_mesh)

    def test_simple_tensors(self):
        with self.local_device_mesh(2, 2):
            x = torch.rand(3, 4)
            y = x + x
            log("%s %s", x, y)
            z = torch.std_mean(x)
            log("%s", z)

    def test_errors(self):
        t = torch.rand(3, 4)
        with self.local_device_mesh(2, 2) as device_mesh:
            y = torch.rand(3, 4)
            with self.assertRaisesRegex(TypeError, 'LOCAL_TENSOR'):
                t.add(y)
            with self.assertRaisesRegex(TypeError, 'WRONG_MESH'):
                sm = device_mesh(host=0)
                with active_mesh(sm):
                    x = torch.rand(3, 4)
                    x.add(y)

    def test_mesh_index(self):
        fake_processes = tuple(range(0, 2*3*4))
        dm = DeviceMesh(None, fake_processes, {'a': 2, 'b': 3, 'c': 4})
        self.assertEqual(0, dm(a=0, b=0, c=0).processes[0])
        x = dm(a=0, b=0)
        self.assertEqual(x.processes, fake_processes[0:4])
        self.assertEqual(x.dims, {'c': 4})
        x = dm(c=slice(None, None, 2))
        self.assertEqual(x.processes, fake_processes[::2])
        x = dm(b=2, c=3)
        self.assertEqual(x.processes, (11, 23))

    def test_sub_mesh(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            h0 = device_mesh(host=0)
            h1 = device_mesh(host=1)
            with active_mesh(h0):
                x = torch.rand(3, 4)
                log('x: %s', x)
            with active_mesh(h1):
                y = torch.rand(3, 4)
                log('y: %s', y)
                with self.assertRaisesRegex(TypeError, 'WRONG_MESH'):
                    log("x: %s", x)


    def test_user_call(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4)
            y = rlist((x + 1, x))
            log("%s", y)

            # resume monday:
            # 1. tensor ctor resource guard (done)
            # 2. __torch_dispatch__ forward of normal ops (done)
            # 3. collectives created for device mesh
            # 4. implement comms APIs
            # 5. transfer tensor back, and simple future to wait for result.

    @patch('torch.distributed.new_group', new=lambda ranks, use_local_synchronization: ranks)
    def test_worker_mesh_init(self):
        from controller.worker import DeviceMesh as WorkerDeviceMesh
        wdm = WorkerDeviceMesh({'a': 3, 'b': 4}, ranks=list(range(3*4)), index=1)
        a, b = wdm.dims['a'], wdm.dims['b']
        self.assertEqual(b.process_group, [0, 1, 2, 3])
        self.assertEqual(b.rank, 1)

        self.assertEqual(a.process_group, [1, 5, 9])
        self.assertEqual(a.rank, 0)


        wdm = WorkerDeviceMesh({'a': 3, 'b': 4}, ranks=list(range(3*4)), index=6)
        a, b = wdm.dims['a'], wdm.dims['b']
        self.assertEqual(b.process_group, [4, 5, 6, 7])
        self.assertEqual(b.rank, 2)
        self.assertEqual(a.process_group, [2, 6, 10])
        self.assertEqual(a.rank, 1)

        wdm = WorkerDeviceMesh({'a': 3, 'b': 4, 'c': 2}, ranks=list(range(3*4*2)), index=10)
        print(wdm.dims)

    def test_reduce(self):

        with self.local_device_mesh(2, 2) as device_mesh:
            x = 12*2*device_mesh.rank('host') + 12*device_mesh.rank('gpu') + torch.arange(12).reshape(3, 4)
            x = x.cuda()
            y = x.reduce('gpu', 'sum')
            g = x.reduce('gpu', 'stack')
            with self.assertRaisesRegex(TypeError, 'When scattering'):
                x = x.reduce('gpu', 'sum', scatter=True)
            x = x.reshape(2, 6)
            atoa = x.reduce('gpu', 'stack', scatter=True)
            rs = x.reduce('gpu', 'sum', scatter=True)
            log("x: %s\ny:%s\ng:%s\natoa:%s\nrs:%s\n", x, y, g, atoa, rs)

    def test_fetch(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            h = device_mesh.rank('host')
            g = device_mesh.rank('gpu')
            for hi in range(2):
                for gi in range(2):
                    x, y = fetch_shard((h, g), dict(host=hi, gpu=gi)).result()
                    with active_mesh(None):
                        self.assertTrue((hi, gi) == (x.item(), y.item()))

    def test_remote_exception(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4)
            y = torch.rand(3, 4)
            z = do_bogus_tensor_work(x, y)
            a = z + x
            b = x + y            
            with self.assertRaisesRegex(RemoteException, 'do_bogus_tensor_work'):
                r = fetch_shard(a).result(timeout=10)
            # but values not dependent on z are fine
            fetch_shard(b).result(timeout=10)

    def test_distributed_error(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4).cuda()
            y = torch.rand(3, 4).cuda()
            # z is broken on rank 1 but not others
            z = do_bogus_tensor_work(x, y, fail_rank=1)
            # test that rank 1 is still doing work despite z failing
            a = (x + y).reduce('gpu')
            fetch_shard(a).result()
            # but z itself should fail, even if we do not fetch it from rank 1
            # (since fetch shard says we first want to assert the whole tensor is correct)
            with self.assertRaisesRegex(RemoteException, 'do_bogus_tensor_work'):
                fetch_shard(z).result()
            # try to reduce z, which should fail, but ranks that are not 1 do not 
            # know about the failure. Rank 1 should still participate in the reduce
            # to unblock work.
            rz = z.reduce('gpu')
            # but we should see the error message still retrieving it because it is
            # dependent on an error.
            with self.assertRaisesRegex(RemoteException, 'do_bogus_tensor_work'):
                fetch_shard(rz).result()
            # however, we should still be able to compute and get a result back
            # from host 1, signaling that the reduction didn't get cuda compute stuck.
            fetch_shard(2*x, {'gpu': 1, 'host': 0}).result()

    def test_future(self):
        the_time = 0
        the_messages = []
        class MockController:
            def _read_messages(self, timeout):
                nonlocal the_time
                if not the_messages:
                    return
                time, action = the_messages[0]
                if timeout is None or time <= the_time + timeout:
                    the_time = time
                    action()
                    the_messages.pop(0)
                else:
                    the_time += timeout
            def _request_status(self):
                pass

        ctrl = MockController()

        def time():
            return the_time

        with patch('time.time', time):
            f = Future(ctrl)
            the_messages = [(1, lambda: f._set_result(4))]
            self.assertTrue(not f.done())
            with self.assertRaises(TimeoutError):
                f.result(timeout=.5)
            self.assertEqual(4, f.result(timeout=1))
            self.assertIsNone(f.exception())
            self.assertTrue(f.done())
            f = Future(ctrl)
            the_messages = [(1, lambda: None), (2, lambda: f._set_result(3))]
            the_time = 0
            self.assertEqual(3, f.result())
            f = Future(ctrl)
            re = RemoteException(0, Exception(), [], [])

            the_messages = [(1, lambda: None), (2, lambda: f._set_result(re))]
            the_time = 0
            self.assertIsNotNone(f.exception())

            f = Future(ctrl)
            the_messages = [(0, lambda: None), (.2, lambda: f._set_result(7))]
            the_time = 0
            self.assertEqual(7, f.result(timeout=.3))

    def test_mutate(self):

        with self.local_device_mesh(2, 2) as device_mesh:
            x = torch.rand(3, 4).cuda()
            x.abs_()
            s = Stream('other')
            b = s.borrow(x)
            with self.assertRaisesRegex(ValueError, "would be mutated"):
                x.abs_()
            with active_stream(s):
                c = b.add(b)
            b.drop()
            x.abs_()
            b = s.borrow(x, mutable=True)
            with active_stream(s):
                b.abs_()
            b.drop()
            # del b
            x.abs_()

    def test_movement(self):
        with self.local_device_mesh(2, 2) as device_mesh:
            
            sm0 = device_mesh(host=0)
            sm1 = device_mesh(host=1)

            with active_mesh(sm0):
                x = torch.rand(3, 4, device='cuda')
                log("x: %s", x)
                y = x.to_mesh(sm1)

            a = torch.rand(3, 4, device='cuda')

            b = a.slice_mesh(host=0)
            d = b.to_mesh(sm0)
            c = b.to_mesh(sm1)
            with active_mesh(sm0):
                log("d: %s", d)
            with active_mesh(sm1):
                log("y: %s", y)
                log("c: %s", c)


    def test_simple_examples(self):
        # `local_device_mesh` is just a helper for testing
        # that sets up the worker processes/host managers/etc.
        # locally. For 'real' programs the initial device_mesh
        # will be provided at program start.
        with self.local_device_mesh(2, 2, activate=False) as device_mesh:

            print(device_mesh)
            # <DeviceMesh(('host', 'gpu'), (2, 2)) at 0x7fa6175b3bb0>
            h0 = device_mesh(host=0)
            h1 = device_mesh(host=1)


            # Device Meshes are multi-dimension lists with named
            # dimensions. On startup they will initially have a host and gpu dimension.


            # When there is no active device mesh, compute is local
            t = torch.rand(1)
            print(t)
            # tensor([0.6893])

            # now _all_ compute is done on the device mesh within this context
            with active_mesh(device_mesh):
                x = torch.rand(3, 4)
                y = x + x
                print(y)
                # DTensor(mesh=..., stream=<Stream('main') at 0x7f79451b3c10>, fake=FakeTensor(..., size=(3, 4)))
                with self.assertRaisesRegex(TypeError, 'LOCAL_TENSOR'):
                    z = y.add(t)
                    """
                    TypeError: Mismatched arguments to distributed tensor operation:

                    torch.ops.aten.add.Tensor(., LOCAL_TENSOR)

                    active_mesh = <DeviceMesh(('host', 'gpu'), (2, 2)) at 0x7f1a468bbe20>
                    active_stream = <Stream('main') at 0x7f1a468bbd30>
                    LOCAL_TENSOR: A local (non-distributed) tensor is being passed while a device_mesh is active.
                    If you want to do local tensor compute use `with active_mesh(None):`
                    """
                # we can use helper functions to get the worker machines to log tensor info they have
                log("Y: %s", y)
                # worker_0: Y: tensor([[0.7125, 0.9058, 0.8245, 0.7008],
                # worker_0:         [1.3899, 0.9606, 0.4697, 1.9011],
                # worker_0:         [1.7506, 1.9513, 0.5936, 1.2739]])
                # worker_3: Y: tensor([[1.1647, 1.8845, 1.7686, 0.2304],
                # worker_3:         [0.4569, 0.9294, 0.0358, 0.8630],
                # worker_3:         [1.3946, 1.3274, 1.1046, 0.3136]])

                # log isn't special, it is just a remote function call.
                #  Note how we use strings to name functions on the controller.
                #  This is so the controller doesn't have to load modules
                #  That might only be installed on the workers, or cannot work
                #  without initializing a cuda context.
                #     log = RemoteFunction('controller.worker.log')

                # If you have a remote function that returns tensors, then you can
                # specify a type propagation function when creating it.

                #





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # this is normally done by the host process in a singal handler
        # but we are using threads to run the host managers in this
        # test setup, it is necessary to have this process issue the
        # kills of any running worker processes
        local.interrupt()
        raise
