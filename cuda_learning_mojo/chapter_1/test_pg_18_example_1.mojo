from gpu.host import DeviceContext
from gpu import thread_idx
from testing import TestSuite


def test_hello_world():
    fn kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel](grid_dim=1, block_dim=(10))
        ctx.synchronize()


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
