from sys.info import size_of
from random import rand, seed
from gpu.host import DeviceContext, get_gpu_target
from gpu import thread_idx


alias Float32Ptr = UnsafePointer[Float32]


fn initialize_data(ptr: Float32Ptr, n_elem: UInt):
    rand[Float32.dtype](ptr, n_elem)


fn sum_arrays_on_gpu(a: Float32Ptr, b: Float32Ptr, c: Float32Ptr, n_elem:UInt):
    idx: Int = thread_idx.x
    if idx < n_elem:
        c[idx] = a[idx] + b[idx]


fn main() raises:
    var n_elem:UInt = 1024
    var n_bytes:UInt = n_elem * UInt(size_of[Float32]())

    # Initialize the host values
    var h_A = Float32Ptr.alloc(n_bytes)
    var h_B = Float32Ptr.alloc(n_bytes)
    var h_C = Float32Ptr.alloc(n_bytes)
    initialize_data(h_A, n_elem)
    initialize_data(h_B, n_elem)
    # Initialize the device values
    with DeviceContext() as ctx:
        var d_A = ctx.enqueue_create_buffer[Float32.dtype](n_bytes)
        var d_B = ctx.enqueue_create_buffer[Float32.dtype](n_bytes)
        var d_C = ctx.enqueue_create_buffer[Float32.dtype](n_bytes)

        ctx.enqueue_copy[Float32.dtype](d_A, h_A)
        ctx.enqueue_copy[Float32.dtype](d_B, h_B)

        ctx.enqueue_function[sum_arrays_on_gpu](
            d_A,
            d_B,
            d_C,
            n_elem,
            grid_dim=1, 
            block_dim=n_elem
        )
        ctx.enqueue_copy(h_C, d_C)
        ctx.synchronize()

    for i in range(n_elem):
        print(h_A[i], ' + ', h_B[i], ' = ', h_C[i])

