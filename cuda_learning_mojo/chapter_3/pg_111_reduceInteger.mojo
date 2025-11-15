import sys
import time
from random import rand, seed, randint
from gpu import thread_idx, block_idx, block_dim, WARP_SIZE, barrier
from gpu.host import DeviceContext, Dim, HostBuffer
from gpu.host.info import GPUInfo, _accelerator_arch
from memory import memset
from gpu.random import Random


alias Int32Ptr = UnsafePointer[Int32]


struct Timer:
    var fn_name: String
    var start_sec: Float64

    fn __init__(out self, var fn_name: String):
        self.fn_name = fn_name
        self.start_sec = time.perf_counter_ns() / 1e9

    fn __enter__(ref self) -> ref [self] Self:
        return self

    fn __exit__(self):
        var end_sec = time.perf_counter_ns() / 1e9
        print(
            self.fn_name, "elapsed time: ", end_sec - self.start_sec, "seconds"
        )


fn initialize_data[n_elem: Int](ptr: Int32Ptr):
    randint(ptr, n_elem, 0, 255)


fn host_accumulate_to_index_0(out_data: Int32Ptr, n: Int):
    for i in range(1, n):
        out_data[0] += out_data[i]


fn host_reduce_neighbored[n: Int](in_data: Int32Ptr, mut out_data: Int32Ptr):
    var num: Int32 = 0
    for i in range(n):
        num += in_data[i]
    out_data[0] = num


fn reduce_neighbored[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    var block_idx_start = block_idx.x * block_dim.x
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    if idx >= n:
        return

    var stride: UInt = 1
    while stride < block_dim.x:
        if tid % (UInt(2) * stride) == 0:
            i_data[tid] += i_data[tid + stride]

        barrier()

        stride *= 2

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_neighbored_less[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    var block_idx_start = block_idx.x * block_dim.x
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    if idx >= n:
        return

    var stride: UInt = 1
    while stride < block_dim.x:
        var index: UInt = UInt(2) * stride * tid
        if index < block_dim.x:
            i_data[index] += i_data[index + stride]

        barrier()

        stride *= 2

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_interleaved[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    var block_idx_start = block_idx.x * block_dim.x
    var idx: UInt = block_idx_start + thread_idx.x
    var i_data: Int32Ptr = in_data + block_idx_start

    if idx >= n:
        return

    var stride: UInt = UInt(Int(block_dim.x / UInt(2)))
    while stride > 0:
        if tid < stride:
            i_data[tid] += i_data[tid + stride]

        barrier()
        stride >>= 1

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_2[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    # Access 2 blocks per thread.
    var block_idx_start = block_idx.x * block_dim.x * 2
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    if idx + block_dim.x < n:
        in_data[idx] += in_data[idx + block_dim.x]
    barrier()

    var stride: UInt = UInt(Int(block_dim.x / 2))
    while stride > 0:
        if tid < stride:
            i_data[tid] += i_data[tid + stride]

        barrier()
        stride >>= 1

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_4[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    var unrolled_factor: UInt = 4
    # Access 2 blocks per thread.
    var block_idx_start = block_idx.x * block_dim.x * unrolled_factor
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    # print('stage 1')
    if idx + block_dim.x * (unrolled_factor - 1) < n:
        # Can we make this a SIMD?
        a1: Int32 = in_data[idx]
        a2: Int32 = in_data[idx + block_dim.x]
        a3: Int32 = in_data[idx + block_dim.x * 2]
        a4: Int32 = in_data[idx + block_dim.x * 3]
        in_data[idx] = a1 + a2 + a3 + a4
    barrier()
    # print('stage 2')

    var stride: UInt = UInt(Int(block_dim.x / 2))
    while stride > 0:
        if tid < stride:
            i_data[tid] += i_data[tid + stride]

        barrier()
        stride >>= 1

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_8[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    var unrolled_factor: UInt = 8
    # Access 2 blocks per thread.
    var block_idx_start = block_idx.x * block_dim.x * unrolled_factor
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    if idx + block_dim.x * (unrolled_factor - 1) < n:
        # Can we make this a SIMD?
        a1: Int32 = in_data[idx]
        a2: Int32 = in_data[idx + block_dim.x]
        a3: Int32 = in_data[idx + block_dim.x * 2]
        a4: Int32 = in_data[idx + block_dim.x * 3]
        a5: Int32 = in_data[idx + block_dim.x * 4]
        a6: Int32 = in_data[idx + block_dim.x * 5]
        a7: Int32 = in_data[idx + block_dim.x * 6]
        a8: Int32 = in_data[idx + block_dim.x * 7]
        in_data[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    barrier()

    var stride: UInt = UInt(Int(block_dim.x / 2))
    while stride > 0:
        if tid < stride:
            i_data[tid] += i_data[tid + stride]

        barrier()
        stride >>= 1

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_8_last_warp[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    var unrolled_factor: UInt = 8
    # Access 2 blocks per thread.
    var block_idx_start = block_idx.x * block_dim.x * unrolled_factor
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    if idx + block_dim.x * (unrolled_factor - 1) < n:
        a1: Int32 = in_data[idx]
        a2: Int32 = in_data[idx + block_dim.x]
        a3: Int32 = in_data[idx + block_dim.x * 2]
        a4: Int32 = in_data[idx + block_dim.x * 3]
        a5: Int32 = in_data[idx + block_dim.x * 4]
        a6: Int32 = in_data[idx + block_dim.x * 5]
        a7: Int32 = in_data[idx + block_dim.x * 6]
        a8: Int32 = in_data[idx + block_dim.x * 7]
        in_data[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    barrier()

    var stride: UInt = UInt(Int(block_dim.x / 2))
    while stride > UInt(WARP_SIZE):
        if tid < stride:
            i_data[tid] += i_data[tid + stride]

        barrier()
        stride >>= 1

    if tid < UInt(WARP_SIZE):
        # Can we make this a SIMD?
        alias strides = [32, 16, 8, 4, 2, 1]
        @parameter
        for stride in strides:
            # print('adding stride',stride)
            i_data.store[volatile=True](
                tid, 
                i_data.load[volatile=True](tid) + i_data.load[volatile=True](tid + UInt(stride))
            )

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_8_unroll_everything[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr):
    var tid: UInt = thread_idx.x
    var unrolled_factor: UInt = 8
    # Access 2 blocks per thread.
    var block_idx_start = block_idx.x * block_dim.x * unrolled_factor
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    if idx + block_dim.x * (unrolled_factor - 1) < n:
        a1: Int32 = in_data[idx]
        a2: Int32 = in_data[idx + block_dim.x]
        a3: Int32 = in_data[idx + block_dim.x * 2]
        a4: Int32 = in_data[idx + block_dim.x * 3]
        a5: Int32 = in_data[idx + block_dim.x * 4]
        a6: Int32 = in_data[idx + block_dim.x * 5]
        a7: Int32 = in_data[idx + block_dim.x * 6]
        a8: Int32 = in_data[idx + block_dim.x * 7]
        in_data[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    barrier()

    if (block_dim.x >= 1024 and tid < 512):
        i_data[tid] += i_data[tid + 512]
    barrier()
    if (block_dim.x >= 512 and tid < 256):
        i_data[tid] += i_data[tid + 256]
    barrier()
    if (block_dim.x >= 256 and tid < 128):
        i_data[tid] += i_data[tid + 128]
    barrier()
    if (block_dim.x >= 128 and tid < 64):
        i_data[tid] += i_data[tid + 64]
    barrier()

    if tid < UInt(WARP_SIZE):
        # Can we make this a SIMD?
        alias strides = [32, 16, 8, 4, 2, 1]
        @parameter
        for stride in strides:
            # print('adding stride',stride)
            i_data.store[volatile=True](
                tid, 
                i_data.load[volatile=True](tid) + i_data.load[volatile=True](tid + UInt(stride))
            )

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn reduce_unrolling_8_unroll_everything_templated[n: UInt](in_data: Int32Ptr, out_data: Int32Ptr, block_size:UInt):
    var tid: UInt = thread_idx.x
    var unrolled_factor: UInt = 8
    # Access 2 blocks per thread.
    var block_idx_start = block_idx.x * block_dim.x * unrolled_factor
    var idx: UInt = block_idx_start + thread_idx.x

    var i_data: Int32Ptr = in_data + block_idx_start

    if idx + block_dim.x * (unrolled_factor - 1) < n:
        a1: Int32 = in_data[idx]
        a2: Int32 = in_data[idx + block_dim.x]
        a3: Int32 = in_data[idx + block_dim.x * 2]
        a4: Int32 = in_data[idx + block_dim.x * 3]
        a5: Int32 = in_data[idx + block_dim.x * 4]
        a6: Int32 = in_data[idx + block_dim.x * 5]
        a7: Int32 = in_data[idx + block_dim.x * 6]
        a8: Int32 = in_data[idx + block_dim.x * 7]
        in_data[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    barrier()

    alias block_sizes:List[UInt] = [1024, 512, 256, 128, 64]
    @parameter
    for i in range(0, len(block_sizes) - 1):
        var current = materialize[block_sizes[i]]()
        if block_size >= current:
            var next:UInt = materialize[block_sizes[i + 1]]()
            if tid < next:
                i_data[tid] += i_data[tid + next]
            barrier()

    if tid < UInt(WARP_SIZE):
        # Can we make this a SIMD?
        # Note: volatile is not the best method to do this.
        # Better to use higher level API / signalling.
        alias strides = [32, 16, 8, 4, 2, 1]
        @parameter
        for stride in strides:
            i_data.store[volatile=True](
                tid, 
                i_data.load[volatile=True](tid) + i_data.load[volatile=True](tid + UInt(stride))
            )

    if tid == 0:
        out_data[block_idx.x] = i_data[0]


fn check_result(h_C: Int32Ptr, h_GpuCheck: Int32Ptr):
    print(h_C[0], h_GpuCheck[0])
    if h_C[0] != h_GpuCheck[0]:
        print("Error at index", 0, ":", h_C[0], "!=", h_GpuCheck[0])
        print("Test Failed")
        return

    print("\tTest passed")


fn test_function_reduce[
	input_size: Int,  # Change this to be inferrable.
	function: fn (Int32Ptr, Int32Ptr) -> None,
](
    mut ctx: DeviceContext,
    function_name: String,
    h_in_data: HostBuffer[Int32.dtype],
    h_out_data_checked: HostBuffer[Int32.dtype],
    grid: Dim,
    block: Dim,
    output_size: Int = input_size,
    pass_block_size: Bool = False
) raises:
    print("===Begin " + function_name + "===")
    with Timer("\t" + function_name):
        h_out_data = ctx.enqueue_create_host_buffer[Int32.dtype](output_size)
        d_in_data = ctx.enqueue_create_buffer[Int32.dtype](input_size)
        d_out_data = ctx.enqueue_create_buffer[Int32.dtype](output_size)
        ctx.synchronize()

        ctx.enqueue_copy(d_in_data, h_in_data)
        ctx.synchronize()

        ctx.enqueue_function_checked[function, function](
            d_in_data, d_out_data, grid_dim=grid, block_dim=block
        )

        ctx.enqueue_copy(h_out_data, d_out_data)

        with Timer(function_name + "_sync"):
            ctx.synchronize()
            host_accumulate_to_index_0(h_out_data.unsafe_ptr(),n=output_size)

        check_result(
            h_out_data.unsafe_ptr(), h_out_data_checked.unsafe_ptr()
        )
    print("===End " + function_name + "===")


fn test_function_reduce[
	input_size: Int,  # Change this to be inferrable.
	function: fn (Int32Ptr, Int32Ptr, UInt) -> None,
](
    mut ctx: DeviceContext,
    function_name: String,
    h_in_data: HostBuffer[Int32.dtype],
    h_out_data_checked: HostBuffer[Int32.dtype],
    grid: Dim,
    block: Dim,
    output_size: Int = input_size
) raises:
    print("===Begin " + function_name + "===")
    with Timer("\t" + function_name):
        h_out_data = ctx.enqueue_create_host_buffer[Int32.dtype](output_size)
        d_in_data = ctx.enqueue_create_buffer[Int32.dtype](input_size)
        d_out_data = ctx.enqueue_create_buffer[Int32.dtype](output_size)
        ctx.synchronize()

        ctx.enqueue_copy(d_in_data, h_in_data)
        ctx.synchronize()

        ctx.enqueue_function_checked[function, function](
            d_in_data, d_out_data, UInt(block.x()), grid_dim=grid, block_dim=block
        )

        ctx.enqueue_copy(h_out_data, d_out_data)

        with Timer(function_name + "_sync"):
            ctx.synchronize()
            host_accumulate_to_index_0(h_out_data.unsafe_ptr(),n=output_size)

        check_result(
            h_out_data.unsafe_ptr(), h_out_data_checked.unsafe_ptr()
        )
    print("===End " + function_name + "===")


fn main() raises:
    print(sys.argv()[0], "Starting...")
    seed(Int(time.perf_counter_ns()))

    alias dev: Int = 0
    gpu_info: GPUInfo = GPUInfo.from_name[_accelerator_arch()]()
    print("Device ", dev, ":", gpu_info)

    alias nx = 1 << 24
    alias n_bytes = nx * sys.info.size_of[Int32]()
    var dim_x = 512
    if len(sys.argv()) > 1:
        dim_x = Int(sys.argv()[1])
    print("Data size:", nx)
    print("Dimensions: x: ", dim_x, "y:", 1)

    var block: Dim = {dim_x, 1}
    var grid: Dim = {
        (nx + block.x() - 1) // block.x(),
        1,
    }
    # var grid = Dim(1)
    # var block = Dim(1)

    print(
        "Execution Configure (block:",
        block.x(),
        block.y(),
        "grid:",
        grid.x(),
        grid.y(),
        ")",
    )

    with DeviceContext(dev) as ctx:
        with Timer("host_initalization"):
            h_out_data_checked = ctx.enqueue_create_host_buffer[Int32.dtype](nx)
            h_in_data = ctx.enqueue_create_host_buffer[Int32.dtype](nx)
        with Timer("host_sync"):
            ctx.synchronize()

        initialize_data[nx](h_in_data.unsafe_ptr())

        with Timer("host_reduce_neighbored"):
            host_reduce_neighbored[nx](
                h_in_data.unsafe_ptr(), h_out_data_checked._host_ptr
            )

        test_function_reduce[nx, reduce_neighbored[UInt(nx)]](
            ctx,
            "reduce_neighbored",
            h_in_data,
            h_out_data_checked,
            grid=grid,
            block=block,
        )

        test_function_reduce[nx, reduce_neighbored_less[UInt(nx)]](
            ctx,
            "reduce_neighbored_less",
            h_in_data,
            h_out_data_checked,
            grid=grid,
            block=block,
        )

        test_function_reduce[nx, reduce_unrolling_interleaved[UInt(nx)]](
            ctx,
            "reduce_unrolling_interleaved",
            h_in_data,
            h_out_data_checked,
            grid=grid,
            block=block,
        )

        var small_grid:Dim = {Int(Int(grid.x()) / 2), 1}
        # Need to be able to do half nx for some operations.
        test_function_reduce[
            nx,
            reduce_unrolling_2[UInt(nx)]
        ](ctx, 'reduce_unrolling_2' ,h_in_data,h_out_data_checked,
            grid=small_grid,
            block=block,
            output_size=small_grid.x()
        )
        # Need to be able to do half nx for some operations.
        var small_grid_4:Dim = {Int(Int(grid.x()) / 4), 1}
        test_function_reduce[
            nx,
            reduce_unrolling_4[UInt(nx)]
        ](ctx, 'reduce_unrolling_4' ,h_in_data,h_out_data_checked,
            grid=small_grid_4,
            block=block,
            output_size=small_grid_4.x()
        )
        # Need to be able to do half nx for some operations.
        var small_grid_8:Dim = {Int(Int(grid.x()) / 8), 1}
        test_function_reduce[
            nx,
            reduce_unrolling_8[UInt(nx)]
        ](ctx, 'reduce_unrolling_8' ,h_in_data,h_out_data_checked,
            grid=small_grid_8,
            block=block,
            output_size=small_grid_8.x()
        )

        # # Need to be able to do half nx for some operations.
        test_function_reduce[
        	nx,
        	reduce_unrolling_8_last_warp[UInt(nx)]
        ](ctx, 'reduce_unrolling_8_last_warp' ,h_in_data,h_out_data_checked,
            grid=small_grid_8,
            block=block,
            output_size = small_grid_8.x()
        )

        # # Need to be able to do half nx for some operations.
        test_function_reduce[
        	nx,
        	reduce_unrolling_8_unroll_everything[UInt(nx)]
        ](ctx, 'reduce_unrolling_8_unroll_everything' ,h_in_data,h_out_data_checked,
            grid=small_grid_8,
            block=block,
            output_size = small_grid_8.x()
        )

        # # Need to be able to do half nx for some operations.
        test_function_reduce[
        	nx,
        	reduce_unrolling_8_unroll_everything_templated[UInt(nx)]
        ](ctx, 'reduce_unrolling_8_unroll_everything_templated' ,h_in_data,h_out_data_checked,
            grid=small_grid_8,
            block=block,
            output_size = small_grid_8.x()
        )


