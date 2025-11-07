import sys
import time
from random import rand, seed
from gpu import thread_idx, block_idx, block_dim, WARP_SIZE
from gpu.host import DeviceContext, Dim
from gpu.host.info import GPUInfo, _accelerator_arch
from memory import memset
from gpu.random import Random


alias Float32Ptr = UnsafePointer[Float32]


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


fn initialize_data[n_elem: Int](ptr: Float32Ptr):
    var rng = Random(seed=time.perf_counter_ns())
    alias left_over = Int(n_elem % 4)
    var tracker: Int = 0

    for i in range(0, n_elem - left_over, 4):
        var uniform_values = rng.step_uniform()

        @parameter
        for k in range(4):
            ptr[i + k] = uniform_values[k]
            tracker = i + k

    @parameter
    if left_over > 0:
        var uniform_values = rng.step_uniform()

        @parameter
        for k in range(left_over):
            ptr[tracker + k] = uniform_values[k]


fn host_sum_matrix_2d[n: Int](a: Float32Ptr, b: Float32Ptr, c: Float32Ptr):
    for i in range(n):
        c[i] = a[i] + b[i]


fn sum_matrix_2d[
    nx: UInt, ny: UInt
](a: Float32Ptr, b: Float32Ptr, c: Float32Ptr):
	var ix: UInt = block_idx.x * block_dim.x + thread_idx.x
	var iy: UInt = block_idx.y * block_dim.y + thread_idx.y
	var idx = iy * nx + ix

	if ix < nx and iy < ny:
		c[idx] = a[idx] + b[idx]


fn check_result(h_C: Float32Ptr, h_GpuCheck: Float32Ptr, n_elem: Int):
    for i in range(n_elem):
        if abs(h_C[i] - h_GpuCheck[i]) > 1e-5:
            print("Error at index", i, ":", h_C[i], "!=", h_GpuCheck[i])
            print('Test Failed')
            return

    print("Test passed")


fn main() raises:
	print(sys.argv()[0], "Starting...")

	alias dev: Int = 0
	gpu_info: GPUInfo = GPUInfo.from_name[_accelerator_arch()]()
	print("Device ", dev, ":", gpu_info)

	alias nx = 1 << 14
	alias ny = 1 << 14
	alias nxy = nx * ny
	alias n_bytes = nxy * sys.info.size_of[Float32]()
	var dim_x = 16
	var dim_y = 16
	if len(sys.argv()) > 1:
		dim_x = Int(sys.argv()[1])
	if len(sys.argv()) > 2:
		dim_y = Int(sys.argv()[2])
	print("Data size:", nxy)
	print("Dimensions: x: ", dim_x, "y:", dim_y)

	var block: Dim = {dim_x, dim_y}
	var grid: Dim = {
		(nx + block.x() - 1) // block.x(),
		(ny + block.y() - 1) // block.y(),
	}

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
			h_c = ctx.enqueue_create_host_buffer[Float32.dtype](nxy)
			h_c_checked = ctx.enqueue_create_host_buffer[Float32.dtype](nxy)
			h_a = ctx.enqueue_create_host_buffer[Float32.dtype](nxy)
			h_b = ctx.enqueue_create_host_buffer[Float32.dtype](nxy)
		with Timer("host_sync"):
			ctx.synchronize()

		initialize_data[nxy](h_a.unsafe_ptr())
		initialize_data[nxy](h_b.unsafe_ptr())
		with Timer("hostSumMatrix"):
			host_sum_matrix_2d[nxy](
				h_a.unsafe_ptr(), h_b.unsafe_ptr(), h_c_checked.unsafe_ptr()
			)

		with Timer("gpu_initialization"):
			d_a = ctx.enqueue_create_buffer[Float32.dtype](nxy)
			d_b = ctx.enqueue_create_buffer[Float32.dtype](nxy)
			d_c = ctx.enqueue_create_buffer[Float32.dtype](nxy)

			ctx.enqueue_copy(d_a, h_a)
			ctx.enqueue_copy(d_b, h_b)

			ctx.enqueue_function[sum_matrix_2d[UInt(nx), UInt(ny)]](
				d_a, d_b, d_c, grid_dim=grid, block_dim=block
			)
			# This copy isn't happening it looks like.
			ctx.enqueue_copy(h_c, d_c)

		with Timer("synchronize"):
			ctx.synchronize()

		check_result(h_c.unsafe_ptr(), h_c_checked.unsafe_ptr(), n_elem=nxy)
